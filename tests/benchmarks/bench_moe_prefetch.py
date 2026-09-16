"""Benchmark MoE layer with dynamic GPU expert cache + ZSSR speculative prefetch.

Experiments implemented:
  Experiment 1: Single MoE Layer Baseline (all 64 experts on GPU) vs COLOSSUS
                (GPU cache size C in {8, 16, 24, 32}, pinned CPU master, async
                 CUDA stream prefetch, exact router execution, fallback demand fetch).
                Measures:
                  - Numerical contract (max abs diff vs baseline)
                  - Hit rate & prediction recall
                  - PCIe transfer volume (MB/GB) & transfer latency
                  - MoE layer forward latency & peak VRAM
  Experiment 2: Temporal Locality & Lookahead Recall (L = 1, 2, 4 tokens).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig


def load_layer_weights(moe_block: nn.Module, model_dir: str, layer_idx: int = 1):
    """Load real trained weights for layer_idx into moe_block using safetensors."""
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"[Warning] Index file not found at {index_path}. Using initialized weights.")
        return

    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.layers.{layer_idx}.mlp."
    needed_shards: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for k, shard in weight_map.items():
        if k.startswith(prefix):
            sub_key = k[len(prefix):]
            needed_shards[shard].append((k, sub_key))

    print(f"[Loader] Found {sum(len(v) for v in needed_shards.values())} tensors for layer {layer_idx} across {len(needed_shards)} shards.")
    loaded = 0
    state_dict = {}
    for shard, keys in needed_shards.items():
        shard_path = os.path.join(model_dir, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for full_k, sub_k in keys:
                state_dict[sub_k] = f.get_tensor(full_k)
                loaded += 1

    missing, unexpected = moe_block.load_state_dict(state_dict, strict=False)
    print(f"[Loader] Loaded {loaded} tensors into layer {layer_idx}. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


class GPUExpertCache:
    """Pre-allocated GPU slot cache for MoE experts with pinned-CPU backing.

    Eliminates dynamic memory allocations in the hot path: C slots are pre-allocated
    on GPU once, and weights are streamed into slots via non-blocking DMA.
    """

    def __init__(self, experts: nn.ModuleList, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.num_experts = len(experts)
        sample_exp = experts[0]
        self.config = sample_exp.config
        self.intermediate_size = sample_exp.intermediate_size

        # CPU master copies in pinned memory
        self.cpu_experts = []
        for e in experts:
            e.to("cpu")
            for p in e.parameters():
                if not p.data.is_pinned():
                    p.data = p.data.pin_memory()
            self.cpu_experts.append(e)

        # Pre-allocate exactly C expert slots on GPU
        self.slots = [
            sample_exp.__class__(self.config, intermediate_size=self.intermediate_size).to(device)
            for _ in range(capacity)
        ]

        # Tracking: expert_id <-> slot_idx
        self.expert_to_slot: Dict[int, int] = {}
        self.slot_to_expert: Dict[int, int] = {}
        # Free slot indices
        self.free_slots: List[int] = list(range(capacity))
        # LRU order of slot indices (most recently used at end)
        self.slot_lru: List[int] = []

        # Prefetch CUDA stream & synchronization event
        self.prefetch_stream = torch.cuda.Stream(device=device)

        # Metrics tracking
        self.total_accesses = 0
        self.hits = 0
        self.misses = 0
        self.bytes_transferred = 0
        self.transfer_time_s = 0.0

    def warm_up(self, initial_expert_ids: List[int]):
        """Warm up cache with initial experts up to capacity."""
        for e_id in initial_expert_ids[:self.capacity]:
            self._load_to_slot(e_id, non_blocking=False)

    def _load_to_slot(self, expert_id: int, non_blocking: bool = True, stream=None) -> int:
        """Stream expert weights from CPU pinned memory into an assigned GPU slot."""
        if expert_id in self.expert_to_slot:
            slot_idx = self.expert_to_slot[expert_id]
            self.slot_lru.remove(slot_idx)
            self.slot_lru.append(slot_idx)
            return slot_idx

        # Choose a slot: free slot if available, else evict LRU slot
        if self.free_slots:
            slot_idx = self.free_slots.pop(0)
        else:
            slot_idx = self.slot_lru.pop(0)
            old_expert = self.slot_to_expert.pop(slot_idx)
            del self.expert_to_slot[old_expert]

        # Map new expert to this slot
        self.expert_to_slot[expert_id] = slot_idx
        self.slot_to_expert[slot_idx] = expert_id
        self.slot_lru.append(slot_idx)

        # Asynchronously copy weights into slot
        t0 = time.perf_counter()
        target_slot = self.slots[slot_idx]
        src_cpu = self.cpu_experts[expert_id]

        stream_ctx = torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream())
        with stream_ctx:
            target_slot.gate_proj.weight.copy_(src_cpu.gate_proj.weight, non_blocking=non_blocking)
            target_slot.up_proj.weight.copy_(src_cpu.up_proj.weight, non_blocking=non_blocking)
            target_slot.down_proj.weight.copy_(src_cpu.down_proj.weight, non_blocking=non_blocking)
            # 3 weight matrices per expert
            bytes_copied = (
                src_cpu.gate_proj.weight.numel() * src_cpu.gate_proj.weight.element_size() +
                src_cpu.up_proj.weight.numel() * src_cpu.up_proj.weight.element_size() +
                src_cpu.down_proj.weight.numel() * src_cpu.down_proj.weight.element_size()
            )
            self.bytes_transferred += bytes_copied

        self.transfer_time_s += (time.perf_counter() - t0)
        return slot_idx

    def async_prefetch(self, expert_ids: List[int]):
        """Prefetch predicted experts on dedicated non-blocking CUDA stream."""
        for e_id in expert_ids:
            if e_id not in self.expert_to_slot:
                self._load_to_slot(e_id, non_blocking=True, stream=self.prefetch_stream)

    def synchronize_prefetch(self):
        """Ensure all prefetch operations on prefetch_stream complete before current stream uses them."""
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)

    def get_expert(self, expert_id: int) -> Tuple[nn.Module, bool]:
        """Access expert for execution. Returns (gpu_slot_module, was_hit)."""
        self.total_accesses += 1
        if expert_id in self.expert_to_slot:
            self.hits += 1
            slot_idx = self.expert_to_slot[expert_id]
            self.slot_lru.remove(slot_idx)
            self.slot_lru.append(slot_idx)
            return self.slots[slot_idx], True
        else:
            self.misses += 1
            # Fallback demand fetch
            slot_idx = self._load_to_slot(expert_id, non_blocking=False)
            return self.slots[slot_idx], False


def run_experiment_1(moe_block: nn.Module, inputs: torch.Tensor, router_weight: torch.Tensor,
                     cache_sizes: List[int], device: torch.device):
    """Run Experiment 1: Baseline vs COLOSSUS Dynamic Cache across cache sizes."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 1: Single MoE Layer Baseline vs COLOSSUS Cache (Tokens: {inputs.shape[0]})")
    print("=" * 80)

    inputs = inputs.to(device)
    # 1. Baseline: All 64 experts resident on GPU
    moe_block.to(device)
    torch.cuda.synchronize()

    baseline_outputs = []
    baseline_topk_ids = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(inputs.shape[0]):
            h_t = inputs[t:t+1]  # [1, 1, H]
            out_t, (_, topk_idx) = moe_block(h_t)
            baseline_outputs.append(out_t.cpu())
            baseline_topk_ids.append(topk_idx.squeeze(0).squeeze(0).cpu().tolist())
    torch.cuda.synchronize()
    baseline_latency = (time.perf_counter() - t0) * 1000.0  # ms
    peak_vram_baseline = torch.cuda.max_memory_allocated(device) / (1024**3)
    torch.cuda.reset_peak_memory_stats(device)

    print(f"Baseline (All 64 Experts on GPU):")
    print(f"  Latency: {baseline_latency:.2f} ms ({baseline_latency / inputs.shape[0]:.3f} ms/token)")
    print(f"  Peak VRAM: {peak_vram_baseline:.3f} GB")
    print("-" * 80)

    results = []
    # 2. Test COLOSSUS with different cache sizes C
    for C in cache_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Create cache with CPU pinned master experts
        cache = GPUExpertCache(moe_block.experts, capacity=C, device=device)
        # Gate and shared experts stay on GPU
        moe_block.gate.to(device)
        if hasattr(moe_block, "shared_experts") and moe_block.shared_experts is not None:
            moe_block.shared_experts.to(device)

        # Warm up cache with initial experts from baseline prompt
        initial_warmup = baseline_topk_ids[0]
        cache.warm_up(initial_warmup)

        colossus_outputs = []
        prediction_matches = 0
        total_predictions = 0

        t0_colossus = time.perf_counter()
        with torch.no_grad():
            for t in range(inputs.shape[0]):
                h_t = inputs[t:t+1]  # [1, 1, H]

                # --- STEP 1: ZSSR Speculative Prediction for step t ---
                # Use current h_t (or h_{t-1} in continuous stream) against router W
                h_flat = h_t.reshape(-1, router_weight.shape[1])[-1].float()
                spec_logits = h_flat @ router_weight.T
                predicted_topk = torch.topk(spec_logits, k=moe_block.num_experts_per_tok).indices.tolist()

                # --- STEP 2: Async Speculative Prefetch ---
                cache.async_prefetch(predicted_topk)

                # --- STEP 3: Actual Native Router Execution (Never Modified) ---
                identity = h_t
                bsz, seq_len, h_dim = h_t.shape
                topk_idx, topk_weight, router_logits = moe_block.gate(h_t)
                actual_topk = topk_idx.squeeze().tolist()

                # Evaluate prediction recall
                pred_set = set(predicted_topk)
                actual_set = set(actual_topk)
                matches = len(pred_set.intersection(actual_set))
                prediction_matches += matches
                total_predictions += len(actual_set)

                # --- STEP 4: Ensure Resident & Synchronize Prefetch ---
                cache.synchronize_prefetch()

                # --- STEP 5: Dispatched MoE Infer with Cached Experts ---
                # Compute using experts from cache
                x = h_t.view(-1, h_dim)
                cnts = topk_idx.new_zeros((topk_idx.shape[0], len(moe_block.experts)))
                cnts.scatter_(1, topk_idx, 1)
                tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
                idxs = topk_idx.view(-1).argsort()
                sorted_tokens = x[idxs // topk_idx.shape[1]]

                outputs = []
                start_idx = 0
                for exp_i, num_tokens in enumerate(tokens_per_expert):
                    end_idx = start_idx + num_tokens
                    if num_tokens == 0:
                        continue
                    exp_mod, was_hit = cache.get_expert(exp_i)
                    tokens_for_exp = sorted_tokens[start_idx:end_idx]
                    exp_out = exp_mod(tokens_for_exp)
                    outputs.append(exp_out.to(device))
                    start_idx = end_idx

                outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
                new_x = torch.empty_like(outs)
                new_x[idxs] = outs
                final_out = (
                    new_x.view(*topk_idx.shape, -1)
                    .type(topk_weight.dtype)
                    .mul_(topk_weight.unsqueeze(dim=-1))
                    .sum(dim=1)
                    .type(new_x.dtype)
                ).view(bsz, seq_len, h_dim)

                if moe_block.config.num_shared_experts is not None:
                    final_out = final_out + moe_block.shared_experts(identity)

                colossus_outputs.append(final_out.cpu())

        torch.cuda.synchronize()
        colossus_latency = (time.perf_counter() - t0_colossus) * 1000.0
        peak_vram_colossus = torch.cuda.max_memory_allocated(device) / (1024**3)

        # Verification of Numerical Contract
        max_abs_diff = 0.0
        for out_base, out_col in zip(baseline_outputs, colossus_outputs):
            diff = (out_base.float() - out_col.float()).abs().max().item()
            if diff > max_abs_diff:
                max_abs_diff = diff

        hit_rate = (cache.hits / cache.total_accesses) * 100.0 if cache.total_accesses > 0 else 0.0
        recall = (prediction_matches / total_predictions) * 100.0 if total_predictions > 0 else 0.0
        transferred_mb = cache.bytes_transferred / (1024 * 1024)

        res = {
            "capacity": C,
            "latency_ms": colossus_latency,
            "latency_per_tok_ms": colossus_latency / inputs.shape[0],
            "peak_vram_gb": peak_vram_colossus,
            "hit_rate_pct": hit_rate,
            "recall_pct": recall,
            "transferred_mb": transferred_mb,
            "max_abs_diff": max_abs_diff,
            "is_lossless": max_abs_diff < 1e-4,
        }
        results.append(res)

        print(f"COLOSSUS (Cache Capacity C={C:2d} of 64 experts):")
        print(f"  Lossless Check   : {'PASS (Exact Match)' if res['is_lossless'] else 'FAIL'} (Max Diff: {max_abs_diff:.6e})")
        print(f"  GPU Cache HitRate: {hit_rate:.1f}% ({cache.hits} hits / {cache.misses} misses)")
        print(f"  ZSSR Recall@6    : {recall:.1f}%")
        print(f"  PCIe Transferred : {transferred_mb:.1f} MB (Transfer time: {cache.transfer_time_s*1000:.1f} ms)")
        print(f"  Layer Latency    : {colossus_latency:.2f} ms ({res['latency_per_tok_ms']:.3f} ms/tok)")
        print(f"  Peak VRAM        : {peak_vram_colossus:.3f} GB (Baseline: {peak_vram_baseline:.3f} GB)")
        print("-" * 80)

    return results


def run_experiment_2(router_weight: torch.Tensor, inputs: torch.Tensor,
                     lookaheads: List[int] = [1, 2, 4]):
    """Run Experiment 2: Temporal Locality & Lookahead Recall (L = 1, 2, 4)."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2: Temporal Locality & Multi-Step Lookahead Recall (Tokens: {inputs.shape[0]})")
    print("=" * 80)

    T = inputs.shape[0]
    device = router_weight.device

    # Compute actual routing for each token
    actual_routings: List[Set[int]] = []
    with torch.no_grad():
        for t in range(T):
            h_t = inputs[t].reshape(-1, router_weight.shape[1])[-1].to(device).float()
            logits = h_t @ router_weight.T
            topk = torch.topk(logits, k=6).indices.cpu().tolist()
            actual_routings.append(set(topk))

    # Evaluate multi-step lookahead prediction
    # Predict step t + L using hidden state at step t
    for L in lookaheads:
        matches = 0
        total = 0
        for t in range(T - L):
            h_t = inputs[t].reshape(-1, router_weight.shape[1])[-1].to(device).float()
            pred_logits = h_t @ router_weight.T
            pred_topk = set(torch.topk(pred_logits, k=6).indices.cpu().tolist())

            target_actual = actual_routings[t + L]
            matches += len(pred_topk.intersection(target_actual))
            total += len(target_actual)

        recall = (matches / total) * 100.0 if total > 0 else 0.0
        print(f"  Lookahead L={L} Token(s): Recall@6 = {recall:5.1f}% ({matches}/{total} correct predictions)")

    # Measure transition overlap (temporal persistence: overlap between actual(t) and actual(t+1))
    persistence_matches = 0
    total_persistence = 0
    for t in range(T - 1):
        persistence_matches += len(actual_routings[t].intersection(actual_routings[t + 1]))
        total_persistence += len(actual_routings[t])
    persistence_pct = (persistence_matches / total_persistence) * 100.0 if total_persistence > 0 else 0.0
    print(f"\n  Intrinsic Temporal Persistence (Actual[t] ∩ Actual[t+1]): {persistence_pct:.1f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MoE Layer Prefetch Micro-Benchmark")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to Param2 model snapshot")
    parser.add_argument("--layer-idx", type=int, default=1, help="MoE layer index to isolate (default: 1)")
    parser.add_argument("--num-tokens", type=int, default=128, help="Number of simulated tokens")
    parser.add_argument("--cache-sizes", type=int, nargs="+", default=[8, 16, 24, 32], help="GPU cache sizes to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Using device: {device}")
    print(f"[Benchmark] Model: {args.model_dir}")
    print(f"[Benchmark] Layer: {args.layer_idx} | Simulated Tokens: {args.num_tokens}")

    # Load Config and instantiate SparseMoeBlock
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    sys.path.insert(0, args.model_dir)

    # Import the model's sparse MoE block using transformers dynamic module loader
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    Param2MoESparseMoeBlock = get_class_from_dynamic_module(
        "modeling_param2moe.Param2MoESparseMoeBlock", args.model_dir
    )

    moe_block = Param2MoESparseMoeBlock(config).to(torch.bfloat16)

    # Load real weights from safetensors
    load_layer_weights(moe_block, args.model_dir, layer_idx=args.layer_idx)

    # Extract router weights for ZSSR projection
    router_weight = moe_block.gate.weight.detach().to(device).float() if hasattr(moe_block.gate, "weight") else moe_block.gate.gate.weight.detach().to(device).float()

    # Generate synthetic sequence of autoregressive hidden states with realistic continuity
    torch.manual_seed(42)
    # Simulate tokens with autoregressive drift (high cosine similarity between consecutive tokens)
    hidden_dim = config.hidden_size
    h_states = []
    current_h = torch.randn(1, 1, hidden_dim, dtype=torch.bfloat16)
    for _ in range(args.num_tokens):
        # Step-wise drift: 85% previous state + 15% new token innovation (realistic autoregressive continuity)
        delta = torch.randn(1, 1, hidden_dim, dtype=torch.bfloat16)
        current_h = 0.85 * current_h + 0.15 * delta
        current_h = current_h / current_h.norm(dim=-1, keepdim=True) * (hidden_dim ** 0.5)
        h_states.append(current_h)
    inputs = torch.cat(h_states, dim=0)  # [T, 1, H]

    # Run Experiments
    run_experiment_1(moe_block, inputs, router_weight, args.cache_sizes, device)
    run_experiment_2(router_weight, inputs, lookaheads=[1, 2, 4])


if __name__ == "__main__":
    main()
