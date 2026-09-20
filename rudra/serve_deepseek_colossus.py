#!/usr/bin/env python3
"""
serve_deepseek_colossus.py
==========================
Full End-to-End Serving Engine for DeepSeek-Coder-V2 (236B MoE, 160 Experts)
Consolidating an 8x H100 Cluster onto 2x NVIDIA H100 NVL (Hopper SM 9.0)
via COLOSSUS Dynamic Expert Streaming.

Pipeline Partitioning:
- GPU 0 (NVIDIA H100 NVL 94GB):
  - Embeddings (model.embed_tokens)
  - Layer 0: Dense Decoder Layer (MLA + Dense MLP + RMSNorms)
  - Layers 1..29: 29 MoE Layers (MLA + Shared Experts + C=12 Dynamic Slots)
- GPU 1 (NVIDIA H100 NVL 94GB):
  - NVLink P2P Activation Bridge
  - Layers 30..59: 30 MoE Layers (MLA + Shared Experts + C=12 Dynamic Slots)
  - Final RMSNorm (model.norm)
  - LM Head (lm_head)
- Host DDR5 (503 GB RAM / Page Cache):
  - 445 GB cold expert pool accessed via zero-copy mmap safetensors
  - Dynamic PCIe Gen5 DMA streaming for cache misses
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility Polyfills for DeepSeek-V2 with Modern Transformers (v5.x)
# ─────────────────────────────────────────────────────────────────────────────
def get_usable_length(self, *args, **kwargs):
    layer_idx = 0
    if len(args) > 1 and isinstance(args[1], int):
        layer_idx = args[1]
    elif "layer_idx" in kwargs:
        layer_idx = kwargs["layer_idx"]
    return self.get_seq_length(layer_idx)

DynamicCache.get_usable_length = get_usable_length

_orig_to_causal_4d = AttentionMaskConverter.to_causal_4d

def patched_to_causal_4d(self, batch_size, query_length, key_value_length, dtype, device="cpu"):
    mask = _orig_to_causal_4d(self, batch_size, query_length, key_value_length, dtype, device)
    if mask is None:
        mask = torch.zeros((batch_size, 1, query_length, key_value_length), dtype=dtype, device=device)
    return mask

AttentionMaskConverter.to_causal_4d = patched_to_causal_4d

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# 1. P2P NVLink Bridge between Layer 29 (GPU 0) and Layer 30 (GPU 1)
# ─────────────────────────────────────────────────────────────────────────────
class Layer30P2PBridge(nn.Module):
    """Bridges GPU 0 and GPU 1 over NVLink for Layer 30."""
    def __init__(self, inner_layer: nn.Module, target_dev: torch.device):
        super().__init__()
        self.inner_layer = inner_layer
        self.target_dev = target_dev

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if hidden_states.device != self.target_dev:
            hidden_states = hidden_states.to(self.target_dev, non_blocking=True)
        # Also transfer any tensor in args or kwargs if on dev0
        new_args = [
            a.to(self.target_dev, non_blocking=True) if isinstance(a, torch.Tensor) and a.device != self.target_dev else a
            for a in args
        ]
        new_kwargs = {
            k: (v.to(self.target_dev, non_blocking=True) if isinstance(v, torch.Tensor) and v.device != self.target_dev else v)
            for k, v in kwargs.items()
        }
        return self.inner_layer(hidden_states, *new_args, **new_kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner_layer, name)



# Set expandable segments to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────────────────────────────────────
# 2. COLOSSUS Fast Dynamic Expert Slot (Instant HBM Allocation)
# ─────────────────────────────────────────────────────────────────────────────
class FastExpertSlot(nn.Module):
    """Direct HBM-allocated slot module for DeepSeek-V2 MoE expert."""
    def __init__(self, cfg, device: torch.device):
        super().__init__()
        H = cfg.hidden_size
        I = cfg.moe_intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False, device=device, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(H, I, bias=False, device=device, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(I, H, bias=False, device=device, dtype=torch.bfloat16)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekColossusMoEWrapper(nn.Module):
    """
    Dynamic Slot Residency MoE Wrapper with Heterogeneous Compute-to-Data Engine:
    - Permanently resident: Gate, Shared Experts on GPU HBM.
    - Dynamic pool: C dynamic slots allocated on GPU HBM.
    - Cold experts with small token load (M <= cpu_token_threshold):
      Directly evaluated on CPU via AVX-512 in native BF16 without PCIe weight DMA (saving 45 MB per miss).
    - Heavy experts (M > cpu_token_threshold):
      Streamed over PCIe Gen5 DMA into dynamic GPU slots.
    """
    def __init__(
        self,
        layer_idx: int,
        moe_module: nn.Module,
        cfg,
        device: torch.device,
        capacity: int,
        handles: Dict[str, any],
        weight_map: Dict[str, str],
        dma_stream: torch.cuda.Stream,
        enable_hetero: bool = True,
        cpu_token_threshold: int = 1,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.device = device
        self.capacity = capacity
        self.handles = handles
        self.weight_map = weight_map
        self.dma_stream = dma_stream
        self.enable_hetero = enable_hetero
        self.cpu_token_threshold = cpu_token_threshold

        # Resident modules on GPU
        self.gate = moe_module.gate
        self.shared_experts = moe_module.shared_experts

        # Dynamic slots on GPU (allocated directly in HBM)
        self.slots: List[nn.Module] = nn.ModuleList([
            FastExpertSlot(cfg, device=device)
            for _ in range(capacity)
        ])

        self.slot_to_expert: Dict[int, int] = {}
        self.expert_to_slot: Dict[int, int] = {}
        self.slot_lru: List[int] = list(range(capacity))

        # Heterogeneous activation streaming buffers
        self.max_cpu_batch = 64
        self.cpu_act_in = torch.empty((self.max_cpu_batch, cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
        self.cpu_act_out = torch.empty((self.max_cpu_batch, cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
        self.act_stream = torch.cuda.Stream(device=device)
        self.cpu_expert_weights: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Metrics
        self.hits = 0
        self.misses = 0
        self.cpu_dispatches = 0
        self.dma_bytes = 0
        self.expert_freq: Dict[int, int] = {}
        self.hetero_max_tokens = 8
        self.hetero_freq_retain = 3

    def _get_cpu_expert_weights(self, expert_id: int):
        if expert_id not in self.cpu_expert_weights:
            pfx = f"model.layers.{self.layer_idx}.mlp.experts.{expert_id}"
            k_gate = f"{pfx}.gate_proj.weight"
            k_up = f"{pfx}.up_proj.weight"
            k_down = f"{pfx}.down_proj.weight"

            shard_gate = self.weight_map[k_gate]
            shard_up = self.weight_map[k_up]
            shard_down = self.weight_map[k_down]

            t_gate = self.handles[shard_gate].get_tensor(k_gate)
            t_up = self.handles[shard_up].get_tensor(k_up)
            t_down = self.handles[shard_down].get_tensor(k_down)
            self.cpu_expert_weights[expert_id] = (t_gate, t_up, t_down)
        return self.cpu_expert_weights[expert_id]

    def _cpu_expert_exec(self, expert_id: int, toks_gpu: torch.Tensor) -> torch.Tensor:
        """Executes cold expert on CPU in native BF16, streaming activations over PCIe."""
        M = toks_gpu.shape[0]
        if M > self.max_cpu_batch:
            self.max_cpu_batch = M
            self.cpu_act_in = torch.empty((self.max_cpu_batch, self.cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)
            self.cpu_act_out = torch.empty((self.max_cpu_batch, self.cfg.hidden_size), dtype=torch.bfloat16, pin_memory=True)

        with torch.cuda.stream(self.act_stream):
            self.cpu_act_in[:M].copy_(toks_gpu, non_blocking=True)
        self.act_stream.synchronize()

        Wg, Wu, Wd = self._get_cpu_expert_weights(expert_id)
        toks_cpu = self.cpu_act_in[:M]

        with torch.no_grad():
            h_g = F.linear(toks_cpu, Wg)
            h_u = F.linear(toks_cpu, Wu)
            act = F.silu(h_g) * h_u
            y = F.linear(act, Wd)
            self.cpu_act_out[:M].copy_(y)

        out = torch.empty_like(toks_gpu)
        with torch.cuda.stream(self.act_stream):
            out.copy_(self.cpu_act_out[:M], non_blocking=True)
        torch.cuda.current_stream(self.device).wait_stream(self.act_stream)
        return out

    def warm_up_slots(self, initial_experts: List[int]):
        """Pre-populates dynamic slots with initial experts."""
        for slot_idx, exp_id in enumerate(initial_experts[:self.capacity]):
            self._load_expert_to_slot(exp_id, slot_idx)

    def _load_expert_to_slot(self, expert_id: int, slot_idx: int):
        slot_mod = self.slots[slot_idx]
        t_gate, t_up, t_down = self._get_cpu_expert_weights(expert_id)

        with torch.no_grad():
            with torch.cuda.stream(self.dma_stream):
                slot_mod.gate_proj.weight.copy_(t_gate, non_blocking=True)
                slot_mod.up_proj.weight.copy_(t_up, non_blocking=True)
                slot_mod.down_proj.weight.copy_(t_down, non_blocking=True)

        if slot_idx in self.slot_to_expert:
            old_exp = self.slot_to_expert[slot_idx]
            self.expert_to_slot.pop(old_exp, None)

        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.dma_bytes += (t_gate.nbytes + t_up.nbytes + t_down.nbytes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        # 1. Gate routing (returns topk_idx, topk_weight, aux_loss)
        topk_indices, topk_weights, _ = self.gate(hidden_states)
        needed_experts = topk_indices.unique().tolist()
        flat_topk = topk_indices.view(-1)

        # 2. Compute shared experts (permanently resident)
        shared_out = self.shared_experts(identity)

        # 3. Dynamic Slot Management & Expert Streaming
        gpu_hits = set([e for e in needed_experts if e in self.expert_to_slot])
        miss_ids = [e for e in needed_experts if e not in self.expert_to_slot]

        # Protect resident hits in LRU
        for exp_id in gpu_hits:
            self.hits += 1
            slot_idx = self.expert_to_slot[exp_id]
            self.slot_lru.remove(slot_idx)
            self.slot_lru.append(slot_idx)

        cpu_ids = set()
        gpu_misses = []
        # Job 1814 fix: freq-aware retention + prefill guard. Frequent experts (seen>=3)
        # stay on GPU even if M small; prefill (N>8) never goes CPU (fixes 67s prefill).
        n_tokens = flat_topk.numel() // max(1, topk_indices.shape[-1])
        use_hetero = self.enable_hetero and n_tokens <= self.hetero_max_tokens
        for _e in needed_experts:
            self.expert_freq[_e] = self.expert_freq.get(_e, 0) + 1
        if use_hetero:
            for exp_id in miss_ids:
                tok_count = (flat_topk == exp_id).sum().item()
                if tok_count <= self.cpu_token_threshold and self.expert_freq.get(exp_id, 0) < self.hetero_freq_retain:
                    cpu_ids.add(exp_id)
                else:
                    gpu_misses.append(exp_id)
        else:
            gpu_misses = miss_ids

        # Stream heavy misses into GPU slots
        if gpu_misses:
            for exp_id in gpu_misses:
                self.misses += 1
                slot_idx = self.slot_lru.pop(0)
                self._load_expert_to_slot(exp_id, slot_idx)
                self.slot_lru.append(slot_idx)
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)

        # 4. Compute routed experts
        cnts = topk_indices.new_zeros((topk_indices.shape[0], self.cfg.n_routed_experts))
        cnts.scatter_(1, topk_indices, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_indices.view(-1).argsort()
        flat_x = hidden_states.view(-1, hidden_states.shape[-1])
        sorted_tokens = flat_x[idxs // topk_indices.shape[1]]

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue

            tokens_for_this = sorted_tokens[start_idx:end_idx]
            if i in cpu_ids:
                self.cpu_dispatches += 1
                out_this = self._cpu_expert_exec(i, tokens_for_this)
            else:
                if i in self.expert_to_slot:
                    slot_idx = self.expert_to_slot[i]
                else:
                    self.misses += 1
                    slot_idx = self.slot_lru.pop(0)
                    self._load_expert_to_slot(i, slot_idx)
                    self.slot_lru.append(slot_idx)
                    torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)
                out_this = self.slots[slot_idx](tokens_for_this)

            outputs.append(out_this)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_indices.shape, -1)
            .type(topk_weights.dtype)
            .mul_(topk_weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return shared_out + final_out.view(*orig_shape)


# ─────────────────────────────────────────────────────────────────────────────
# 3. End-to-End Serving Engine Execution
# ─────────────────────────────────────────────────────────────────────────────
def serve_deepseek(args):
    print("=" * 80)
    print("  COLOSSUS PRODUCTION SERVING: DeepSeek-Coder-V2 (236B MoE)")
    if args.num_gpus == 1:
        print("  Hardware Target: 1x NVIDIA H100 NVL (Consolidating 8-GPU Cluster onto 1 GPU!)")
    else:
        print("  Hardware Target: 2x NVIDIA H100 NVL (Consolidating 8-GPU Cluster)")
    print("=" * 80)

    assert torch.cuda.is_available(), "CUDA GPU required!"
    if args.num_gpus == 1:
        dev0 = torch.device("cuda:0")
        dev1 = dev0
        p0 = torch.cuda.get_device_properties(dev0)
        p1 = p0
        print(f"  GPU 0: {p0.name} | Total HBM3: {p0.total_memory / (1024**3):.1f} GB")
    else:
        assert torch.cuda.device_count() >= 2, "Dual GPUs required for 2-GPU serving!"
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1")
        p0 = torch.cuda.get_device_properties(dev0)
        p1 = torch.cuda.get_device_properties(dev1)
        print(f"  GPU 0: {p0.name} | Total HBM3: {p0.total_memory / (1024**3):.1f} GB")
        print(f"  GPU 1: {p1.name} | Total HBM3: {p1.total_memory / (1024**3):.1f} GB")
    print(f"  Dynamic Slot Capacity C = {args.capacity} slots per MoE layer")


    # Load Tokenizer & Config
    print(f"\n[1] Loading Tokenizer & Architecture Config from {MODEL_PATH}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  Loaded in {time.time() - t0:.2f}s | Vocab: {len(tokenizer):,} | Layers: {cfg.num_hidden_layers} | Experts: {cfg.n_routed_experts} (top-{cfg.num_experts_per_tok})")

    # Open Safetensors Handles (Direct Zero-Copy mmap)
    print(f"\n[2] Pre-opening Safetensors Shard Handles (Host DDR5 Zero-Copy mmap)...")
    t0 = time.time()
    with open(f"{MODEL_PATH}/model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(list(set(weight_map.values())))
    handles = {s: safe_open(os.path.join(MODEL_PATH, s), framework="pt", device="cpu") for s in shards}
    print(f"  Pre-opened {len(handles)} shards in {time.time() - t0:.2f}s.")

    # Instantiate Meta Model Skeleton
    print(f"\n[3] Instantiating Meta Model Skeleton...")
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
    print(f"  Instantiated 60-layer skeleton in {time.time() - t0:.2f}s.")

    # Load Non-Routed Weights to Respective GPUs
    print(f"\n[4] Materializing Non-Routed Weights (~25 GB) across Dual H100 NVL...")
    t0 = time.time()
    non_routed_keys = {k: v for k, v in weight_map.items() if not (".mlp.experts." in k and ".shared_experts" not in k)}

    for shard_file in sorted(list(set(non_routed_keys.values()))):
        handle = handles[shard_file]
        for k in handle.keys():
            if k not in non_routed_keys:
                continue
            if k.startswith("model.embed_tokens."):
                target_dev = dev0
            elif k.startswith("model.layers."):
                l_idx = int(k.split(".")[2])
                target_dev = dev0 if (args.num_gpus == 1 or l_idx < 30) else dev1
            else:
                target_dev = dev0 if args.num_gpus == 1 else dev1
            t = handle.get_tensor(k)
            set_module_tensor_to_device(model, k, target_dev, value=t.to(torch.bfloat16))

    torch.cuda.synchronize(dev0)
    if args.num_gpus > 1:
        torch.cuda.synchronize(dev1)
    print(f"  Non-routed parameters materialized in {time.time() - t0:.2f}s.")
    print(f"    GPU 0 Allocated: {torch.cuda.memory_allocated(dev0) / (1024**3):.2f} GB")
    if args.num_gpus > 1:
        print(f"    GPU 1 Allocated: {torch.cuda.memory_allocated(dev1) / (1024**3):.2f} GB")

    # Re-initialize RoPE rotary embeddings on target devices to eliminate meta buffers
    print(f"\n[4.5] Initializing RoPE Rotary Embeddings on Target GPUs...")
    for l_idx in range(cfg.num_hidden_layers):
        layer = model.model.layers[l_idx]
        target_dev = dev0 if (args.num_gpus == 1 or l_idx < 30) else dev1
        layer.self_attn._init_rope()
        layer.self_attn.rotary_emb.to(target_dev)

    # Install COLOSSUS Dynamic Slot Wrappers for Layers 1..59
    print(f"\n[5] Installing COLOSSUS Dynamic Slot Wrappers (Layers 1..59)...")
    t0 = time.time()
    dma_stream0 = torch.cuda.Stream(device=dev0)
    dma_stream1 = dma_stream0 if args.num_gpus == 1 else torch.cuda.Stream(device=dev1)
    colossus_wrappers: List[DeepSeekColossusMoEWrapper] = []

    for l_idx in range(1, cfg.num_hidden_layers):
        layer = model.model.layers[l_idx]
        dev = dev0 if (args.num_gpus == 1 or l_idx < 30) else dev1
        dma_stream = dma_stream0 if (args.num_gpus == 1 or l_idx < 30) else dma_stream1
        cap = args.capacity if dev == dev0 else min(args.capacity, args.capacity_gpu1)

        wrapper = DeepSeekColossusMoEWrapper(
            layer_idx=l_idx,
            moe_module=layer.mlp,
            cfg=cfg,
            device=dev,
            capacity=cap,
            handles=handles,
            weight_map=weight_map,
            dma_stream=dma_stream,
            enable_hetero=args.enable_hetero,
            cpu_token_threshold=args.cpu_token_threshold,
        )

        if args.warm_slots:
            # Pre-warm slots with first C experts
            wrapper.warm_up_slots(list(range(cap)))

        layer.mlp = wrapper
        colossus_wrappers.append(wrapper)

    torch.cuda.synchronize(dev0)
    if args.num_gpus > 1:
        torch.cuda.synchronize(dev1)
    print(f"  Installed {len(colossus_wrappers)} COLOSSUS wrappers in {time.time() - t0:.2f}s.")
    print(f"    GPU 0 Allocated (with C={args.capacity} slots): {torch.cuda.memory_allocated(dev0) / (1024**3):.2f} GB")
    if args.num_gpus > 1:
        print(f"    GPU 1 Allocated (with C={args.capacity_gpu1} slots): {torch.cuda.memory_allocated(dev1) / (1024**3):.2f} GB")

    if args.num_gpus > 1:
        # Install P2P Bridge on all layers 30..59 so hidden_states, position_ids, attention_mask are on dev1
        print(f"\n[6] Installing NVLink P2P Hooks on Layers 30..59...")
        def gpu1_pre_hook(module, args, kwargs):
            new_args = [
                a.to(dev1, non_blocking=True) if isinstance(a, torch.Tensor) and a.device != dev1 else a
                for a in args
            ]
            new_kwargs = {
                k: (v.to(dev1, non_blocking=True) if isinstance(v, torch.Tensor) and v.device != dev1 else v)
                for k, v in kwargs.items()
            }
            return tuple(new_args), new_kwargs

        for l_idx in range(30, cfg.num_hidden_layers):
            model.model.layers[l_idx].register_forward_pre_hook(gpu1_pre_hook, with_kwargs=True)
    else:
        print(f"\n[6] Single GPU serving on {p0.name} (No cross-GPU NVLink hooks needed)")
    model.eval()



    # ─────────────────────────────────────────────────────────────────────────
    # Run End-to-End Generation Benchmark
    # ─────────────────────────────────────────────────────────────────────────
    prompt = args.prompt
    print(f"\n[7] Starting Generation Benchmark:")
    print(f"  Prompt: {repr(prompt)}")
    print(f"  Max New Tokens: {args.max_new_tokens}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(dev0)
    prompt_len = input_ids.shape[1]
    print(f"  Prompt Length: {prompt_len} tokens")

    # Reset cache metrics before generation
    for w in colossus_wrappers:
        w.hits = 0
        w.misses = 0
        w.cpu_dispatches = 0
        w.dma_bytes = 0

    generated_ids = input_ids.clone()

    # 1. Prefill Phase
    print(f"\n  --- Prefill Phase ({prompt_len} tokens) ---")
    torch.cuda.synchronize(dev0)
    torch.cuda.synchronize(dev1)
    t_prefill_start = time.perf_counter()

    from transformers.cache_utils import DynamicCache
    past_key_values = DynamicCache() if args.use_cache else None

    with torch.no_grad():
        out = model(input_ids=generated_ids, past_key_values=past_key_values, use_cache=args.use_cache)
        logits = out.logits  # [1, prompt_len, vocab_size] on dev1
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True).to(dev0)
        past_key_values = getattr(out, "past_key_values", None)


    torch.cuda.synchronize(dev0)
    torch.cuda.synchronize(dev1)
    t_prefill = time.perf_counter() - t_prefill_start
    prefill_tps = prompt_len / t_prefill
    print(f"  Prefill Time : {t_prefill*1000:.2f} ms ({prefill_tps:.2f} tok/s)")
    sys.stdout.flush()

    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # 2. Decode Phase (Token-by-Token)
    print(f"\n  --- Decode Phase ({args.max_new_tokens - 1} tokens) [KV-Cache: {args.use_cache}] ---")
    sys.stdout.flush()
    decode_latencies = []
    decode_step_details = []

    last_hits = sum(w.hits for w in colossus_wrappers)
    last_misses = sum(w.misses for w in colossus_wrappers)
    last_cpu = sum(w.cpu_dispatches for w in colossus_wrappers)
    last_dma = sum(w.dma_bytes for w in colossus_wrappers)

    for step in range(args.max_new_tokens - 1):
        torch.cuda.synchronize(dev0)
        torch.cuda.synchronize(dev1)
        t_step_start = time.perf_counter()

        with torch.no_grad():
            if args.use_cache and past_key_values is not None:
                out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            else:
                out = model(input_ids=generated_ids, use_cache=False)
            logits = out.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True).to(dev0)
            if args.use_cache:
                past_key_values = getattr(out, "past_key_values", None)

        torch.cuda.synchronize(dev0)
        torch.cuda.synchronize(dev1)
        t_step = time.perf_counter() - t_step_start
        decode_latencies.append(t_step)

        # Per-step cache metrics
        cur_hits = sum(w.hits for w in colossus_wrappers)
        cur_misses = sum(w.misses for w in colossus_wrappers)
        cur_cpu = sum(w.cpu_dispatches for w in colossus_wrappers)
        cur_dma = sum(w.dma_bytes for w in colossus_wrappers)

        step_hits = cur_hits - last_hits
        step_misses = cur_misses - last_misses
        step_cpu = cur_cpu - last_cpu
        step_lookups = step_hits + step_misses + step_cpu
        step_hit_rate = (step_hits / step_lookups * 100.0) if step_lookups > 0 else 0.0
        step_dma_mb = (cur_dma - last_dma) / (1024**2)
        step_dma_saved_mb = step_cpu * 45.0

        last_hits, last_misses, last_cpu, last_dma = cur_hits, cur_misses, cur_cpu, cur_dma

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tok_str = tokenizer.decode(next_token[0], skip_special_tokens=False)
        print(f"    Token {step+1:2d}/{args.max_new_tokens-1:2d} | Latency: {t_step*1000:6.1f} ms | Hits: {step_hits:3d}/{step_lookups:3d} ({step_hit_rate:5.1f}%) | CPU Fallback: {step_cpu:2d} ({step_dma_saved_mb:5.1f}MB saved) | Misses: {step_misses:2d} | DMA: {step_dma_mb:5.1f} MB | Tok: {repr(tok_str)}")
        sys.stdout.flush()

        decode_step_details.append({
            "step": step + 1,
            "latency_ms": t_step * 1000,
            "step_hits": step_hits,
            "step_misses": step_misses,
            "step_cpu_dispatches": step_cpu,
            "step_dma_saved_mb": step_dma_saved_mb,
            "step_hit_rate_pct": step_hit_rate,
            "step_dma_mb": step_dma_mb,
            "token": tok_str,
        })




    # ─────────────────────────────────────────────────────────────────────────
    # Summary Metrics
    # ─────────────────────────────────────────────────────────────────────────
    total_decode_time = sum(decode_latencies)
    avg_decode_lat = total_decode_time / len(decode_latencies) if decode_latencies else 0.0
    decode_tps = 1.0 / avg_decode_lat if avg_decode_lat > 0 else 0.0

    total_hits = sum(w.hits for w in colossus_wrappers)
    total_misses = sum(w.misses for w in colossus_wrappers)
    total_cpu = sum(w.cpu_dispatches for w in colossus_wrappers)
    total_lookups = total_hits + total_misses + total_cpu
    hit_rate = (total_hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
    total_dma_mb = sum(w.dma_bytes for w in colossus_wrappers) / (1024**2)
    total_dma_saved_mb = total_cpu * 45.0

    peak_hbm0 = torch.cuda.max_memory_allocated(dev0) / (1024**3)
    peak_hbm1 = torch.cuda.max_memory_allocated(dev1) / (1024**3)

    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 80)
    print("  COLOSSUS SERVING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Model                     : DeepSeek-Coder-V2-Instruct (236B MoE)")
    print(f"  Uncompressed Weights      : 471.5 GB BF16")
    if args.num_gpus == 1:
        print(f"  Hardware Footprint        : 1x NVIDIA H100 NVL (Consolidating 8-GPU Cluster onto 1 GPU!)")
    else:
        print(f"  Hardware Footprint        : 2x NVIDIA H100 NVL (Consolidating 8-GPU Cluster)")
    print(f"  Dynamic Slot Capacity C   : {args.capacity} slots / layer ({(args.capacity / cfg.n_routed_experts)*100:.1f}% expert residency)")
    print(f"  Routed Expert Reduction   : {(1.0 - args.capacity / cfg.n_routed_experts)*100:.1f}% reduction in routed expert GPU memory")
    print(f"  Compute-to-Data Engine    : {'Enabled (Threshold <= ' + str(args.cpu_token_threshold) + ')' if args.enable_hetero else 'Disabled'}")
    print("-" * 80)
    print(f"  Prefill Latency           : {t_prefill*1000:7.1f} ms ({prefill_tps:5.1f} tok/s for {prompt_len} tokens)")
    print(f"  Decode Latency (Avg)      : {avg_decode_lat*1000:7.1f} ms / token")
    print(f"  Decode Throughput         : {decode_tps:7.2f} tokens / sec")
    print(f"  Cache Hits                : {total_hits:,} ({hit_rate:.1f}%)")
    print(f"  CPU Cold Fallbacks        : {total_cpu:,} ({total_dma_saved_mb:,.1f} MB PCIe DMA avoided!)")
    print(f"  Cache Misses (Cold DMA)   : {total_misses:,}")
    print(f"  Total PCIe DMA Transferred: {total_dma_mb:7.1f} MB")
    print(f"  Peak VRAM GPU 0           : {peak_hbm0:7.2f} GB / 93.1 GB")
    if args.num_gpus > 1:
        print(f"  Peak VRAM GPU 1           : {peak_hbm1:7.2f} GB / 93.1 GB")
    print("-" * 80)
    print(f"  Generated Text Output:")
    print(f"  {repr(gen_text)}")
    print("=" * 80)

    # Save to JSON
    results = {
        "model": "DeepSeek-Coder-V2-Instruct",
        "parameters": "236B",
        "routed_experts": 160,
        "active_experts": 6,
        "shared_experts": 2,
        "num_gpus": args.num_gpus,
        "gpus": [p0.name] if args.num_gpus == 1 else [p0.name, p1.name],
        "capacity_slots": args.capacity,
        "residency_reduction_pct": (1.0 - args.capacity / cfg.n_routed_experts) * 100.0,
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "generated_tokens": args.max_new_tokens,
        "prefill_ms": t_prefill * 1000,
        "prefill_tps": prefill_tps,
        "decode_avg_ms": avg_decode_lat * 1000,
        "decode_tps": decode_tps,
        "decode_latencies_ms": [l * 1000 for l in decode_latencies],
        "decode_step_details": decode_step_details,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_cpu_dispatches": total_cpu,
        "total_dma_saved_mb": total_dma_saved_mb,
        "hit_rate_pct": hit_rate,
        "total_dma_mb": total_dma_mb,
        "enable_hetero": args.enable_hetero,
        "cpu_token_threshold": args.cpu_token_threshold,
        "peak_vram_gpu0_gb": peak_hbm0,
        "peak_vram_gpu1_gb": peak_hbm1 if args.num_gpus > 1 else None,
        "generated_text": gen_text,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved benchmark results to: {args.output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="def quicksort(arr):")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--capacity_gpu1", type=int, default=7)
    parser.add_argument("--warm_slots", action="store_true", default=False)
    parser.add_argument("--enable_hetero", action="store_true", default=True)
    parser.add_argument("--no_hetero", dest="enable_hetero", action="store_false")
    parser.add_argument("--cpu_token_threshold", type=int, default=1)

    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--no_cache", dest="use_cache", action="store_false")
    parser.add_argument("--output_json", type=str, default="/home/palakm/MoEServingSim/aditya/deepseek_serving_colossus.json")
    args = parser.parse_args()


    serve_deepseek(args)
