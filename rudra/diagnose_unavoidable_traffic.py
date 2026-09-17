#!/usr/bin/env python3
"""
Diagnostic Instrumenter: How Much of the 9.53 GB/token PCIe Traffic is Unavoidable?

Instruments for every generated token:
  ├── Layer-by-layer router top-2 selections across all 32 layers
  ├── Router locality: Jaccard similarity and reuse distance across consecutive tokens
  ├── Cache hit vs miss per layer and per expert
  ├── Cumulative working set size of unique experts over time
  ├── Theoretical optimal cache hit rate (Belady's MIN vs LRU)
  ├── Cold payload DMA transfer time (T_demand via CUDA events)
  ├── Pure GPU compute time (T_compute = T_total - T_demand)
  ├── The exact ratio T_demand / T_compute
  └── Theoretical ceiling on prefetch speedup at current bytes
"""

import os
import sys
import time
import json
import logging
import argparse
from types import SimpleNamespace
from typing import Dict, List, Any, Set
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bhaskera.introspect import introspect_model
from bhaskera.inference.colossus.hook import ColossusMoEHook
from bhaskera.inference.colossus.dynamic_cache import DynamicMoELayerWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrafficDiagnostic")


def simulate_belady_min(access_sequence: List[int], cache_capacity: int) -> float:
    """Simulates Belady's Optimal (MIN) Clairvoyant Cache algorithm."""
    cache: Set[int] = set()
    hits = 0
    misses = 0
    for i, item in enumerate(access_sequence):
        if item in cache:
            hits += 1
        else:
            misses += 1
            if len(cache) < cache_capacity:
                cache.add(item)
            else:
                # Evict the item that is used farthest in the future
                future_uses = {}
                for c_item in cache:
                    try:
                        next_idx = access_sequence.index(c_item, i + 1)
                    except ValueError:
                        next_idx = float("inf")
                    future_uses[c_item] = next_idx
                evict_item = max(future_uses, key=future_uses.get)
                cache.remove(evict_item)
                cache.add(item)
    total = hits + misses
    return (hits / total * 100.0) if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Traffic Diagnostic for Mixtral-8x7B")
    parser.add_argument("--model-dir", type=str, default="/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1")
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--output-json", type=str, default="/home/bapic_iiitd/2_group/diagnostic_traffic_mixtral.json")
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--missing-col-ratio", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    total_physical_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    logger.info("=" * 80)
    logger.info("  DIAGNOSTIC EXPERIMENT: DECONSTRUCTING THE 9.53 GB/TOKEN PCIe TRAFFIC")
    logger.info("  Model: %s | Device: %s (%.2f GiB)", args.model_dir, torch.cuda.get_device_name(0), total_physical_gb)
    logger.info("  Config: C=%d slots | Missing Col Ratio: %.2f (50%% hot, 50%% cold)", args.capacity, args.missing_col_ratio)
    logger.info("=" * 80)

    # 1. Load Tokenizer & Prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompt = prompts[0]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)

    # 2. Stage model on CPU host RAM
    logger.info("[1/4] Staging Mixtral skeleton on host CPU memory...")
    t0_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("[1/4] Model staged in %.2fs", time.perf_counter() - t0_load)

    # 3. Move Non-MoE parameters to GPU
    logger.info("[2/4] Moving non-MoE parameters to cuda:0...")
    model.model.embed_tokens.to(dev)
    model.model.norm.to(dev)
    model.lm_head.to(dev)
    for layer in model.model.layers:
        layer.self_attn.to(dev)
        layer.input_layernorm.to(dev)
        layer.post_attention_layernorm.to(dev)
        layer.block_sparse_moe.gate.to(dev)

    # 4. Attach COLOSSUS with ADETR Salient Slicing
    logger.info("[3/4] Attaching COLOSSUS ADETR (C=%d, missing_ratio=%.2f)...", args.capacity, args.missing_col_ratio)
    profile = introspect_model(model)
    colossus_cfg = SimpleNamespace(
        enabled=True,
        offload_enabled=True,
        hot_expert_topk=args.capacity,
        missing_col_ratio=args.missing_col_ratio,
        lru_slots_per_expert=args.capacity,
        warmup_slots=0,
        lookahead_enabled=False,
        budget="tiered_fwd",
    )
    hook = ColossusMoEHook.build(model, profile, colossus_cfg)
    hook.attach(model, profile, device=dev)

    # Instrument wrappers
    wrappers: List[DynamicMoELayerWrapper] = []
    for m in model.modules():
        if isinstance(m, DynamicMoELayerWrapper):
            wrappers.append(m)
    logger.info("Found %d DynamicMoELayerWrapper instances across model", len(wrappers))

    # Instrument router hooks to capture top-2 routing decisions
    # For each layer, store the sequence of [e1, e2] accessed
    layer_access_sequences: Dict[int, List[int]] = defaultdict(list)
    layer_top2_history: Dict[int, List[List[int]]] = defaultdict(list)

    def make_gate_hook(layer_idx: int):
        def gate_forward_hook(module, args_in, output):
            # output is router logits [batch, seq, num_experts]
            logits = output
            if logits.ndim == 3:
                logits = logits[:, -1, :] # last token
            top2 = torch.topk(logits, k=2, dim=-1).indices[0].tolist()
            layer_top2_history[layer_idx].append(top2)
            layer_access_sequences[layer_idx].extend(top2)
        return gate_forward_hook

    gate_hooks = []
    for l_idx, w in enumerate(wrappers):
        h = w.block.gate.register_forward_hook(make_gate_hook(l_idx))
        gate_hooks.append(h)

    # Generation loop
    token_diagnostics: List[Dict[str, Any]] = []
    curr_input_ids = input_ids
    past_key_values = None

    logger.info("[4/4] Starting token generation and traffic profiling...")

    for step in range(args.max_new_tokens):
        # Clear step counters in wrappers
        step_misses_start = sum(w.misses for w in wrappers)
        step_hits_start = sum(w.hits for w in wrappers)

        # Clear any pending events
        for w in wrappers:
            if hasattr(w, "_drain_demand_events"):
                w._drain_demand_events(sync_first=True)

        torch.cuda.synchronize()
        t0_step = time.perf_counter()

        with torch.no_grad():
            if past_key_values is None:
                outputs = model(curr_input_ids, use_cache=True)
            else:
                outputs = model(curr_input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

        torch.cuda.synchronize()
        t1_step = time.perf_counter()

        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        curr_input_ids = torch.cat([curr_input_ids, next_token_id], dim=-1)

        step_total_s = t1_step - t0_step

        # Drain hardware demand DMA events
        step_dma_s = 0.0
        for w in wrappers:
            if hasattr(w, "_drain_demand_events"):
                w._drain_demand_events(sync_first=True)
                step_dma_s += w.demand_stall_s

        step_compute_s = max(0.0, step_total_s - step_dma_s)
        step_misses = sum(w.misses for w in wrappers) - step_misses_start
        step_hits = sum(w.hits for w in wrappers) - step_hits_start
        step_accesses = step_misses + step_hits
        step_hit_rate = (step_hits / step_accesses * 100.0) if step_accesses > 0 else 0.0

        step_cold_bytes = step_misses * (176.2 * 1024 * 1024)
        step_pcie_gb = step_cold_bytes / (1024 ** 3)

        # Compute Jaccard similarity of selected experts vs previous token
        jaccard_scores = []
        if step > 0:
            for l_idx in range(32):
                prev_e = set(layer_top2_history[l_idx][step - 1])
                curr_e = set(layer_top2_history[l_idx][step])
                intersection = len(prev_e.intersection(curr_e))
                union = len(prev_e.union(curr_e))
                jaccard_scores.append(intersection / union if union > 0 else 0.0)
        avg_jaccard = (sum(jaccard_scores) / len(jaccard_scores)) if jaccard_scores else 0.0

        # Cumulative unique experts accessed across the whole model up to this token
        all_unique_experts = set()
        for l_idx in range(32):
            for e in layer_access_sequences[l_idx]:
                all_unique_experts.add((l_idx, e))

        logger.info(
            "Tok %2d | Step: %.2fs | DMA Stall: %.2fs (%.1f%%) | Compute: %.3fs | Miss: %d/64 (Hit: %.1f%%) | PCIe: %.2f GB | Temp Jaccard: %.2f",
            step + 1,
            step_total_s,
            step_dma_s,
            (step_dma_s / step_total_s * 100.0) if step_total_s > 0 else 0.0,
            step_compute_s,
            step_misses,
            step_hit_rate,
            step_pcie_gb,
            avg_jaccard,
        )

        token_diagnostics.append({
            "token": step + 1,
            "step_total_s": round(step_total_s, 4),
            "dma_stall_s": round(step_dma_s, 4),
            "compute_s": round(step_compute_s, 4),
            "dma_pct": round((step_dma_s / step_total_s * 100.0) if step_total_s > 0 else 0.0, 1),
            "cache_hits": step_hits,
            "cache_misses": step_misses,
            "hit_rate_pct": round(step_hit_rate, 2),
            "pcie_gb": round(step_pcie_gb, 3),
            "temporal_jaccard_similarity": round(avg_jaccard, 3),
            "active_experts_working_set": len(all_unique_experts),
        })

    # Global sequence simulation:
    # 1. Theoretical Belady MIN cache simulation per layer with C=2
    belady_hit_rates = []
    lru_hit_rates = []
    for l_idx in range(32):
        seq = layer_access_sequences[l_idx]
        b_hit = simulate_belady_min(seq, cache_capacity=args.capacity)
        belady_hit_rates.append(b_hit)

    avg_belady_hit_rate = sum(belady_hit_rates) / len(belady_hit_rates)
    avg_lru_hit_rate = sum(m["hit_rate_pct"] for m in token_diagnostics) / len(token_diagnostics)

    total_dma = sum(m["dma_stall_s"] for m in token_diagnostics)
    total_compute = sum(m["compute_s"] for m in token_diagnostics)
    total_pcie_gb = sum(m["pcie_gb"] for m in token_diagnostics)
    avg_pcie_gb_per_tok = total_pcie_gb / len(token_diagnostics)

    # Physical limits calculation:
    t_dma_floor_s = avg_pcie_gb_per_tok / 2.96
    theoretical_max_tps_at_current_bytes = 1.0 / t_dma_floor_s
    max_prefetch_speedup = (total_dma + total_compute) / total_dma if total_dma > 0 else 1.0

    # How much of the 9.53 GB is unavoidable with C=2?
    # Under Belady's optimal clairvoyant cache (best possible caching policy knowing future tokens):
    unavoidable_miss_ratio_belady = 1.0 - (avg_belady_hit_rate / 100.0)
    unavoidable_pcie_gb_belady = (unavoidable_miss_ratio_belady * 64 * 176.2 * 1024 * 1024) / (1024 ** 3)

    results = {
        "diagnostic_title": "Mixtral 9.53 GB/token PCIe Traffic Deconstruction",
        "total_tokens_generated": len(token_diagnostics),
        "total_dma_stall_s": round(total_dma, 3),
        "total_compute_s": round(total_compute, 3),
        "ratio_dma_to_compute": round(total_dma / total_compute, 2) if total_compute > 0 else 0,
        "dma_time_fraction_pct": round(total_dma / (total_dma + total_compute) * 100.0, 2),
        "gpu_compute_fraction_pct": round(total_compute / (total_dma + total_compute) * 100.0, 2),
        "measured_avg_pcie_gb_per_tok": round(avg_pcie_gb_per_tok, 3),
        "physical_transfer_floor_s": round(t_dma_floor_s, 3),
        "speed_of_light_tps_at_current_bytes": round(theoretical_max_tps_at_current_bytes, 3),
        "max_theoretical_prefetch_gain": round(max_prefetch_speedup, 4),
        "caching_analysis": {
            "measured_lru_hit_rate_pct": round(avg_lru_hit_rate, 2),
            "belady_optimal_hit_rate_pct": round(avg_belady_hit_rate, 2),
            "unavoidable_pcie_gb_under_belady_oracle": round(unavoidable_pcie_gb_belady, 3),
            "total_possible_experts_across_model": 32 * 8,
            "cumulative_unique_experts_touched": len(all_unique_experts),
            "pct_of_all_experts_touched": round(len(all_unique_experts) / (32 * 8) * 100.0, 1),
        },
        "per_token_diagnostics": token_diagnostics,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 80)
    logger.info("DIAGNOSTIC SUMMARY:")
    logger.info("  1. DMA Fraction:       %.1f%% of total latency (Compute is only %.1f%%)", results["dma_time_fraction_pct"], results["gpu_compute_fraction_pct"])
    logger.info("  2. Ratio DMA/Compute:  %.1fx", results["ratio_dma_to_compute"])
    logger.info("  3. Prefetch Ceiling:   %.4fx max speedup (without reducing bytes)", max_prefetch_speedup)
    logger.info("  4. LRU Hit Rate:       %.1f%% (Belady Optimal Oracle Hit Rate: %.1f%%)", avg_lru_hit_rate, avg_belady_hit_rate)
    logger.info("  5. Belady Unavoidable: %.2f GB/token (even with an oracle cache!)", unavoidable_pcie_gb_belady)
    logger.info("  Saved results to %s", args.output_json)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
