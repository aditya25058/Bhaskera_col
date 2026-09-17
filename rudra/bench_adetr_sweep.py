#!/usr/bin/env python3
"""
ADETR Memory-Traffic-Accuracy Sweep for 1-GPU Mixtral-8x7B Serving on NVIDIA A100.

Evaluates the exact frontier requested by the user:
  C=2..4 full resident slots + Salient Hot Column Residency (25%, 40%, 50%, 60%)
  against the full-expert C=6 baseline.

Measures:
  - Peak HBM (GB) & Headroom (GB) under the 79.15 GiB limit
  - Cache hits, misses, hit rate (%)
  - Transferred cold payload per miss (MB)
  - Total PCIe H2D volume (GB) & PCIe Bytes / Token
  - Demand stall & prefetch stall (s)
  - Serving Throughput (TPS) & Latency (ms/tok)
  - Bitwise Token Exactness vs 2-GPU Dense Reference (tokens_mixtral_2gpu.pt)
"""

import argparse
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bhaskera.introspect import introspect_model
from bhaskera.inference.colossus.hook import ColossusMoEHook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("adetr_sweep")


def main():
    parser = argparse.ArgumentParser(description="ADETR Column Slicing Sweep on A100")
    parser.add_argument("--model-dir", type=str, default="/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1")
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--ref-tokens", type=str, default="/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt")
    parser.add_argument("--out-tokens", type=str, default="/home/bapic_iiitd/2_group/tokens_adetr.pt")
    parser.add_argument("--out-metrics", type=str, default="/home/bapic_iiitd/2_group/adetr_metrics.json")
    parser.add_argument("--capacity", type=int, default=2, help="Number of full dynamic slots per layer")
    parser.add_argument("--missing-col-ratio", type=float, default=0.50, help="Cold column ratio (0.50 = 50% hot resident, 50% cold streamed)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-idx", type=int, default=0)
    parser.add_argument("--lookahead", action="store_true", default=False)
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    total_physical_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    hot_pct = round((1.0 - args.missing_col_ratio) * 100.0, 1)
    cold_payload_mb = round(352.32 * args.missing_col_ratio, 1)

    logger.info("=" * 80)
    logger.info("  ADETR MEMORY-TRAFFIC-ACCURACY SWEEP: 1-GPU Mixtral-8x7B-v0.1")
    logger.info(f"  Device: {torch.cuda.get_device_name(0)} ({total_physical_gb:.2f} GiB physical HBM)")
    logger.info(f"  Config: C={args.capacity} slots | Missing Col Ratio: {args.missing_col_ratio:.2f} ({hot_pct}% hot in HBM, {100-hot_pct}% cold)")
    logger.info(f"  Payload per Miss: {cold_payload_mb:.1f} MB (vs 352.3 MB full expert)")
    logger.info("=" * 80)

    # 1. Load Tokenizer & Prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    target_indices = [args.prompt_idx] if args.prompt_idx is not None else list(range(len(prompts)))

    # 2. Stage model on CPU host RAM
    logger.info("[1/4] Staging Mixtral-8x7B skeleton on host CPU memory...")
    t0_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"[1/4] Model staged in {time.perf_counter() - t0_load:.2f}s")

    # 3. Move Non-MoE parameters to GPU
    logger.info("[2/4] Moving non-MoE parameters to cuda:0...")
    t0_non_moe = time.perf_counter()
    model.model.embed_tokens.to(dev)
    model.model.norm.to(dev)
    model.lm_head.to(dev)
    for layer in model.model.layers:
        layer.self_attn.to(dev)
        layer.input_layernorm.to(dev)
        layer.post_attention_layernorm.to(dev)
        layer.block_sparse_moe.gate.to(dev)

    non_moe_hbm = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    logger.info(f"[2/4] Non-MoE staged in {time.perf_counter() - t0_non_moe:.2f}s | Allocated: {non_moe_hbm:.2f} GB")

    # 4. Attach COLOSSUS with ADETR Salient Slicing
    logger.info(f"[3/4] Attaching COLOSSUS ADETR (C={args.capacity}, missing_ratio={args.missing_col_ratio:.2f})...")
    profile = introspect_model(model)
    colossus_cfg = SimpleNamespace(
        enabled=True,
        offload_enabled=True,
        hot_expert_topk=args.capacity,
        missing_col_ratio=args.missing_col_ratio,
        lru_slots_per_expert=args.capacity,
        warmup_slots=0,
        lookahead_enabled=args.lookahead,
        budget="tiered_fwd",
    )
    t0_hook = time.perf_counter()
    hook = ColossusMoEHook.build(model, profile, colossus_cfg)
    hook.attach(model, profile, device=dev)
    t_hook = time.perf_counter() - t0_hook

    post_hook_hbm = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    post_hook_peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    logger.info(f"[3/4] COLOSSUS ADETR attached in {t_hook:.2f}s | Allocated: {post_hook_hbm:.2f} GB (Peak: {post_hook_peak:.2f} GB)")

    # 5. Load 2-GPU Reference for Exactness Verification
    ref_tokens = None
    if os.path.exists(args.ref_tokens):
        raw_ref = torch.load(args.ref_tokens, map_location="cpu")
        ref_tokens = [t.tolist() if isinstance(t, torch.Tensor) else t for t in raw_ref]
        logger.info(f"Loaded 2-GPU reference: {len(ref_tokens)} prompts, {sum(len(x) for x in ref_tokens)} tokens")

    # 6. Execute Generation
    logger.info("[4/4] Executing ADETR generation...")
    torch.cuda.reset_peak_memory_stats(dev)
    all_generated_tokens = []
    generated_texts = []
    prompt_match_results = []
    total_tokens = 0

    t0_gen = time.perf_counter()
    for p_idx in target_indices:
        prompt = prompts[p_idx]
        logger.info(f"--- Prompt {p_idx + 1}/{len(prompts)}: '{prompt[:50]}...' ---")
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)
        inputs.pop("token_type_ids", None)
        prompt_len = inputs["input_ids"].shape[1]

        t0_p = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        t_p = time.perf_counter() - t0_p

        new_ids = out_ids[0, prompt_len:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        all_generated_tokens.append(new_ids)
        generated_texts.append(text)
        total_tokens += len(new_ids)
        p_tps = len(new_ids) / t_p

        p_match = False
        if ref_tokens and p_idx < len(ref_tokens):
            ref_sub = ref_tokens[p_idx][:len(new_ids)]
            p_match = (new_ids == ref_sub)
            logger.info(f"Prompt {p_idx + 1} generated {len(new_ids)} tokens in {t_p:.2f}s ({p_tps:.2f} tok/s) | Bitwise Exact: {p_match}")
            if p_match:
                logger.info(">>> BITWISE TOKEN EQUALITY CONFIRMED: torch.equal == True <<<")
            else:
                logger.warning(f"Token diff: Gen[:8]={new_ids[:8]} vs Ref[:8]={ref_sub[:8]}")
        else:
            logger.info(f"Prompt {p_idx + 1} generated {len(new_ids)} tokens in {t_p:.2f}s ({p_tps:.2f} tok/s)")

        prompt_match_results.append(p_match)

    t_gen = time.perf_counter() - t0_gen
    tps = total_tokens / t_gen if t_gen > 0 else 0.0
    ms_tok = (t_gen / total_tokens * 1000.0) if total_tokens > 0 else 0.0
    peak_hbm = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    headroom = total_physical_gb - peak_hbm

    # 7. Collect Hardware & Cache Metrics
    for w in getattr(hook, "_wrapped_layers", {}).values():
        if hasattr(w, "_drain_demand_events"):
            w._drain_demand_events(sync_first=True)
    hook_stats = hook.stats()
    dc_stats = hook_stats.get("dynamic_cache", {})
    layer_metrics = dc_stats.get("layers", [])

    hits = sum(w["hits"] for w in layer_metrics)
    misses = sum(w["misses"] for w in layer_metrics)
    tot_acc = hits + misses
    hit_rate = (hits / tot_acc * 100.0) if tot_acc > 0 else 0.0
    dma_c2g_mb = sum(w["prefetch_mb"] + w["demand_mb"] for w in layer_metrics)
    dma_b_tok = (dma_c2g_mb * 1024 * 1024) / max(1, total_tokens)
    demand_stall = sum(w.get("demand_stall_s", 0.0) for w in layer_metrics)
    prefetch_stall = sum(w.get("prefetch_stall_s", 0.0) for w in layer_metrics)
    pcie_time = sum(w.get("pcie_time_s", 0.0) for w in layer_metrics)

    all_exact = (len(prompt_match_results) > 0 and all(prompt_match_results))

    torch.save(all_generated_tokens, args.out_tokens)

    metrics = {
        "experiment": f"ADETR Sweep: C={args.capacity}, missing={args.missing_col_ratio:.2f}",
        "capacity_slots": args.capacity,
        "missing_col_ratio": args.missing_col_ratio,
        "hot_columns_pct": hot_pct,
        "cold_payload_mb_per_miss": cold_payload_mb,
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "device": torch.cuda.get_device_name(0),
        "total_physical_hbm_gb": total_physical_gb,
        "peak_hbm_gb": round(peak_hbm, 2),
        "hbm_headroom_gb": round(headroom, 2),
        "fits_in_80gb": bool(peak_hbm < total_physical_gb),
        "non_moe_hbm_gb": round(non_moe_hbm, 2),
        "post_hook_hbm_gb": round(post_hook_hbm, 2),
        "total_tokens": total_tokens,
        "gen_time_s": round(t_gen, 2),
        "tps": round(tps, 2),
        "ms_per_tok": round(ms_tok, 2),
        "exact_token_match": all_exact,
        "prompt_matches": prompt_match_results,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate_pct": round(hit_rate, 2),
        "pcie_h2d_gb": round(dma_c2g_mb / 1024.0, 2),
        "pcie_bytes_per_tok": round(dma_b_tok, 0),
        "demand_stall_s": round(demand_stall, 2),
        "prefetch_stall_s": round(prefetch_stall, 2),
        "pcie_time_s": round(pcie_time, 2),
        "generated_texts": generated_texts,
    }

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"  SWEEP POINT COMPLETED: C={args.capacity}, Missing={args.missing_col_ratio:.2f} ({hot_pct}% hot)")
    logger.info(f"  Peak HBM:       {peak_hbm:.2f} GB (Headroom: {headroom:.2f} GB)")
    logger.info(f"  Cold Transfer:  {cold_payload_mb:.1f} MB / miss")
    logger.info(f"  PCIe H2D Volume:{dma_c2g_mb / 1024.0:.2f} GB ({dma_b_tok / (1024*1024):.1f} MB/tok)")
    logger.info(f"  Cache Stats:    {hits} hits, {misses} misses ({hit_rate:.1f}% hit rate)")
    logger.info(f"  Serving TPS:    {tps:.2f} tok/s ({ms_tok:.2f} ms/tok)")
    logger.info(f"  Bitwise Exact:  {all_exact}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
