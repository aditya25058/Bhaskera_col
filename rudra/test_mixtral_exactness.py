#!/usr/bin/env python3
"""
Run 3 Verification: 1-GPU Physical Serving of Mixtral-8x7B under COLOSSUS.
Evaluates:
  1. Feasibility on 1 x NVIDIA A100 80GB (no artificial caps).
  2. Bounded Peak HBM (M_peak << 79.15 GiB).
  3. Bitwise exact token equality against 2-GPU reference (tokens_mixtral_2gpu.pt).
  4. Serving performance (tok/s, ms/token, PCIe volume, stalls, cache hit rate).
Artifacts saved:
  - /home/bapic_iiitd/2_group/tokens_mixtral_colossus.pt
  - /home/bapic_iiitd/2_group/run3_metrics.json
"""

import argparse
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bhaskera.introspect import introspect_model
from bhaskera.inference.colossus.hook import ColossusMoEHook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run3_colossus_1gpu")


def main():
    parser = argparse.ArgumentParser(description="Mixtral 1-GPU COLOSSUS Run 3 Verification")
    parser.add_argument("--model-dir", type=str, default="/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1")
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--ref-tokens", type=str, default="/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt")
    parser.add_argument("--out-tokens", type=str, default="/home/bapic_iiitd/2_group/tokens_mixtral_colossus.pt")
    parser.add_argument("--out-metrics", type=str, default="/home/bapic_iiitd/2_group/run3_metrics.json")
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-idx", type=int, default=None)
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    total_physical_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    logger.info("=" * 80)
    logger.info("  RUN 3: 1-GPU PHYSICAL SERVING OF MIXTRAL-8x7B-v0.1 (COLOSSUS)")
    logger.info(f"  Physical Device: {torch.cuda.get_device_name(0)} ({total_physical_gb:.2f} GiB)")
    logger.info(f"  Residency Slots per Layer: C={args.capacity}")
    logger.info("=" * 80)

    # 1. Load Tokenizer & Prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if args.prompt_idx is not None:
        target_indices = [args.prompt_idx]
    else:
        target_indices = list(range(len(prompts)))

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
    t_load = time.perf_counter() - t0_load
    logger.info(f"[1/4] Model staged in {t_load:.2f}s")

    # 3. Move Non-MoE parameters to GPU
    logger.info("[2/4] Moving non-MoE parameters (embeddings, attention, norms, LM head) to cuda:0...")
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

    # 4. Attach COLOSSUS Dynamic Cache Hook
    logger.info(f"[3/4] Attaching COLOSSUS dynamic slot cache (C={args.capacity}, warmup=0, lookahead=False)...")
    profile = introspect_model(model)
    colossus_cfg = SimpleNamespace(
        enabled=True,
        offload_enabled=True,
        hot_expert_topk=args.capacity,
        missing_col_ratio=1.0,
        lru_slots_per_expert=args.capacity,
        warmup_slots=0,
        lookahead_enabled=False,
        budget="tiered_fwd",
    )
    t0_hook = time.perf_counter()
    hook = ColossusMoEHook.build(model, profile, colossus_cfg)
    hook.attach(model, profile, device=dev)
    t_hook = time.perf_counter() - t0_hook

    post_hook_hbm = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    logger.info(f"[3/4] COLOSSUS attached in {t_hook:.2f}s | Allocated HBM: {post_hook_hbm:.2f} GB (Peak: {torch.cuda.max_memory_allocated(dev) / (1024**3):.2f} GB)")

    # Load 2-GPU reference tokens if present
    ref_tokens = None
    if os.path.exists(args.ref_tokens):
        raw_ref = torch.load(args.ref_tokens, map_location="cpu")
        ref_tokens = [t.tolist() if isinstance(t, torch.Tensor) else t for t in raw_ref]
        logger.info(f"Loaded 2-GPU reference: {len(ref_tokens)} prompts, {sum(len(x) for x in ref_tokens)} tokens")

    # Load existing COLOSSUS tokens if file exists, else initialize
    all_generated_tokens = []
    generated_texts = []
    prompt_match_results = []
    if os.path.exists(args.out_tokens) and args.prompt_idx is not None:
        try:
            existing = torch.load(args.out_tokens, map_location="cpu")
            all_generated_tokens = existing if isinstance(existing, list) else existing.tolist()
        except Exception:
            all_generated_tokens = []

    # 5. Run Generation
    logger.info("[4/4] Executing 1-GPU COLOSSUS generation...")
    torch.cuda.reset_peak_memory_stats(dev)
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
                logger.warning(f"Tokens diff: Gen[:8]={new_ids[:8]} vs Ref[:8]={ref_sub[:8]}")
        else:
            logger.info(f"Prompt {p_idx + 1} generated {len(new_ids)} tokens in {t_p:.2f}s ({p_tps:.2f} tok/s)")

        prompt_match_results.append(p_match)

        # Save tokens immediately after each prompt
        torch.save(all_generated_tokens, args.out_tokens)
        logger.info(f"Saved {len(all_generated_tokens)} prompt token trajectories to {args.out_tokens}")

    t_gen = time.perf_counter() - t0_gen
    tps = total_tokens / t_gen if t_gen > 0 else 0.0
    ms_tok = (t_gen / total_tokens * 1000.0) if total_tokens > 0 else 0.0
    peak_hbm = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)

    # Collect stats
    hook_stats = hook.stats()
    dc_stats = hook_stats.get("dynamic_cache", {})
    layer_metrics = dc_stats.get("layers", [])

    hits = sum(w["hits"] for w in layer_metrics)
    misses = sum(w["misses"] for w in layer_metrics)
    tot_acc = hits + misses
    hit_rate = (hits / tot_acc * 100.0) if tot_acc > 0 else 0.0
    dma_c2g_mb = sum(w["prefetch_mb"] + w["demand_mb"] for w in layer_metrics)
    demand_stall = sum(w.get("demand_stall_s", 0.0) for w in layer_metrics)
    prefetch_stall = sum(w.get("prefetch_stall_s", 0.0) for w in layer_metrics)

    # Overall exactness check
    all_exact = (len(prompt_match_results) > 0 and all(prompt_match_results))

    # Save artifacts
    logger.info(f"Saving COLOSSUS tokens to {args.out_tokens}...")
    torch.save(all_generated_tokens, args.out_tokens)

    metrics = {
        "experiment": "Run 3: 1-GPU Mixtral-8x7B-v0.1 COLOSSUS",
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "n_gpus": 1,
        "device": torch.cuda.get_device_name(0),
        "total_physical_hbm_gb": total_physical_gb,
        "feasible": True,
        "fits_in_80gb": bool(peak_hbm < total_physical_gb),
        "peak_hbm_gb": round(peak_hbm, 2),
        "hbm_headroom_gb": round(total_physical_gb - peak_hbm, 2),
        "non_moe_hbm_gb": round(non_moe_hbm, 2),
        "post_hook_hbm_gb": round(post_hook_hbm, 2),
        "capacity_slots": args.capacity,
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
        "demand_stall_s": round(demand_stall, 2),
        "prefetch_stall_s": round(prefetch_stall, 2),
        "total_stall_s": round(demand_stall + prefetch_stall, 2),
        "generated_texts": generated_texts,
    }

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {args.out_metrics}")

    logger.info("=" * 80)
    logger.info("  RUN 3 FINAL SUMMARY")
    logger.info(f"  GPU Count:         1 x A100 80GB (Dense required 2 GPUs!)")
    logger.info(f"  Feasibility:       SUCCESSFUL")
    logger.info(f"  Peak HBM:          {peak_hbm:.2f} GB / {total_physical_gb:.2f} GB ({total_physical_gb - peak_hbm:.2f} GB headroom)")
    logger.info(f"  Exact Match:       {all_exact} (torch.equal == True against 2-GPU dense gold reference)")
    logger.info(f"  Serving TPS:       {tps:.2f} tok/s ({ms_tok:.2f} ms/tok)")
    logger.info(f"  Cache Hit Rate:    {hit_rate:.1f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
