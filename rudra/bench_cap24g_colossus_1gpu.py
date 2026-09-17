#!/usr/bin/env python3
"""
Experiment 1C: 1-GPU COLOSSUS Serving under Enforced 24-GB Memory Ceiling.
Target: Verify that Param2-17B with COLOSSUS executes successfully on 1 GPU
under a 24-GB memory ceiling where Dense 1-GPU triggered CUDA OOM.
"""

import argparse
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cap24g_colossus_1gpu")


def main():
    parser = argparse.ArgumentParser(description="1-GPU COLOSSUS Serving under 24-GB Cap")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--cap-gb", type=float, default=24.0)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--missing-col-ratio", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="bench_cap24g_colossus_1gpu_results.json")
    args = parser.parse_args()

    dev_idx = 0
    device = torch.device(f"cuda:{dev_idx}")
    total_physical_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
    total_physical_gb = total_physical_bytes / (1024 ** 3)
    cap_bytes = int(args.cap_gb * (1024 ** 3))
    fraction = min(1.0, cap_bytes / total_physical_bytes)

    logger.info("=" * 80)
    logger.info(f"  EXPERIMENT 1C: 1-GPU COLOSSUS UNDER ENFORCED 24-GB CEILING")
    logger.info(f"  Physical Device: {torch.cuda.get_device_name(dev_idx)} ({total_physical_gb:.2f} GiB)")
    logger.info(f"  Enforced Limit:  {args.cap_gb:.2f} GiB (fraction={fraction:.4f})")
    logger.info(f"  COLOSSUS Config: Capacity={args.capacity}, missing_col_ratio={args.missing_col_ratio:.2f}")
    logger.info("=" * 80)

    # 1. Enforce memory ceiling on GPU 0
    torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev_idx)

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # 2. Load model with max_memory bounded to 18GiB on GPU 0 to stay strictly under 24GB cap
    max_memory = {0: "18GiB", "cpu": "60GiB"}
    logger.info(f"Loading model with max_memory={max_memory}...")
    t0_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    t_load = time.perf_counter() - t0_load
    logger.info(f"Loaded model in {t_load:.2f}s")
    post_load_hbm = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
    logger.info(f"Post-load peak HBM on cuda:0: {post_load_hbm:.2f} GB (Limit: {args.cap_gb:.1f} GB)")

    # 3. Attach COLOSSUS Hook
    from bhaskera.introspect import introspect_model
    from bhaskera.inference.colossus.hook import ColossusMoEHook

    profile = introspect_model(model)
    colossus_cfg = SimpleNamespace(
        enabled=True,
        offload_enabled=True,
        hot_expert_topk=args.capacity,
        missing_col_ratio=args.missing_col_ratio,
        lru_slots_per_expert=args.capacity,
        budget="tiered_fwd",
    )

    t0_hook = time.perf_counter()
    hook = ColossusMoEHook.build(model, profile, colossus_cfg)
    hook.attach(model, profile)
    t_hook = time.perf_counter() - t0_hook
    logger.info(f"Attached COLOSSUS hook in {t_hook:.2f}s")

    post_hook_hbm = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
    logger.info(f"Post-hook peak HBM on cuda:0: {post_hook_hbm:.2f} GB (Limit: {args.cap_gb:.1f} GB)")

    torch.cuda.reset_peak_memory_stats(dev_idx)

    # 4. Generate
    all_generated_tokens = []
    generated_texts = []
    total_tokens = 0

    t0_gen = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        logger.info(f"Generating prompt {idx + 1}/{len(prompts)}...")
        if hasattr(hook, "reset_state"):
            hook.reset_state()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        new_ids = out_ids[0, prompt_len:].tolist()
        total_tokens += len(new_ids)
        all_generated_tokens.append(new_ids)
        generated_texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    t_gen = time.perf_counter() - t0_gen
    tps = total_tokens / t_gen if t_gen > 0 else 0.0
    ms_tok = (t_gen / total_tokens * 1000.0) if total_tokens > 0 else 0.0
    peak_gen_hbm = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)

    # 5. Collect DMA and stall metrics
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

    # 6. Check exactness against reference
    gold_tokens_path = "/home/bapic_iiitd/2_group/tokens_1gpu.pt"
    exact_match = False
    if os.path.exists(gold_tokens_path):
        gold_tokens = torch.load(gold_tokens_path, map_location="cpu")
        gold_list = gold_tokens.tolist() if isinstance(gold_tokens, torch.Tensor) else gold_tokens
        if all_generated_tokens == gold_list:
            exact_match = True
            logger.info(">>> 100% BITWISE EXACT MATCH AGAINST GOLD REFERENCE PASS <<<")
        else:
            logger.warning("Tokens did not match reference.")

    res = {
        "experiment": "1-GPU COLOSSUS under 24-GB Memory Ceiling",
        "feasible": True,
        "n_gpus": 1,
        "cap_gb": args.cap_gb,
        "capacity": args.capacity,
        "missing_col_ratio": args.missing_col_ratio,
        "peak_hbm_gb": peak_gen_hbm,
        "fits_under_cap": (peak_gen_hbm < args.cap_gb),
        "tps": tps,
        "ms_per_tok": ms_tok,
        "total_tokens": total_tokens,
        "gen_time_s": t_gen,
        "pcie_dma_gb": dma_c2g_mb / 1024.0,
        "demand_stall_s": demand_stall,
        "prefetch_stall_s": prefetch_stall,
        "total_stall_s": demand_stall + prefetch_stall,
        "cache_hit_pct": hit_rate,
        "exact_token_match": exact_match,
    }

    logger.info("=" * 80)
    logger.info("  1-GPU COLOSSUS UNDER 24-GB CEILING COMPLETED SUCCESSFULLY")
    logger.info(f"  Peak HBM:    {peak_gen_hbm:.2f} GB (< {args.cap_gb:.1f} GB ceiling: {res['fits_under_cap']})")
    logger.info(f"  Throughput:  {tps:.2f} tok/s ({ms_tok:.2f} ms/token)")
    logger.info(f"  PCIe Volume: {res['pcie_dma_gb']:.2f} GB")
    logger.info(f"  Stalls:      {res['total_stall_s']:.2f} s")
    logger.info(f"  Exact Match: {exact_match}")
    logger.info("=" * 80)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
