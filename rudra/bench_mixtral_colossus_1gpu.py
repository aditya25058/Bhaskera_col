#!/usr/bin/env python3
"""
Experiment 3: 1-GPU COLOSSUS Serving of Mixtral-8x7B-v0.1 on NVIDIA A100 80GB.
Target: Measure the empirical memory ledger on 1 GPU and discover the optimal
residency operating point where:
  N_COLOSSUS = 1 < N_dense = 2
with exact token equality and viable serving throughput.
"""

import argparse
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mixtral_colossus_1gpu")


def run_mixtral_colossus_config(
    model_dir: str,
    prompts: List[str],
    capacity: int,
    missing_col_ratio: float,
    max_new_tokens: int = 64,
    ref_tokens_path: Optional[str] = "/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt",
) -> Dict:
    dev_idx = 0
    device = torch.device(f"cuda:{dev_idx}")
    total_physical_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
    total_physical_gb = total_physical_bytes / (1024 ** 3)

    logger.info("=" * 80)
    logger.info(f"  TESTING 1-GPU MIXTRAL COLOSSUS: C={capacity}, missing_col_ratio={missing_col_ratio:.2f}")
    logger.info(f"  Physical Device: {torch.cuda.get_device_name(dev_idx)} ({total_physical_gb:.2f} GiB)")
    logger.info("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev_idx)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # 1. Load model with bounded max_memory on GPU 0 to prevent 93.4 GB OOM spike during init
    max_memory = {0: "65GiB", "cpu": "120GiB"}
    logger.info(f"Loading Mixtral skeleton with max_memory={max_memory}...")
    t0_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    t_load = time.perf_counter() - t0_load
    base_loaded_hbm = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
    logger.info(f"Model staged in {t_load:.2f}s. Initial HBM on cuda:0: {base_loaded_hbm:.2f} GB")

    # 2. Attach COLOSSUS Hook
    from bhaskera.introspect import introspect_model
    from bhaskera.inference.colossus.hook import ColossusMoEHook

    profile = introspect_model(model)
    colossus_cfg = SimpleNamespace(
        enabled=True,
        offload_enabled=True,
        hot_expert_topk=capacity,
        missing_col_ratio=missing_col_ratio,
        lru_slots_per_expert=capacity,
        budget="tiered_fwd",
    )

    t0_hook = time.perf_counter()
    hook = ColossusMoEHook.build(model, profile, colossus_cfg)
    hook.attach(model, profile)
    t_hook = time.perf_counter() - t0_hook
    post_hook_hbm = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
    logger.info(f"Attached COLOSSUS in {t_hook:.2f}s. Post-hook HBM: {post_hook_hbm:.2f} GB")

    torch.cuda.reset_peak_memory_stats(dev_idx)

    # 3. Generate
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
                max_new_tokens=max_new_tokens,
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

    # 4. Metrics & Stalls
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

    # 5. Exactness check against 2-GPU reference
    exact_match = False
    if ref_tokens_path and os.path.exists(ref_tokens_path):
        ref_tokens = torch.load(ref_tokens_path, map_location="cpu")
        ref_list = ref_tokens.tolist() if isinstance(ref_tokens, torch.Tensor) else ref_tokens
        if all_generated_tokens == ref_list:
            exact_match = True
            logger.info(">>> 100% BITWISE EXACT MATCH AGAINST 2-GPU DENSE REFERENCE PASS <<<")
        else:
            logger.warning("Generated tokens did not match 2-GPU reference.")

    res = {
        "feasible": True,
        "n_gpus": 1,
        "capacity": capacity,
        "missing_col_ratio": missing_col_ratio,
        "peak_hbm_gb": peak_gen_hbm,
        "fits_in_80gb": (peak_gen_hbm < 80.0),
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
        "tokens": all_generated_tokens,
        "texts": generated_texts,
    }

    logger.info("=" * 80)
    logger.info(f"  CONFIG C={capacity}, m={missing_col_ratio:.2f} RESULT:")
    logger.info(f"  Peak HBM:    {peak_gen_hbm:.2f} GB (< 80 GB: {res['fits_in_80gb']})")
    logger.info(f"  Throughput:  {tps:.2f} tok/s ({ms_tok:.2f} ms/token)")
    logger.info(f"  PCIe Volume: {res['pcie_dma_gb']:.2f} GB")
    logger.info(f"  Stalls:      {res['total_stall_s']:.2f} s")
    logger.info(f"  Exact Match: {exact_match}")
    logger.info("=" * 80)

    del model
    del hook
    torch.cuda.empty_cache()

    return res


def main():
    parser = argparse.ArgumentParser(description="1-GPU Mixtral COLOSSUS Residency Search")
    parser.add_argument("--model-dir", type=str, default="/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1")
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--output-json", type=str, default="/home/bapic_iiitd/2_group/bench_mixtral_colossus_1gpu_results.json")
    parser.add_argument("--ref-tokens", type=str, default="/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt")
    args = parser.parse_args()

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Sweep configurations:
    # 1. Whole expert baseline: C=2, m=1.00 (minimal memory footprint)
    # 2. Column-granular conservative: C=2, m=0.60
    # 3. Column-granular balanced: C=2, m=0.50
    # 4. Column-granular C=4: C=4, m=0.60
    configs = [
        (2, 1.00),
        (2, 0.60),
        (2, 0.50),
        (4, 0.60),
    ]

    results = []
    for c, m in configs:
        try:
            res = run_mixtral_colossus_config(
                model_dir=args.model_dir,
                prompts=prompts,
                capacity=c,
                missing_col_ratio=m,
                ref_tokens_path=args.ref_tokens,
            )
            results.append(res)
        except Exception as e:
            logger.error(f"Config C={c}, m={m} failed: {e}")
            results.append({
                "feasible": False,
                "capacity": c,
                "missing_col_ratio": m,
                "error": str(e),
            })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"configs": results}, f, indent=2)
    logger.info(f"Sweep results written to {args.output_json}")


if __name__ == "__main__":
    main()
