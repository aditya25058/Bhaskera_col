#!/usr/bin/env python3
"""
Experiment 1B: 2-GPU Dense Layer-Sharded Serving under Software-Enforced 24-GB Ceiling per GPU.
Target: Verify that Param2-17B Dense serving executes successfully without OOM across 2 GPUs
when each GPU is constrained to a 24-GB memory ceiling.
"""

import argparse
import json
import logging
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cap24g_dense_2gpu")


def main():
    parser = argparse.ArgumentParser(description="2-GPU Dense Serving under 24-GB/GPU Ceiling")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--cap-gb", type=float, default=24.0, help="Per-device CUDA memory limit in GiB")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="bench_cap24g_dense_2gpu_results.json")
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        logger.error(f"Requires at least 2 GPUs, found {torch.cuda.device_count()}")
        sys.exit(1)

    # Enforce 24 GB ceiling on both devices
    for dev_idx in [0, 1]:
        total_physical_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
        cap_bytes = int(args.cap_gb * (1024 ** 3))
        fraction = min(1.0, cap_bytes / total_physical_bytes)
        torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)
        torch.cuda.reset_peak_memory_stats(dev_idx)
        logger.info(f"GPU {dev_idx}: Set memory fraction {fraction:.4f} ({args.cap_gb:.2f} GiB cap)")

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Frozen layer sharding allocation
    max_memory = {0: "18GiB", 1: "23GiB"}
    logger.info(f"Loading model with max_memory={max_memory} under 24-GB per-GPU limit...")
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
    logger.info(f"Loaded 2-GPU model in {t_load:.2f}s")

    peak_load_gpu0 = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    peak_load_gpu1 = torch.cuda.max_memory_allocated(1) / (1024 ** 3)
    logger.info(f"Post-load peak: GPU 0 = {peak_load_gpu0:.2f} GB | GPU 1 = {peak_load_gpu1:.2f} GB")

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_peak_memory_stats(1)

    all_generated_tokens = []
    generated_texts = []
    total_tokens = 0

    t0_gen = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        logger.info(f"Generating prompt {idx + 1}/{len(prompts)}...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
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

    peak_gpu0 = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    peak_gpu1 = torch.cuda.max_memory_allocated(1) / (1024 ** 3)
    max_hbm = max(peak_gpu0, peak_gpu1)

    result_data = {
        "experiment": "2-GPU Dense Layer-Sharded under 24GB/GPU Ceiling",
        "feasible": True,
        "n_gpus": 2,
        "cap_gb_per_gpu": args.cap_gb,
        "gpu0_peak_hbm_gb": peak_gpu0,
        "gpu1_peak_hbm_gb": peak_gpu1,
        "max_peak_hbm_gb": max_hbm,
        "tps": tps,
        "ms_per_tok": ms_tok,
        "total_tokens": total_tokens,
        "gen_time_s": t_gen,
    }

    logger.info("=" * 80)
    logger.info("  2-GPU DENSE SERVING UNDER 24-GB CEILING COMPLETED SUCCESSFULLY")
    logger.info(f"  GPU 0 Peak HBM: {peak_gpu0:.2f} GB (< {args.cap_gb:.1f} GB)")
    logger.info(f"  GPU 1 Peak HBM: {peak_gpu1:.2f} GB (< {args.cap_gb:.1f} GB)")
    logger.info(f"  Throughput:     {tps:.2f} tok/s ({ms_tok:.2f} ms/token)")
    logger.info("=" * 80)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
