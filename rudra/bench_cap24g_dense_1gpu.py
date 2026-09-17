#!/usr/bin/env python3
"""
Experiment 1A: 1-GPU Dense Serving under Software-Enforced 24-GB Memory Ceiling.
Target: Verify that Param2-17B Dense serving triggers torch.OutOfMemoryError on 1 GPU
when constrained to a 24-GB memory ceiling.
"""

import argparse
import json
import logging
import os
import sys
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cap24g_dense_1gpu")


def main():
    parser = argparse.ArgumentParser(description="1-GPU Dense Serving under 24-GB Ceiling")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--cap-gb", type=float, default=24.0, help="Per-process CUDA memory limit in GiB")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="bench_cap24g_dense_1gpu_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is not available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_physical_bytes = torch.cuda.get_device_properties(0).total_memory
    total_physical_gb = total_physical_bytes / (1024 ** 3)
    cap_bytes = int(args.cap_gb * (1024 ** 3))
    fraction = min(1.0, cap_bytes / total_physical_bytes)

    logger.info("=" * 80)
    logger.info("  EXPERIMENT 1A: 1-GPU DENSE SERVING UNDER ENFORCED 24-GB CEILING")
    logger.info(f"  Physical Device: {device_name} ({total_physical_gb:.2f} GiB)")
    logger.info(f"  Enforced Limit:  {args.cap_gb:.2f} GiB ({fraction:.4f} of total)")
    logger.info("=" * 80)

    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # Load prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    result_data = {
        "experiment": "1-GPU Dense under 24GB Ceiling",
        "device": device_name,
        "physical_total_gb": total_physical_gb,
        "enforced_cap_gb": args.cap_gb,
        "fraction": fraction,
        "oom_triggered": False,
        "feasible": False,
        "stage_of_failure": None,
        "error_type": None,
        "error_message": None,
        "peak_hbm_gb": 0.0,
        "allocated_at_failure_gb": 0.0,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # 1. Attempt Model Load under 24 GB Cap
    logger.info("Attempting to load Param2-17B densely onto cuda:0 under 24 GB cap...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        model.eval()
        peak_load = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        logger.info(f"Model loaded. Peak memory during load: {peak_load:.2f} GB")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_str = str(e)
        if "out of memory" in err_str.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            logger.warning(">>> SUCCESSFUL OOM REPRODUCTION AT MODEL LOAD <<<")
            logger.warning(f"Error: {e}")
            logger.warning(f"Allocated at crash: {alloc:.2f} GB | Peak: {peak:.2f} GB | Cap: {args.cap_gb:.2f} GB")
            result_data["oom_triggered"] = True
            result_data["stage_of_failure"] = "model_load"
            result_data["error_type"] = type(e).__name__
            result_data["error_message"] = err_str
            result_data["allocated_at_failure_gb"] = alloc
            result_data["peak_hbm_gb"] = peak
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
            logger.info(f"Results written to {args.output_json}")
            sys.exit(0)
        else:
            logger.error(f"Unexpected error during model load: {e}")
            raise e

    # 2. Attempt Generation if model load unexpectedly fit
    logger.info("Model loaded without OOM; attempting forward generation step...")
    try:
        for idx, prompt in enumerate(prompts):
            logger.info(f"Generating prompt {idx + 1}...")
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            inputs.pop("token_type_ids", None)
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        logger.info(f"Generation unexpectedly completed! Peak HBM: {peak:.2f} GB")
        result_data["feasible"] = True
        result_data["peak_hbm_gb"] = peak
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_str = str(e)
        if "out of memory" in err_str.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            logger.warning(">>> SUCCESSFUL OOM REPRODUCTION AT GENERATION <<<")
            logger.warning(f"Error: {e}")
            result_data["oom_triggered"] = True
            result_data["stage_of_failure"] = "generation"
            result_data["error_type"] = type(e).__name__
            result_data["error_message"] = err_str
            result_data["allocated_at_failure_gb"] = alloc
            result_data["peak_hbm_gb"] = peak
        else:
            raise e

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
