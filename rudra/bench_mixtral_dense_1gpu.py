#!/usr/bin/env python3
"""
Experiment: 1-GPU Dense Serving of Mixtral-8x7B-v0.1 on NVIDIA A100 80GB PCIe.
Target: Verify physical hardware CUDA Out of Memory (OOM) on 1 × 80GB GPU
with NO software-imposed memory ceilings.
"""

import argparse
import json
import logging
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mixtral_dense_1gpu")


def main():
    parser = argparse.ArgumentParser(description="1-GPU Dense Mixtral-8x7B (Hardware OOM Test)")
    parser.add_argument("--model-dir", type=str, default="/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1")
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--output-json", type=str, default="/home/bapic_iiitd/2_group/bench_mixtral_dense_1gpu_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_physical_bytes = torch.cuda.get_device_properties(0).total_memory
    total_physical_gb = total_physical_bytes / (1024 ** 3)

    logger.info("=" * 80)
    logger.info("  1-GPU DENSE SERVING TEST (PHYSICAL SILICON CEILING)")
    logger.info(f"  Model:           {args.model_dir}")
    logger.info(f"  Physical Device: {device_name} ({total_physical_gb:.2f} GiB)")
    logger.info("  Memory Cap:      NONE (Full 80-GB hardware capacity exposed)")
    logger.info("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    result_data = {
        "experiment": "1-GPU Dense Mixtral-8x7B Physical Silicon Constraint",
        "model": args.model_dir,
        "device": device_name,
        "physical_total_gb": total_physical_gb,
        "artificial_cap_imposed": False,
        "oom_triggered": False,
        "feasible": False,
        "stage_of_failure": None,
        "error_type": None,
        "error_message": None,
        "peak_hbm_gb": 0.0,
        "allocated_at_failure_gb": 0.0,
    }

    # Attempt to load model densely onto cuda:0
    logger.info("Attempting to load Mixtral-8x7B (93.4 GB weights) densely onto cuda:0...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        model.eval()
        peak_load = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        logger.info(f"Model loaded unexpectedly! Peak memory: {peak_load:.2f} GB")
        result_data["feasible"] = True
        result_data["peak_hbm_gb"] = peak_load
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_str = str(e)
        if "out of memory" in err_str.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            logger.warning("=" * 80)
            logger.warning(">>> SUCCESSFUL PHYSICAL HARDWARE OOM REPRODUCTION <<<")
            logger.warning(f"Error: {e}")
            logger.warning(f"Allocated at crash: {alloc:.2f} GB | Peak: {peak:.2f} GB | Physical Device: {total_physical_gb:.2f} GB")
            logger.warning("=" * 80)
            result_data["oom_triggered"] = True
            result_data["stage_of_failure"] = "model_load"
            result_data["error_type"] = type(e).__name__
            result_data["error_message"] = err_str
            result_data["allocated_at_failure_gb"] = alloc
            result_data["peak_hbm_gb"] = peak
        else:
            logger.error(f"Unexpected error: {e}")
            raise e

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    logger.info(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
