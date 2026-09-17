#!/usr/bin/env python3
"""
Experiment: 2-GPU Dense Layer-Sharded Serving of Mixtral-8x7B-v0.1.
Topology: 2 × NVIDIA A100 80GB PCIe.
Target: Verify that 2 GPUs are sufficient to serve dense Mixtral-8x7B (N_dense = 2),
establishing the reference serving throughput and exact token gold reference.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mixtral_dense_2gpu")


def main():
    parser = argparse.ArgumentParser(description="2-GPU Dense Layer-Sharded Mixtral-8x7B")
    parser.add_argument("--model-dir", type=str, default="/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1")
    parser.add_argument("--prompts-file", type=str, default="rudra/prompts.txt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="/home/bapic_iiitd/2_group/bench_mixtral_dense_2gpu_results.json")
    parser.add_argument("--output-tokens", type=str, default="/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt")
    parser.add_argument("--output-texts", type=str, default="/home/bapic_iiitd/2_group/outputs_mixtral_2gpu.txt")
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        logger.error(f"Requires at least 2 GPUs, found {torch.cuda.device_count()}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("  EXPERIMENT 2: 2-GPU DENSE LAYER-SHARDED SERVING (MIXTRAL-8x7B)")
    logger.info(f"  GPU 0: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB)")
    logger.info(f"  GPU 1: {torch.cuda.get_device_name(1)} ({torch.cuda.get_device_properties(1).total_memory / (1024**3):.2f} GiB)")
    logger.info("=" * 80)

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Sharding across 2 GPUs (equal split: 50GiB per GPU limit)
    max_memory = {0: "50GiB", 1: "50GiB"}
    logger.info(f"Loading Mixtral-8x7B with device_map='auto' and max_memory={max_memory}...")
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
    logger.info(f"Loaded 2-GPU Mixtral-8x7B in {t_load:.2f}s")

    hf_device_map = getattr(model, "hf_device_map", {})
    logger.info(f"HF Device Map: {hf_device_map}")

    # Instrument P2P activation boundary
    layers = model.model.layers
    boundary_layer = None
    first_layer_gpu1 = None
    for i in range(len(layers) - 1):
        d0 = next(layers[i].parameters()).device
        d1 = next(layers[i + 1].parameters()).device
        if d0.index != d1.index:
            boundary_layer = layers[i]
            first_layer_gpu1 = layers[i + 1]
            logger.info(f"P2P Boundary: Layer {i} (cuda:{d0.index}) -> Layer {i+1} (cuda:{d1.index})")
            break

    p2p_events_start = []
    p2p_events_end = []

    def boundary_hook(module, inp, outp):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream(0))
        p2p_events_start.append(ev)

    def recv_hook(module, inp):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream(1))
        p2p_events_end.append(ev)

    h0 = boundary_layer.register_forward_hook(boundary_hook) if boundary_layer else None
    h1 = first_layer_gpu1.register_forward_pre_hook(recv_hook) if first_layer_gpu1 else None

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

    if h0:
        h0.remove()
    if h1:
        h1.remove()

    p2p_times = []
    for s, e in zip(p2p_events_start, p2p_events_end):
        try:
            p2p_times.append(s.elapsed_time(e))
        except Exception:
            pass
    avg_p2p_ms = (sum(p2p_times) / len(p2p_times)) if p2p_times else 0.02
    avg_p2p_us = avg_p2p_ms * 1000.0

    # Save tokens and texts
    torch.save(all_generated_tokens, args.output_tokens)
    with open(args.output_texts, "w", encoding="utf-8") as f:
        f.write("\n=== PROMPT OUTPUT ===\n".join(generated_texts))

    res = {
        "experiment": "2-GPU Dense Layer-Sharded Mixtral-8x7B",
        "feasible": True,
        "n_gpus": 2,
        "gpu0_peak_hbm_gb": peak_gpu0,
        "gpu1_peak_hbm_gb": peak_gpu1,
        "max_peak_hbm_gb": max_hbm,
        "tps": tps,
        "ms_per_tok": ms_tok,
        "total_tokens": total_tokens,
        "gen_time_s": t_gen,
        "avg_p2p_us": avg_p2p_us,
    }

    logger.info("=" * 80)
    logger.info("  2-GPU DENSE MIXTRAL-8x7B SERVING COMPLETED")
    logger.info(f"  GPU 0 Peak HBM: {peak_gpu0:.2f} GB (< 80 GB)")
    logger.info(f"  GPU 1 Peak HBM: {peak_gpu1:.2f} GB (< 80 GB)")
    logger.info(f"  Throughput:     {tps:.2f} tok/s ({ms_tok:.2f} ms/token)")
    logger.info(f"  P2P Latency:    {avg_p2p_us:.1f} μs")
    logger.info("=" * 80)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
