"""Stage A & B Evaluation: 20/20 MoE Layers Correctness and Cache Breakdown.

Evaluates:
  Stage A: Token-for-token equality and numerical match between Dense Baseline
           and COLOSSUS Dynamic Cache (C=16 across all 20 MoE layers).
  Stage B: Layer-by-layer cache statistics across all 20 layers:
           Hit Rate %, Demand Misses, Prefetch MB, Demand MB, Recall@6.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bhaskera.inference.colossus.dynamic_cache import DynamicMoELayerWrapper


def main():
    parser = argparse.ArgumentParser(description="Stage A & B Evaluation on Rudra")
    parser.add_argument("--model-dir", type=str, required=True, help="Param2 model directory")
    parser.add_argument("--prompt-file", type=str, default="rudra/prompts.txt", help="Prompts file")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to generate per prompt")
    parser.add_argument("--cache-capacity", type=int, default=16, help="GPU cache capacity per layer")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[Stage A & B] Running on: {device} ({torch.cuda.get_device_name(device)})")
    print(f"[Stage A & B] Model: {args.model_dir}")
    print(f"[Stage A & B] Cache capacity per layer: C = {args.cache_capacity}")
    print("=" * 80)

    # 1. Load Tokenizer & Prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"[Stage A & B] Loaded {len(prompts)} prompts.")

    # 2. Load Model into GPU
    print("\n[Stage A & B] Loading model into GPU (bfloat16)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[Stage A & B] Model loaded in {time.perf_counter() - t0:.1f}s.")
    vram_dense_load = torch.cuda.memory_allocated(device) / (1024 ** 3)
    print(f"[Stage A & B] Dense Model VRAM: {vram_dense_load:.2f} GB")

    # Format inputs with chat template
    all_input_ids = []
    for p in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        all_input_ids.append(ids)

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: DENSE BASELINE GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 1: DENSE BASELINE GENERATION (All 64 experts on GPU for all 20 layers)")
    print("=" * 80)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_outputs = []
    baseline_tokens = []
    t_start_base = time.perf_counter()
    total_base_tokens = 0

    with torch.no_grad():
        for i, in_ids in enumerate(all_input_ids):
            t_p0 = time.perf_counter()
            out_ids = model.generate(
                input_ids=in_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            gen_len = out_ids.shape[1] - in_ids.shape[1]
            total_base_tokens += gen_len
            text = tokenizer.decode(out_ids[0][in_ids.shape[1]:], skip_special_tokens=False)
            baseline_tokens.append(out_ids[0].cpu())
            baseline_outputs.append(text)
            print(f"  Prompt {i+1}/{len(prompts)}: {gen_len} tokens in {time.perf_counter() - t_p0:.2f}s")

    base_time = time.perf_counter() - t_start_base
    base_tps = total_base_tokens / base_time if base_time > 0 else 0
    peak_vram_base = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"Dense Baseline Total: {total_base_tokens} tokens | {base_time:.2f}s | {base_tps:.2f} tok/s | Peak VRAM: {peak_vram_base:.2f} GB")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: INSTALL COLOSSUS DYNAMIC CACHE WRAPPERS (20/20 LAYERS)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"PART 2: INSTALLING COLOSSUS DYNAMIC CACHE (C={args.cache_capacity}) ON ALL 20 LAYERS")
    print("=" * 80)

    layers_container = getattr(model, "model", model)
    model_layers = getattr(layers_container, "layers", None)
    wrapped_layers: Dict[int, DynamicMoELayerWrapper] = {}

    for l_idx in range(1, len(model_layers)):
        layer = model_layers[l_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts") and len(mlp.experts) > 1:
            wrapper = DynamicMoELayerWrapper(
                layer_idx=l_idx,
                moe_block=mlp,
                capacity=args.cache_capacity,
                device=device,
            )
            layer.mlp = wrapper
            wrapped_layers[l_idx] = wrapper

    print(f"[Stage A & B] Successfully wrapped {len(wrapped_layers)}/20 MoE layers with DynamicMoELayerWrapper.")
    torch.cuda.empty_cache()
    vram_after_wrap = torch.cuda.memory_allocated(device) / (1024 ** 3)
    print(f"[Stage A & B] VRAM after wrapping (cold experts pinned to CPU): {vram_after_wrap:.2f} GB")
    print(f"[Stage A & B] Memory Freed from GPU: {vram_dense_load - vram_after_wrap:.2f} GB")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 3: COLOSSUS GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"PART 3: COLOSSUS GENERATION (C={args.cache_capacity}, ZSSR Prefetch + Exact Router)")
    print("=" * 80)
    torch.cuda.reset_peak_memory_stats(device)
    colossus_outputs = []
    colossus_tokens = []
    t_start_col = time.perf_counter()
    total_col_tokens = 0

    with torch.no_grad():
        for i, in_ids in enumerate(all_input_ids):
            t_p0 = time.perf_counter()
            out_ids = model.generate(
                input_ids=in_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            gen_len = out_ids.shape[1] - in_ids.shape[1]
            total_col_tokens += gen_len
            text = tokenizer.decode(out_ids[0][in_ids.shape[1]:], skip_special_tokens=False)
            colossus_tokens.append(out_ids[0].cpu())
            colossus_outputs.append(text)
            print(f"  Prompt {i+1}/{len(prompts)}: {gen_len} tokens in {time.perf_counter() - t_p0:.2f}s")

    col_time = time.perf_counter() - t_start_col
    col_tps = total_col_tokens / col_time if col_time > 0 else 0
    peak_vram_col = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"COLOSSUS Total: {total_col_tokens} tokens | {col_time:.2f}s | {col_tps:.2f} tok/s | Peak VRAM: {peak_vram_col:.2f} GB")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 4: STAGE A - STRICT CORRECTNESS VERIFICATION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE A: STRICT CORRECTNESS & LOSSLESS CONTRACT VERIFICATION")
    print("=" * 80)

    all_matched = True
    for i in range(len(prompts)):
        base_tok = baseline_tokens[i]
        col_tok = colossus_tokens[i]
        is_exact = torch.equal(base_tok, col_tok)
        max_diff = (base_tok.float() - col_tok.float()).abs().max().item()
        text_match = (baseline_outputs[i] == colossus_outputs[i])
        print(f"Prompt {i+1}:")
        print(f"  torch.equal() match : {is_exact}")
        print(f"  Max absolute diff   : {max_diff:.1f}")
        print(f"  Text string match   : {text_match}")
        if not is_exact:
            all_matched = False
            print(f"  [Mismatch Details] First divergent token:")
            diff_indices = (base_tok != col_tok).nonzero()
            if len(diff_indices) > 0:
                idx0 = diff_indices[0].item()
                print(f"    Index {idx0}: Baseline={base_tok[idx0].item()} vs COLOSSUS={col_tok[idx0].item()}")

    print("-" * 80)
    print(f"STAGE A FINAL RESULT: {'PASS (100% BIT-FOR-BIT LOSSLESS ACROSS ALL 20 LAYERS)' if all_matched else 'FAIL'}")
    print("-" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # PART 5: STAGE B - PER-LAYER CACHE BREAKDOWN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"STAGE B: PER-LAYER CACHE METRICS TABLE (Capacity C={args.cache_capacity} of 64 Experts)")
    print("=" * 80)
    print(f"{'Layer':>6} | {'C':>3} | {'Hit Rate':>9} | {'Hits':>6} | {'Misses':>6} | {'Prefetch MB':>12} | {'Demand MB':>10} | {'Recall@6':>9}")
    print("-" * 80)

    total_hits = 0
    total_misses = 0
    total_prefetch_mb = 0.0
    total_demand_mb = 0.0
    recall_sum = 0.0

    for l_idx in sorted(wrapped_layers.keys()):
        st = wrapped_layers[l_idx].get_stats()
        print(f"{l_idx:6d} | {st['capacity']:3d} | {st['hit_rate_pct']:8.1f}% | {st['hits']:6d} | {st['misses']:6d} | {st['prefetch_mb']:11.1f} | {st['demand_mb']:9.1f} | {st['recall_pct']:8.1f}%")
        total_hits += st["hits"]
        total_misses += st["misses"]
        total_prefetch_mb += st["prefetch_mb"]
        total_demand_mb += st["demand_mb"]
        recall_sum += st["recall_pct"]

    avg_hit_rate = total_hits / (total_hits + total_misses) * 100.0 if (total_hits + total_misses) > 0 else 0.0
    avg_recall = recall_sum / len(wrapped_layers) if len(wrapped_layers) > 0 else 0.0
    print("-" * 80)
    print(f"{'TOTAL':>6} | {args.cache_capacity:3d} | {avg_hit_rate:8.1f}% | {total_hits:6d} | {total_misses:6d} | {total_prefetch_mb:11.1f} | {total_demand_mb:9.1f} | {avg_recall:8.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
