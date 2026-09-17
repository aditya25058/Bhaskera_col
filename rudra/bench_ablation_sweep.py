#!/usr/bin/env python3
"""
bench_ablation_sweep.py — Component Ablation: Where Does COLOSSUS Help?
========================================================================

Runs 5 progressive configurations sequentially in one GPU session to
isolate the contribution of each COLOSSUS component:

  A. Bhaskera Baseline         (COLOSSUS off)
  B. + COLOSSUS shadow-only    (observe-only, no offload)
  C. + Whole-expert offload    (C=32, 100% missing, L+4 lookahead)
  D. + Column-granular 25%     (C=32, 25% missing, L+4)
  E. + Column-granular 10%     (C=32, 10% missing, L+4)

Each configuration:
  - Loads from scratch (fresh engine)
  - Generates with identical prompts and greedy decoding
  - Records throughput, latency, VRAM, stalls, DMA, correctness
  - Verifies output against the baseline reference

Usage:
    python rudra/bench_ablation_sweep.py \
        --model /path/to/Param2-17B \
        --prompt-file rudra/prompts.txt \
        --max-tokens 64
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")


def _count_tokens(texts: List[str], tokenizer) -> int:
    total = 0
    for t in texts:
        try:
            total += len(tokenizer.encode(t, add_special_tokens=False))
        except Exception:
            total += max(1, int(len(t.split()) * 0.9))
    return total


def _gpu_topology() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,"
             "pcie.link.width.current,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown GPU"


def _clear_gpu():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration definitions
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    {
        "label": "A. Bhaskera Baseline",
        "short": "baseline",
        "colossus_enabled": False,
        "offload_enabled": False,
        "hot_expert_topk": 32,
        "missing_col_ratio": 1.0,
    },
    {
        "label": "B. + COLOSSUS Shadow",
        "short": "shadow",
        "colossus_enabled": True,
        "offload_enabled": False,
        "hot_expert_topk": 32,
        "missing_col_ratio": 1.0,
    },
    {
        "label": "C. + Whole-Expert Offload (C=32)",
        "short": "whole_c32",
        "colossus_enabled": True,
        "offload_enabled": True,
        "hot_expert_topk": 32,
        "missing_col_ratio": 1.0,
    },
    {
        "label": "D. + Column-Granular 25%",
        "short": "col_25",
        "colossus_enabled": True,
        "offload_enabled": True,
        "hot_expert_topk": 32,
        "missing_col_ratio": 0.25,
    },
    {
        "label": "E. + Column-Granular 10%",
        "short": "col_10",
        "colossus_enabled": True,
        "offload_enabled": True,
        "hot_expert_topk": 32,
        "missing_col_ratio": 0.10,
    },
]


def run_single(
    label: str,
    model_dir: str,
    prompts: List[str],
    max_tokens: int,
    colossus_enabled: bool = False,
    offload_enabled: bool = False,
    hot_expert_topk: int = 32,
    missing_col_ratio: float = 1.0,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one configuration and return metrics."""
    import torch
    from bhaskera.config import Config

    logger.info(f"\n{'='*72}")
    logger.info(f"  ABLATION: {label}")
    logger.info(f"  COLOSSUS={colossus_enabled}  offload={offload_enabled}  "
                f"C={hot_expert_topk}  missing={missing_col_ratio}")
    logger.info(f"{'='*72}\n")

    cfg = Config()
    cfg.model.name = model_dir
    cfg.model.dtype = "bfloat16"
    cfg.model.trust_remote_code = True
    cfg.inference.max_new_tokens = max_tokens
    cfg.inference.do_sample = False
    cfg.inference.batch_size = 1
    cfg.inference.device = "auto"
    cfg.inference.kv_cache = "turboquant"
    cfg.inference.turboquant.enabled = True
    cfg.inference.turboquant.key_bits = 4
    cfg.inference.turboquant.value_bits = 2
    cfg.inference.turboquant.residual_window = 128
    cfg.inference.turboquant.protected_layers = 2
    cfg.inference.torch_compile = False
    cfg.inference.speculative.enabled = False

    cfg.inference.colossus.enabled = colossus_enabled
    if colossus_enabled:
        cfg.inference.colossus.replica = "int4_row"
        cfg.inference.colossus.budget = "tiered_fwd"
        cfg.inference.colossus.top_k_experts = 8
        cfg.inference.colossus.lru_slots_per_expert = 32
        cfg.inference.colossus.offload_enabled = offload_enabled
        cfg.inference.colossus.hot_expert_topk = hot_expert_topk
        cfg.inference.colossus.missing_col_ratio = missing_col_ratio

    os.environ["BHASKERA_BACKEND"] = "hf"

    from bhaskera.inference import InferenceEngine
    t_load = time.perf_counter()
    engine = InferenceEngine(cfg, model_name=model_dir)
    engine.load()
    t_load = time.perf_counter() - t_load

    tokenizer = None
    try:
        tokenizer = engine._backend._tok
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_tokens, do_sample=False)
    t_gen = time.perf_counter() - t0

    n_tokens = _count_tokens(outputs, tokenizer) if tokenizer else sum(max(1, int(len(o.split()) * 0.9)) for o in outputs)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

    col_stats = engine.colossus_status() or {}
    dc = col_stats.get("dynamic_cache", {})

    result = {
        "label": label,
        "gen_time_s": t_gen,
        "n_tokens": n_tokens,
        "throughput_tok_s": n_tokens / t_gen if t_gen > 0 else 0,
        "latency_ms_tok": (t_gen / n_tokens * 1000) if n_tokens > 0 else 0,
        "peak_vram_gb": peak_vram,
        "demand_stall_s": dc.get("demand_stall_s", 0.0),
        "prefetch_stall_s": dc.get("prefetch_stall_s", 0.0),
        "pcie_time_s": dc.get("pcie_time_s", 0.0),
        "total_dma_gb": (dc.get("prefetch_mb", 0) + dc.get("demand_mb", 0)) / 1024,
        "hit_rate_pct": dc.get("hit_rate_pct", 0.0),
        "recall_pct": dc.get("recall_pct", 0.0),
        "missing_col_mean": dc.get("missing_col_mean", 0.0),
        "outputs": outputs,
    }

    if output_file:
        with open(output_file, "w") as f:
            for o in outputs:
                f.write(o.replace("\n", "\\n") + "\n")

    logger.info(f"[{label}] {n_tokens} tok in {t_gen:.2f}s = {result['throughput_tok_s']:.1f} tok/s | "
                f"VRAM: {peak_vram:.2f} GB | Demand stall: {result['demand_stall_s']:.2f}s")

    del engine
    _clear_gpu()
    return result


def print_ablation_table(results: List[Dict], ref_outputs: List[str], gpu_info: str):
    """Print the full ablation comparison table."""
    print(f"\n{'='*120}")
    print(f"  COMPONENT ABLATION: Bhaskera + COLOSSUS Progressive Configuration Sweep")
    print(f"  Model: Param2-17B-A2.4B-Thinking | GPU: {gpu_info}")
    print(f"{'='*120}")

    # Header
    header = f"{'Config':<32} {'tok/s':>8} {'ms/tok':>8} {'VRAM GB':>8} {'DMA GB':>8} {'DemStall':>9} {'PrefStall':>10} {'HitRate':>8} {'Recall':>8} {'Exact?':>8}"
    print(f"\n{header}")
    print("-" * 120)

    for r in results:
        # Correctness check vs baseline
        exact = "---"
        if ref_outputs is not None and r['outputs']:
            match = all(a == b for a, b in zip(ref_outputs, r['outputs']))
            exact = "PASS" if match else "FAIL"
            if r['label'].startswith("A."):
                exact = "REF"

        print(f"{r['label']:<32} "
              f"{r['throughput_tok_s']:>8.1f} "
              f"{r['latency_ms_tok']:>8.1f} "
              f"{r['peak_vram_gb']:>8.2f} "
              f"{r['total_dma_gb']:>8.1f} "
              f"{r['demand_stall_s']:>8.2f}s "
              f"{r['prefetch_stall_s']:>9.2f}s "
              f"{r['hit_rate_pct']:>7.1f}% "
              f"{r['recall_pct']:>7.1f}% "
              f"{exact:>8}")

    print("-" * 120)

    # Speedup summary vs baseline
    if len(results) >= 2:
        base_tps = results[0]['throughput_tok_s']
        if base_tps > 0:
            print(f"\nSpeedups vs Baseline ({base_tps:.1f} tok/s):")
            for r in results[1:]:
                speedup = r['throughput_tok_s'] / base_tps
                print(f"  {r['label']:<32} {speedup:.2f}x")

    # Stall reduction summary
    if len(results) >= 3 and results[2]['demand_stall_s'] > 0:
        whole_stall = results[2]['demand_stall_s'] + results[2]['prefetch_stall_s']
        print(f"\nStall reduction vs Whole-Expert (Config C, total stall = {whole_stall:.2f}s):")
        for r in results[3:]:
            r_stall = r['demand_stall_s'] + r['prefetch_stall_s']
            reduction = (1 - r_stall / whole_stall) * 100 if whole_stall > 0 else 0
            print(f"  {r['label']:<32} {r_stall:.2f}s ({reduction:.1f}% reduction)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="COLOSSUS Component Ablation Sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt-file", required=True, help="Path to prompts.txt")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--json-out", default=None, help="Optional JSON results path")
    args = parser.parse_args()

    prompts_path = Path(args.prompt_file)
    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(prompts)} prompts")

    gpu_info = _gpu_topology()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    ref_outputs = None

    for cfg_def in ABLATION_CONFIGS:
        result = run_single(
            label=cfg_def["label"],
            model_dir=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            colossus_enabled=cfg_def["colossus_enabled"],
            offload_enabled=cfg_def["offload_enabled"],
            hot_expert_topk=cfg_def["hot_expert_topk"],
            missing_col_ratio=cfg_def["missing_col_ratio"],
            output_file=str(out_dir / f"ablation_{cfg_def['short']}.txt"),
        )
        results.append(result)
        if ref_outputs is None:
            ref_outputs = result['outputs']

    print_ablation_table(results, ref_outputs, gpu_info)

    if args.json_out:
        export = {
            "gpu": gpu_info,
            "prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "configs": [
                {k: v for k, v in r.items() if k != "outputs"}
                for r in results
            ],
        }
        with open(args.json_out, "w") as f:
            json.dump(export, f, indent=2)
        logger.info(f"JSON saved to {args.json_out}")


if __name__ == "__main__":
    main()
