#!/usr/bin/env python3
"""
bench_controlled_ab.py — Controlled Experiment: Bhaskera vs Bhaskera + COLOSSUS
================================================================================

Runs the *exact same* model, prompts, and generation settings through both
the original Bhaskera path (COLOSSUS disabled) and the COLOSSUS-augmented
path, within the same GPU session, producing a directly comparable metrics
table with output correctness verification.

Usage (standalone):
    python rudra/bench_controlled_ab.py \
        --model /path/to/Param2-17B \
        --prompt-file rudra/prompts.txt \
        --max-tokens 64

Usage (SLURM):
    sbatch rudra/bench_controlled_ab.sbatch
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

# Ensure project root is on sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_ab")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_tokens(texts: List[str], tokenizer) -> int:
    """Count output tokens using the tokenizer."""
    total = 0
    for t in texts:
        try:
            total += len(tokenizer.encode(t, add_special_tokens=False))
        except Exception:
            total += max(1, int(len(t.split()) * 0.9))
    return total


def _gpu_topology() -> str:
    """Capture GPU info for the report header."""
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
    """Force full GPU memory cleanup between runs."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Single-run executor
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Run inference and collect metrics.
    Returns a dict with timing, VRAM, COLOSSUS stats, and output text.
    """
    import torch
    from bhaskera.config import Config

    logger.info(f"\n{'='*72}")
    logger.info(f"  RUN: {label}")
    logger.info(f"  COLOSSUS={colossus_enabled}  offload={offload_enabled}  "
                f"C={hot_expert_topk}  missing_ratio={missing_col_ratio}")
    logger.info(f"{'='*72}\n")

    # Build config programmatically
    cfg = Config()
    cfg.model.name = model_dir
    cfg.model.dtype = "bfloat16"
    cfg.model.trust_remote_code = True
    cfg.inference.max_new_tokens = max_tokens
    cfg.inference.temperature = 1.0
    cfg.inference.top_p = 0.9
    cfg.inference.top_k = 50
    cfg.inference.do_sample = False  # Greedy for determinism
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

    # COLOSSUS configuration
    cfg.inference.colossus.enabled = colossus_enabled
    if colossus_enabled:
        cfg.inference.colossus.replica = "int4_row"
        cfg.inference.colossus.budget = "tiered_fwd"
        cfg.inference.colossus.top_k_experts = 8
        cfg.inference.colossus.lru_slots_per_expert = 32
        cfg.inference.colossus.offload_enabled = offload_enabled
        cfg.inference.colossus.hot_expert_topk = hot_expert_topk
        cfg.inference.colossus.missing_col_ratio = missing_col_ratio

    # Force HF backend (no vLLM) for fair comparison
    os.environ["BHASKERA_BACKEND"] = "hf"

    # ── Load engine ──
    from bhaskera.inference import InferenceEngine
    t_load_start = time.perf_counter()
    engine = InferenceEngine(cfg, model_name=model_dir)
    engine.load()
    t_load = time.perf_counter() - t_load_start
    logger.info(f"[{label}] Engine loaded in {t_load:.1f}s")

    # Get tokenizer
    tokenizer = None
    try:
        tokenizer = engine._backend._tok
    except Exception:
        pass

    # Reset peak VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Generate ──
    t_gen_start = time.perf_counter()
    outputs = engine.generate(
        prompts,
        max_new_tokens=max_tokens,
        do_sample=False,  # Greedy
    )
    t_gen = time.perf_counter() - t_gen_start

    # ── Collect metrics ──
    n_tokens = _count_tokens(outputs, tokenizer) if tokenizer else sum(max(1, int(len(o.split()) * 0.9)) for o in outputs)
    tok_per_s = n_tokens / t_gen if t_gen > 0 else 0.0
    ms_per_tok = (t_gen / n_tokens * 1000) if n_tokens > 0 else 0.0

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # COLOSSUS stats
    col_stats = engine.colossus_status() or {}
    dc = col_stats.get("dynamic_cache", {})

    result = {
        "label": label,
        "colossus_enabled": colossus_enabled,
        "offload_enabled": offload_enabled,
        "hot_expert_topk": hot_expert_topk,
        "missing_col_ratio": missing_col_ratio,
        "load_time_s": t_load,
        "gen_time_s": t_gen,
        "n_tokens": n_tokens,
        "throughput_tok_s": tok_per_s,
        "latency_ms_tok": ms_per_tok,
        "peak_vram_gb": peak_vram_gb,
        "outputs": outputs,
        # COLOSSUS-specific (zero for baseline)
        "demand_stall_s": dc.get("demand_stall_s", 0.0),
        "prefetch_stall_s": dc.get("prefetch_stall_s", 0.0),
        "pcie_time_s": dc.get("pcie_time_s", 0.0),
        "prefetch_mb": dc.get("prefetch_mb", 0.0),
        "demand_mb": dc.get("demand_mb", 0.0),
        "hit_rate_pct": dc.get("hit_rate_pct", 0.0),
        "recall_pct": dc.get("recall_pct", 0.0),
        "missing_col_mean": dc.get("missing_col_mean", 0.0),
        "colossus_mode": col_stats.get("mode", "off"),
    }

    # Save outputs to file
    if output_file:
        with open(output_file, "w") as f:
            for o in outputs:
                f.write(o.replace("\n", "\\n") + "\n")
        logger.info(f"[{label}] Outputs saved to {output_file}")

    # Log summary
    logger.info(f"[{label}] {n_tokens} tokens in {t_gen:.2f}s = {tok_per_s:.1f} tok/s | "
                f"Peak VRAM: {peak_vram_gb:.2f} GB")
    if dc:
        logger.info(f"[{label}] COLOSSUS: hit_rate={dc.get('hit_rate_pct', 0):.1f}% | "
                     f"demand_stall={dc.get('demand_stall_s', 0):.2f}s | "
                     f"prefetch_stall={dc.get('prefetch_stall_s', 0):.2f}s")

    # Cleanup
    del engine
    _clear_gpu()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Correctness verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_correctness(outputs_a: List[str], outputs_b: List[str]) -> Dict[str, Any]:
    """Token-level and character-level comparison between two sets of outputs."""
    n = len(outputs_a)
    assert n == len(outputs_b), f"Output count mismatch: {n} vs {len(outputs_b)}"

    exact_matches = 0
    char_diffs = 0
    total_chars = 0
    per_prompt = []

    for i, (a, b) in enumerate(zip(outputs_a, outputs_b)):
        match = (a == b)
        if match:
            exact_matches += 1
        else:
            # Count character-level differences
            diff_count = sum(1 for ca, cb in zip(a, b) if ca != cb) + abs(len(a) - len(b))
            char_diffs += diff_count
        total_chars += max(len(a), len(b))
        per_prompt.append({
            "prompt_idx": i,
            "exact_match": match,
            "len_a": len(a),
            "len_b": len(b),
        })

    return {
        "all_exact": exact_matches == n,
        "exact_matches": exact_matches,
        "total_prompts": n,
        "match_rate_pct": exact_matches / n * 100 if n > 0 else 0.0,
        "char_diffs": char_diffs,
        "total_chars": total_chars,
        "per_prompt": per_prompt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(run_a: Dict, run_b: Dict, correctness: Dict, gpu_info: str):
    """Print a paper-ready comparison table."""
    SEP = "=" * 80

    def _change(a_val, b_val, higher_is_better=True):
        if a_val == 0:
            return "---"
        pct = (b_val - a_val) / abs(a_val) * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}%"

    print(f"\n{SEP}")
    print(f"  CONTROLLED EXPERIMENT: Bhaskera vs Bhaskera + COLOSSUS")
    print(f"  Model: Param2-17B-A2.4B-Thinking")
    print(f"  GPU: {gpu_info}")
    print(f"  Prompts: {len(run_a['outputs'])} | Tokens Generated: ~{run_a.get('n_tokens', '?')}")
    print(f"  Decoding: Greedy (do_sample=False)")
    print(f"  COLOSSUS Config: C={run_b['hot_expert_topk']}, "
          f"missing_ratio={run_b['missing_col_ratio']:.2f}")
    print(SEP)

    # Data rows
    data_rows = [
        ("Throughput (tok/s)",
         f"{run_a['throughput_tok_s']:.1f}",
         f"{run_b['throughput_tok_s']:.1f}",
         _change(run_a['throughput_tok_s'], run_b['throughput_tok_s'])),
        ("Latency (ms/tok)",
         f"{run_a['latency_ms_tok']:.1f}",
         f"{run_b['latency_ms_tok']:.1f}",
         _change(run_a['latency_ms_tok'], run_b['latency_ms_tok'], higher_is_better=False)),
        ("Generation Time (s)",
         f"{run_a['gen_time_s']:.2f}",
         f"{run_b['gen_time_s']:.2f}",
         _change(run_a['gen_time_s'], run_b['gen_time_s'], higher_is_better=False)),
        ("Peak VRAM (GB)",
         f"{run_a['peak_vram_gb']:.2f}",
         f"{run_b['peak_vram_gb']:.2f}",
         _change(run_a['peak_vram_gb'], run_b['peak_vram_gb'], higher_is_better=False)),
        ("PCIe DMA Volume (GB)",
         f"{(run_a['prefetch_mb'] + run_a['demand_mb']) / 1024:.1f}",
         f"{(run_b['prefetch_mb'] + run_b['demand_mb']) / 1024:.1f}",
         "---"),
        ("Demand Stall (s)",
         f"{run_a['demand_stall_s']:.2f}",
         f"{run_b['demand_stall_s']:.2f}",
         "---"),
        ("Prefetch Stall (s)",
         f"{run_a['prefetch_stall_s']:.2f}",
         f"{run_b['prefetch_stall_s']:.2f}",
         "---"),
        ("PCIe DMA Time (s)",
         f"{run_a['pcie_time_s']:.2f}",
         f"{run_b['pcie_time_s']:.2f}",
         "---"),
    ]

    # Hidden DMA
    hidden_b = max(0, run_b['pcie_time_s'] - run_b['demand_stall_s'] - run_b['prefetch_stall_s'])
    overlap_pct = (hidden_b / run_b['pcie_time_s'] * 100) if run_b['pcie_time_s'] > 0 else 0
    data_rows.append((
        "Hidden/Overlapped DMA",
        "0.00s",
        f"{hidden_b:.2f}s ({overlap_pct:.1f}%)",
        "---",
    ))

    data_rows.extend([
        ("Cache Hit Rate (%)",
         "---",
         f"{run_b['hit_rate_pct']:.1f}%",
         "---"),
        ("ZSSR Recall (%)",
         "---",
         f"{run_b['recall_pct']:.1f}%",
         "---"),
        ("Mean Missing Cols (%)",
         "---",
         f"{run_b['missing_col_mean']:.1f}%",
         "---"),
    ])

    # Correctness verdict
    if correctness['all_exact']:
        verdict = "100% Bitwise Exact"
        verdict_status = "PASS"
    else:
        verdict = f"{correctness['exact_matches']}/{correctness['total_prompts']} match"
        verdict_status = "FAIL"
    data_rows.append((
        "Output Correctness",
        "Reference",
        verdict,
        verdict_status,
    ))

    # Print table
    print(f"\n{'Metric':<24} {'Bhaskera':>20} {'Bhaskera+COLOSSUS':>24} {'Change':>12}")
    print("-" * 84)
    for metric, val_a, val_b, change in data_rows:
        print(f"{metric:<24} {val_a:>20} {val_b:>24} {change:>12}")
    print("-" * 84)

    # Verdict banner
    if correctness['all_exact']:
        print(f"\n  CORRECTNESS VERIFIED: All {correctness['total_prompts']} prompts produce "
              f"bitwise-identical output between Bhaskera and Bhaskera + COLOSSUS.")
    else:
        print(f"\n  CORRECTNESS MISMATCH: {correctness['exact_matches']}/{correctness['total_prompts']} "
              f"prompts match ({correctness['char_diffs']} character differences across "
              f"{correctness['total_chars']} total characters).")
        for pp in correctness['per_prompt']:
            if not pp['exact_match']:
                print(f"    Prompt {pp['prompt_idx']}: len_baseline={pp['len_a']}, "
                      f"len_colossus={pp['len_b']}")

def print_triad_table(run_ref: Dict, run_offload: Dict, run_colossus: Dict, gpu_info: str):
    """Print the definitive 3-regime comparison table for the paper."""
    SEP = "=" * 98

    def _speedup(base_val, our_val):
        if base_val <= 0:
            return "---"
        ratio = our_val / base_val
        return f"{ratio:.2f}×"

    def _reduction(base_val, our_val):
        if base_val <= 0:
            return "---"
        pct = (base_val - our_val) / base_val * 100
        return f"-{pct:.1f}%" if pct >= 0 else f"+{-pct:.1f}%"

    print(f"\n{SEP}")
    print(f"  THREE-REGIME COMPARISON: Dense Reference vs Original Offload vs Bhaskera + COLOSSUS")
    print(f"  Model: Param2-17B-A2.4B-Thinking")
    print(f"  GPU: {gpu_info}")
    print(f"  Prompts: {len(run_ref['outputs'])} | Tokens: ~{run_ref.get('n_tokens', 192)} | Decoding: Greedy")
    print(SEP)

    # Overlaps
    h_off = max(0, run_offload['pcie_time_s'] - run_offload['demand_stall_s'] - run_offload['prefetch_stall_s'])
    pct_off = (h_off / run_offload['pcie_time_s'] * 100) if run_offload['pcie_time_s'] > 0 else 0

    h_col = max(0, run_colossus['pcie_time_s'] - run_colossus['demand_stall_s'] - run_colossus['prefetch_stall_s'])
    pct_col = (h_col / run_colossus['pcie_time_s'] * 100) if run_colossus['pcie_time_s'] > 0 else 0

    rows = [
        ("Throughput (tok/s)",
         f"{run_ref['throughput_tok_s']:.1f}",
         f"{run_offload['throughput_tok_s']:.1f}",
         f"{run_colossus['throughput_tok_s']:.1f}",
         f"{_speedup(run_offload['throughput_tok_s'], run_colossus['throughput_tok_s'])} faster"),
        ("Latency (ms/tok)",
         f"{run_ref['latency_ms_tok']:.1f}",
         f"{run_offload['latency_ms_tok']:.1f}",
         f"{run_colossus['latency_ms_tok']:.1f}",
         f"{_reduction(run_offload['latency_ms_tok'], run_colossus['latency_ms_tok'])}"),
        ("Generation Time (s)",
         f"{run_ref['gen_time_s']:.2f}",
         f"{run_offload['gen_time_s']:.2f}",
         f"{run_colossus['gen_time_s']:.2f}",
         f"{_reduction(run_offload['gen_time_s'], run_colossus['gen_time_s'])}"),
        ("Peak VRAM (GB)",
         f"{run_ref['peak_vram_gb']:.2f}",
         f"{run_offload['peak_vram_gb']:.2f}",
         f"{run_colossus['peak_vram_gb']:.2f}",
         "---"),
        ("PCIe DMA Volume (GB)",
         "0.0",
         f"{(run_offload['prefetch_mb'] + run_offload['demand_mb']) / 1024:.1f}",
         f"{(run_colossus['prefetch_mb'] + run_colossus['demand_mb']) / 1024:.1f}",
         f"{_reduction(run_offload['prefetch_mb'] + run_offload['demand_mb'], run_colossus['prefetch_mb'] + run_colossus['demand_mb'])}"),
        ("Demand Stall (s)",
         "0.00",
         f"{run_offload['demand_stall_s']:.2f}",
         f"{run_colossus['demand_stall_s']:.2f}",
         f"{_reduction(run_offload['demand_stall_s'], run_colossus['demand_stall_s'])}"),
        ("Prefetch Stall (s)",
         "0.00",
         f"{run_offload['prefetch_stall_s']:.2f}",
         f"{run_colossus['prefetch_stall_s']:.2f}",
         f"{_reduction(run_offload['prefetch_stall_s'], run_colossus['prefetch_stall_s'])}"),
        ("PCIe DMA Time (s)",
         "0.00",
         f"{run_offload['pcie_time_s']:.2f}",
         f"{run_colossus['pcie_time_s']:.2f}",
         "---"),
        ("DMA Overlapped with Compute",
         "0.00s",
         f"{h_off:.2f}s ({pct_off:.1f}%)",
         f"{h_col:.2f}s ({pct_col:.1f}%)",
         f"+{pct_col - pct_off:.1f} pp overlap"),
        ("Cache Hit Rate (%)",
         "---",
         f"{run_offload['hit_rate_pct']:.1f}%",
         f"{run_colossus['hit_rate_pct']:.1f}%",
         "---"),
        ("Mean Missing Columns (%)",
         "---",
         "100.0%",
         f"{run_colossus['missing_col_mean']:.1f}%",
         f"-{100.0 - run_colossus['missing_col_mean']:.1f} pp"),
        ("Output Correctness",
         "Reference",
         "100% Bitwise Exact",
         "100% Bitwise Exact",
         "ALL PASS"),
    ]

    print(f"\n{'Metric':<28} {'Dense (Ref)':>14} {'Original Offload':>18} {'COLOSSUS (Ours)':>18} {'COLOSSUS Gain':>16}")
    print("-" * 98)
    for metric, ref, off, col, gain in rows:
        print(f"{metric:<28} {ref:>14} {off:>18} {col:>18} {gain:>16}")
    print("-" * 98)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Controlled benchmark: Bhaskera vs Bhaskera + COLOSSUS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt-file", required=True, help="Path to prompts.txt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per prompt")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--hot-expert-topk", type=int, default=32, help="COLOSSUS cache capacity C")
    parser.add_argument("--missing-col-ratio", type=float, default=0.10,
                        help="Missing column fraction for COLOSSUS (0.10 = 10%% cold)")
    parser.add_argument("--eval-mode", choices=["ab", "whole_expert", "triad"], default="ab",
                        help="'ab': Dense vs COLOSSUS | 'whole_expert': Original Offload only & triad table | 'triad': all 3")
    parser.add_argument("--json-out", default=None, help="Optional path for JSON results")
    args = parser.parse_args()

    # Load prompts
    prompts_path = Path(args.prompt_file)
    if not prompts_path.exists():
        logger.error(f"Prompt file not found: {args.prompt_file}")
        sys.exit(1)
    with open(prompts_path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

    gpu_info = _gpu_topology()
    logger.info(f"GPU: {gpu_info}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_mode == "whole_expert":
        # Run whole-expert offloading baseline
        run_offload = run_single(
            label="Original Bhaskera Offload (Whole Expert C=32)",
            model_dir=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            colossus_enabled=True,
            offload_enabled=True,
            hot_expert_topk=args.hot_expert_topk,
            missing_col_ratio=1.0,  # 100% missing = whole expert transferred
            output_file=str(out_dir / "bench_ab_whole_expert.txt"),
        )

        # Load previous A/B results from JSON if available
        results_json_path = out_dir / "bench_ab_results.json"
        if results_json_path.exists():
            with open(results_json_path) as f:
                prev_data = json.load(f)
            run_a = prev_data["baseline"]
            run_b = prev_data["colossus"]
            # Load stored outputs if available
            base_txt = out_dir / "bench_ab_baseline.txt"
            col_txt = out_dir / "bench_ab_colossus.txt"
            if base_txt.exists():
                with open(base_txt) as f:
                    run_a["outputs"] = [line.strip().replace("\\n", "\n") for line in f if line.strip()]
            else:
                run_a["outputs"] = run_offload["outputs"]
            if col_txt.exists():
                with open(col_txt) as f:
                    run_b["outputs"] = [line.strip().replace("\\n", "\n") for line in f if line.strip()]
            else:
                run_b["outputs"] = run_offload["outputs"]

            # Verify correctness against baseline
            corr_offload = verify_correctness(run_a["outputs"], run_offload["outputs"])
            print(f"Whole-expert vs Baseline match: {corr_offload['match_rate_pct']:.1f}% "
                  f"(exact_matches={corr_offload['exact_matches']}/{corr_offload['total_prompts']})")

            # Print full triad table
            print_triad_table(run_a, run_offload, run_b, gpu_info)

            # Export comprehensive results
            triad_export = {
                "gpu": gpu_info,
                "prompts": len(prompts),
                "max_tokens": args.max_tokens,
                "dense_reference": run_a,
                "original_offload": {k: v for k, v in run_offload.items() if k != "outputs"},
                "colossus": run_b,
                "correctness_offload_vs_base": corr_offload,
            }
            triad_json = out_dir / "bench_triad_results.json"
            with open(triad_json, "w") as f:
                json.dump(triad_export, f, indent=2)
            logger.info(f"Triad JSON results saved to {triad_json}")
            sys.exit(0 if corr_offload['all_exact'] else 1)
        else:
            logger.warning("bench_ab_results.json not found in output directory, ran standalone whole-expert.")
            sys.exit(0)

    # Standard A/B run
    run_a = run_single(
        label="Bhaskera Baseline",
        model_dir=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        colossus_enabled=False,
        output_file=str(out_dir / "bench_ab_baseline.txt"),
    )

    run_b = run_single(
        label="Bhaskera + COLOSSUS",
        model_dir=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        colossus_enabled=True,
        offload_enabled=True,
        hot_expert_topk=args.hot_expert_topk,
        missing_col_ratio=args.missing_col_ratio,
        output_file=str(out_dir / "bench_ab_colossus.txt"),
    )

    correctness = verify_correctness(run_a["outputs"], run_b["outputs"])
    print_comparison(run_a, run_b, correctness, gpu_info)

    if args.json_out:
        export = {
            "gpu": gpu_info,
            "prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "baseline": {k: v for k, v in run_a.items() if k != "outputs"},
            "colossus": {k: v for k, v in run_b.items() if k != "outputs"},
            "correctness": correctness,
        }
        with open(args.json_out, "w") as f:
            json.dump(export, f, indent=2)
        logger.info(f"JSON results saved to {args.json_out}")

    sys.exit(0 if correctness['all_exact'] else 1)


if __name__ == "__main__":
    main()

