"""Compare Base vs +COLOSSUS inference runs on Rudra A100.

Reads log files and output files, calculates throughput, VRAM, and text similarity,
and outputs a comparative markdown table and TABLE.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_log(path: Path) -> dict:
    info = {
        "tok_per_sec": 0.0,
        "elapsed": 0.0,
        "tokens": 0,
        "peak_vram_gb": 0.0,
        "colossus_mode": "off",
        "layers_hooked": 0,
        "shadow_steps": 0,
        "hits": 0,
        "misses": 0,
    }
    if not path.exists():
        return info

    text = path.read_text(encoding="utf-8", errors="ignore")

    # Tokens / sec / elapsed
    m = re.search(r"Generated \d+ response\(s\) \|\s*(\d+) tokens \|\s*([\d\.]+)s \|\s*([\d\.]+) tok/s", text)
    if m:
        info["tokens"] = int(m.group(1))
        info["elapsed"] = float(m.group(2))
        info["tok_per_sec"] = float(m.group(3))

    # Peak VRAM
    m_vram = re.search(r"Peak VRAM:\s*([\d\.]+)\s*GB", text)
    if m_vram:
        info["peak_vram_gb"] = float(m_vram.group(1))

    # COLOSSUS MoE stats
    m_col = re.search(r"COLOSSUS MoE: mode=(\w+) \| layers=(\d+) \| steps=(\d+) \| hits=(\d+) \| misses=(\d+)", text)
    if m_col:
        info["colossus_mode"] = m_col.group(1)
        info["layers_hooked"] = int(m_col.group(2))
        info["shadow_steps"] = int(m_col.group(3))
        info["hits"] = int(m_col.group(4))
        info["misses"] = int(m_col.group(5))

    return info


def compare_outputs(base_out_path: Path, col_out_path: Path) -> dict:
    if not base_out_path.exists() or not col_out_path.exists():
        return {"exact_match_ratio": 0.0, "total_prompts": 0}

    base_lines = [l.strip() for l in base_out_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    col_lines = [l.strip() for l in col_out_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]

    matches = sum(1 for b, c in zip(base_lines, col_lines) if b == c)
    total = max(len(base_lines), len(col_lines), 1)

    return {
        "exact_match_ratio": matches / total,
        "matches": matches,
        "total_prompts": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log", default="/home/bapic_iiitd/2_group/infer_base.out")
    parser.add_argument("--col-log", default="/home/bapic_iiitd/2_group/infer_colossus.out")
    parser.add_argument("--base-out", default="/home/bapic_iiitd/2_group/outputs_base.txt")
    parser.add_argument("--col-out", default="/home/bapic_iiitd/2_group/outputs_colossus.txt")
    parser.add_argument("--csv", default="/home/bapic_iiitd/2_group/TABLE.csv")
    args = parser.parse_args()

    base = parse_log(Path(args.base_log))
    col = parse_log(Path(args.col_log))
    out_comp = compare_outputs(Path(args.base_out), Path(args.col_out))

    hit_rate = 0.0
    total_lookups = col["hits"] + col["misses"]
    if total_lookups > 0:
        hit_rate = (col["hits"] / total_lookups) * 100

    speedup = col["tok_per_sec"] / base["tok_per_sec"] if base["tok_per_sec"] > 0 else 1.0

    print("\n" + "=" * 60)
    print("PAIRED EVALUATION RESULT: Base vs +COLOSSUS (Param2-17B on Rudra A100)")
    print("=" * 60)
    print(f"| Metric                    | Base (Dense) | +COLOSSUS    |")
    print(f"|---------------------------|--------------|--------------|")
    print(f"| Throughput (tok/s)        | {base['tok_per_sec']:<12.1f} | {col['tok_per_sec']:<12.1f} |")
    print(f"| Speedup                   | 1.00×        | {speedup:<12.2f}×|")
    print(f"| Peak VRAM (GB)            | {base['peak_vram_gb']:<12.2f} | {col['peak_vram_gb']:<12.2f} |")
    print(f"| Output Text Exact Match   | Ref (100%)   | {out_comp['exact_match_ratio']*100:<11.1f}% |")
    print(f"| Hooked MoE Layers         | 0            | {col['layers_hooked']:<12d} |")
    print(f"| Shadow Prediction Steps   | 0            | {col['shadow_steps']:<12d} |")
    print(f"| Directory Hit-Rate (%)    | N/A          | {hit_rate:<11.1f}% |")
    print("=" * 60 + "\n")

    # Write CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Base_Dense", "Plus_COLOSSUS"])
        writer.writerow(["tok_per_sec", base["tok_per_sec"], col["tok_per_sec"]])
        writer.writerow(["speedup", 1.0, speedup])
        writer.writerow(["peak_vram_gb", base["peak_vram_gb"], col["peak_vram_gb"]])
        writer.writerow(["exact_match_ratio", 1.0, out_comp["exact_match_ratio"]])
        writer.writerow(["layers_hooked", 0, col["layers_hooked"]])
        writer.writerow(["shadow_steps", 0, col["shadow_steps"]])
        writer.writerow(["hit_rate_pct", 0.0, hit_rate])

    print(f"Saved comparison to {args.csv}")


if __name__ == "__main__":
    main()
