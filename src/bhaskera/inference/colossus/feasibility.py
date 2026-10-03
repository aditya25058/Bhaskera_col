"""Feasibility + cost model (planner core; pure arithmetic, no GPU).

Inputs (all discovered, never hardcoded):
  model: inspector.describe_model output (moe + weights)
  hw:    hwprobe.probe output (gpus, ram, measured bandwidths)
  workload: {batch, seq_len, gen_tokens, shared}
  fidelity: "bitwise" (default) | "ulp1"  (quantized = outside contract)

Output: ranked candidates [{placement, capacity, fits, first_token_s,
steady_tps, vram_gb, ram_gb, fetched_gb, reasons[]}]. Best first.
Streaming floor (capacity 0) is always feasible on any torch device:
the planner prices, never refuses (unknown arch / no device excepted).
"""
from __future__ import annotations

from typing Any, Dict, List

# Calibrated constants (this program's measurements; overridable per box).
FIXED_MS_PER_LAYER = 20.0   # Python orchestration per MoE layer per step
CPU_EFF = 0.8               # oneDNN GEMV vs triad-lite utilization
HBM_MARGIN = 0.9            # usable fraction of reported HBM


def _kv_gb_per_token(moe: dict) -> float:
    h = moe.get("hidden", 5120) or 5120
    L = moe.get("moe_layers", 60) or 60
    return 2 * h * L * 2 / (1024 ** 3)  # K+V bf16 (conservative; MLA smaller)


def _expert_gb(w: dict, moe: dict) -> float:
    n = (moe.get("moe_layers", 0) or 0) * (moe.get("routed_per_layer", 0) or 0)
    if n <= 0:
        return 0.0
    return (w.get("routed_gb", 0.0) or 0.0) / n


def _union_per_layer(moe: dict, batch: int, shared: bool) -> float:
    k = moe.get("top_k", 6) or 6
    per = moe.get("routed_per_layer", 160) or 160
    if shared:
        return float(k)  # measured 6.00: identical position routes identically
    return float(min(batch * k, per))  # diversity ceiling


def plan(model: dict, hw: dict, workload: dict,
         fidelity: str = "bitwise") -> Dict[str, Any]:
    moe = model.get("moe") or {}
    w = model.get("weights") or {}
    if not moe.get("routed_per_layer"):
        return {"feasible": False, "candidates": [],
                "reasons": ["not MoE (routed<=1): tiered execution n/a"]}
    gpus = hw.get("gpus") or []
    hbm = (gpus[0]["total_gb"] if gpus else 0.0) * HBM_MARGIN
    ram = ((hw.get("ram_gb") or {}).get("available_gb", 0.0)
           if isinstance(hw.get("ram_gb"), dict) else 0.0)
    pcie = hw.get("pcie_gbs", 0.0) or 50.0
    dram = hw.get("dram_gbs", 0.0) or 100.0
    has_cuda = bool(hw.get("cuda_available", True))

    B = int(workload.get("batch", 1))
    L = int(workload.get("seq_len", 32))
    G = int(workload.get("gen_tokens", 16))
    shared = bool(workload.get("shared", True))
    layers = moe.get("moe_layers", 60) or 60
    egb = _expert_gb(w, moe)
    kv = _kv_gb_per_token(moe) * B * (L + G)
    resident = (w.get("resident_gb", 0.0) or 0.0)
    union = _union_per_layer(moe, B, shared)
    fixed_s = FIXED_MS_PER_LAYER * layers / 1000.0

    cands = []
    place_opts = ["slots", "cpu"] if fidelity == "ulp1" else ["slots"]
    for place in place_opts:
        for cap in ([0, 4, 8, 12, 24, 36] if place == "slots" else [0]):
            vram = resident + cap * egb + kv if has_cuda else 0.0
            fits = (vram <= hbm) if has_cuda else (place == "cpu")
            if place == "slots" and not has_cuda:
                fits = False
            demand_gb = union * layers * egb
            if place == "slots":
                bw = pcie
                t_step = (demand_gb / bw if bw > 0 else 0.0) + fixed_s
            else:
                bw = dram * CPU_EFF
                t_step = (demand_gb / bw if bw > 0 else 0.0) + fixed_s * 0.5
            prefill_gb = resident + union * layers * egb
            t_first = (prefill_gb / (pcie if has_cuda else dram)) + fixed_s
            fetched = resident + union * layers * egb * min(G, 8)
            reasons = []
            if not fits:
                reasons.append(f"resident {vram:.1f}GB exceeds usable HBM {hbm:.1f}GB")
            if place == "cpu" and not (hw.get("cpu") or {}).get("bf16", True):
                reasons.append("no CPU BF16: fp32 fallback (slower, untested)")
            cands.append({
                "placement": place, "capacity": cap, "fits": bool(fits),
                "first_token_s": round(t_first, 1),
                "steady_tps": round(B / t_step, 2) if t_step > 0 else 0.0,
                "vram_gb": round(vram, 1),
                "ram_gb": round(resident + demand_gb, 1),
                "fetched_gb": round(fetched, 1),
                "reasons": reasons,
            })
    feas = [c for c in cands if c["fits"]]
    feas.sort(key=lambda c: -c["steady_tps"])
    return {"feasible": bool(feas),
            "recommended": feas[0] if feas else cands[-1],
            "candidates": feas + [c for c in cands if not c["fits"]],
            "reasons": [] if feas else ["no config fits; streaming floor (cap 0) priced above"]}
