"""Frequency-budgeted tier assignment for column residency (adaptive-f).

Routing skew is strong: a minority of experts dominates traffic. Model:
- every expert is assigned exactly one tier (fraction);
- each tier has a fixed pool of slots (LRU within tier);
- HBM = sum over tiers of (pool_slots x fraction) full-expert equivalents.

Simplest effective shape (prototype default): pin the top-K experts per
layer at f=1.0 with K dedicated slots (pool >= members = never evict),
everyone else thin-frac in a shared thin pool. Same HBM budget buys
pinned-hot coverage plus thin tail instead of uniform churn.

Input: `--log-routing` JSON (list of {layer, experts} per step).
Output: {"pools": {frac: slots}, "layers": [[f_e...] per layer]}.
"""
from __future__ import annotations

import json
from collections import Counter


def load_routing_log(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def expert_frequencies(entries: list, n_layers: int) -> list[Counter]:
    freqs = [Counter() for _ in range(n_layers)]
    for e in entries:
        lyr = e["layer"] if isinstance(e, dict) else e[0]
        exps = e["experts"] if isinstance(e, dict) else e[1]
        if 0 <= lyr < n_layers:
            freqs[lyr].update(exps)
    return freqs


def assign_tiers(freq: Counter, n_experts: int, pin_top: int,
                 thin_frac: float) -> list[float]:
    """Top-pin_top experts -> 1.0, rest -> thin_frac (expert-id order)."""
    ranked = [e for e, _ in freq.most_common()]
    ranked += [e for e in range(n_experts) if e not in freq]
    pinned = set(ranked[:pin_top])
    return [1.0 if e in pinned else thin_frac for e in range(n_experts)]


def hbm_equiv(pools: dict) -> float:
    return sum(float(f) * n for f, n in pools.items())


def build_tier_file(routing_path: str, n_layers: int, n_experts: int,
                    pin_top: int, thin_slots: int, thin_frac: float,
                    out_path: str) -> dict:
    entries = load_routing_log(routing_path)
    freqs = expert_frequencies(entries, n_layers)
    layers = [assign_tiers(f, n_experts, pin_top, thin_frac) for f in freqs]
    pools = {"1.0": pin_top, str(thin_frac): thin_slots}
    doc = {"pools": pools, "layers": layers, "n_experts": n_experts,
           "hbm_equiv": hbm_equiv({1.0: pin_top, thin_frac: thin_slots})}
    with open(out_path, "w") as f:
        json.dump(doc, f)
    return doc


def load_tier_file(path: str) -> dict[int, list[float]]:
    with open(path) as f:
        doc = json.load(f)
    return {i: l for i, l in enumerate(doc["layers"])}


def load_tier_pools(path: str) -> dict[float, int]:
    with open(path) as f:
        doc = json.load(f)
    return {float(k): int(v) for k, v in doc.get("pools", {}).items()}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Adaptive-f tier assignment")
    p.add_argument("routing", help="--log-routing JSON")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--experts", type=int, required=True)
    p.add_argument("--pin-top", type=int, default=6,
                   help="top-K experts/layer pinned at f=1.0")
    p.add_argument("--thin-slots", type=int, default=12)
    p.add_argument("--thin-frac", type=float, default=0.1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    doc = build_tier_file(a.routing, a.layers, a.experts, a.pin_top,
                          a.thin_slots, a.thin_frac, a.out)
    print(f"tiers -> {a.out} | hbm_equiv={doc['hbm_equiv']:.1f} "
          f"full-expert slots/layer")
