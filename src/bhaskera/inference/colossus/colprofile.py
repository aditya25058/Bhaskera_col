"""Frequency-budgeted tier assignment for column residency (adaptive-f).

Routing skew is strong: a minority of experts dominates traffic. Under a
fixed HBM budget (in full-expert slot equivalents), spend residency where
traffic is: frequent experts get f=1.0 (never cold-fetch), rare experts
get thin fractions. Same HBM bytes, much less cold DMA than uniform-f.

Input: `--log-routing` JSON (list of {layer, experts} per step).
Output: JSON {layers: [[f_e...] per layer], budget, tiers, coverage}.
Coverage note: tiers are per-LAYER expert ranks (routing is per-layer).
"""
from __future__ import annotations

import json
from collections import Counter


DEFAULT_TIERS = (1.0, 0.5, 0.25, 0.1)


def load_routing_log(path: str) -> list:
    with open(path) as f:
        entries = json.load(f)
    return entries


def expert_frequencies(entries: list, n_layers: int) -> list[Counter]:
    """Per-layer Counter(expert -> selections) over routing entries."""
    freqs = [Counter() for _ in range(n_layers)]
    for e in entries:
        lyr = e["layer"] if isinstance(e, dict) else e[0]
        exps = e["experts"] if isinstance(e, dict) else e[1]
        if 0 <= lyr < n_layers:
            freqs[lyr].update(exps)
    return freqs


def assign_tiers(freq: Counter, n_experts: int, budget: float,
                 tiers: tuple = DEFAULT_TIERS) -> list[float]:
    """Greedy water-filling under budget (full-expert equivalents).

    Everyone starts at the thinnest tier; remaining budget upgrades
    experts in frequency rank order (1.0 first, then 0.5, ...).
    Returns per-expert fractions in expert-id order. Budget respected
    exactly (sum <= budget + 1e-9).
    """
    tiers = sorted(tiers, reverse=True)
    base = tiers[-1]
    fracs = [base] * n_experts
    spent = base * n_experts
    if spent > budget:
        raise ValueError(f"budget {budget} below base {spent}")
    ranked = [e for e, _ in freq.most_common()]
    ranked += [e for e in range(n_experts) if e not in freq]
    for tier in tiers[:-1]:
        extra = tier - base
        for e in ranked:
            if fracs[e] >= tier:
                continue
            if spent + (tier - fracs[e]) <= budget + 1e-9:
                spent += tier - fracs[e]
                fracs[e] = tier
            else:
                break
    return fracs


def build_tier_file(routing_path: str, n_layers: int, n_experts: int,
                    budget: float, out_path: str,
                    tiers: tuple = DEFAULT_TIERS) -> dict:
    entries = load_routing_log(routing_path)
    freqs = expert_frequencies(entries, n_layers)
    layers = [assign_tiers(f, n_experts, budget, tiers) for f in freqs]
    doc = {"layers": layers, "budget": budget, "tiers": list(tiers),
           "n_experts": n_experts,
           "spent": [round(sum(l), 3) for l in layers]}
    with open(out_path, "w") as f:
        json.dump(doc, f)
    return doc


def load_tier_file(path: str) -> dict[int, list[float]]:
    with open(path) as f:
        doc = json.load(f)
    return {i: l for i, l in enumerate(doc["layers"])}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Adaptive-f tier assignment")
    p.add_argument("routing", help="--log-routing JSON")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--experts", type=int, required=True)
    p.add_argument("--budget", type=float, required=True,
                   help="HBM budget in full-expert slot equivalents/layer")
    p.add_argument("--out", required=True)
    p.add_argument("--tiers", type=float, nargs="+", default=list(DEFAULT_TIERS))
    a = p.parse_args()
    doc = build_tier_file(a.routing, a.layers, a.experts, a.budget, a.out,
                          tuple(a.tiers))
    print(f"tiers -> {a.out} | spent/layer: {doc['spent'][:3]}...")
