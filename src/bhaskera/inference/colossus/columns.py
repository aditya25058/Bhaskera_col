"""Batched ZSSR column scorer (from ``zssr_standalone``).

Replaces the per-expert Python loop::

    for e in top8: h @ Wg_e ; h @ Wu_e ; silu ; * ; square ; topk

with a single batched GEMM over stacked expert weights::

    G = silu(h @ Wg8^T) ; U = h @ Wu8^T ; E = (G*U)^2 ; topk(E, k, dim=-1)

Shapes (per token): ``h [H]``, ``Wg8/Wu8 [E,I,H]`` -> ``E [E,I]``.
Honors per-expert budgets (e.g. tiered ``50x5+25x3``) by slicing each
row's top-k. ``torch`` is imported lazily.
"""
from __future__ import annotations


def stack_expert_weights(gate_dict, up_dict, experts, device=None, dtype=None):
    """Stack per-expert [I,H] weights -> [E,I,H] contiguous tensor."""
    import torch

    del dtype  # kept for API parity; caller casts beforehand
    Wg = torch.stack([gate_dict[e] for e in experts]).to(device).contiguous()
    Wu = torch.stack([up_dict[e] for e in experts]).to(device).contiguous()
    return Wg, Wu


def columns_batched(h, Wg8, Wu8, budgets):
    """Batched scorer. Returns ``{row: [col ids]}``.

    One bmm per branch + one topk. Keep indices on-GPU in serving
    (no ``.tolist()`` sync on the hot path) — the ``.tolist()`` here is
    for the CPU/offline path and tests only.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        h3 = h.reshape(1, 1, -1)  # [1,1,H] broadcasts over stacked experts
        G = F.silu(h3 @ Wg8.transpose(1, 2)).squeeze(1)  # [E,I]
        U = (h3 @ Wu8.transpose(1, 2)).squeeze(1)        # [E,I]
        E = (G * U).pow(2).squeeze(0)                    # [E,I]
        kmax = max(budgets)
        top = torch.topk(E, k=kmax, dim=-1).indices.tolist()
        return {r: top[r][:k] for r, k in enumerate(budgets)}


def columns_loop(h, gate_dict, up_dict, experts, budgets):
    """Reference per-expert loop (bit-identical to batched)."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        if h.dim() == 1:
            h = h.unsqueeze(0)
        out = {}
        for r, (e, k) in enumerate(zip(experts, budgets)):
            g = F.silu(h @ gate_dict[e].T)
            u = h @ up_dict[e].T
            energy = (g * u).pow(2).squeeze(0)
            out[r] = torch.topk(energy, k=k).indices.tolist()
        return out
