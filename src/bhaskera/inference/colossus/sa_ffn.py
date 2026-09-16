"""SA-FFN: lossless split execution for MoE experts (from COLOSSUS v3).

Standard SwiGLU expert (per token ``x [H]``)::

    y = ((silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T)   # Wg,Wu [I,H], Wd [H,I]

COLOSSUS partitions columns into cached (resident in GPU LRU) and missed
(fetched via prefetch/demand)::

    y = y_cached + y_missed

where each term evaluates only its column subset. Because SwiGLU columns
are independent before the ``Wd`` projection, the sum is mathematically
identical to the dense expert — no approximation. ``verify_lossless``
checks ``L_inf == 0`` / cosine ``== 1``.

``torch`` is imported lazily so CPU-only hosts can import this module.
"""
from __future__ import annotations


def dense_expert_forward(x, Wg, Wu, Wd):
    """Reference dense SwiGLU expert. ``x``: [H] or [B,H]."""
    import torch
    import torch.nn.functional as F

    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
    g = F.silu(x @ Wg.T)
    u = x @ Wu.T
    y = (g * u) @ Wd.T
    return y.squeeze(0) if squeeze else y


def sa_expert_forward(x, cols_cached, cols_missed, Wg, Wu, Wd):
    """Split execution. Returns ``(y_cached, y_missed, y_total)``."""
    import torch
    import torch.nn.functional as F

    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
    dev = x.device
    H = x.shape[-1]
    y_cached = torch.zeros(x.shape[:-1] + (H,), device=dev, dtype=x.dtype)
    y_missed = torch.zeros_like(y_cached)
    if cols_cached:
        g = F.silu(x @ Wg[cols_cached].T)
        u = x @ Wu[cols_cached].T
        y_cached = (g * u) @ Wd[:, cols_cached].T
    if cols_missed:
        g = F.silu(x @ Wg[cols_missed].T)
        u = x @ Wu[cols_missed].T
        y_missed = (g * u) @ Wd[:, cols_missed].T
    y_total = y_cached + y_missed
    if squeeze:
        y_cached, y_missed, y_total = y_cached.squeeze(0), y_missed.squeeze(0), y_total.squeeze(0)
    return y_cached, y_missed, y_total


def verify_lossless(y_dense, y_sa):
    """Returns ``(linf, l1, cosine)`` between dense and SA-FFN outputs."""
    import torch

    d = y_dense.float().reshape(-1)
    s = y_sa.float().reshape(-1)
    linf = (d - s).abs().max().item()
    l1 = (d - s).abs().mean().item()
    cos = (d @ s / (d.norm() * s.norm() + 1e-12)).item()
    return linf, l1, cos
