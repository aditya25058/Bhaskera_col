"""
bhaskera.inference.colossus
============================
COLOSSUS + ZSSR column-level MoE offload for Bhaskera inference.

Frozen scoring architecture (from ``zssr_standalone``):

* Layer 0: Markov/frequency fallback (no previous hidden state exists).
* Layers >= 1: zero-shot target-router projection
  ``logits = h_{L-1} @ W_L.T`` -> Top-8 experts (zero trainable params),
  then SwiGLU-energy column ranking
  ``E = (silu(h @ Wg) * (h @ Wu))^2`` over a resident INT8 (H100) /
  INT4-per-row (24GB) scoring replica — 0 bytes/token scoring DMA.
* Fixed column packets (``50x5+25x3`` = 325 cols or ``40x8`` = 320 cols)
  -> GPU timestamp LRU directory -> async H2D prefetch inside the
  ~136us MHA overlap window. Max 2 syncs/layer. Fresh every token.
* Exact SA-FFN execution ``y = y_cached + y_missed`` (lossless).

This package is intentionally additive and default-off: importing it
never touches the dense ``InferenceEngine`` path unless
``cfg.inference.colossus.enabled`` is set. ``torch`` is imported lazily
inside functions so ``bhaskera.config`` round-trips work on CPU-only
hosts (e.g. login nodes, CI).
"""
from __future__ import annotations

from .columns import columns_batched, columns_loop, stack_expert_weights
from .directory import ColumnDirectory, plan_fixed_packets
from .sa_ffn import dense_expert_forward, sa_expert_forward, verify_lossless
from .predictor import ZSSRPredictor, quantize_int8

__all__ = [
    "ZSSRPredictor",
    "quantize_int8",
    "columns_batched",
    "columns_loop",
    "stack_expert_weights",
    "ColumnDirectory",
    "plan_fixed_packets",
    "dense_expert_forward",
    "sa_expert_forward",
    "verify_lossless",
]

try:  # torch-only; keeps CPU-only config imports working
    from .hook import ColossusMoEHook
    from .offload import ExpertOffloadManager

    __all__.extend(["ColossusMoEHook", "ExpertOffloadManager"])
except ImportError:  # pragma: no cover
    pass

