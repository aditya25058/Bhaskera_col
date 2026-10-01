"""
bhaskera.inference.colossus
============================
Tiered exact MoE execution (PR 2): model-agnostic interface (PR 1) plus
slot/CPU placement, mmap loading, and the huge-model serve loop.
Additive and default-off: importing it changes no existing code path.
"""
from __future__ import annotations

from .interface import GATE_DEEPSEEK_3TUPLE, GATE_LOGITS, MoELayerSpec, find_proj
from .interface import classify_role, expert_weight_keys, experts_of, find_moe_block
from .loading import ShardHandles, ShardMap, is_routed_expert_key, materialize, split_routed
from .placement import FastSlot, TieredMoEWrapper, wrap_moe_layers
from .serve import install_cache_compat, prepare_model, serve_huge_moe

__all__ = [
    "MoELayerSpec",
    "find_proj",
    "classify_role",
    "expert_weight_keys",
    "experts_of",
    "find_moe_block",
    "GATE_DEEPSEEK_3TUPLE",
    "GATE_LOGITS",
    "ShardHandles",
    "ShardMap",
    "is_routed_expert_key",
    "materialize",
    "split_routed",
    "FastSlot",
    "TieredMoEWrapper",
    "wrap_moe_layers",
    "install_cache_compat",
    "prepare_model",
    "serve_huge_moe",
]
