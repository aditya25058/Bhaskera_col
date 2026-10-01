"""
bhaskera.inference.colossus
============================
Model-agnostic MoE interface for tiered exact execution (PR 1).

Translates any MoE block into a uniform spec the execution machinery
consumes — no model names anywhere in this package. Additive and inert:
importing it changes no existing code path.
"""
from __future__ import annotations

from .interface import GATE_DEEPSEEK_3TUPLE, GATE_LOGITS, MoELayerSpec, find_proj
from .interface import classify_role, expert_weight_keys, experts_of, find_moe_block

__all__ = [
    "MoELayerSpec",
    "find_proj",
    "classify_role",
    "expert_weight_keys",
    "experts_of",
    "find_moe_block",
    "GATE_DEEPSEEK_3TUPLE",
    "GATE_LOGITS",
]
