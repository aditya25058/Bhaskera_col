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
from .interface import normalize_activation
from .columns import columns_batched, columns_loop, stack_expert_weights
from .directory import ColumnDirectory, plan_fixed_packets
from .sa_ffn import dense_expert_forward, sa_expert_forward, verify_lossless
from .predictor import ZSSRPredictor, quantize_int8
from .loading import ShardHandles, ShardMap, is_routed_expert_key, materialize, split_routed
from .placement import FastSlot, TieredMoEWrapper, apply_gate, wrap_moe_layers
from .remote import RangeFetcher, RemoteShardHandles
from .inspector import describe_model, inspect_weights, write_inspect_json
from .serve import install_cache_compat, prepare_model, serve_huge_moe

__all__ = [
    "MoELayerSpec",
    "find_proj",
    "classify_role",
    "expert_weight_keys",
    "experts_of",
    "find_moe_block",
    "normalize_activation",
    "GATE_DEEPSEEK_3TUPLE",
    "GATE_LOGITS",
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
    "ShardHandles",
    "ShardMap",
    "is_routed_expert_key",
    "materialize",
    "split_routed",
    "FastSlot",
    "TieredMoEWrapper",
    "apply_gate",
    "RangeFetcher",
    "RemoteShardHandles",
    "wrap_moe_layers",
    "describe_model",
    "inspect_weights",
    "write_inspect_json",
    "install_cache_compat",
    "prepare_model",
    "serve_huge_moe",
]

try:  # torch-only legacy hook; keeps CPU-only config imports working
    from .hook import ColossusMoEHook
    from .offload import ExpertOffloadManager

    __all__.extend(["ColossusMoEHook", "ExpertOffloadManager"])
except ImportError:  # pragma: no cover
    pass
