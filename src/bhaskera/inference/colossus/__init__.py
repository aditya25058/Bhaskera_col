"""
bhaskera.inference.colossus
============================
Tiered exact MoE execution (PR 2): model-agnostic interface (PR 1) plus
slot/CPU placement, mmap loading, and the huge-model serve loop.
Additive and default-off: importing it changes no existing code path.
"""
from __future__ import annotations

from .columns import columns_batched, columns_loop, stack_expert_weights
from .directory import ColumnDirectory, plan_fixed_packets
from .feasibility import plan as feasibility_plan
from .feasibility import render_table
from .hwprobe import probe as hardware_probe
from .hwprobe import write_probe_json
from .inspector import describe_model, inspect_weights, write_inspect_json
from .interface import (
    GATE_DEEPSEEK_3TUPLE,
    GATE_LOGITS,
    MoELayerSpec,
    classify_role,
    expert_weight_keys,
    experts_of,
    find_moe_block,
    find_proj,
    normalize_activation,
)
from .loading import ShardHandles, ShardMap, is_routed_expert_key, materialize, split_routed
from .placement import FastSlot, TieredMoEWrapper, apply_gate, wrap_moe_layers
from .predictor import ZSSRPredictor, quantize_int8
from .remote import RangeFetcher, RemoteShardHandles, TieredHandles
from .sa_ffn import dense_expert_forward, sa_expert_forward, verify_lossless
from .serve import install_cache_compat, prepare_model, serve_huge_moe

__all__ = [
    "GATE_DEEPSEEK_3TUPLE",
    "GATE_LOGITS",
    "ColumnDirectory",
    "FastSlot",
    "MoELayerSpec",
    "RangeFetcher",
    "RemoteShardHandles",
    "ShardHandles",
    "ShardMap",
    "TieredHandles",
    "TieredMoEWrapper",
    "ZSSRPredictor",
    "apply_gate",
    "classify_role",
    "columns_batched",
    "columns_loop",
    "dense_expert_forward",
    "describe_model",
    "expert_weight_keys",
    "experts_of",
    "feasibility_plan",
    "find_moe_block",
    "find_proj",
    "hardware_probe",
    "inspect_weights",
    "install_cache_compat",
    "is_routed_expert_key",
    "materialize",
    "normalize_activation",
    "plan_fixed_packets",
    "prepare_model",
    "quantize_int8",
    "render_table",
    "sa_expert_forward",
    "serve_huge_moe",
    "split_routed",
    "stack_expert_weights",
    "verify_lossless",
    "wrap_moe_layers",
    "write_inspect_json",
    "write_probe_json",
]

try:  # torch-only legacy hook; keeps CPU-only config imports working
    from .hook import ColossusMoEHook
    from .offload import ExpertOffloadManager

    __all__.extend(["ColossusMoEHook", "ExpertOffloadManager"])
except ImportError:  # pragma: no cover
    pass
