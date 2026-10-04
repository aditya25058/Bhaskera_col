"""Canonical MoE model description (Phase 2 schema).

`describe_model` merges architecture (ModelProfile) with weight statistics
(index + shard headers, no model load) into a JSON-serializable description
that the feasibility planner consumes. Model-agnostic: everything derives
from the profile and the index, never from names.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .interface import classify_role
from .loading import ShardHandles, split_routed

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I64": 8, "I32": 4}


def inspect_weights(model_dir: str) -> dict[str, Any]:
    """Weight statistics from the safetensors index alone (no tensors read)."""
    handles = ShardHandles.open(model_dir)
    return inspect_handles(handles)


def inspect_handles(handles) -> dict[str, Any]:
    """Weight statistics from any handles (local mmap or remote Range)."""
    resident, routed = split_routed(dict(handles.weight_map))
    routed_set = set(routed)
    nbytes = {"total": 0, "resident": 0, "routed": 0}
    dtypes: dict[str, int] = {}
    hidden, inter = 0, 0
    for k in handles.weight_map:
        info = handles.header(k)
        b, e = info["data_offsets"]
        n = e - b
        nbytes["total"] += n
        tag = info.get("dtype", "BF16")
        dtypes[tag] = dtypes.get(tag, 0) + n
        shape = info.get("shape", [])
        if k in routed_set:
            nbytes["routed"] += n
            if not inter and len(shape) == 2 and \
                    classify_role(k.split(".")[-2] + ".weight") in ("gate", "up"):
                inter = int(shape[0])
        else:
            nbytes["resident"] += n
            if len(shape) == 2 and any(
                    t in k for t in ("embed_tokens", "word_embeddings", "wte")):
                hidden = int(shape[1])
    top_dtype = max(dtypes, key=dtypes.get) if dtypes else "BF16"
    return {
        "total_gb": nbytes["total"] / (1024 ** 3),
        "resident_gb": nbytes["resident"] / (1024 ** 3),
        "routed_gb": nbytes["routed"] / (1024 ** 3),
        "dtype": top_dtype,
        "shards": len(handles.shards()),
        "tensors": len(handles.weight_map),
        "expert_tensors": len(routed),
        "hidden": hidden,
        "intermediate": inter,
    }


def describe_model(model_dir: str = "", profile: Any = None,
                   name: str | None = None, handles=None) -> dict[str, Any]:
    """Canonical description: architecture (profile) + weights.

    Weights come from the local index, or from a prebuilt handles object
    (e.g. remote Range handles: headers only, no tensors read).
    """
    if handles is None:
        w = inspect_weights(model_dir)
    else:
        w = inspect_handles(handles)
    desc: dict[str, Any] = {
        "model": {"name": name or os.path.basename((model_dir or "").rstrip("/"))},
        "weights": w,
        "moe": None,
        "capabilities": {"tiered_execution": False, "cpu_placement": False,
                         "reasons": []},
    }
    if profile is None:
        desc["capabilities"]["reasons"].append("no profile: architecture unknown")
        return desc
    n_routed = int(getattr(profile, "num_experts", 0) or 0)
    moe = {
        "is_moe": bool(getattr(profile, "is_moe", n_routed > 1)),
        "model_type": str(getattr(profile, "model_type", "") or ""),
        "routed_per_layer": n_routed,
        "shared": int(getattr(profile, "num_shared_experts", 0) or 0),
        "top_k": int(getattr(profile, "experts_per_token", 0) or 0),
        "moe_layers": int(getattr(profile, "num_hidden_layers", 0) or 0),
        "hidden": w.get("hidden", 0),
        "intermediate": w.get("intermediate", 0),
        "has_aux_loss": bool(getattr(profile, "has_aux_loss", False)),
    }
    desc["moe"] = moe
    caps = desc["capabilities"]
    if n_routed > 1:
        caps["tiered_execution"] = True
    else:
        caps["reasons"].append("not MoE (routed<=1): tiered execution n/a")
    if w["dtype"] in ("BF16", "F16", "F32"):
        caps["cpu_placement"] = True
    else:
        caps["reasons"].append(f"dtype {w['dtype']}: no verified CPU kernels")
    if not moe["moe_layers"]:
        caps["reasons"].append("layer count unknown: set num_hidden_layers")
    return desc


def write_inspect_json(model_dir: str, out_path: str, profile: Any = None,
                       name: str | None = None) -> dict[str, Any]:
    desc = describe_model(model_dir, profile, name)
    with open(out_path, "w") as f:
        json.dump(desc, f, indent=2)
    return desc
