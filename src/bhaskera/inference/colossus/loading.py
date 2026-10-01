"""Huge-model weight loading for COLOSSUS tiers (chunk 2).

Models larger than HBM cannot be `model.to(device)` wholesale. Instead:

1. `ShardHandles.open(model_dir)`: mmap every safetensors shard once
   (zero-copy views, page-cache backed) + the index weight map.
2. `split_routed(weight_map)`: partition weight keys into resident
   (non-routed: attention, norms, embeddings, shared experts, head) vs
   tiered (routed experts, streamed/computed on demand).
3. `materialize(model, keys, ...)`: copy resident keys to GPU.

Expert-key detection is structural (container leaf names + index), mirroring
`bhaskera.introspect` conventions without importing its privates at runtime
(a test asserts the two stay in sync).
"""
from __future__ import annotations

import json
import mmap
import os
import re
from typing import Any, Dict, List, Tuple

import torch

# Mirror of bhaskera.introspect._EXPERT_LEAF_NAMES / _SHARED_EXPERT_HINTS.
# test_loading.py::test_conventions_in_sync guards drift (import there, not here).
EXPERT_CONTAINER_RES = tuple(
    re.compile(r"\." + name + r"\.\d+\.") for name in ("experts", "local_experts", "routed_experts")
)
SHARED_HINTS = ("shared_expert", "shared_experts")


def is_routed_expert_key(key: str) -> bool:
    """True iff key names a weight inside a routed (non-shared) expert."""
    low = key.lower()
    if any(h in low for h in SHARED_HINTS):
        return False
    return any(rx.search(key) is not None for rx in EXPERT_CONTAINER_RES)


def split_routed(weight_map: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """-> (resident_keys, routed_keys), both sorted for determinism."""
    resident, routed = [], []
    for k in weight_map:
        (routed if is_routed_expert_key(k) else resident).append(k)
    return sorted(resident), sorted(routed)


class ShardMap:
    """One aligned mmap per safetensors shard; zero-copy tensor views."""

    _cache: Dict[str, "ShardMap"] = {}

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.header_len = int.from_bytes(self.f.read(8), "little")
        self.header = json.loads(self.f.read(self.header_len))
        self.data_start = 8 + self.header_len
        # ACCESS_COPY: writeable view, no disk I/O for reads (torch needs it).
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_COPY)
        self.u8 = torch.frombuffer(self.mm, dtype=torch.uint8)

    @classmethod
    def get(cls, path: str) -> "ShardMap":
        if path not in cls._cache:
            cls._cache[path] = ShardMap(path)
        return cls._cache[path]

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()

    def view_tensor(self, key: str) -> torch.Tensor:
        info = self.header[key]
        b, e = info["data_offsets"]
        s = self.data_start + b
        dtype = _safetensors_dtype(info.get("dtype", "BF16"))
        return self.u8[s:s + (e - b)].view(dtype).reshape(info["shape"])


def _safetensors_dtype(tag: str) -> torch.dtype:
    return {"BF16": torch.bfloat16, "F16": torch.float16,
            "F32": torch.float32, "U8": torch.uint8,
            "I64": torch.int64}.get(tag.upper(), torch.bfloat16)


class ShardHandles:
    """Index-driven mmap handles for a sharded safetensors model directory."""

    def __init__(self, model_dir: str, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._maps: Dict[str, ShardMap] = {}
        self._safe: Dict[str, Any] = {}  # shard -> persistent safe_open handle

    @classmethod
    def open(cls, model_dir: str) -> "ShardHandles":
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        return cls(model_dir, weight_map)

    def _safe_handle(self, shard: str):
        """Persistent safe_open per shard (open once; per-call open costs ms)."""
        h = self._safe.get(shard)
        if h is None:
            from safetensors import safe_open
            h = safe_open(os.path.join(self.model_dir, shard),
                          framework="pt", device="cpu")
            self._safe[shard] = h
        return h

    def get_tensor(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        try:
            return self._safe_handle(shard).get_tensor(key)
        except Exception:
            return self.header_view(key)[0]

    def header_view(self, key: str):
        """(tensor view, info dict) for a key without persistent handles."""
        shard = self.weight_map[key]
        sm = self._maps.get(shard)
        if sm is None:
            sm = ShardMap.get(os.path.join(self.model_dir, shard))
            self._maps[shard] = sm
        return sm.view_tensor(key), sm.header[key]

    def header(self, key: str) -> dict:
        return self.header_view(key)[1]

    def shards(self) -> List[str]:
        return sorted(set(self.weight_map.values()))


def set_module_tensor(model: torch.nn.Module, dotted: str, value: torch.Tensor,
                      device: torch.device | None = None,
                      dtype: torch.dtype | None = None) -> None:
    """Version-proof replacement for transformers' removed
    set_module_tensor_to_device: walk dotted path, replace param in-place.
    Never queries the existing (possibly meta) tensor for device/dtype."""
    parts = dotted.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    v = value
    if dtype is not None or device is not None:
        v = v.to(device=device or v.device, dtype=dtype or v.dtype)
    # Fresh Parameter (never .data-assign: meta->device is rejected, and
    # replacement keeps inference graphs clean; no optim state exists here).
    setattr(mod, parts[-1], torch.nn.Parameter(v, requires_grad=False))


def materialize(model: torch.nn.Module, keys: List[str], handles: ShardHandles,
                device: torch.device, dtype: torch.dtype = torch.bfloat16) -> int:
    """Copy resident keys into the (meta/empty) model on device. Returns bytes."""
    nbytes = 0
    with torch.no_grad():
        for k in keys:
            t = handles.get_tensor(k)
            nbytes += t.nbytes
            set_module_tensor(model, k, t, device=device, dtype=dtype)
    return nbytes
