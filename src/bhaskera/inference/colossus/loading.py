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
from typing import Any

import torch

# Mirror of bhaskera.introspect._EXPERT_LEAF_NAMES / _SHARED_EXPERT_HINTS.
# test_loading.py::test_conventions_in_sync guards drift (import there, not here).
EXPERT_CONTAINER_RES = tuple(
    re.compile(r"\." + name + r"\.\d+\.") for name in ("experts", "local_experts", "routed_experts")
)
SHARED_HINTS = ("shared_expert", "shared_experts")

# Grouped-weight experts: container leaf directly followed by the fused
# parameter (no expert index). Single-sourced in grouped.py; the
# sync test pins the import (no duplicated literal).
from .grouped import GROUPED_KEY_RES  # noqa: E402


def is_grouped_expert_key(key: str) -> bool:
    """True iff key names a fused grouped-expert tensor (representation 2)."""
    low = key.lower()
    if any(h in low for h in SHARED_HINTS):
        return False
    return any(rx.search(key) is not None for rx in GROUPED_KEY_RES)


def is_routed_expert_key(key: str) -> bool:
    """True iff key names a weight inside a routed (non-shared) expert."""
    low = key.lower()
    if any(h in low for h in SHARED_HINTS):
        return False
    return any(rx.search(key) is not None for rx in EXPERT_CONTAINER_RES)


def split_routed(weight_map: dict[str, str]) -> tuple[list[str], list[str]]:
    """-> (resident_keys, routed_keys), both sorted for determinism.

    Routed = per-expert keys (representation 1) + fused grouped-expert
    tensors (representation 2). Shared-expert hints stay resident.
    """
    resident, routed = [], []
    for k in weight_map:
        routed_key = is_routed_expert_key(k) or is_grouped_expert_key(k)
        (routed if routed_key else resident).append(k)
    return sorted(resident), sorted(routed)


def prefault_shards(weight_map: dict[str, str], model_dir: str,
                    verbose: bool = False) -> dict:
    """Advise the kernel to page in every weight shard (cold-start tax).

    `POSIX_FADV_WILLNEED` per shard: non-blocking readahead hint, no
    memory pressure, no behavior change — converts thousands of scattered
    demand major-faults during materialize/decode into one sequential
    streaming read. Returns {"shards": n, "gb": size, "advise_s": wall}.
    Conditional win: helps cold boxes with RAM >> model; on
    marginal-cache boxes the streaming readahead races demand I/O (§25).
    """
    import logging as _logging
    import time as _time
    log = _logging.getLogger("bhaskera.loading")
    shards = sorted(set(weight_map.values()))
    total = 0
    t0 = _time.perf_counter()
    for shard in shards:
        path = os.path.join(model_dir, shard)
        try:
            total += os.path.getsize(path)
            with open(path, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
        except Exception as e:
            log.warning("prefault %s: %s", shard, e)
    dt = _time.perf_counter() - t0
    stats = {"shards": len(shards), "gb": total / 1e9, "advise_s": dt}
    if verbose:
        log.info("prefault: %d shards %.1f GB advise %.1fs",
                 stats["shards"], stats["gb"], dt)
    return stats


class ShardMap:
    """One aligned mmap per safetensors shard; zero-copy tensor views."""

    _cache: dict[str, ShardMap] = {}

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
    def get(cls, path: str) -> ShardMap:
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

    def __init__(self, model_dir: str, weight_map: dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._maps: dict[str, ShardMap] = {}
        self._safe: dict[str, Any] = {}  # shard -> persistent safe_open handle

    @classmethod
    def open(cls, model_dir: str) -> ShardHandles:
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        return cls(model_dir, weight_map)

    @classmethod
    def open_partial(cls, model_dir: str) -> ShardHandles:
        """Index-less: scan *.safetensors headers (partial checkouts welcome)."""
        import glob
        weight_map: dict[str, str] = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            fn = os.path.basename(path)
            sm = ShardMap.get(path)
            for k in sm.header:
                if k != "__metadata__":
                    weight_map[k] = fn
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
        """Header info only (no tensor materialization)."""
        shard = self.weight_map[key]
        sm = self._maps.get(shard)
        if sm is None:
            sm = ShardMap.get(os.path.join(self.model_dir, shard))
            self._maps[shard] = sm
        return sm.header[key]

    def shards(self) -> list[str]:
        return sorted(set(self.weight_map.values()))


def set_module_tensor(model: torch.nn.Module, dotted: str, value: torch.Tensor,
                      device: torch.device | None = None,
                      dtype: torch.dtype | None = None) -> None:
    """Version-proof replacement for transformers' removed
    set_module_tensor_to_device: walk dotted path, replace param in-place.
    Never queries the existing (possibly meta) tensor for device/dtype.
    Tries the model root, then common decoder containers, tolerating index
    keys with or without the top-level prefix."""
    parts = dotted.split(".")
    last_exc = None
    for root in (model, getattr(model, "model", None),
                 getattr(model, "transformer", None)):
        if root is None:
            continue
        try:
            mod = root
            for p in parts[:-1]:
                mod = getattr(mod, p)
            v = value
            if dtype is not None or device is not None:
                v = v.to(device=device or v.device, dtype=dtype or v.dtype)
            # Fresh Parameter (never .data-assign: meta->device is rejected,
            # and replacement keeps inference graphs clean).
            setattr(mod, parts[-1], torch.nn.Parameter(v, requires_grad=False))
            return
        except AttributeError as e:
            last_exc = e
            continue
    raise AttributeError(
        f"cannot resolve {dotted!r} on {type(model).__name__}: {last_exc}")


def materialize(model: torch.nn.Module, keys: list[str], handles: ShardHandles,
                device: torch.device, dtype: torch.dtype = torch.bfloat16) -> int:
    """Copy resident keys into the (meta/empty) model on device. Returns bytes."""
    nbytes = 0
    with torch.no_grad():
        for k in keys:
            t = handles.get_tensor(k)
            nbytes += t.nbytes
            set_module_tensor(model, k, t, device=device, dtype=dtype)
    return nbytes
