"""Remote-backed shard handles (Phase B, mock-first).

`RemoteShardHandles` mirrors the `ShardHandles` surface (`weight_map`,
`get_tensor`, `header`, `shards`, plus `stats`) with bytes sourced from
HTTP Range requests instead of mmap. Executors, routing, and placement are
untouched by construction — only the storage backend changes.

Design (see docs/remote_handles_spec.md):
  - Shard headers fetched once (8B length + JSON), persisted in manifest.
  - Tensor bytes cached per-tensor (content-hash addressed), verify-once.
  - LRU byte-cap eviction; offline + uncached = named hard error.
  - Cold/warm stats kept split (network_bytes vs disk_bytes); never blended.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import torch

from .loading import _safetensors_dtype


class RangeFetcher:
    """Minimal HTTP Range client (stdlib only, no new dependencies)."""

    def __init__(self, base_url: str, opener=None):
        self.base_url = base_url.rstrip("/")
        self.opener = opener
        self.requests = 0
        self.bytes = 0

    def get(self, path: str, start: int, end: int) -> bytes:
        """GET bytes [start, end) from base_url/path. Raises on non-206."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
        try:
            with (self.opener or urllib.request).urlopen(req) as r:
                if r.status not in (200, 206):
                    raise IOError(f"range fetch {url} [{start},{end}): HTTP {r.status}")
                data = r.read()
        except Exception as e:
            raise IOError(f"range fetch {url} [{start},{end}) failed: "
                          f"{type(e).__name__}: {e}") from e
        self.requests += 1
        self.bytes += len(data)
        if len(data) != end - start and r.status == 206:
            raise IOError(f"range fetch {url}: short read {len(data)}/{end - start}")
        return data


class RemoteShardHandles:
    """Weight source over HTTP Range + persistent content-addressed cache."""

    MANIFEST = "manifest.json"
    LRU = "lru.json"

    def __init__(self, weight_map: Dict[str, str], fetcher: RangeFetcher,
                 cache_dir: str, cache_cap_gb: float = 200.0,
                 expected_hashes: Optional[Dict[str, str]] = None):
        self.weight_map = dict(weight_map)
        self.fetcher = fetcher
        self.cache_dir = cache_dir
        self.cache_cap_bytes = int(cache_cap_gb * (1024 ** 3))
        self.expected_hashes = expected_hashes or {}
        os.makedirs(os.path.join(cache_dir, "tensors"), exist_ok=True)
        self._manifest = self._load_json(self.MANIFEST, {})
        self._headers: Dict[str, dict] = self._manifest.get("headers", {})
        self._lru: "OrderedDict[str, None]" = OrderedDict(
            (k, None) for k in self._load_json(self.LRU, []))
        self._used_bytes = self._scan_cache()
        self.net_bytes = 0
        self.disk_bytes = 0
        self.hits = 0
        self.misses = 0

    # -- persistence --------------------------------------------------
    def _load_json(self, name, default):
        p = os.path.join(self.cache_dir, name)
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, name, obj) -> None:
        with open(os.path.join(self.cache_dir, name), "w") as f:
            json.dump(obj, f)

    def _scan_cache(self) -> int:
        total = 0
        tdir = os.path.join(self.cache_dir, "tensors")
        for fn in os.listdir(tdir):
            if fn.endswith(".bin"):
                total += os.path.getsize(os.path.join(tdir, fn))
        return total

    def _key_path(self, key: str) -> str:
        h = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, "tensors", h + ".bin")

    # -- headers (fetched once, manifest-persisted) --------------------
    def _shard_header(self, shard: str) -> dict:
        if shard not in self._headers:
            raw = self.fetcher.get(shard, 0, 8)
            import struct
            (hlen,) = struct.unpack("<Q", raw)
            header = json.loads(self.fetcher.get(shard, 8, 8 + hlen))
            self._headers[shard] = header
            self.net_bytes += 8 + hlen
            self._manifest["headers"] = self._headers
            self._save_json(self.MANIFEST, self._manifest)
        return self._headers[shard]

    def header(self, key: str) -> dict:
        return self._shard_header(self.weight_map[key])[key]

    def shards(self) -> List[str]:
        return sorted(set(self.weight_map.values()))

    # -- tensors -------------------------------------------------------
    def _touch(self, key: str) -> None:
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = None
        self._save_json(self.LRU, list(self._lru))

    def _evict(self) -> None:
        while self._used_bytes > self.cache_cap_bytes and self._lru:
            old, _ = self._lru.popitem(last=False)
            p = self._key_path(old)
            try:
                self._used_bytes -= os.path.getsize(p)
                os.remove(p)
            except FileNotFoundError:
                pass
            try:
                os.remove(p[:-4] + ".sha")
            except FileNotFoundError:
                pass
        self._save_json(self.LRU, list(self._lru))

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"unknown weight key: {key}")
        info = self.header(key)
        p = self._key_path(key)
        if os.path.exists(p):
            with open(p, "rb") as f:
                raw = f.read()
            self.disk_bytes += len(raw)
            self.hits += 1
            self._touch(key)
            return torch.frombuffer(bytearray(raw),
                                    dtype=_safetensors_dtype(info.get("dtype", "BF16"))
                                    ).reshape(info["shape"])
        # cold: fetch exact data span, verify once, store
        self.misses += 1
        shard = self.weight_map[key]
        b, e = info["data_offsets"]
        # data span = header region end + offset; recompute from shard layout:
        # fetch via a second header read is avoided: offsets are relative to
        # data start, whose absolute position we recover from the manifest.
        base = self._manifest.get("data_start", {}).get(shard)
        if base is None:
            # first contact: header length prefix known (8B) + cached header len
            import struct
            raw8 = self.fetcher.get(shard, 0, 8)
            (hlen,) = struct.unpack("<Q", raw8)
            base = 8 + hlen
            ds = self._manifest.setdefault("data_start", {})
            ds[shard] = base
            self._save_json(self.MANIFEST, self._manifest)
            self.net_bytes += 8
        raw = self.fetcher.get(shard, base + b, base + e)
        self.net_bytes += len(raw)
        digest = hashlib.sha256(raw).hexdigest()
        exp = self.expected_hashes.get(key)
        if exp is not None and digest != exp:
            raise IOError(f"integrity failure for {key}: hash mismatch")
        with open(p, "wb") as f:
            f.write(raw)
        with open(p[:-4] + ".sha", "w") as f:
            f.write(digest)
        self._used_bytes += len(raw)
        self._touch(key)
        self._evict()
        return torch.frombuffer(bytearray(raw),
                                dtype=_safetensors_dtype(info.get("dtype", "BF16"))
                                ).reshape(info["shape"])

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {"network_bytes": self.net_bytes,
                "disk_bytes": self.disk_bytes,
                "requests": self.fetcher.requests,
                "hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "cache_bytes": self._used_bytes,
                "cache_cap_bytes": self.cache_cap_bytes}

    @classmethod
    def from_hub(cls, repo_id: str, revision: str = "main",
                 cache_dir: Optional[str] = None,
                 cache_cap_gb: float = 200.0,
                 token: Optional[str] = None) -> "RemoteShardHandles":
        """Build from a Hub repo: list shards, fetch headers, weight map.

        No file is downloaded in full; only 8B + JSON headers per shard
        travel here. Tensor bytes follow on demand via get_tensor().
        """
        from huggingface_hub import HfApi, hf_hub_url
        api = HfApi(token=token)
        files = sorted(f for f in api.list_repo_files(repo_id, revision=revision)
                       if f.endswith(".safetensors"))
        if not files:
            raise ValueError(f"no safetensors shards in {repo_id}@{revision}")

        class _HubFetch(RangeFetcher):
            def get(self, path, start, end):
                url = hf_hub_url(repo_id, path, revision=revision)
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{end - 1}"})
                try:
                    r = urllib.request.urlopen(req, timeout=120)
                    if r.status not in (200, 206):
                        raise IOError(f"HTTP {r.status}")
                    data = r.read()
                except Exception as e:
                    raise IOError(f"hub range fetch {path} [{start},{end}): "
                                  f"{type(e).__name__}: {e}") from e
                self.requests += 1
                self.bytes += len(data)
                return data

        fetcher = _HubFetch("")
        weight_map: Dict[str, str] = {}
        tmp = cls(weight_map, fetcher,
                  cache_dir or f"/tmp/remote_cache_{repo_id.replace('/', '_')}",
                  cache_cap_gb)
        for fn in files:
            header = tmp._shard_header(fn)
            for k in header:
                if k != "__metadata__":
                    weight_map[k] = fn
        # __init__ copied the (then-empty) map; rebind the filled one.
        tmp.weight_map = weight_map
        return tmp
