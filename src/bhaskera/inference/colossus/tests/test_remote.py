"""Mock HTTP Range server + RemoteShardHandles gates (Phase B, mock-first).

Proves, with zero network beyond localhost:
  1. remote bytes == local mmap bytes (exactness gate),
  2. warm pass performs zero HTTP (manifest + cache persistence),
  3. eviction under tiny cap stays exact (refetch path),
  4. offline + uncached raises the named hard error.
CPU-only.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from bhaskera.inference.colossus.loading import ShardHandles
from bhaskera.inference.colossus.remote import (
    RangeFetcher,
    RemoteShardHandles,
    TieredHandles,
)


class RangeHandler(BaseHTTPRequestHandler):
    root = ""
    hits = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = os.path.join(type(self).root, self.path.lstrip("/").split("?")[0])
        if not os.path.isfile(path):
            self.send_response(404)
            self.end_headers()
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size
        if rng and rng.startswith("bytes="):
            s, e = rng[len("bytes="):].split("-")
            start = int(s)
            end = int(e) + 1 if e else size
        type(self).hits += 1
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start)
        code = 206 if rng else 200
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{size}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def served_shard(tmp_path):
    from safetensors.torch import save_file
    d = tmp_path / "srv"
    d.mkdir()
    tensors = {
        "blk.experts.0.gate_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16).clone(),
        "blk.experts.0.down_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16).clone(),
        "blk.embed.weight": torch.randn(4, 16, dtype=torch.bfloat16).clone(),
    }
    save_file(tensors, str(d / "shard.safetensors"))
    wm = {k: "shard.safetensors" for k in tensors}
    with open(d / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": wm}, f)
    RangeHandler.root = str(d)
    RangeHandler.hits = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}", dict(wm), str(d)
    srv.shutdown()
    th.join()


def _remote(base_url, wm, cache, cap_gb=200.0):
    return RemoteShardHandles(dict(wm), RangeFetcher(base_url), cache,
                              cache_cap_gb=cap_gb)


def test_fetch_exact(served_shard, tmp_path):
    base_url, wm, srvdir = served_shard
    # local reference straight from the served files (same bytes the mock serves)
    import glob

    from safetensors import safe_open
    ref = {}
    for fn in glob.glob(srvdir + "/*.safetensors"):
        with safe_open(fn, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                ref[k] = fh.get_tensor(k)
    r = _remote(base_url, wm, str(tmp_path / "c1"))
    for k, v in ref.items():
        got = r.get_tensor(k)
        assert torch.equal(got.cpu(), v.cpu()), k
    s = r.stats()
    assert s["misses"] == len(ref) and s["hits"] == 0
    assert s["network_bytes"] > 0


def test_warm_pass_zero_http(served_shard, tmp_path):
    base_url, wm, _ = served_shard
    cache = str(tmp_path / "c2")
    r1 = _remote(base_url, wm, cache)
    keys = list(wm)
    for k in keys:
        r1.get_tensor(k)
    n1 = r1.fetcher.requests
    assert n1 > 0
    # brand-new instance, same cache dir: manifest + tensors persist
    r2 = _remote(base_url, wm, cache)
    for k in keys:
        r2.get_tensor(k)
    assert r2.fetcher.requests == 0, "warm pass must perform zero HTTP"
    assert r2.stats()["hits"] == len(keys)


def test_eviction_stays_exact(served_shard, tmp_path):
    base_url, wm, _ = served_shard
    import glob

    from safetensors import safe_open
    ref = {}
    # reference tensors (re-read served files)
    srvdir = RangeHandler.root
    for fn in glob.glob(srvdir + "/*.safetensors"):
        with safe_open(fn, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                ref[k] = fh.get_tensor(k)
    # cap smaller than one tensor (~256 B) forces evict-every-time
    r = _remote(base_url, wm, str(tmp_path / "c3"), cap_gb=256 / (1024 ** 3))
    for k, v in ref.items():
        assert torch.equal(r.get_tensor(k).cpu(), v.cpu()), k
    s = r.stats()
    assert s["cache_bytes"] <= 512, s
    # still exact after eviction churn
    for k, v in ref.items():
        assert torch.equal(r.get_tensor(k).cpu(), v.cpu()), k


def test_offline_uncached_raises(served_shard, tmp_path):
    base_url, wm, _ = served_shard
    r = _remote("http://127.0.0.1:9", wm, str(tmp_path / "c4"))  # dead port
    k = list(wm)[0]
    with pytest.raises(IOError):
        r.get_tensor(k)
    # cached key survives outage: fetch one key while live, then it is local
    r2 = _remote(base_url, wm, str(tmp_path / "c5"))
    r2.get_tensor(k)
    assert r2.stats()["misses"] == 1


class PartialLocal:
    """ShardHandles-shaped stub exposing only a subset of keys (simulates a
    partial checkout: some tensors on disk, the rest remote-only)."""

    def __init__(self, full: ShardHandles, keep):
        self._full = full
        self.weight_map = {k: v for k, v in full.weight_map.items() if k in keep}

    def shards(self):
        return sorted(set(self.weight_map.values()))

    def header(self, key):
        return self._full.header(key)

    def get_tensor(self, key):
        return self._full.get_tensor(key)


def test_tiered_local_first_no_remote(served_shard, tmp_path):
    base_url, wm, srvdir = served_shard
    local = ShardHandles.open(srvdir)
    keep = [list(wm)[0]]
    r = RemoteShardHandles(dict(wm), RangeFetcher(base_url),
                           str(tmp_path / "ct1"))
    t = TieredHandles(PartialLocal(local, keep), r)
    a = t.get_tensor(keep[0])
    b = local.get_tensor(keep[0])
    assert torch.equal(a.cpu(), b.cpu())
    assert r.fetcher.requests == 0, "local-first must not touch network"
    assert t.stats()["local_hits"] == 1


def test_tiered_remote_fill_exact(served_shard, tmp_path):
    base_url, wm, srvdir = served_shard
    local = ShardHandles.open(srvdir)
    keep = [list(wm)[0]]
    r = RemoteShardHandles(dict(wm), RangeFetcher(base_url),
                           str(tmp_path / "ct2"))
    t = TieredHandles(PartialLocal(local, keep), r)
    missing = [k for k in wm if k != keep[0]]
    for k in missing:
        assert torch.equal(t.get_tensor(k).cpu(), local.get_tensor(k).cpu()), k
    s = t.stats()
    assert s["remote_hits"] == len(missing)
    # full union accounted across tiers
    assert set(t.weight_map) == set(wm)
