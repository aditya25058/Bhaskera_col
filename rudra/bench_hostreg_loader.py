#!/usr/bin/env python3
"""Phase 1c-R loader benchmark: A/B/C on identical 45MB expert transfers.

A: current safe_open + mmap (pageable source, driver-staged DMA)
B: custom aligned mmap (whole-shard mapping, unregistered)
C: custom aligned mmap + ONE-TIME cudaHostRegister of whole shard mapping(s)
Measures: registration time/ms, DMA ms (CUDA events), effective GB/s,
exact byte identity vs A, registration call counts. No server integration.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import time

import torch
from safetensors import safe_open


def load_shard_index(model_path):
    with open(f"{model_path}/model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def expert_keys(layer_idx, expert_id):
    pfx = f"model.layers.{layer_idx}.mlp.experts.{expert_id}"
    return [f"{pfx}.gate_proj.weight", f"{pfx}.up_proj.weight", f"{pfx}.down_proj.weight"]


class ShardMap:
    """Whole-file aligned mmap + parsed safetensors header. One mapping per shard."""

    _cache = {}

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        raw_len = self.f.read(8)
        self.header_len = int.from_bytes(raw_len, "little")
        self.header = json.loads(self.f.read(self.header_len))
        self.data_start = 8 + self.header_len
        # ACCESS_COPY: writeable mapping view (no disk I/O for reads, private
        # pages only materialize on write — we never write), required because
        # torch.frombuffer needs a writeable buffer object.
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_COPY)
        # Whole-mapping uint8 view (zero-copy); slices below are views, not copies.
        self.u8 = torch.frombuffer(self.mm, dtype=torch.uint8)

    @classmethod
    def get(cls, path):
        if path not in cls._cache:
            cls._cache[path] = ShardMap(path)
        return cls._cache[path]

    def view_tensor(self, key):
        """Zero-copy BF16 view into the mapping (for DMA timing, no CPU copy)."""
        info = self.header[key]
        b, e = info["data_offsets"]
        n = e - b
        s = self.data_start + b
        return self.u8[s:s + n].view(torch.bfloat16).reshape(info["shape"])


_DTYPE_BYTES = {"BF16": 2}


def get_cudart():
    lib = ctypes.util.find_library("cudart")
    return ctypes.CDLL(lib) if lib else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device(args.device)
    torch.cuda.init()
    cu = get_cudart()
    assert cu is not None, "libcudart not found"
    cu.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    cu.cudaHostUnregister.argtypes = [ctypes.c_void_p]

    wm = load_shard_index(args.model)
    keys = expert_keys(args.layer, args.expert)
    print(f"expert L{args.layer}E{args.expert}:")
    for k in keys:
        print(f"  {k.split('.')[-2]} <- {wm[k]}")

    # Reference tensors via path A (also exactness ground truth)
    handles = {}
    ref = []
    for k in keys:
        sh = wm[k]
        if sh not in handles:
            handles[sh] = safe_open(f"{args.model}/{sh}", framework="pt")
        ref.append(handles[sh].get_tensor(k))
    total_mb = sum(t.nbytes for t in ref) / (1024 ** 2)
    print(f"total payload: {total_mb:.1f} MB")

    # Exactness: custom views == safe_open tensors
    views = [ShardMap.get(f"{args.model}/{wm[k]}").view_tensor(k) for k in keys]
    exact = all(torch.equal(r.cpu(), v.cpu()) for r, v in zip(ref, views))
    print(f"byte identity custom-vs-safe_open: {exact}")
    assert exact, "custom loader mismatch!"

    gpu = [torch.empty_like(ref[0]).to(dev), torch.empty_like(ref[1]).to(dev), torch.empty_like(ref[2]).to(dev)]
    stream = torch.cuda.Stream(device=dev)
    N = args.iters

    def bench(srcs, label):
        for _ in range(3):  # warmup
            with torch.cuda.stream(stream):
                for g, s in zip(gpu, srcs):
                    g.copy_(s, non_blocking=True)
        torch.cuda.synchronize(dev)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record(stream)
        with torch.cuda.stream(stream):
            for _ in range(N):
                for g, s in zip(gpu, srcs):
                    g.copy_(s, non_blocking=True)
        ev1.record(stream)
        ev1.synchronize()
        ms = ev0.elapsed_time(ev1) / N
        gbps = (total_mb / 1024) / (ms / 1000.0)
        print(f"{label}: {ms:.4f} ms/iter ({total_mb:.1f} MB) -> {gbps:.2f} GB/s")
        return ms, gbps

    out = {"payload_mb": total_mb, "iters": N}
    ms_a, gb_a = bench(ref, "A safe_open+mmap   ")
    out["A"] = {"ms": ms_a, "gbps": gb_a}
    ms_b, gb_b = bench(views, "B custom mmap      ")
    out["B"] = {"ms": ms_b, "gbps": gb_b}

    # C: register each distinct whole-shard mapping ONCE
    t0 = time.perf_counter()
    reg_calls, reg_bytes = 0, 0
    for path, sm in ShardMap._cache.items():
        err = cu.cudaHostRegister(ctypes.c_void_p(sm.u8.data_ptr()),
                                  ctypes.c_size_t(sm.u8.nbytes), ctypes.c_uint(0))
        assert int(err) == 0, f"register failed for {path}: {err}"
        reg_calls += 1
        reg_bytes += sm.u8.nbytes
    reg_ms = (time.perf_counter() - t0) * 1000.0
    print(f"C registration: {reg_calls} calls, {reg_bytes / 1024**3:.2f} GB, {reg_ms:.2f} ms "
          f"({reg_ms / max(1, reg_calls):.2f} ms/call)")
    ms_c, gb_c = bench(views, "C mmap+HostRegister")
    out["C"] = {"ms": ms_c, "gbps": gb_c, "reg_calls": reg_calls,
                "reg_gb": reg_bytes / 1024**3, "reg_ms": reg_ms}

    # Exactness after registered DMA
    with torch.cuda.stream(stream):
        for g, s in zip(gpu, views):
            g.copy_(s, non_blocking=True)
    torch.cuda.synchronize(dev)
    exact_gpu = all(torch.equal(g.cpu(), r.cpu()) for g, r in zip(gpu, ref))
    print(f"byte identity post-DMA: {exact_gpu}")
    out["exact"] = bool(exact and exact_gpu)
    print("PASS" if out["exact"] else "FAIL", "| A->C speedup: %.2fx" % (ms_a / ms_c))
    with open("/home/palakm/MoEServingSim/aditya/hostreg_loader_bench.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
