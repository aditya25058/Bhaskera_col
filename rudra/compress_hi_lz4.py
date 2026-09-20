#!/usr/bin/env python3
"""Offline LZ4-HC compressor for BF16 hi-bytes (Phase 2, lossless wire compression).

For every routed-expert tensor in DeepSeek-Coder-V2, splits BF16 bytes into
lo (mantissa noise, incompressible -> DMA raw from original shards) and hi
(sign+exp, structured) and compresses hi with LZ4-HC-max. Output per shard:
  <out_dir>/<shard>.lz4h      concatenated hi-comp blobs
  <out_dir>/index.json        {tensor_key: {blob, offset, comp_len, hi_len, shape}}
Runtime DMA per expert: 22.5MB lo (strided) + ~12.9MB hi-comp instead of 45MB.
Exact: hi-comp decompresses bit-identically; interleave reproduces BF16.
Multiprocessing over experts. One-time cost.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import lz4.frame
import torch
from safetensors import safe_open


def expert_keys(cfg_n_layers, n_routed):
    for l in range(1, cfg_n_layers):
        for e in range(n_routed):
            pfx = f"model.layers.{l}.mlp.experts.{e}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield f"{pfx}.{proj}.weight"


def compress_shard(args):
    model_path, shard, keys, level = args
    h = safe_open(os.path.join(model_path, shard), framework="pt")
    out = []
    for key in keys:
        t = h.get_tensor(key)
        raw = bytes(t.untyped_storage())[:t.nbytes]
        u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        hi = u8[1::2].numpy().tobytes()
        blob = lz4.frame.compress(hi, compression_level=level)
        out.append((key, blob, len(hi), tuple(t.shape)))
    return shard, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct")
    ap.add_argument("--out", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-LZ4H")
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--experts", type=int, default=160)
    ap.add_argument("--level", type=int, default=12)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0=all, else first N tensors (smoke)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wm = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    keys = [k for k in expert_keys(args.layers, args.experts) if k in wm]
    if args.limit:
        keys = keys[:args.limit]
    by_shard = {}
    for k in keys:
        by_shard.setdefault(wm[k], []).append(k)
    print(f"tensors: {len(keys)} in {len(by_shard)} shards, workers: {args.workers}, level: {args.level}", flush=True)

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        jobs = [(args.model, sh, ks, args.level) for sh, ks in sorted(by_shard.items())]
        shard_out = pool.map(compress_shard, jobs, chunksize=1)
    print(f"compressed in {time.perf_counter() - t0:.1f}s", flush=True)

    # Pack per-shard containers + index
    index = {}
    files = {}
    n_tensors = 0
    for shard, items in shard_out:
        fn = shard.replace(".safetensors", ".lz4h")
        if fn not in files:
            files[fn] = open(os.path.join(args.out, fn), "wb")
        for key, blob, hi_len, shape in items:
            off = files[fn].tell()
            files[fn].write(blob)
            index[key] = {"blob": fn, "offset": off, "comp_len": len(blob),
                          "hi_len": hi_len, "shape": list(shape)}
            n_tensors += 1
    for f in files.values():
        f.close()
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f)
    raw_hi = sum(v["hi_len"] for v in index.values())
    comp = sum(v["comp_len"] for v in index.values())
    print(f"tensors {n_tensors}: hi raw {raw_hi / 2**30:.2f}GB -> comp {comp / 2**30:.2f}GB "
          f"(ratio {comp / raw_hi:.3f}); files: {len(files)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
