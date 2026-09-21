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


def expert_keys(cfg_n_layers, n_routed):
    for l in range(1, cfg_n_layers):
        for e in range(n_routed):
            pfx = f"model.layers.{l}.mlp.experts.{e}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield f"{pfx}.{proj}.weight"


def read_shard_header(path):
    import struct
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(hlen)), 8 + hlen


def compress_shard(args):
    # Worker writes its shard container directly; returns only metadata.
    # (Parent gathering 28k blobs would OOM: ~126GB pickled.)
    # Reads via os.pread (6.4GB/s proven): torch bytes(storage) path runs at
    # 0.5MB/s on mmap-backed safetensors tensors on this host (root cause TBD).
    model_path, shard, keys, level, out_dir = args
    header, data_start = read_shard_header(os.path.join(model_path, shard))
    fd = os.open(os.path.join(model_path, shard), os.O_RDONLY)
    fn = shard.replace(".safetensors", ".lz4h")
    meta = []
    try:
        with open(os.path.join(out_dir, fn), "wb") as f:
            for key in keys:
                b, e = header[key]["data_offsets"]
                n = e - b
                raw = bytearray(n)
                mv = memoryview(raw)
                off = data_start + b
                while n > 0:
                    chunk = os.pread(fd, n, off)
                    if not chunk:
                        raise IOError(f"short read {key}")
                    mv[len(mv) - n:len(mv) - n + len(chunk)] = chunk
                    off += len(chunk)
                    n -= len(chunk)
                u8 = torch.frombuffer(raw, dtype=torch.uint8)
                hi = u8[1::2].numpy().tobytes()
                blob = lz4.frame.compress(hi, compression_level=level)
                pos = f.tell()
                f.write(blob)
                meta.append((key, pos, len(blob), len(hi), tuple(header[key]["shape"])))
                del raw, u8, hi, blob
    finally:
        os.close(fd)
    return shard, fn, meta


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
        jobs = [(args.model, sh, ks, args.level, args.out) for sh, ks in sorted(by_shard.items())]
        shard_out = pool.map(compress_shard, jobs, chunksize=1)
    print(f"compressed in {time.perf_counter() - t0:.1f}s", flush=True)

    # Assemble index from worker-returned metadata (small)
    index = {}
    files = set()
    n_tensors = 0
    for shard, fn, items in shard_out:
        files.add(fn)
        for key, off, comp_len, hi_len, shape in items:
            index[key] = {"blob": fn, "offset": off, "comp_len": comp_len,
                          "hi_len": hi_len, "shape": list(shape)}
            n_tensors += 1
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f)
    raw_hi = sum(v["hi_len"] for v in index.values())
    comp = sum(v["comp_len"] for v in index.values())
    print(f"tensors {n_tensors}: hi raw {raw_hi / 2**30:.2f}GB -> comp {comp / 2**30:.2f}GB "
          f"(ratio {comp / raw_hi:.3f}); files: {len(files)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
