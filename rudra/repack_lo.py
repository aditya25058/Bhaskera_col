#!/usr/bin/env python3
"""Offline lo-byte repack (Phase 2b, contiguous transport for ANS path).

Reads expert tensors sequentially per shard (readahead-friendly), extracts lo
(even) bytes contiguously, writes per-shard .lob files + lo_index.json
{same-tensor-key: {lob, offset, len}} into the ANS store dir.
Single process: kind to the shared disk. ~10 min for 236GB lo.
Runtime DMA becomes 2 contiguous transfers (lo + comp) instead of strided.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct")
    ap.add_argument("--store", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-ANSH")
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--experts", type=int, default=160)
    args = ap.parse_args()

    wm = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    keys = []
    for l in range(1, args.layers):
        for e in range(args.experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                k = f"model.layers.{l}.mlp.experts.{e}.{proj}.weight"
                if k in wm:
                    keys.append(k)
    by_shard = {}
    for k in keys:
        by_shard.setdefault(wm[k], []).append(k)
    print(f"tensors: {len(keys)} in {len(by_shard)} shards", flush=True)

    index = {}
    t0 = time.perf_counter()
    total_lo = 0
    for shard, ks in sorted(by_shard.items()):
        path = os.path.join(args.model, shard)
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
            data_start = 8 + hlen
            # One sequential pass per shard: read whole expert byte ranges,
            # gather lo bytes, write contiguous container.
            fn = shard.replace(".safetensors", ".lob")
            with open(os.path.join(args.store, fn), "wb") as out:
                for k in ks:
                    b, e = header[k]["data_offsets"]
                    f.seek(data_start + b)
                    raw = f.read(e - b)
                    assert len(raw) == e - b, (k, len(raw))
                    lob = raw[0::2]
                    off = out.tell()
                    out.write(lob)
                    index[k] = {"lob": fn, "offset": off, "len": len(lob),
                                "shape": list(header[k]["shape"])}
                    total_lo += len(lob)
        done = len(index)
        if done % 5000 == 0:
            print(f"  ...{done}/{len(keys)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.store, "lo_index.json"), "w") as f:
        json.dump(index, f)
    print(f"DONE tensors {len(index)} in {time.perf_counter() - t0:.1f}s: "
          f"lo {total_lo / 2**30:.2f}GB", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
