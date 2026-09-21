#!/usr/bin/env python3
"""GPU ANS encoder for BF16 hi-bytes (Phase 2, lossless wire compression).

Reads expert tensors via os.pread (proven 6.4GB/s path), uploads hi bytes,
encodes with nvcomp ANS on GPU (native bitstream, GPU-decodable), writes
per-shard containers + index.json with the SAME schema as compress_hi_lz4
plus {"codec": "ans"} per entry. Single process (one GPU), sequential reads.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch

_VENV_SITE = "/home/palakm/MoEServingSim/aditya/venv/lib/python3.10/site-packages"


def load_nvcomp():
    import nvidia
    _nvd = os.path.join(_VENV_SITE, "nvidia")
    if _nvd not in list(nvidia.__path__):
        nvidia.__path__.append(_nvd)
    import nvidia.nvcomp as nvcomp
    return nvcomp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct")
    ap.add_argument("--out", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-ANSH")
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--experts", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device(args.device)
    nvcomp = load_nvcomp()
    import torch.utils.dlpack as dlpack
    codec = nvcomp.Codec(algorithm="ans")

    os.makedirs(args.out, exist_ok=True)
    wm = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    keys = []
    for l in range(1, args.layers):
        for e in range(args.experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                k = f"model.layers.{l}.mlp.experts.{e}.{proj}.weight"
                if k in wm:
                    keys.append(k)
    if args.limit:
        keys = keys[:args.limit]
    by_shard = {}
    for k in keys:
        by_shard.setdefault(wm[k], []).append(k)
    print(f"tensors: {len(keys)} in {len(by_shard)} shards", flush=True)

    index = {}
    t0 = time.perf_counter()
    n_done = 0
    for shard, ks in sorted(by_shard.items()):
        path = os.path.join(args.model, shard)
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
            data_start = 8 + hlen
        fd = os.open(path, os.O_RDONLY)
        fn = shard.replace(".safetensors", ".ansh")
        try:
            with open(os.path.join(args.out, fn), "wb") as out:
                for k in ks:
                    b, e = header[k]["data_offsets"]
                    n = e - b
                    raw = bytearray(n)
                    mv = memoryview(raw)
                    off, rem = data_start + b, n
                    while rem > 0:
                        chunk = os.pread(fd, rem, off)
                        if not chunk:
                            raise IOError(f"short read {k}")
                        mv[len(mv) - rem:len(mv) - rem + len(chunk)] = chunk
                        off += len(chunk)
                        rem -= len(chunk)
                    u8 = torch.frombuffer(raw, dtype=torch.uint8)
                    hi = u8[1::2].clone()
                    d_hi = hi.to(dev)
                    enc = codec.encode(nvcomp.as_array(d_hi))
                    enc_b = dlpack.from_dlpack(enc).cpu().numpy().tobytes()[:enc.size]
                    pos = out.tell()
                    out.write(enc_b)
                    index[k] = {"blob": fn, "offset": pos, "comp_len": len(enc_b),
                                "hi_len": len(hi), "shape": list(header[k]["shape"]),
                                "codec": "ans"}
                    n_done += 1
                    del raw, u8, hi, d_hi, enc
        finally:
            os.close(fd)
        if n_done % 2000 == 0:
            print(f"  ...{n_done}/{len(keys)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f)
    raw_hi = sum(v["hi_len"] for v in index.values())
    comp = sum(v["comp_len"] for v in index.values())
    print(f"DONE tensors {n_done} in {time.perf_counter() - t0:.1f}s: hi {raw_hi / 2**30:.2f}GB -> "
          f"{comp / 2**30:.2f}GB (ratio {comp / raw_hi:.3f})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
