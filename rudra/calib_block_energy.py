#!/usr/bin/env python3
"""Static block-energy calibration (B1, zero runtime scoring cost).

Expected SwiGLU energy under isotropic x is proportional to
||gate_col||^2 * ||up_col||^2. Rank 128-row blocks per expert by summed
proxy energy once, offline, on CPU. Output calib.json:
  {"L/E": {"gate_proj": [top-N block ids], "up_proj": [...]}}
Runtime transfers top-N blocks per needed expert (N=1/3 of blocks);
complement via pool/demand. Down kept whole (output-partition axis).
Untimed, one-time, exactness-preserving (ranking only affects movement).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time

import numpy as np

BLOCK = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct")
    ap.add_argument("--out", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-CALIB")
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--experts", type=int, default=160)
    ap.add_argument("--frac", type=float, default=1.0 / 3.0)
    ap.add_argument("--block", type=int, default=BLOCK)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wm = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    import numpy as np
    import torch

    by_shard = {}
    for l in range(1, args.layers):
        for e in range(args.experts):
            for proj in ("gate_proj", "up_proj"):
                k = f"model.layers.{l}.mlp.experts.{e}.{proj}.weight"
                if k in wm:
                    by_shard.setdefault(wm[k], []).append((l, e, proj, k))
    calib = {}
    # Gate and up projections may live in DIFFERENT shards: accumulate per-block
    # squared norms (lightweight) per shard, combine across shards at the end.
    scores = {}  # (l, e, proj) -> [(score, bi)]
    t0 = time.perf_counter()
    n_done = 0
    for shard, ks in sorted(by_shard.items()):
        path = os.path.join(args.model, shard)
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
            data_start = 8 + hlen
            fd = os.open(path, os.O_RDONLY)
            try:
                for l, e, proj, k in ks:
                    b, en = header[k]["data_offsets"]
                    n = en - b
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
                    I0, H0 = header[k]["shape"][0], header[k]["shape"][1]
                    w = torch.frombuffer(raw, dtype=torch.uint8).view(
                        torch.bfloat16).reshape(I0, H0).float().numpy()
                    nb = (I0 + args.block - 1) // args.block
                    bl = []
                    for bi in range(nb):
                        r0, r1 = bi * args.block, min((bi + 1) * args.block, I0)
                        bl.append((float((w[r0:r1] ** 2).sum()), bi))
                    scores[(l, e, proj)] = bl
                    n_done += 1
                    del raw, w
            finally:
                os.close(fd)
        if n_done % 2000 == 0:
            print(f"  ...{n_done} tensors ({time.perf_counter() - t0:.0f}s)", flush=True)
    n_exp = 0
    for l in range(1, args.layers):
        for e in range(args.experts):
            g = scores.get((l, e, "gate_proj"))
            u = scores.get((l, e, "up_proj"))
            if not g or not u:
                continue
            gd = dict((bi, s) for s, bi in g)
            nb = len(g)
            ranked = sorted(((gd.get(bi, 0.0) * s, bi) for s, bi in u), reverse=True)
            ntop = max(2, int(nb * args.frac + 0.5))
            top = sorted(bi for _, bi in ranked[:ntop])
            calib[f"{l}/{e}"] = {"gate_proj": top, "up_proj": top,
                                 "nblocks": nb, "ntop": ntop}
            n_exp += 1
    with open(os.path.join(args.out, "calib.json"), "w") as f:
        json.dump(calib, f)
    print(f"DONE {len(calib)} experts in {time.perf_counter() - t0:.1f}s -> {args.out}/calib.json",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
