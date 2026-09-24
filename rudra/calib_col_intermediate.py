#!/usr/bin/env python3
"""H2a: intermediate-dim energy calibration (down-COLUMNS included).

Per-expert energy over intermediate dim i (1536):
  E[i] = ||gate[i,:]||^2 + ||up[i,:]||^2 + ||down[:,i]||^2
Rank once, offline, on CPU. Output calib_col.json:
  {"L/E": {"hot": [top-512 idxs, ascending], "n": 1536, "ntop": 512}}
Ranking affects movement/split only (exactness-preserving).

Also (--verify N): split-protocol check on N sample experts x 10 tokens:
  y_hot + y_cold (native suborders) vs full native -> max diff + bitwise rate.
Expected: ulp-level (reassociation), NOT bitwise (H1 already killed that).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct")
    ap.add_argument("--out", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-CALIBCOL")
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--experts", type=int, default=160)
    ap.add_argument("--frac", type=float, default=1.0 / 3.0)
    ap.add_argument("--verify", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wm = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
    by_shard = {}
    for l in range(1, args.layers):
        for e in range(args.experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                k = f"model.layers.{l}.mlp.experts.{e}.{proj}.weight"
                if k in wm:
                    by_shard.setdefault(wm[k], []).append((l, e, proj, k))

    energy = {}  # (l,e) -> np f32[1536] accumulator
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
                    acc = energy.setdefault((l, e), None)
                    if proj in ("gate_proj", "up_proj"):
                        part = (w ** 2).sum(axis=1)  # per intermediate row
                    else:
                        part = (w ** 2).sum(axis=0)  # per intermediate COLUMN
                    energy[(l, e)] = part if acc is None else acc + part
                    n_done += 1
                    del raw, w
            finally:
                os.close(fd)
        if n_done % 6000 == 0:
            print(f"  ...{n_done} tensors ({time.perf_counter() - t0:.0f}s)", flush=True)

    calib = {}
    for (l, e), acc in energy.items():
        n = len(acc)
        ntop = max(64, int(n * args.frac + 0.5))
        order = np.argsort(-acc)
        calib[f"{l}/{e}"] = {"hot": sorted(int(i) for i in order[:ntop]),
                             "n": n, "ntop": ntop}
    with open(os.path.join(args.out, "calib_col.json"), "w") as f:
        json.dump(calib, f)
    print(f"DONE {len(calib)} experts in {time.perf_counter() - t0:.1f}s -> {args.out}/calib_col.json",
          flush=True)

    if args.verify > 0:
        from safetensors import safe_open
        torch.set_num_threads(6)
        keys = sorted(calib)[:args.verify]
        g = torch.Generator().manual_seed(0)
        worst, n_eq, n_tot = 0.0, 0, 0
        t1 = time.perf_counter()
        for ke in keys:
            l, e = ke.split("/")
            pfx = f"model.layers.{l}.mlp.experts.{e}."
            with safe_open(os.path.join(args.model, wm[pfx + "gate_proj.weight"]),
                           framework="pt", device="cpu") as fh:
                wg = fh.get_tensor(pfx + "gate_proj.weight")
                wu = fh.get_tensor(pfx + "up_proj.weight")
                wd = fh.get_tensor(pfx + "down_proj.weight")
            hot = torch.tensor(calib[ke]["hot"])
            cold = torch.tensor([i for i in range(calib[ke]["n"]) if i not in set(calib[ke]["hot"])])
            H = wg.shape[1]
            for _ in range(10):
                x = torch.randn(1, H, dtype=torch.bfloat16, generator=g)
                with torch.no_grad():
                    a = torch.nn.functional.silu(torch.nn.functional.linear(x, wg)) * \
                        torch.nn.functional.linear(x, wu)
                    ref = torch.nn.functional.linear(a, wd)
                    yh = torch.nn.functional.linear(a[:, hot], wd[:, hot])
                    yc = torch.nn.functional.linear(a[:, cold], wd[:, cold])
                    y = (yh.float() + yc.float()).to(torch.bfloat16)
                n_eq += bool(torch.equal(y, ref))
                worst = max(worst, float((y.float() - ref.float()).abs().max()))
                n_tot += 1
        print(f"VERIFY {n_eq}/{n_tot} bitwise, worst|diff|={worst:.3e} "
              f"({time.perf_counter() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
