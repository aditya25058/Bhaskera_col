#!/usr/bin/env python3
"""GPU block-ANS encoder (column-granular store, Phase 1C-real).

Per expert tensor (I=1536), splits hi bytes into 128-col blocks (12 blocks),
ANS-encodes each block on GPU (native bitstream, independently decodable),
writes per-shard .bansh containers + block_index.json:
  {(layer, expert, block, proj): {blob, offset, comp_len, hi_len}}
lo blocks need NO new store: they slice arithmetically from the .lob repack.
Single process, sequential reads (disk-kind). Serve decodes per-layer batches.
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
BLOCK = 128


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
    ap.add_argument("--out", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-BLOCKH")
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--experts", type=int, default=160)
    ap.add_argument("--block", type=int, default=BLOCK)
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
                    keys.append((l, e, proj, k))
    if args.limit:
        keys = keys[:args.limit]
    by_shard = {}
    for l, e, proj, k in keys:
        by_shard.setdefault(wm[k], []).append((l, e, proj, k))
    print(f"tensors: {len(keys)} in {len(by_shard)} shards, block={args.block}", flush=True)

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
        fn = shard.replace(".safetensors", ".bansh")
        try:
            with open(os.path.join(args.out, fn), "wb") as out:
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
                    u8 = torch.frombuffer(raw, dtype=torch.uint8)
                    hi = u8[1::2].reshape(I0, H0)
                    nb = (I0 + args.block - 1) // args.block
                    arrs, metas = [], []
                    for bi in range(nb):
                        r0, r1 = bi * args.block, min((bi + 1) * args.block, I0)
                        blk = hi[r0:r1].contiguous()
                        d = blk.to(dev)
                        arrs.append(nvcomp.as_array(d))
                        metas.append((bi, r1 - r0))
                    enc = codec.encode(arrs)
                    if not isinstance(enc, list):
                        enc = [enc]
                    for (bi, nrows), a in zip(metas, enc):
                        # a.size = ACTUAL payload bytes (buffer may be padded to max)
                        eb_full = dlpack.from_dlpack(a).cpu().numpy().tobytes()
                        eb = eb_full[:a.size]
                        pos = out.tell()
                        out.write(eb)
                        index[f"{l}/{e}/{bi}/{proj}"] = {
                            "blob": fn, "offset": pos, "comp_len": len(eb),
                            "hi_len": nrows * H0, "rows": nrows}
                    n_done += 1
                    del raw, u8, hi
        finally:
            os.close(fd)
        if n_done % 2000 == 0:
            print(f"  ...{n_done}/{len(keys)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.out, "block_index.json"), "w") as f:
        json.dump(index, f)
    comp = sum(v["comp_len"] for v in index.values())
    print(f"DONE tensors {n_done} in {time.perf_counter() - t0:.1f}s: comp {comp / 2**30:.2f}GB", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
