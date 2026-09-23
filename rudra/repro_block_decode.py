#!/usr/bin/env python3
"""Bisect block-decode segfault: batched nvcomp decode of 64 small block blobs."""
import json
import os
import sys

sys.path.insert(0, "/home/palakm/MoEServingSim/aditya/Bhaskera_col/rudra")
import mmap as _mmap

import torch

BLOCKH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-BLOCKH"

import nvidia
nvidia.__path__.append("/home/palakm/MoEServingSim/aditya/venv/lib/python3.10/site-packages/nvidia")
import nvidia.nvcomp as _nv
import torch.utils.dlpack as dlpack

dev = torch.device("cuda:0")
torch.cuda.init()
idx = json.load(open(os.path.join(BLOCKH, "block_index.json")))
maps = {}
for fn in sorted(os.listdir(BLOCKH)):
    if fn.endswith(".bansh"):
        path = os.path.join(BLOCKH, fn)
        fh = open(path, "rb")
        mm = _mmap.mmap(fh.fileno(), 0, access=_mmap.ACCESS_COPY)
        maps[fn] = torch.frombuffer(mm, dtype=torch.uint8)

# all blocks of L1E0
keys = sorted([k for k in idx if k.startswith("1/0/")],
              key=lambda k: (k.rsplit("/", 1)[0], k))
print(f"blocks: {len(keys)}", flush=True)
codec = _nv.Codec(algorithm="ans")
comp_gs = []
for k in keys:
    m = idx[k]
    with open(os.path.join(BLOCKH, m["blob"]), "rb") as f:
        f.seek(m["offset"])
        cb = f.read(m["comp_len"])
    comp_gs.append(torch.frombuffer(bytearray(cb), dtype=torch.uint8).to(dev))
print("staged on GPU", flush=True)
arrs = [_nv.as_array(c) for c in comp_gs]
print("arrays built, decoding...", flush=True)
dec = codec.decode(arrs)
print("decodedOK n=", len(dec) if hasattr(dec, "__len__") else "?", flush=True)
tot = 0
for d, k in zip(dec, keys):
    m = idx[k]
    got = dlpack.from_dlpack(d)
    assert got.numel() == m["hi_len"], (k, got.numel(), m["hi_len"])
    tot += 1
print(f"ALL {tot} BLOCKS ROUNDTRIP OK", flush=True)
