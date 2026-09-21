#!/usr/bin/env python3
"""Standalone nvcomp LZ4 batched-decompress check (ctypes, no serve dependency).

Decompresses ONE hi-blob from the LZ4H store on GPU and compares against
lz4.frame CPU output. Validates ctypes signatures before serve integration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_VENV_SITE = "/home/palakm/MoEServingSim/aditya/venv/lib/python3.10/site-packages"


def load_nvcomp():
    """Import official nvidia.nvcomp bindings.

    ~/.local/.../nvidia/__init__.py (regular package) shadows the venv's
    namespace dir, so extend __path__ to expose the venv's nvcomp subpackage.
    """
    import nvidia
    _nvd = os.path.join(_VENV_SITE, "nvidia")
    if _nvd not in list(nvidia.__path__):
        nvidia.__path__.append(_nvd)
    import nvidia.nvcomp as nvcomp
    return nvcomp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-LZ4H")
    ap.add_argument("--key", default="model.layers.1.mlp.experts.0.gate_proj.weight")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device(args.device)
    idx = json.load(open(os.path.join(args.store, "index.json")))
    meta = idx[args.key]
    with open(os.path.join(args.store, meta["blob"]), "rb") as f:
        f.seek(meta["offset"])
        comp = f.read(meta["comp_len"])
    print(f"blob {meta['comp_len']}B -> hi {meta['hi_len']}B")

    import lz4.frame
    ref = lz4.frame.decompress(comp)
    assert len(ref) == meta["hi_len"], (len(ref), meta["hi_len"])

    nvcomp = load_nvcomp()
    codec = nvcomp.Codec(algorithm="lz4")
    d_in = torch.frombuffer(bytearray(comp), dtype=torch.uint8).to(dev)
    src = nvcomp.as_array(d_in)
    res = codec.decode(src)
    print("decode type:", type(res))
    if isinstance(res, (bytes, bytearray, memoryview)):
        got = bytes(res)
    else:
        import torch.utils.dlpack as dlpack
        got = dlpack.from_dlpack(res).cpu().numpy().tobytes()
    print(f"decompressed {len(got)}B equal={got == ref}")
    assert got == ref, "GPU decompress mismatch!"
    print("NVCOMP ROUNDTRIP PASS")


if __name__ == "__main__":
    raise SystemExit(main())
