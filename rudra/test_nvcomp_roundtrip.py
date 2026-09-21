#!/usr/bin/env python3
"""Standalone nvcomp LZ4 batched-decompress check (ctypes, no serve dependency).

Decompresses ONE hi-blob from the LZ4H store on GPU and compares against
lz4.frame CPU output. Validates ctypes signatures before serve integration.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os

import torch


def load_nvcomp():
    import ctypes.util
    lib = ctypes.util.find_library("nvcomp")
    if lib is None:
        import glob
        cands = glob.glob(os.path.expanduser("~") + "/MoEServingSim/aditya/venv/lib/python3.10/site-packages/nvidia/libnvcomp/lib64/libnvcomp.so*")
        cands += ["/usr/local/cuda/lib64/libnvcomp.so"]
        lib = next((c for c in cands if os.path.exists(c)), None)
    assert lib, "libnvcomp not found"
    return ctypes.CDLL(lib)


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

    cu = load_nvcomp()
    # Signatures (nvcomp v5 C API)
    cu.nvcompBatchedLZ4DecompressGetTempSizeSync.argtypes = [ctypes.c_size_t, ctypes.c_size_t,
                                                             ctypes.POINTER(ctypes.c_size_t)]
    cu.nvcompBatchedLZ4DecompressGetTempSizeSync.restype = ctypes.c_int
    cu.nvcompBatchedLZ4DecompressAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    cu.nvcompBatchedLZ4DecompressAsync.restype = ctypes.c_int

    d_in = torch.frombuffer(bytearray(comp), dtype=torch.uint8).to(dev)
    d_out = torch.empty(meta["hi_len"], dtype=torch.uint8, device=dev)
    max_uncomp = meta["hi_len"]
    temp_need = ctypes.c_size_t(0)
    st = cu.nvcompBatchedLZ4DecompressGetTempSizeSync(1, max_uncomp, ctypes.byref(temp_need))
    assert st == 0, st
    d_temp = torch.empty(int(temp_need.value), dtype=torch.uint8, device=dev)

    def dev_ptr_array(ptrs):
        host = torch.tensor(ptrs, dtype=torch.int64)
        return host.to(dev)

    in_ptrs = dev_ptr_array([d_in.data_ptr()])
    in_bytes = dev_ptr_array([len(comp)])
    uncomp_bytes = dev_ptr_array([meta["hi_len"]])
    actual = torch.empty(1, dtype=torch.int64, device=dev)
    out_ptrs = dev_ptr_array([d_out.data_ptr()])
    stream = torch.cuda.current_stream(dev).cuda_stream
    st = cu.nvcompBatchedLZ4DecompressAsync(
        ctypes.c_void_p(in_ptrs.data_ptr()), ctypes.c_void_p(in_bytes.data_ptr()),
        ctypes.c_void_p(uncomp_bytes.data_ptr()), ctypes.c_void_p(actual.data_ptr()),
        1, ctypes.c_void_p(d_temp.data_ptr()), int(temp_need.value),
        ctypes.c_void_p(out_ptrs.data_ptr()), ctypes.c_void_p(stream))
    assert st == 0, st
    torch.cuda.synchronize(dev)
    got = d_out.cpu().numpy().tobytes()
    print(f"actual_uncomp={int(actual.cpu()[0])} equal={got == ref}")
    assert got == ref, "GPU decompress mismatch!"
    print("NVCOMP ROUNDTRIP PASS")


if __name__ == "__main__":
    raise SystemExit(main())
