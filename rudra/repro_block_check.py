#!/usr/bin/env python3
"""Minimal block_self_check repro (no full serve)."""
import json
import os
import sys
import torch

sys.path.insert(0, "/home/palakm/MoEServingSim/aditya/Bhaskera_col/rudra")

MODEL = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"
BLOCKH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-BLOCKH"
ANSH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-ANSH"

from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

# import wrapper class without running serve main
import importlib.util
spec = importlib.util.spec_from_file_location(
    "srv", "/home/palakm/MoEServingSim/aditya/Bhaskera_col/rudra/serve_deepseek_colossus.py")
print("loading module (imports only)...", flush=True)
# NOTE: module has no top-level side effects beyond imports/consts; guard anyway
import unittest.mock as _m  # noqa
srv = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(srv)
except SystemExit:
    pass
print("module loaded", flush=True)

dev = torch.device("cuda:0")
torch.cuda.init()
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
wm = json.load(open(MODEL + "/model.safetensors.index.json"))["weight_map"]
shards = sorted(set(wm.values()))
handles = {s: safe_open(os.path.join(MODEL, s), framework="pt", device="cpu") for s in shards[:3]}

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
moe = model.model.layers[1].mlp.to_empty(device="cpu")
# materialize gate + one expert + shared minimally for the check path
W = srv.DeepSeekColossusMoEWrapper
dma = torch.cuda.Stream(device=dev)
pool = srv.BlockPool(device=dev, budget_gb=1.0, block_cols=128)
wrap = W(layer_idx=1, moe_module=moe, cfg=cfg, device=dev, capacity=2,
         handles=handles, weight_map=wm, dma_stream=dma,
         model_dir=MODEL, block_pool=pool, block_store=BLOCKH, ans_store=ANSH)
print("wrapper built; running block_self_check...", flush=True)
slot_idx = wrap.slot_lru.pop(0)
print("assembling...", flush=True)
desc = wrap._block_assemble(0, slot_idx, kind="demand")
print(f"assembled: {len(desc['misses'])} misses, {len(desc['hits'])} hits", flush=True)
print("fetching+decoding...", flush=True)
fetched = wrap._block_fetch_decode([(p, m) for (p, m) in desc["misses"]])
print(f"fetched: {len(fetched)} blocks", flush=True)
print("installing block 0 only...", flush=True)
import torch as _t
pkey, m, lo, hi = fetched[0][0], fetched[0][1], fetched[0][2], fetched[0][3]
print(f"  lo {tuple(lo.shape)} {lo.device} hi {tuple(hi.shape)} {hi.device}", flush=True)
slot = wrap.slots[slot_idx]
_L, _E, _bi, _pname = pkey
w = {"gate_proj": slot.gate_proj.weight, "up_proj": slot.up_proj.weight,
     "down_proj": slot.down_proj.weight}[_pname]
print(f"  slot w {tuple(w.shape)} {w.device}", flush=True)
w8 = w.view(_t.uint8).reshape(w.shape[0], -1)
print("  view ok", flush=True)
print("  strided lo copy...", flush=True)
w8[0:128, 0::2].copy_(lo.reshape(128, -1))
print("  lo copy ok", flush=True)
print("  strided hi copy...", flush=True)
w8[0:128, 1::2].copy_(hi.reshape(128, -1))
print("  hi copy ok", flush=True)
print("  pool entry clone...", flush=True)
ent = w[0:128].detach().clone()
print("  clone ok", flush=True)
ok = wrap.block_self_check()
print("SELF_CHECK:", ok, flush=True)
