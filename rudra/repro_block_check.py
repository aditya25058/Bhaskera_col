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
print("decoding/installing...", flush=True)
wrap._block_decode_many([desc])
print("decode_many OK", flush=True)
ok = wrap.block_self_check()
print("SELF_CHECK:", ok, flush=True)
