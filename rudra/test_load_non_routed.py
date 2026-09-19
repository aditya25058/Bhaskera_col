#!/usr/bin/env python3
"""
test_load_non_routed.py
=======================
Tests loading all 782 non-routed tensors (~26 GB) across Dual H100 NVL:
- GPU 0: Embeddings, Layer 0 Dense, Layers 1-29 Attention MLA, RMSNorms, Shared Experts
- GPU 1: Layers 30-59 Attention MLA, RMSNorms, Shared Experts, Final RMSNorm, LM Head
"""

import os
import time
import json
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"

print("=" * 80)
print("  TEST LOADING NON-ROUTED TENSORS ON DUAL H100 NVL")
print("=" * 80)

dev0 = torch.device("cuda:0")
dev1 = torch.device("cuda:1")

cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 1. Inspect index map
with open(f"{MODEL_PATH}/model.safetensors.index.json", "r") as f:
    weight_map = json.load(f)["weight_map"]

non_routed_keys = {k: v for k, v in weight_map.items() if not (".mlp.experts." in k and ".shared_experts" not in k)}
print(f"Total weights: {len(weight_map)} | Non-routed weights: {len(non_routed_keys)}")

# Categorize keys by target device
gpu0_keys = {}
gpu1_keys = {}

for k, shard in non_routed_keys.items():
    if k.startswith("model.embed_tokens."):
        gpu0_keys[k] = shard
    elif k.startswith("model.layers."):
        layer_idx = int(k.split(".")[2])
        if layer_idx < 30:
            gpu0_keys[k] = shard
        else:
            gpu1_keys[k] = shard
    else:  # model.norm, lm_head
        gpu1_keys[k] = shard

print(f"GPU 0 target keys: {len(gpu0_keys)}")
print(f"GPU 1 target keys: {len(gpu1_keys)}")

# 2. Instantiate meta model
print("\nInstantiating meta model skeleton...")
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)

# Group shards needed
shards_to_load = sorted(list(set(non_routed_keys.values())))
print(f"Non-routed tensors span {len(shards_to_load)} shards.")

# 3. Load non-routed tensors directly to target GPU and assign to model
from accelerate.utils import set_module_tensor_to_device

t0 = time.perf_counter()
total_loaded_bytes = 0

for shard_file in shards_to_load:
    shard_path = os.path.join(MODEL_PATH, shard_file)
    handle = safe_open(shard_path, framework="pt", device="cpu")
    for k in handle.keys():
        if k in gpu0_keys:
            tensor = handle.get_tensor(k)
            set_module_tensor_to_device(model, k, dev0, value=tensor.to(torch.bfloat16))
            total_loaded_bytes += tensor.nbytes
        elif k in gpu1_keys:
            tensor = handle.get_tensor(k)
            set_module_tensor_to_device(model, k, dev1, value=tensor.to(torch.bfloat16))
            total_loaded_bytes += tensor.nbytes


torch.cuda.synchronize(dev0)
torch.cuda.synchronize(dev1)
dt = time.perf_counter() - t0

gb_loaded = total_loaded_bytes / (1024**3)
rate = gb_loaded / dt
print(f"\nLoaded {gb_loaded:.2f} GB of non-routed tensors across 2 GPUs in {dt:.2f}s ({rate:.2f} GB/s)")
print(f"  GPU 0 Allocated: {torch.cuda.memory_allocated(dev0) / (1024**3):.2f} GB / {torch.cuda.get_device_properties(dev0).total_memory / (1024**3):.1f} GB")
print(f"  GPU 1 Allocated: {torch.cuda.memory_allocated(dev1) / (1024**3):.2f} GB / {torch.cuda.get_device_properties(dev1).total_memory / (1024**3):.1f} GB")
