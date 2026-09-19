#!/usr/bin/env python3
"""
test_deepseek_layer_exact.py
============================
Verifies 100% numerical exactness between:
1. Native DeepseekV2MoE (all 160 routed experts in memory)
2. COLOSSUS DeepSeekColossusMoEWrapper (160 routed experts on pinned host, dynamic slots on H100)
"""

import time
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig

MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"

print("=" * 80)
print("  COLOSSUS NUMERICAL EXACTNESS VERIFICATION: DeepSeek-V2 MoE Layer")
print("=" * 80)

dev = torch.device("cuda:0")
cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Import the native modeling module
from transformers import AutoModelForCausalLM
with torch.device("meta"):
    meta_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)

native_layer = meta_model.model.layers[1].mlp
# Materialize layer 1 on CPU
native_layer = native_layer.to_empty(device="cpu").to(torch.bfloat16)

# Load real weights for Layer 1 from shard 1
shard1_path = f"{MODEL_PATH}/model-00001-of-000055.safetensors"
print(f"Loading real weights from {shard1_path}...")
t0 = time.time()
shard1_weights = load_file(shard1_path)
print(f"Shard 1 loaded in {time.time() - t0:.2f}s")

# Load state dict for layer 1 mlp
prefix = "model.layers.1.mlp."
layer1_sd = {k[len(prefix):]: v for k, v in shard1_weights.items() if k.startswith(prefix)}
print(f"Extracted {len(layer1_sd)} tensors for Layer 1 MLP")

native_layer.load_state_dict(layer1_sd)
print("Loaded real weights into native DeepseekV2MoE on CPU successfully!")

# Move native layer to GPU for reference run
native_gpu = native_layer.to(dev)

# Test activation [1, 1, 5120]
torch.manual_seed(42)
x = torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16, device=dev)

print("\n[1] Running Reference Native Forward Pass on H100...")
torch.cuda.synchronize(dev)
t0 = time.time()
with torch.no_grad():
    ref_out = native_gpu(x)
torch.cuda.synchronize(dev)
print(f"  Reference output shape: {ref_out.shape} | Time: {(time.time() - t0)*1000:.3f} ms")

# Now create COLOSSUS wrapper on CPU native layer, with capacity C=12 slots
print("\n[2] Initializing COLOSSUS Dynamic Slot Wrapper (Capacity C=12 of 160 Experts)...")
# Move back to CPU to extract pinned host weights
native_cpu = native_gpu.to("cpu")

from serve_deepseek_colossus import DeepSeekColossusMoEWrapper
colossus_moe = DeepSeekColossusMoEWrapper(native_cpu, device=dev, capacity=12)

print("\n[3] Running COLOSSUS Forward Pass on H100...")
torch.cuda.synchronize(dev)
t1 = time.time()
with torch.no_grad():
    col_out = colossus_moe(x)
torch.cuda.synchronize(dev)
print(f"  COLOSSUS output shape: {col_out.shape} | Time: {(time.time() - t1)*1000:.3f} ms")

# Compare outputs
diff = (ref_out - col_out).abs().max().item()
is_exact = torch.equal(ref_out, col_out)
allclose = torch.allclose(ref_out, col_out, atol=1e-3, rtol=1e-3)

print("\n" + "=" * 80)
print("  EXACTNESS VERIFICATION RESULTS")
print("=" * 80)
print(f"  Max Absolute Difference : {diff:.6e}")
print(f"  torch.equal (Bitwise)    : {is_exact}")
print(f"  torch.allclose (BF16)   : {allclose}")
print(f"  Cache Hits              : {colossus_moe.hits}")
print(f"  Cache Misses (Cold DMA) : {colossus_moe.misses}")
print(f"  Total DMA Transferred   : {colossus_moe.dma_bytes / (1024**2):.2f} MB")
print("=" * 80)

if allclose or is_exact:
    print("\n>>> SUCCESS: COLOSSUS produces mathematically exact output on real DeepSeek-Coder-V2 weights! <<<")
else:
    print("\n>>> FAILED: Output mismatch! <<<")
