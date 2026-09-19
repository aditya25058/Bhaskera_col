#!/usr/bin/env python3
import time
import torch

assert torch.cuda.is_available()
dev = torch.device("cuda:0")

# 45 MB expert payload (BF16)
n_bytes = int(45.0 * 1024 * 1024)
n_elems = n_bytes // 2

print(f"Testing 45.0 MB transfer to {torch.cuda.get_device_name(0)}...")

# 1. Pinned Host Memory
pinned = torch.empty(n_elems, dtype=torch.bfloat16, pin_memory=True).normal_()
gpu_buf = torch.empty(n_elems, dtype=torch.bfloat16, device=dev)

# Warmup
for _ in range(5):
    gpu_buf.copy_(pinned, non_blocking=True)
torch.cuda.synchronize(dev)

t0 = time.perf_counter()
iters = 30
for _ in range(iters):
    gpu_buf.copy_(pinned, non_blocking=True)
torch.cuda.synchronize(dev)
lat_pinned = (time.perf_counter() - t0) / iters * 1000
bw_pinned = (n_bytes / 1e9) / (lat_pinned / 1000)
print(f"  [1] Pinned H2D DMA        : {lat_pinned:6.3f} ms | Bandwidth: {bw_pinned:6.2f} GB/s")

# 2. Unpinned (Pageable) Host Memory
unpinned = torch.empty(n_elems, dtype=torch.bfloat16).normal_()

for _ in range(5):
    gpu_buf.copy_(unpinned, non_blocking=False)
torch.cuda.synchronize(dev)

t0 = time.perf_counter()
for _ in range(iters):
    gpu_buf.copy_(unpinned, non_blocking=False)
torch.cuda.synchronize(dev)
lat_unpinned = (time.perf_counter() - t0) / iters * 1000
bw_unpinned = (n_bytes / 1e9) / (lat_unpinned / 1000)
print(f"  [2] Unpinned (Pageable)   : {lat_unpinned:6.3f} ms | Bandwidth: {bw_unpinned:6.2f} GB/s")

# 3. Direct from Safetensors shard
from safetensors import safe_open
MODEL_PATH = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct"
shard_path = f"{MODEL_PATH}/model-00001-of-000055.safetensors"
handle = safe_open(shard_path, framework="pt", device="cpu")

# Find an expert weight
exp_key = "model.layers.1.mlp.experts.0.gate_proj.weight"
st_tensor = handle.get_tensor(exp_key)
t_bytes = st_tensor.nbytes
gpu_t = torch.empty_like(st_tensor, device=dev)

for _ in range(5):
    gpu_t.copy_(st_tensor, non_blocking=False)
torch.cuda.synchronize(dev)

t0 = time.perf_counter()
for _ in range(iters):
    gpu_t.copy_(st_tensor, non_blocking=False)
torch.cuda.synchronize(dev)
lat_st = (time.perf_counter() - t0) / iters * 1000
bw_st = (t_bytes / 1e9) / (lat_st / 1000)
print(f"  [3] Direct Safetensors mmap: {lat_st:6.3f} ms for {t_bytes/(1024**2):.1f} MB | Bandwidth: {bw_st:6.2f} GB/s")
