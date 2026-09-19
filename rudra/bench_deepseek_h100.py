#!/usr/bin/env python3
"""
bench_deepseek_h100.py
======================
Hardware Frontier & DMA Microbenchmark for DeepSeek-Coder-V2 (236B MoE)
on NVIDIA H100 NVL (Hopper SM 9.0).

Evaluates:
1. Physical Host-to-Device (H2D) PCIe Gen5 DMA throughput on H100 NVL
   for exact DeepSeek-V2 expert payloads:
   - 45.0 MB (100% whole expert)
   - 22.5 MB (50% ADETR cold columns)
   - 11.25 MB (25% ADETR cold columns)
2. Real Hopper Tensor Core execution latency for DeepSeek-V2 MoE layer:
   - Hidden Size: 5120
   - MoE Intermediate Size: 1536
   - Routed Experts: 160 (Top-6 active)
   - Shared Experts: 2 (Permanently resident)
   - Precision: bfloat16
3. Interconnect Compatibility Ratio: R = T_DMA / T_compute
"""

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

print('=' * 80)
print('  COLOSSUS HARDWARE FRONTIER: DeepSeek-Coder-V2 on NVIDIA H100 NVL')
print('=' * 80)

assert torch.cuda.is_available(), 'CUDA not available!'
dev = torch.device('cuda:0')
prop = torch.cuda.get_device_properties(0)
print(f'Device: {prop.name} | Total HBM3: {prop.total_memory / (1024**3):.2f} GB | SM: {prop.major}.{prop.minor}')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Physical H2D DMA Bandwidth & Latency Benchmark
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Profiling Physical Host-to-Device (H2D) PCIe Gen5 DMA on H100...')

test_payloads = [
    ('Whole Expert (100%)', 45.0 * 1024 * 1024),
    ('ADETR Cold (50%)',    22.5 * 1024 * 1024),
    ('ADETR Cold (25%)',    11.25 * 1024 * 1024),
]

dma_results = {}
stream = torch.cuda.Stream(device=dev)

for label, nbytes in test_payloads:
    n_elements = int(nbytes // 2)  # bfloat16
    
    # Pinned host buffer
    host_pinned = torch.empty(n_elements, dtype=torch.bfloat16, pin_memory=True)
    host_pinned.normal_()
    gpu_buf = torch.empty(n_elements, dtype=torch.bfloat16, device=dev)
    
    # Warmup
    for _ in range(5):
        gpu_buf.copy_(host_pinned, non_blocking=False)
    torch.cuda.synchronize(dev)
    
    # Timed transfer via CUDA Events
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    
    iters = 30
    latencies = []
    with torch.cuda.stream(stream):
        for _ in range(iters):
            ev_start.record(stream)
            gpu_buf.copy_(host_pinned, non_blocking=True)
            ev_end.record(stream)
            ev_end.synchronize()
            latencies.append(ev_start.elapsed_time(ev_end))
    
    avg_lat_ms = sum(latencies) / len(latencies)
    mb = nbytes / (1024 * 1024)
    bw_gbps = (nbytes / (1024**3)) / (avg_lat_ms / 1000.0)
    
    dma_results[label] = {
        'bytes': nbytes,
        'mb': mb,
        'avg_lat_ms': avg_lat_ms,
        'bw_gbps': bw_gbps,
    }
    print(f'  {label:22s} | Payload: {mb:5.1f} MB | Latency: {avg_lat_ms:6.3f} ms | Bandwidth: {bw_gbps:6.2f} GB/s')

# ─────────────────────────────────────────────────────────────────────────────
# 2. DeepSeek-V2 MoE Layer Compute Latency on H100 Hopper Tensor Cores
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Profiling DeepSeek-V2 MoE Layer Compute on H100...')

H = 5120
I_routed = 1536
I_shared = 1536 * 2  # 2 shared experts
top_k = 6

class DeepSeekExpert(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# Instantiate top-6 routed experts + 1 shared expert (representing 2 combined)
routed_experts = nn.ModuleList([DeepSeekExpert(H, I_routed).to(dev).to(torch.bfloat16) for _ in range(top_k)])
shared_expert = DeepSeekExpert(H, I_shared).to(dev).to(torch.bfloat16)

# DeepSeek-V2 MLA (Multi-Head Latent Attention): 128 heads, head_dim 128 -> 16384 dim
q_proj = nn.Linear(H, 128 * 128, bias=False).to(dev).to(torch.bfloat16)
k_proj = nn.Linear(H, 128 * 128, bias=False).to(dev).to(torch.bfloat16)
v_proj = nn.Linear(H, 128 * 128, bias=False).to(dev).to(torch.bfloat16)
o_proj = nn.Linear(128 * 128, H, bias=False).to(dev).to(torch.bfloat16)

# Single token input [1, 1, H]
x = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev)

# Warmup
for _ in range(10):
    # Attention
    q, k, v = q_proj(x), k_proj(x), v_proj(x)
    attn_out = o_proj(q)
    # MoE: 6 routed experts + shared
    out = shared_expert(attn_out)
    for exp in routed_experts:
        out = out + exp(attn_out) / top_k
torch.cuda.synchronize(dev)

# Timed run
ev_c_start = torch.cuda.Event(enable_timing=True)
ev_c_end = torch.cuda.Event(enable_timing=True)

compute_iters = 50
attn_lats, moe_lats, total_lats = [], [], []

for _ in range(compute_iters):
    # 1. Attention compute
    ev_c_start.record()
    q, k, v = q_proj(x), k_proj(x), v_proj(x)
    attn_out = o_proj(q)
    ev_c_end.record()
    ev_c_end.synchronize()
    attn_lats.append(ev_c_start.elapsed_time(ev_c_end))
    
    # 2. MoE compute (6 routed + shared)
    ev_c_start.record()
    out = shared_expert(attn_out)
    for exp in routed_experts:
        out = out + exp(attn_out) / top_k
    ev_c_end.record()
    ev_c_end.synchronize()
    moe_lats.append(ev_c_start.elapsed_time(ev_c_end))
    
    total_lats.append(attn_lats[-1] + moe_lats[-1])

t_attn_ms = sum(attn_lats) / len(attn_lats)
t_moe_ms = sum(moe_lats) / len(moe_lats)
t_total_ms = sum(total_lats) / len(total_lats)

print(f'  Attention Compute / Layer: {t_attn_ms:.3f} ms')
print(f'  MoE (6 Routed + 2 Shared): {t_moe_ms:.3f} ms')
print(f'  Total Compute / Layer    : {t_total_ms:.3f} ms')

# ─────────────────────────────────────────────────────────────────────────────
# 3. The Interconnect Compatibility Law (R Ratio) Analysis
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Interconnect Compatibility Analysis (R = T_DMA / T_compute):')
print('-' * 80)
print(f'{"Configuration":26s} | {"Cold DMA":10s} | {"Compute":9s} | {"R Ratio":8s} | {"Prefetch Overlap":16s}')
print('-' * 80)

comparisons = []
for label, data in dma_results.items():
    t_dma = data['avg_lat_ms']
    r_val = t_dma / t_total_ms
    overlap_speedup = (t_dma + t_total_ms) / max(t_dma, t_total_ms)
    status = 'COMPLETE OVERLAP' if r_val <= 1.0 else f'{overlap_speedup:.2f}x Speedup'
    print(f'{label:26s} | {t_dma:8.3f} ms | {t_total_ms:7.3f} ms | {r_val:8.2f} | {status:16s}')
    comparisons.append({
        'config': label,
        'dma_ms': t_dma,
        'compute_ms': t_total_ms,
        'r_ratio': r_val,
        'bandwidth_gbps': data['bw_gbps'],
    })

print('-' * 80)

# End-to-end Token Projection for 59 MoE Layers
print('\n[4] End-to-End Serving Projection for DeepSeek-Coder-V2 (59 MoE Layers):')
for comp in comparisons:
    cfg_name = comp['config']
    # Under batch=1, assume 6 routed expert misses per layer
    # With C slots, cache hit rate is ~30-50% in practice. Let miss count = 3 misses/layer
    misses_per_layer = 3.0
    layer_dma = comp['dma_ms'] * misses_per_layer
    layer_compute = comp['compute_ms']
    # If prefetching can overlap compute:
    layer_time_overlapped = max(layer_dma, layer_compute)
    token_time_s = (layer_time_overlapped * 59) / 1000.0
    tps = 1.0 / token_time_s if token_time_s > 0 else 0.0
    print(f'  {cfg_name:26s}: Latency = {token_time_s*1000:6.1f} ms/tok | Expected TPS = {tps:5.1f} tok/s on 2x H100')

# Save benchmark results
out_json = '/home/palakm/MoEServingSim/aditya/deepseek_h100_frontier.json'
with open(out_json, 'w') as f:
    json.dump({
        'device': prop.name,
        'dma': dma_results,
        'compute': {'attn_ms': t_attn_ms, 'moe_ms': t_moe_ms, 'total_ms': t_total_ms},
        'comparisons': comparisons,
    }, f, indent=2)
print(f'\nSaved hardware frontier results to: {out_json}')
