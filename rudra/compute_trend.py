# Calculate architectural numbers for models across accelerators
models = [
    {'name': 'Mixtral-8x7B', 'total_params': '46.7B', 'routed_exp': 8, 'active_exp': 2, 'H': 4096, 'I': 14336, 'exp_mb': 336.0, 'layers': 32},
    {'name': 'Param2-17B', 'total_params': '17.0B', 'routed_exp': 64, 'active_exp': 2, 'H': 2048, 'I': 2048, 'exp_mb': 24.0, 'layers': 20},
    {'name': 'Qwen2-57B-A14B', 'total_params': '57.2B', 'routed_exp': 64, 'active_exp': 8, 'H': 3584, 'I': 2560, 'exp_mb': 52.5, 'layers': 28},
    {'name': 'Qwen3-30B-A3B', 'total_params': '30.0B', 'routed_exp': 128, 'active_exp': 8, 'H': 2048, 'I': 768, 'exp_mb': 9.44, 'layers': 48},
    {'name': 'DeepSeek-V2-Lite', 'total_params': '15.7B', 'routed_exp': 64, 'active_exp': 6, 'H': 2048, 'I': 1408, 'exp_mb': 17.3, 'layers': 27},
    {'name': 'DeepSeek-Coder-V2', 'total_params': '236B', 'routed_exp': 160, 'active_exp': 6, 'H': 5120, 'I': 1536, 'exp_mb': 45.0, 'layers': 60},
    {'name': 'DeepSeek-V3', 'total_params': '671B', 'routed_exp': 256, 'active_exp': 8, 'H': 7168, 'I': 2048, 'exp_mb': 88.0, 'layers': 61},
]

# Hardware specs
gpus = [
    {'name': 'RTX 4090 (PCIe 4.0)', 'hbm_gb': 24, 'pcie_bw_gbps': 25.0, 'tflops_bf16': 165},
    {'name': 'A100 (PCIe 4.0)', 'hbm_gb': 80, 'pcie_bw_gbps': 25.0, 'tflops_bf16': 312},
    {'name': 'H100 NVL (PCIe 5.0)', 'hbm_gb': 94, 'pcie_bw_gbps': 51.56, 'tflops_bf16': 989},
]

print(f"{'Model':<20} | {'Param':<6} | {'Exp MB':<8} | {'Act/Tot':<8} | {'GPU':<18} | {'Bus BW':<8} | {'T_DMA (Whole)':<14} | {'T_DMA (ADETR)':<14} | {'T_compute':<10} | {'R (Whole)':<10} | {'R (ADETR)':<10}")
print('-' * 145)

for m in models:
    for g in gpus:
        # Compute FLOPs for 1 token through active experts in 1 layer: 2 * 3 * H * I * active_exp
        flops = 2 * 3 * m['H'] * m['I'] * m['active_exp']
        # Single token decode is memory-bound GEMV on GPU HBM, effective bandwidth ~ 60% of HBM or MFU ~ 5-10%
        # Real measured T_compute on H100 for DeepSeek-Coder-V2 is 0.806 ms (0.262 ms attn + 0.544 ms moe)
        # We scale T_compute relative to H100 measured baseline
        h100_rel = (flops / (2 * 3 * 5120 * 1536 * 6)) * (989.0 / g['tflops_bf16']) * 0.544
        t_compute_ms = max(0.04, h100_rel)
        
        # T_DMA for 1 cold expert (Whole vs ADETR 50%)
        t_dma_whole_ms = (m['exp_mb'] / 1024) / g['pcie_bw_gbps'] * 1000
        t_dma_adetr_ms = (m['exp_mb'] * 0.50 / 1024) / g['pcie_bw_gbps'] * 1000
        
        r_whole = t_dma_whole_ms / t_compute_ms
        r_adetr = t_dma_adetr_ms / t_compute_ms
        
        print(f"{m['name']:<20} | {m['total_params']:<6} | {m['exp_mb']:>6.1f} MB | {m['active_exp']}/{m['routed_exp']:<4} | {g['name']:<18} | {g['pcie_bw_gbps']:>4.1f} GB/s | {t_dma_whole_ms:>10.3f} ms  | {t_dma_adetr_ms:>10.3f} ms  | {t_compute_ms:>7.3f} ms | {r_whole:>9.2f}  | {r_adetr:>9.2f}")
    print('.' * 145)
