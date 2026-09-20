models = [
    {'name': 'Mixtral-8x7B', 'params': 46.7, 'tot_gb': 93.4, 'non_routed_gb': 6.2, 'exp_mb': 336.0, 'K': 2, 'E': 8, 'layers': 32},
    {'name': 'Qwen3-30B-A3B', 'params': 30.0, 'tot_gb': 60.0, 'non_routed_gb': 4.1, 'exp_mb': 9.44, 'K': 8, 'E': 128, 'layers': 48},
    {'name': 'DeepSeek-V2-Lite', 'params': 15.7, 'tot_gb': 31.4, 'non_routed_gb': 3.2, 'exp_mb': 17.3, 'K': 6, 'E': 64, 'layers': 27},
    {'name': 'DeepSeek-Coder-V2', 'params': 236.0, 'tot_gb': 471.5, 'non_routed_gb': 24.3, 'exp_mb': 45.0, 'K': 6, 'E': 160, 'layers': 60},
    {'name': 'DeepSeek-V3', 'params': 671.0, 'tot_gb': 1342.0, 'non_routed_gb': 42.0, 'exp_mb': 88.0, 'K': 8, 'E': 256, 'layers': 61},
]

print(f"{'Model':<18} | {'Dense BF16':<11} | {'Non-Routed':<11} | {'C=K Slot GB':<12} | {'C=2K Slot GB':<13} | {'Fit on 24GB?':<13} | {'Fit on 48GB?':<13} | {'Fit on 80/94GB?':<15}")
print('-' * 115)

for m in models:
    vram_ck = m['non_routed_gb'] + (m['layers'] * m['K'] * m['exp_mb'] / 1024)
    vram_c2k = m['non_routed_gb'] + (m['layers'] * (2 * m['K']) * m['exp_mb'] / 1024)
    
    fit_24 = f"Yes ({vram_ck:.1f}G)" if vram_ck <= 22 else "No"
    fit_48 = f"Yes ({vram_ck:.1f}G)" if vram_ck <= 44 else "No"
    fit_80 = f"Yes ({vram_c2k:.1f}G)" if vram_c2k <= 75 else (f"Yes C=K ({vram_ck:.1f}G)" if vram_ck <= 85 else "No")
    
    print(f"{m['name']:<18} | {m['tot_gb']:>8.1f} GB | {m['non_routed_gb']:>8.1f} GB | {vram_ck:>9.1f} GB | {vram_c2k:>10.1f} GB | {fit_24:<13} | {fit_48:<13} | {fit_80:<15}")
