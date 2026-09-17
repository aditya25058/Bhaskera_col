"""
COLOSSUS Stage 3A & Stage 3B Benchmark:
Column-Level SA-FFN Exactness & Measured PCIe Traffic Reduction.

Geometry: Exact Param2-17B MoE Expert (H=2048, I=2048, dtype=bfloat16, matching 24.3 MB/expert)
Evaluates on CUDA (NVIDIA A100 PCIe Gen 3 x4 on Rudra):
- Stage 3A: Bitwise exactness and numerical comparison:
    1. Native full expert forward: y_native = W_d @ (silu(x @ W_g.T) * (x @ W_u.T))
    2. Decomposed SA-FFN forward: y_decomposed = y_c + y_m
    3. Assembled slot forward: y_assembled (hot columns pinned + cold columns DMA'd into contiguous buffer)
    4. Exactness checks: torch.equal(), max |Δ|, relative error, cosine similarity, bit-level comparison.
- Stage 3B: Measured traffic reduction and hardware DMA timings across missing column fractions:
    - 100% missing (whole-expert baseline)
    - 50% missing columns
    - 25% missing columns
    - 10% missing columns
"""
from __future__ import annotations

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"COLOSSUS STAGE 3A & 3B: SA-FFN COLUMN DECOMPOSITION & TRAFFIC BENCHMARK")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Current PCIe Link: Gen {torch.cuda.get_device_properties(0).major} (A100 PCIe)")
    print("=" * 80)

    # Param2-17B expert dimensions
    H = 2048
    I = 2048
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # 1. Master CPU pinned expert weights
    Wg_master = torch.randn(I, H, dtype=dtype)
    Wu_master = torch.randn(I, H, dtype=dtype)
    Wd_master = torch.randn(H, I, dtype=dtype)

    # Move full reference expert to GPU
    Wg_gpu = Wg_master.to(device)
    Wu_gpu = Wu_master.to(device)
    Wd_gpu = Wd_master.to(device)

    # Synthetic activation input: batch 4, seq_len 1 (or 4 tokens)
    x = torch.randn(4, H, dtype=dtype).to(device)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3A: End-to-End SA-FFN Numerical & Bitwise Comparison
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("STAGE 3A: EXACTNESS COMPARISON (Native Full Expert vs SA-FFN Decomposition)")
    print("-" * 80)

    # 1. Native full expert forward
    # y = (silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
    z_native = F.silu(F.linear(x, Wg_gpu)) * F.linear(x, Wu_gpu)
    y_native = F.linear(z_native, Wd_gpu)

    # Test across multiple missing fractions
    fractions = [0.50, 0.25, 0.10]
    for frac in fractions:
        i_missed = int(I * frac)
        i_cached = I - i_missed
        
        # Partition along intermediate dimension I
        Wg_c, Wg_m = Wg_gpu[:i_cached, :], Wg_gpu[i_cached:, :]
        Wu_c, Wu_m = Wu_gpu[:i_cached, :], Wu_gpu[i_cached:, :]
        Wd_c, Wd_m = Wd_gpu[:, :i_cached], Wd_gpu[:, i_cached:]

        # Decomposed SA-FFN computation:
        # z_c = silu(x @ Wg_c.T) * (x @ Wu_c.T), y_c = z_c @ Wd_c.T
        # z_m = silu(x @ Wg_m.T) * (x @ Wu_m.T), y_m = z_m @ Wd_m.T
        # y_decomposed = y_c + y_m
        z_c = F.silu(F.linear(x, Wg_c)) * F.linear(x, Wu_c)
        y_c = F.linear(z_c, Wd_c)

        z_m = F.silu(F.linear(x, Wg_m)) * F.linear(x, Wu_m)
        y_m = F.linear(z_m, Wd_m)

        y_decomposed = y_c + y_m

        # Slot-assembled forward (Hot columns pre-pinned, cold columns transferred to contiguous slot)
        slot_Wg = torch.empty_like(Wg_gpu)
        slot_Wu = torch.empty_like(Wu_gpu)
        slot_Wd = torch.empty_like(Wd_gpu)

        slot_Wg[:i_cached].copy_(Wg_c)
        slot_Wu[:i_cached].copy_(Wu_c)
        slot_Wd[:, :i_cached].copy_(Wd_c)

        # DMA transfer of missing columns into slot
        slot_Wg[i_cached:].copy_(Wg_m)
        slot_Wu[i_cached:].copy_(Wu_m)
        slot_Wd[:, i_cached:].copy_(Wd_m)

        z_assembled = F.silu(F.linear(x, slot_Wg)) * F.linear(x, slot_Wu)
        y_assembled = F.linear(z_assembled, slot_Wd)

        # Comparisons
        # 1. Assembled vs Native
        is_assembled_exact = torch.equal(y_native, y_assembled)
        diff_assembled = (y_native.float() - y_assembled.float()).abs().max().item()

        # 2. Decomposed vs Native
        is_decomposed_exact = torch.equal(y_native, y_decomposed)
        diff_decomposed = (y_native.float() - y_decomposed.float()).abs().max().item()
        cos_sim = F.cosine_similarity(y_native.float().flatten(), y_decomposed.float().flatten(), dim=0).item()
        rel_err = diff_decomposed / (y_native.float().abs().max().item() + 1e-12)

        print(f"Fraction Missing = {frac*100:4.1f}% ({i_missed}/{I} columns):")
        print(f"  * Assembled Slot vs Native : torch.equal={is_assembled_exact:<5} | max |diff| = {diff_assembled:.6f}")
        print(f"  * Decomposed vs Native     : torch.equal={is_decomposed_exact:<5} | max |diff| = {diff_decomposed:.6f} | rel_err = {rel_err:.2e} | cos = {cos_sim:.10f}")

    # --------------------------------------------------------------------------
    # STAGE 3B: Measured Traffic Reduction & Hardware DMA Latency on Rudra
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 3B: MEASURED HARDWARE DMA TIMINGS & TRAFFIC REDUCTION ON RUDRA A100")
    print("-" * 80)

    if device.type != "cuda":
        print("CUDA not available for Stage 3B hardware DMA benchmark. Exiting.")
        return 0

    # Pin master CPU memory
    cpu_Wg = Wg_master.pin_memory()
    cpu_Wu = Wu_master.pin_memory()
    cpu_Wd = Wd_master.pin_memory()

    # Pre-allocate contiguous GPU receive buffers
    gpu_rx_Wg = torch.empty_like(Wg_gpu)
    gpu_rx_Wu = torch.empty_like(Wu_gpu)
    gpu_rx_Wd = torch.empty_like(Wd_gpu)

    total_expert_bytes = (cpu_Wg.numel() + cpu_Wu.numel() + cpu_Wd.numel()) * 2  # bfloat16 = 2 bytes
    print(f"Total Expert Weight Size: {total_expert_bytes / (1024*1024):.2f} MB (100% baseline)")
    print(f"6 Active Experts Payload: {6 * total_expert_bytes / (1024*1024):.2f} MB\n")

    test_configs = [
        ("Whole Expert (100%)", 1.00),
        ("Columns 50% Missing", 0.50),
        ("Columns 25% Missing", 0.25),
        ("Columns 10% Missing", 0.10),
    ]

    results = []
    stream = torch.cuda.Stream()
    N_ITERS = 100

    print(f"{'Configuration':<22} | {'Missing %':>9} | {'1 Exp DMA':>10} | {'6 Exp DMA':>10} | {'DMA Time (ms)':>13} | {'DMA BW (GB/s)':>13} | {'Fits 5-8ms Window?':>18}")
    print("-" * 110)

    for name, frac in test_configs:
        i_missed = int(I * frac)
        missing_bytes_per_expert = (i_missed * H + i_missed * H + H * i_missed) * 2
        missing_bytes_6_experts = missing_bytes_per_expert * 6

        # Slice CPU pinned tensors
        slice_Wg = cpu_Wg[:i_missed, :]
        slice_Wu = cpu_Wu[:i_missed, :]
        slice_Wd = cpu_Wd[:, :i_missed]

        # Slice target GPU buffers
        target_Wg = gpu_rx_Wg[:i_missed, :]
        target_Wu = gpu_rx_Wu[:i_missed, :]
        target_Wd = gpu_rx_Wd[:, :i_missed]

        # Warmup
        for _ in range(5):
            with torch.cuda.stream(stream):
                target_Wg.copy_(slice_Wg, non_blocking=True)
                target_Wu.copy_(slice_Wu, non_blocking=True)
                target_Wd.copy_(slice_Wd, non_blocking=True)
        torch.cuda.synchronize()

        # Benchmarking hardware DMA transfer time using CUDA Events
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        ev_start.record(stream)
        with torch.cuda.stream(stream):
            for _ in range(N_ITERS):
                # Simulate transferring missing columns for 6 experts
                for _ in range(6):
                    target_Wg.copy_(slice_Wg, non_blocking=True)
                    target_Wu.copy_(slice_Wu, non_blocking=True)
                    target_Wd.copy_(slice_Wd, non_blocking=True)
        ev_end.record(stream)
        ev_end.synchronize()

        total_ms = ev_start.elapsed_time(ev_end)
        avg_dma_ms = total_ms / N_ITERS  # DMA time to transfer 6 experts
        bw_gb_s = (missing_bytes_6_experts / (1024**3)) / (avg_dma_ms / 1000.0)
        fits_window = "YES (PASS)" if avg_dma_ms <= 8.0 else f"NO (+{avg_dma_ms - 8.0:.1f}ms)"

        results.append({
            "name": name,
            "frac": frac,
            "missing_mb_1": missing_bytes_per_expert / (1024 * 1024),
            "missing_mb_6": missing_bytes_6_experts / (1024 * 1024),
            "dma_ms": avg_dma_ms,
            "bw_gb_s": bw_gb_s,
            "fits": fits_window,
        })

        print(f"{name:<22} | {frac*100:8.1f}% | {missing_bytes_per_expert / (1024*1024):8.2f} MB | {missing_bytes_6_experts / (1024*1024):8.2f} MB | {avg_dma_ms:11.2f} ms | {bw_gb_s:11.2f} GB/s | {fits_window:>18}")

    print("=" * 110)
    print("\nSUMMARY OF EMPIRICAL FINDINGS FOR PAPER:")
    for r in results:
        print(f"  * {r['name']}: {r['missing_mb_6']:.1f} MB payload -> {r['dma_ms']:.2f} ms transfer time ({r['fits']})")
    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
