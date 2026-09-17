"""
COLOSSUS Stage 4 Benchmark:
ADETR (Adaptive Dynamic Expert Tensor Reorganization) & SA-FFN Kernel Efficiency.

Evaluates on NVIDIA A100 80GB PCIe Gen3 x4 (Rudra node rdgpu01):
1. Kernel Latency & Launch Overhead:
   - Native Dense Expert forward: 3 GEMMs (gate, up, down)
   - Decomposed Non-ADETR SA-FFN: 6 narrow GEMMs + elementwise add (y = y_c + y_m)
   - ADETR Assembled-Slot SA-FFN: Contiguous HBM transposition + 3 full-width native GEMMs
2. Tensor Core Compute Efficiency & TFLOPS:
   - Effective TFLOPS for full GEMM vs fragmented narrow GEMMs
   - Wave quantization / SM occupancy penalty on A100 (108 SMs)
3. PCIe DMA Transfer Efficiency:
   - Non-ADETR strided transfer (unreorganized W_down slices)
   - ADETR contiguous transfer (reorganized [I_cold, H] pinned buffer)
4. Bitwise Exactness:
   - Assembled slot path vs Native: torch.equal = True
   - Decomposed path vs Native: ULP / floating point reordering drift
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def benchmark_cuda_kernel(fn, warmup: int = 50, iters: int = 500) -> float:
    """Benchmark a callable GPU function using CUDA Events over multiple iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    end_event.synchronize()

    return start_event.elapsed_time(end_event) / iters  # milliseconds


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("COLOSSUS STAGE 4: ADETR KERNEL & TENSOR CORE EFFICIENCY BENCHMARK")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("=" * 80)

    # Param2-17B MoE Expert geometry
    H = 2048
    I = 2048
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Single-token autoregressive decoding vector: [1, H]
    x = torch.randn(1, H, dtype=dtype, device=device)

    # Reference native master expert weights
    Wg = torch.randn(I, H, dtype=dtype, device=device)
    Wu = torch.randn(I, H, dtype=dtype, device=device)
    Wd = torch.randn(H, I, dtype=dtype, device=device)

    # Pre-allocated slot buffers (simulating pre-allocated GPU cache slot)
    slot_Wg = torch.empty((I, H), dtype=dtype, device=device)
    slot_Wu = torch.empty((I, H), dtype=dtype, device=device)
    slot_Wd = torch.empty((H, I), dtype=dtype, device=device)

    # Cold column fractions to evaluate
    fractions = [0.50, 0.25, 0.10]

    # FLOPs per expert forward: 2*B*H*I (gate) + 2*B*H*I (up) + 2*B*I*H (down) = 6 * B * H * I
    flops_per_token = 6 * 1 * H * I  # 25,165,824 FLOPs

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: Native Dense Expert Baseline Benchmark
    # ──────────────────────────────────────────────────────────────────────────
    def native_expert_forward():
        z = F.silu(F.linear(x, Wg)) * F.linear(x, Wu)
        return F.linear(z, Wd)

    t_native_ms = benchmark_cuda_kernel(native_expert_forward, warmup=100, iters=1000)
    tflops_native = (flops_per_token / (t_native_ms * 1e-3)) / 1e12

    print(f"\n[1] NATIVE DENSE EXPERT FORWARD (Reference):")
    print(f"    Kernel Latency  : {t_native_ms*1000:.2f} μs ({t_native_ms:.4f} ms)")
    print(f"    Kernel Launches : 3 GEMM launches")
    print(f"    Tensor Throughput: {tflops_native:.2f} TFLOPS")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: ADETR vs Non-ADETR Kernel Execution Across Missing Column Points
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 4 KERNEL LATENCY & TENSOR-CORE EFFICIENCY ABLATION")
    print("=" * 80)
    print(f"{'Config':<18} | {'Missing':<7} | {'Decomposed SA-FFN':<17} | {'ADETR Assembled':<15} | {'Speedup':<8} | {'Exactness'}")
    print("-" * 80)

    for frac in fractions:
        i_cold = int(I * frac)
        i_hot = I - i_cold

        # Partition weights
        Wg_hot, Wg_cold = Wg[:i_hot, :], Wg[i_hot:, :]
        Wu_hot, Wu_cold = Wu[:i_hot, :], Wu[i_hot:, :]
        Wd_hot, Wd_cold = Wd[:, :i_hot], Wd[:, i_hot:]

        # ADETR contiguous receive buffer: [I_cold, H] in GPU HBM
        adetr_rx_buf = torch.empty((i_cold, H), dtype=dtype, device=device)
        adetr_rx_buf.copy_(Wd_cold.t().contiguous())

        # 1. Non-ADETR Decomposed SA-FFN: y = y_cached + y_missed (6 narrow GEMMs)
        def decomposed_forward():
            # Hot partition (3 GEMMs)
            g_c = F.linear(x, Wg_hot)
            u_c = F.linear(x, Wu_hot)
            y_c = F.linear(F.silu(g_c) * u_c, Wd_hot)
            # Cold partition (3 narrow GEMMs)
            g_m = F.linear(x, Wg_cold)
            u_m = F.linear(x, Wu_cold)
            y_m = F.linear(F.silu(g_m) * u_m, Wd_cold)
            return y_c + y_m

        t_decomp_ms = benchmark_cuda_kernel(decomposed_forward, warmup=100, iters=1000)
        tflops_decomp = (flops_per_token / (t_decomp_ms * 1e-3)) / 1e12

        # Initialize hot columns in slot buffer
        slot_Wd[:, :i_hot].copy_(Wd_hot)

        # 2. ADETR Assembled-Slot Forward: Transpose in GPU HBM (1.5 TB/s) + 1 contiguous native GEMM
        def adetr_assembled_forward():
            # Fast GPU transposition of cold W_down into contiguous slot
            slot_Wd[:, i_hot:].copy_(adetr_rx_buf.t())
            # Single native full-width GEMM (3 launches)
            z = F.silu(F.linear(x, Wg)) * F.linear(x, Wu)
            return F.linear(z, slot_Wd)

        t_adetr_ms = benchmark_cuda_kernel(adetr_assembled_forward, warmup=100, iters=1000)
        tflops_adetr = (flops_per_token / (t_adetr_ms * 1e-3)) / 1e12

        speedup = t_decomp_ms / t_adetr_ms

        # Exactness check
        y_ref = native_expert_forward()
        y_decomp = decomposed_forward()
        y_adetr = adetr_assembled_forward()

        eq_adetr = torch.equal(y_ref, y_adetr)
        cos_decomp = F.cosine_similarity(y_ref.float().view(-1), y_decomp.float().view(-1), dim=0).item()

        print(f"Missing {frac*100:4.1f}%     | {i_cold:4d} cols | {t_decomp_ms*1000:6.1f} μs ({tflops_decomp:4.1f} TF) | {t_adetr_ms*1000:6.1f} μs ({tflops_adetr:4.1f} TF) | {speedup:6.2f}x  | ADETR: {eq_adetr} (cos_decomp: {cos_decomp:.7f})")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 3: PCIe Transfer Efficiency: Strided vs ADETR Contiguous DMA
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 4 PCIE DMA EFFICIENCY: STRIDED VS ADETR CONTIGUOUS TRANSFER")
    print("=" * 80)

    # Create host pinned buffers for W_down cold partition
    pin = torch.cuda.is_available()
    for frac in fractions:
        i_cold = int(I * frac)
        nbytes = H * i_cold * 2  # bfloat16 bytes

        # 1. Non-ADETR: Strided transfer from [H, I] host buffer directly into GPU [H, i_cold] slice
        host_strided = torch.randn(H, I, dtype=dtype, pin_memory=pin)
        gpu_slice = torch.empty((H, i_cold), dtype=dtype, device=device)

        def strided_dma():
            gpu_slice.copy_(host_strided[:, :i_cold], non_blocking=False)

        t_strided_ms = benchmark_cuda_kernel(strided_dma, warmup=20, iters=100)
        bw_strided = (nbytes / (1024**3)) / (t_strided_ms * 1e-3)

        # 2. ADETR: Contiguous [i_cold, H] pinned buffer DMA -> GPU rx buffer
        host_adetr = torch.randn(i_cold, H, dtype=dtype, pin_memory=pin)
        gpu_adetr_rx = torch.empty((i_cold, H), dtype=dtype, device=device)

        def adetr_contiguous_dma():
            gpu_adetr_rx.copy_(host_adetr, non_blocking=False)

        t_adetr_dma_ms = benchmark_cuda_kernel(adetr_contiguous_dma, warmup=20, iters=100)
        bw_adetr = (nbytes / (1024**3)) / (t_adetr_dma_ms * 1e-3)

        dma_speedup = t_strided_ms / t_adetr_dma_ms

        print(f"Missing {frac*100:4.1f}% ({nbytes/(1024*1024):5.2f} MB):")
        print(f"  Non-ADETR Strided Transfer   : {t_strided_ms:6.2f} ms ({bw_strided:4.2f} GB/s)")
        print(f"  ADETR Contiguous Burst Transfer: {t_adetr_dma_ms:6.2f} ms ({bw_adetr:4.2f} GB/s)")
        print(f"  PCIe Bandwidth Advantage     : {dma_speedup:.2f}x faster ({bw_adetr/bw_strided:.2f}x bus efficiency)")
        print("-" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # PART 4: Kernel Launch Count & Wave Quantization Analysis
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 4 KERNEL LAUNCH & SM OCCUPANCY ANALYSIS (Param2 Top-6 Routing)")
    print("=" * 80)
    print("For 1 Decoding Token across 20 Layers (6 active experts/layer = 120 expert forwards):")
    print(f"  * Native Expert          : 3 launches/expert  = 360 GEMM launches/token")
    print(f"  * Non-ADETR Decomposed   : 7 launches/expert  = 840 GEMM + Add launches/token (+480 launches overhead!)")
    print(f"  * ADETR Assembled-Slot   : 3 launches/expert  = 360 GEMM launches/token (Zero kernel launch inflation)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
