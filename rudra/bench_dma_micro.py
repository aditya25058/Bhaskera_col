#!/usr/bin/env python3
"""
DMA Microbenchmark: Empirical PCIe Gen4 x16 Transfer Bandwidth & Allocation Gate
for 1-GPU COLOSSUS Mixtral-8x7B Serving on Param Rudra (NVIDIA A100 80GB).

Measures:
  1. Allocation latency for pinned host buffers (single expert 352 MB up to 22.5 GB).
  2. Unpinned vs Pinned H2D DMA bandwidth (GB/s) & latency (ms).
  3. Coalesced (1x352 MB) vs Fragmented (3x117 MB: gate, up, down) transfer speed.
  4. Stream overlap efficiency: DMA transfer concurrent with simulated attention compute.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any

import torch

def format_gb_s(n_bytes: int, duration_s: float) -> float:
    return (n_bytes / (1024 ** 3)) / duration_s

def main():
    parser = argparse.ArgumentParser(description="DMA Microbenchmark on A100")
    parser.add_argument("--out-json", type=str, default="/home/bapic_iiitd/2_group/dma_microbench_results.json")
    parser.add_argument("--iters", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this node.")
        sys.exit(1)

    device = torch.device("cuda:0")
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    total_hbm_gb = props.total_memory / (1024 ** 3)

    print("=" * 80)
    print(f"  DMA MICROBENCHMARK: NVIDIA A100 PCIe Bandwidth & Allocation Gate")
    print(f"  Device: {gpu_name} ({total_hbm_gb:.2f} GiB HBM)")
    print(f"  PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print("=" * 80)

    results: Dict[str, Any] = {
        "gpu_name": gpu_name,
        "total_hbm_gb": total_hbm_gb,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    # -------------------------------------------------------------------------
    # Expert Tensor Geometry for Mixtral-8x7B (bfloat16)
    # -------------------------------------------------------------------------
    hidden_size = 4096
    intermediate_size = 14336
    dtype = torch.bfloat16
    elem_bytes = 2

    # Projections per expert:
    # gate: [14336, 4096] = 58,720,256 elements = 117,440,512 bytes (~112.0 MiB)
    # up:   [14336, 4096] = 58,720,256 elements = 117,440,512 bytes (~112.0 MiB)
    # down: [4096, 14336] = 58,720,256 elements = 117,440,512 bytes (~112.0 MiB)
    expert_elems = intermediate_size * hidden_size * 3
    expert_bytes = expert_elems * elem_bytes
    expert_mb = expert_bytes / (1024 ** 2)

    print(f"\n[Geometry] Mixtral-8x7B single expert (gate + up + down, bf16):")
    print(f"  Total elements: {expert_elems:,}")
    print(f"  Total bytes:    {expert_bytes:,} ({expert_mb:.2f} MiB / {expert_bytes/(1024**3):.4f} GiB)")

    # -------------------------------------------------------------------------
    # GATE 1: Pinned Memory Allocation Latency
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("GATE 1: Pinned Host Memory Allocation Overhead")
    print("-" * 80)

    alloc_sizes = [
        ("Single Expert Buffer", expert_bytes),
        ("Dual Stream Staging (2x Expert)", expert_bytes * 2),
        ("1 Layer (8 Experts)", expert_bytes * 8),
        ("Offload Target (22.5 GB = 64 Experts)", int(22.5 * (1024 ** 3))),
    ]

    alloc_results = {}
    for name, size_b in alloc_sizes:
        t0 = time.perf_counter()
        buf = torch.empty(size_b, dtype=torch.uint8, device="cpu", pin_memory=True)
        t_alloc = time.perf_counter() - t0
        alloc_results[name] = {
            "size_bytes": size_b,
            "size_gb": round(size_b / (1024 ** 3), 3),
            "alloc_time_s": round(t_alloc, 4),
            "alloc_time_ms": round(t_alloc * 1000.0, 2),
        }
        print(f"  {name:40s}: {size_b / (1024**3):6.2f} GiB allocated in {t_alloc * 1000.0:8.2f} ms ({t_alloc:.4f}s)")
        del buf

    results["gate1_allocation"] = alloc_results

    # -------------------------------------------------------------------------
    # GATE 2: Unpinned vs Pinned Transfer Bandwidth (Single Expert 352 MB)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("GATE 2: Unpinned vs Pinned Transfer Bandwidth (Coalesced 352.3 MB)")
    print("-" * 80)

    # Pre-allocate source and destination tensors
    unpinned_src = torch.randn(expert_elems, dtype=dtype, device="cpu")
    pinned_src = torch.empty(expert_elems, dtype=dtype, device="cpu", pin_memory=True)
    pinned_src.copy_(unpinned_src)
    gpu_dst = torch.empty(expert_elems, dtype=dtype, device=device)

    # Test 2A: Unpinned Synchronous Transfer
    for _ in range(args.warmup):
        gpu_dst.copy_(unpinned_src, non_blocking=False)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        gpu_dst.copy_(unpinned_src, non_blocking=False)
    torch.cuda.synchronize(device)
    t_unpinned = (time.perf_counter() - t0) / args.iters
    bw_unpinned = format_gb_s(expert_bytes, t_unpinned)
    print(f"  [2A] Unpinned Host -> GPU (blocking)    : {t_unpinned * 1000.0:6.2f} ms | {bw_unpinned:6.2f} GB/s")

    # Test 2B: Pinned Synchronous Transfer
    for _ in range(args.warmup):
        gpu_dst.copy_(pinned_src, non_blocking=False)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        gpu_dst.copy_(pinned_src, non_blocking=False)
    torch.cuda.synchronize(device)
    t_pinned_sync = (time.perf_counter() - t0) / args.iters
    bw_pinned_sync = format_gb_s(expert_bytes, t_pinned_sync)
    print(f"  [2B] Pinned Host -> GPU (blocking)      : {t_pinned_sync * 1000.0:6.2f} ms | {bw_pinned_sync:6.2f} GB/s")

    # Test 2C: Pinned Dedicated Stream Async DMA
    dma_stream = torch.cuda.Stream(device=device)
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for _ in range(args.warmup):
        with torch.cuda.stream(dma_stream):
            gpu_dst.copy_(pinned_src, non_blocking=True)
    dma_stream.synchronize()

    dma_times = []
    for _ in range(args.iters):
        with torch.cuda.stream(dma_stream):
            ev_start.record(dma_stream)
            gpu_dst.copy_(pinned_src, non_blocking=True)
            ev_end.record(dma_stream)
        dma_stream.synchronize()
        dma_times.append(ev_start.elapsed_time(ev_end))

    dma_times.sort()
    t_pinned_async = (sum(dma_times) / len(dma_times)) / 1000.0  # seconds
    t_pinned_async_p50 = dma_times[len(dma_times) // 2] / 1000.0
    bw_pinned_async = format_gb_s(expert_bytes, t_pinned_async)
    bw_pinned_async_p50 = format_gb_s(expert_bytes, t_pinned_async_p50)
    print(f"  [2C] Pinned Host -> GPU (async stream)  : {t_pinned_async * 1000.0:6.2f} ms | {bw_pinned_async:6.2f} GB/s (p50: {bw_pinned_async_p50:.2f} GB/s)")

    results["gate2_pinned_vs_unpinned"] = {
        "unpinned_ms": round(t_unpinned * 1000.0, 2),
        "unpinned_gb_s": round(bw_unpinned, 2),
        "pinned_sync_ms": round(t_pinned_sync * 1000.0, 2),
        "pinned_sync_gb_s": round(bw_pinned_sync, 2),
        "pinned_async_ms": round(t_pinned_async * 1000.0, 2),
        "pinned_async_gb_s": round(bw_pinned_async, 2),
        "pinned_speedup": round(bw_pinned_async / max(0.01, bw_unpinned), 2),
    }

    # -------------------------------------------------------------------------
    # GATE 3: Coalesced (1x352 MB) vs Fragmented (3x117 MB)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("GATE 3: Coalesced vs Fragmented Multi-Tensor DMA")
    print("-" * 80)

    # 3 Separate Tensors (Current COLOSSUS layout: gate, up, down)
    g_pin = torch.empty((intermediate_size, hidden_size), dtype=dtype, device="cpu", pin_memory=True)
    u_pin = torch.empty((intermediate_size, hidden_size), dtype=dtype, device="cpu", pin_memory=True)
    d_pin = torch.empty((hidden_size, intermediate_size), dtype=dtype, device="cpu", pin_memory=True)

    g_gpu = torch.empty((intermediate_size, hidden_size), dtype=dtype, device=device)
    u_gpu = torch.empty((intermediate_size, hidden_size), dtype=dtype, device=device)
    d_gpu = torch.empty((hidden_size, intermediate_size), dtype=dtype, device=device)

    frag_times = []
    for _ in range(args.warmup):
        with torch.cuda.stream(dma_stream):
            g_gpu.copy_(g_pin, non_blocking=True)
            u_gpu.copy_(u_pin, non_blocking=True)
            d_gpu.copy_(d_pin, non_blocking=True)
    dma_stream.synchronize()

    for _ in range(args.iters):
        with torch.cuda.stream(dma_stream):
            ev_start.record(dma_stream)
            g_gpu.copy_(g_pin, non_blocking=True)
            u_gpu.copy_(u_pin, non_blocking=True)
            d_gpu.copy_(d_pin, non_blocking=True)
            ev_end.record(dma_stream)
        dma_stream.synchronize()
        frag_times.append(ev_start.elapsed_time(ev_end))

    t_frag = (sum(frag_times) / len(frag_times)) / 1000.0
    bw_frag = format_gb_s(expert_bytes, t_frag)
    print(f"  [3A] Fragmented 3x Tensors (gate,up,down): {t_frag * 1000.0:6.2f} ms | {bw_frag:6.2f} GB/s")
    print(f"  [3B] Coalesced 1x Tensor   (contiguous) : {t_pinned_async * 1000.0:6.2f} ms | {bw_pinned_async:6.2f} GB/s")
    overhead_diff_ms = (t_frag - t_pinned_async) * 1000.0
    print(f"  -> Coalescing gain: {bw_pinned_async / bw_frag:.2f}x bandwidth ({overhead_diff_ms:+.2f} ms driver overhead per expert)")

    results["gate3_coalesced_vs_fragmented"] = {
        "fragmented_3x_ms": round(t_frag * 1000.0, 2),
        "fragmented_3x_gb_s": round(bw_frag, 2),
        "coalesced_1x_ms": round(t_pinned_async * 1000.0, 2),
        "coalesced_1x_gb_s": round(bw_pinned_async, 2),
        "driver_overhead_delta_ms": round(overhead_diff_ms, 2),
    }

    # -------------------------------------------------------------------------
    # GATE 4: Overlapped Computation + DMA Transfer
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("GATE 4: Computation-Transfer Overlap Benchmark")
    print("-" * 80)

    # Simulate realistic transformer compute (Multi-Head Attention GEMM)
    # e.g., matmul [4096, 4096] x [4096, 4096] repeated to simulate ~15-20 ms attention
    a_mat = torch.randn(4096, 4096, dtype=dtype, device=device)
    b_mat = torch.randn(4096, 4096, dtype=dtype, device=device)

    def run_compute_kernel(reps=10):
        c = a_mat
        for _ in range(reps):
            c = torch.matmul(c, b_mat)
        return c

    # Calibrate compute reps to reach ~15 ms
    reps = 8
    for _ in range(args.warmup):
        run_compute_kernel(reps)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_compute_kernel(reps)
    torch.cuda.synchronize(device)
    t_compute_alone = (time.perf_counter() - t0) / args.iters
    print(f"  Simulated Attention Compute Alone  : {t_compute_alone * 1000.0:6.2f} ms")
    print(f"  Dedicated Pinned DMA Alone         : {t_pinned_async * 1000.0:6.2f} ms")

    # Now run both CONCURRENTLY:
    # DMA runs on dma_stream, Compute runs on default_stream
    default_stream = torch.cuda.current_stream(device)
    ev_wall_start = torch.cuda.Event(enable_timing=True)
    ev_wall_end = torch.cuda.Event(enable_timing=True)

    for _ in range(args.warmup):
        with torch.cuda.stream(dma_stream):
            gpu_dst.copy_(pinned_src, non_blocking=True)
        run_compute_kernel(reps)
        dma_stream.synchronize()
        torch.cuda.synchronize(device)

    overlap_times = []
    for _ in range(args.iters):
        ev_wall_start.record(default_stream)
        # 1. Launch DMA on background stream
        with torch.cuda.stream(dma_stream):
            gpu_dst.copy_(pinned_src, non_blocking=True)
        # 2. Concurrently execute compute on main stream
        run_compute_kernel(reps)
        # 3. Synchronize both streams
        dma_stream.synchronize()
        torch.cuda.synchronize(device)
        ev_wall_end.record(default_stream)
        ev_wall_end.synchronize()
        overlap_times.append(ev_wall_start.elapsed_time(ev_wall_end))

    t_overlapped = (sum(overlap_times) / len(overlap_times)) / 1000.0
    sum_isolated = t_compute_alone + t_pinned_async
    overlap_savings_ms = (sum_isolated - t_overlapped) * 1000.0
    overlap_eff_pct = (sum_isolated - t_overlapped) / min(t_compute_alone, t_pinned_async) * 100.0

    print(f"  Concurrently Overlapped Wallclock  : {t_overlapped * 1000.0:6.2f} ms (vs {sum_isolated * 1000.0:.2f} ms serialized)")
    print(f"  Latency Hidden Behind Compute      : {overlap_savings_ms:6.2f} ms ({overlap_eff_pct:5.1f}% overlap efficiency)")

    results["gate4_overlap"] = {
        "compute_alone_ms": round(t_compute_alone * 1000.0, 2),
        "dma_alone_ms": round(t_pinned_async * 1000.0, 2),
        "serialized_sum_ms": round(sum_isolated * 1000.0, 2),
        "overlapped_wallclock_ms": round(t_overlapped * 1000.0, 2),
        "hidden_latency_ms": round(overlap_savings_ms, 2),
        "overlap_efficiency_pct": round(overlap_eff_pct, 1),
    }

    # -------------------------------------------------------------------------
    # Theoretical Model Verification & Projection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EMPIRICAL HARDWARE VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"  Measured PCIe Gen4 x16 Pinned DMA Bandwidth : {bw_pinned_async:.2f} GB/s")
    print(f"  Transfer Time per 352.3 MB Mixtral Expert   : {t_pinned_async * 1000.0:.2f} ms")
    print(f"  22.5 GB Contiguous Pinned Host Buffer Alloc : {alloc_results['Offload Target (22.5 GB = 64 Experts)']['alloc_time_ms']:.2f} ms ({alloc_results['Offload Target (22.5 GB = 64 Experts)']['alloc_time_s']:.3f} s)")
    print(f"  Speedup of Pinned DMA over Unpinned Copy    : {bw_pinned_async / max(0.01, bw_unpinned):.2f}x")
    print(f"  Overlap Efficiency Behind Attention         : {overlap_eff_pct:.1f}%")
    print("=" * 80)

    # Save to JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Detailed microbenchmark metrics saved to: {args.out_json}")

if __name__ == "__main__":
    main()
