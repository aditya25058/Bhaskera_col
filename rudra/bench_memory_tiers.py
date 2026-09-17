#!/usr/bin/env python3
"""
bench_memory_tiers.py — Empirical Communication Path & Memory Matrix
====================================================================

Measures measured (not theoretical) bandwidth and latency across the 
physical communication paths on the target GPU node:
  1. HBM -> HBM (Local intra-GPU copy)
  2. GPU 0 -> GPU 1 (Intra-node P2P / PCIe peer transfer)
  3. CPU DRAM -> GPU (H2D pinned memory PCIe DMA)
  4. GPU -> CPU DRAM (D2H pinned memory PCIe DMA)
  5. Node Network (TCP socket line rate)

Exact payload sizes evaluated:
  - 4 KB   (Param2-17B single-token activation [1, 1, 2048] bfloat16)
  - 64 KB
  - 256 KB
  - 1 MB
  - 4 MB
  - 16 MB  (~Param2-17B 10% cold columns payload: 14.34 MB)
  - 64 MB
  - 256 MB (Bulk line-rate saturation)
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from typing import Dict, List
import torch

# Force unbuffered output so SLURM logs update in real-time
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

PAYLOADS = [
    ("4 KB", 4 * 1024),
    ("64 KB", 64 * 1024),
    ("256 KB", 256 * 1024),
    ("1 MB", 1 * 1024 * 1024),
    ("4 MB", 4 * 1024 * 1024),
    ("16 MB", 16 * 1024 * 1024),
    ("64 MB", 64 * 1024 * 1024),
    ("256 MB", 256 * 1024 * 1024),
]

WARMUP = 5
ITERS = 25


def bench_hbm(device_id: int = 0) -> List[Dict]:
    """Path: HBM -> HBM (Local intra-GPU copy)."""
    results = []
    device = torch.device(f"cuda:{device_id}")
    for name, size_bytes in PAYLOADS:
        n_elements = size_bytes // 2  # bfloat16
        src = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        dst = torch.empty(n_elements, dtype=torch.bfloat16, device=device)

        for _ in range(WARMUP):
            dst.copy_(src)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(ITERS):
            dst.copy_(src)
        end.record()
        torch.cuda.synchronize(device)

        avg_ms = start.elapsed_time(end) / ITERS
        gb_s = (size_bytes / (1024 ** 3)) / (avg_ms / 1000.0) if avg_ms > 0 else 0.0

        res = {
            "path": "HBM -> HBM",
            "direction": "local",
            "payload": name,
            "bytes": size_bytes,
            "latency_ms": avg_ms,
            "latency_us": avg_ms * 1000.0,
            "bandwidth_gbs": gb_s,
        }
        results.append(res)
        lat_str = f"{res['latency_us']:.1f} μs" if res['latency_us'] < 1000 else f"{res['latency_ms']:.2f} ms"
        bw_str = f"{res['bandwidth_gbs']:.2f} GB/s" if res['bandwidth_gbs'] >= 1.0 else f"{res['bandwidth_gbs']*1024:.1f} MB/s"
        print(f"  [HBM] {name:<10}: {bw_str:>14} | {lat_str:>12}", flush=True)

        del src, dst
        torch.cuda.empty_cache()
    return results


def bench_p2p() -> List[Dict]:
    """Path: GPU 0 -> GPU 1 (P2P / Peer PCIe transfer)."""
    results = []
    if torch.cuda.device_count() < 2:
        return results

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    can_access = torch.cuda.can_device_access_peer(0, 1)
    boundary_name = "P2P" if can_access else "PCIe Peer"

    for name, size_bytes in PAYLOADS:
        n_elements = size_bytes // 2
        src = torch.empty(n_elements, dtype=torch.bfloat16, device=dev0)
        dst = torch.empty(n_elements, dtype=torch.bfloat16, device=dev1)

        for _ in range(WARMUP):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(dev0)
        torch.cuda.synchronize(dev1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(ITERS):
            dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize(dev1)

        avg_ms = start.elapsed_time(end) / ITERS
        gb_s = (size_bytes / (1024 ** 3)) / (avg_ms / 1000.0) if avg_ms > 0 else 0.0

        res = {
            "path": "GPU 0 -> GPU 1",
            "direction": boundary_name,
            "payload": name,
            "bytes": size_bytes,
            "latency_ms": avg_ms,
            "latency_us": avg_ms * 1000.0,
            "bandwidth_gbs": gb_s,
        }
        results.append(res)
        lat_str = f"{res['latency_us']:.1f} μs" if res['latency_us'] < 1000 else f"{res['latency_ms']:.2f} ms"
        bw_str = f"{res['bandwidth_gbs']:.2f} GB/s" if res['bandwidth_gbs'] >= 1.0 else f"{res['bandwidth_gbs']*1024:.1f} MB/s"
        print(f"  [P2P] {name:<10}: {bw_str:>14} | {lat_str:>12}", flush=True)

        del src, dst
        torch.cuda.empty_cache()
    return results


def bench_dma(direction: str = "H2D", device_id: int = 0) -> List[Dict]:
    """Path: CPU DRAM <-> GPU PCIe DMA (Pinned memory)."""
    results = []
    device = torch.device(f"cuda:{device_id}")

    for name, size_bytes in PAYLOADS:
        n_elements = size_bytes // 2
        cpu_buf = torch.empty(n_elements, dtype=torch.bfloat16, pin_memory=True)
        gpu_buf = torch.empty(n_elements, dtype=torch.bfloat16, device=device)

        src = cpu_buf if direction == "H2D" else gpu_buf
        dst = gpu_buf if direction == "H2D" else cpu_buf
        path_name = "CPU -> GPU" if direction == "H2D" else "GPU -> CPU"

        for _ in range(WARMUP):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(ITERS):
            dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize(device)

        avg_ms = start.elapsed_time(end) / ITERS
        gb_s = (size_bytes / (1024 ** 3)) / (avg_ms / 1000.0) if avg_ms > 0 else 0.0

        res = {
            "path": path_name,
            "direction": direction,
            "payload": name,
            "bytes": size_bytes,
            "latency_ms": avg_ms,
            "latency_us": avg_ms * 1000.0,
            "bandwidth_gbs": gb_s,
        }
        results.append(res)
        lat_str = f"{res['latency_us']:.1f} μs" if res['latency_us'] < 1000 else f"{res['latency_ms']:.2f} ms"
        bw_str = f"{res['bandwidth_gbs']:.2f} GB/s" if res['bandwidth_gbs'] >= 1.0 else f"{res['bandwidth_gbs']*1024:.1f} MB/s"
        print(f"  [{direction}] {name:<10}: {bw_str:>14} | {lat_str:>12}", flush=True)

        del cpu_buf, gpu_buf
        torch.cuda.empty_cache()
    return results


def bench_tcp_socket() -> List[Dict]:
    """Path: Node Network (TCP socket line rate via threaded client/server)."""
    results = []
    t_iters = 10

    for name, size_bytes in PAYLOADS:
        data = b"x" * size_bytes

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        conn, _ = server.accept()

        def _receiver():
            for _ in range(t_iters):
                received = 0
                while received < size_bytes:
                    chunk = conn.recv(min(65536, size_bytes - received))
                    if not chunk:
                        return
                    received += len(chunk)

        rx_thread = threading.Thread(target=_receiver)
        rx_thread.start()

        t_start = time.perf_counter()
        for _ in range(t_iters):
            client.sendall(data)
        rx_thread.join()
        t_total = time.perf_counter() - t_start

        conn.close()
        client.close()
        server.close()

        avg_ms = (t_total / t_iters) * 1000.0
        gb_s = (size_bytes / (1024 ** 3)) / (avg_ms / 1000.0) if avg_ms > 0 else 0.0

        res = {
            "path": "Node -> Node",
            "direction": "TCP Socket",
            "payload": name,
            "bytes": size_bytes,
            "latency_ms": avg_ms,
            "latency_us": avg_ms * 1000.0,
            "bandwidth_gbs": gb_s,
        }
        results.append(res)
        lat_str = f"{res['latency_us']:.1f} μs" if res['latency_us'] < 1000 else f"{res['latency_ms']:.2f} ms"
        bw_str = f"{res['bandwidth_gbs']:.2f} GB/s" if res['bandwidth_gbs'] >= 1.0 else f"{res['bandwidth_gbs']*1024:.1f} MB/s"
        print(f"  [TCP] {name:<10}: {bw_str:>14} | {lat_str:>12}", flush=True)

    return results


def main():
    print("=" * 86, flush=True)
    print("  EMPIRICAL COMMUNICATION PATH & MEMORY MATRIX BENCHMARK", flush=True)
    print(f"  Node: {socket.gethostname()}", flush=True)
    print(f"  PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}", flush=True)
    print(f"  Visible GPUs: {torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB)", flush=True)
    if torch.cuda.device_count() >= 2:
        can_p2p = torch.cuda.can_device_access_peer(0, 1)
        print(f"  P2P Access (GPU 0 <-> GPU 1): {'Supported' if can_p2p else 'Not Supported / PCIe Routed'}", flush=True)
    print("=" * 86, flush=True)

    all_results = []
    print("\n[1/5] Profiling Path: HBM -> HBM (Local)...", flush=True)
    all_results.extend(bench_hbm(0))

    if torch.cuda.device_count() >= 2:
        print("\n[2/5] Profiling Path: GPU 0 -> GPU 1 (P2P)...", flush=True)
        all_results.extend(bench_p2p())
    else:
        print("\n[2/5] GPU 0 -> GPU 1 skipped (requires 2 allocated GPUs)", flush=True)

    print("\n[3/5] Profiling Path: CPU -> GPU (H2D PCIe DMA)...", flush=True)
    all_results.extend(bench_dma("H2D", 0))

    print("\n[4/5] Profiling Path: GPU -> CPU (D2H PCIe DMA)...", flush=True)
    all_results.extend(bench_dma("D2H", 0))

    print("\n[5/5] Profiling Path: Node -> Node (TCP Socket)...", flush=True)
    all_results.extend(bench_tcp_socket())

    # Formatted Paper-Ready Table
    print("\n" + "=" * 86, flush=True)
    print(f"{'Path':<16} {'Payload':<10} {'Bandwidth':>16} {'Latency':>16} {'Direction':>18}", flush=True)
    print("-" * 86, flush=True)
    for r in all_results:
        lat_str = f"{r['latency_us']:.1f} μs" if r['latency_us'] < 1000 else f"{r['latency_ms']:.2f} ms"
        if r['bandwidth_gbs'] >= 1.0:
            bw_str = f"{r['bandwidth_gbs']:.2f} GB/s"
        elif r['bandwidth_gbs'] >= 0.001:
            bw_str = f"{r['bandwidth_gbs']*1024:.1f} MB/s"
        else:
            bw_str = f"{r['bandwidth_gbs']*1024*1024:.1f} KB/s"
        print(f"{r['path']:<16} {r['payload']:<10} {bw_str:>16} {lat_str:>16} {r['direction']:>18}", flush=True)
    print("-" * 86, flush=True)


if __name__ == "__main__":
    main()
