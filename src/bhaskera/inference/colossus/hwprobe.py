"""Hardware capability probe (planner input contract).

Discovers GPUs/HBM, CPU/RAM, and MEASURES the bandwidths the cost model
needs (PCIe DMA, DRAM triad-lite) — names are displayed, numbers decide.
Fast by design (<30 s): small transfers, no model weights, no training.

Output: JSON-serializable dict (see probe()). CPU-only fields always work;
CUDA fields degrade gracefully to absent (CPU-only hosts supported).
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any, Dict

import torch


def _cpu_info() -> Dict[str, Any]:
    import multiprocessing
    info: Dict[str, Any] = {"cores": multiprocessing.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    flags = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except Exception:
        pass
    info["bf16"] = ("avx512_bf16" in flags) or ("bf16" in flags)
    info["arch"] = platform.machine()
    return info


def _ram_info() -> Dict[str, Any]:
    out = {"total_gb": 0.0, "available_gb": 0.0}
    try:
        with open("/proc/meminfo") as f:
            kv = {}
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    kv[p[0].rstrip(":")] = int(p[1])  # kB
        out["total_gb"] = kv.get("MemTotal", 0) / (1024 ** 2)
        out["available_gb"] = kv.get("MemAvailable", kv.get("MemFree", 0)) / (1024 ** 2)
    except Exception:
        pass
    return out


def _gpu_info() -> list:
    gpus = []
    if not torch.cuda.is_available():
        return gpus
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append({"index": i, "name": p.name,
                     "total_gb": p.total_memory / (1024 ** 3),
                     "capability": f"{p.major}.{p.minor}",
                     "multiprocessors": p.multi_processor_count})
    return gpus


def _measure_dram_gbs(n_gb: int = 1, iters: int = 5) -> float:
    """Triad-lite a=b+c over n_gb (float32); best-of-iters GB/s."""
    n = (n_gb * 1024 ** 3) // 4
    a = torch.empty(n, dtype=torch.float32)
    b = torch.rand(n, dtype=torch.float32)
    c = torch.rand(n, dtype=torch.float32)
    for _ in range(2):
        a.copy_(b).add_(c)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        a.copy_(b).add_(c)
        best = min(best, time.perf_counter() - t0)
    return (3 * n * 4) / best / 1e9


def _measure_pcie_gbs(size_mb: int = 256, iters: int = 3) -> float:
    """Pinned-host -> device copy; best-of-iters GB/s. 0.0 without CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    dev = torch.device("cuda:0")
    n = (size_mb * 1024 * 1024) // 4
    h = torch.empty(n, dtype=torch.float32, pin_memory=True)
    d = torch.empty(n, dtype=torch.float32, device=dev)
    h.random_()
    torch.cuda.synchronize(dev)
    best = float("inf")
    for _ in range(iters):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        d.copy_(h, non_blocking=True)
        torch.cuda.synchronize(dev)
        best = min(best, time.perf_counter() - t0)
    del d
    return (n * 4) / best / 1e9


def probe(fast: bool = True) -> Dict[str, Any]:
    """Hardware capability dict (JSON-serializable)."""
    gpus = _gpu_info()
    pcie = _measure_pcie_gbs(64 if fast else 256, 2 if fast else 3)
    return {
        "gpus": gpus,
        "cpu": _cpu_info(),
        "ram_gb": _ram_info(),
        "pcie_gbs": round(pcie, 1),
        "dram_gbs": round(_measure_dram_gbs(1, 4), 1),
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__.split("+")[0],
    }


def write_probe_json(out_path: str, fast: bool = True) -> Dict[str, Any]:
    import json
    desc = probe(fast=fast)
    with open(out_path, "w") as f:
        json.dump(desc, f, indent=2)
    return desc
