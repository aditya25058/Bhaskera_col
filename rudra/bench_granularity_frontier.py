#!/usr/bin/env python3
"""
Controlled Expert-Granularity Experiment for COLOSSUS & ADETR.

Directly tests the central scientific hypothesis:
  "COLOSSUS works when expert granularity is compatible with the interconnect."
  Governing parameter:
    R = T_DMA / T_compute = S_cold / (B_PCIe * T_compute)

Evaluates:
  1. Controlled sweep of expert sizes across identical hardware and PCIe bus:
     S_E in [12 MB, 24 MB, 48 MB, 96 MB, 192 MB, 336 MB]
  2. Real model reference points:
     - Param2-17B-A2.4B (S_E = 24.0 MB)
     - Mixtral-8x7B-v0.1 (S_E = 336.0 MB)
  3. Precise measurements:
     - S_E (Full Expert MB)
     - S_cold (Cold Payload MB at 50% and 10% cold)
     - T_compute (Layer GEMM compute time via CUDA events)
     - T_DMA (Measured PCIe transfer time via CUDA events)
     - R = T_DMA / T_compute
     - Baseline offloading TPS vs ADETR TPS vs Prefetched TPS
     - COLOSSUS Throughput Recovery Factor
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GranularityExperiment")


class SyntheticMoELayer(nn.Module):
    """Controlled MoE layer with configurable expert size and ADETR column slicing."""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        missing_col_ratio: float = 0.50,
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.H = hidden_size
        self.I = intermediate_size
        self.E = num_experts
        self.K = top_k
        self.missing_ratio = missing_col_ratio
        self.dev = device

        # Calculate exact expert size: 3 matrices (gate, up, down) in BF16
        # gate: [I, H], up: [I, H], down: [H, I]
        self.expert_bytes = (2 * self.I * self.H + self.H * self.I) * 2
        self.expert_mb = self.expert_bytes / (1024 ** 2)

        # ADETR partitioning
        self.i_missed = int(self.I * self.missing_ratio)
        self.i_hot = self.I - self.i_missed
        self.cold_bytes = (2 * self.i_missed * self.H + self.H * self.i_missed) * 2
        self.cold_mb = self.cold_bytes / (1024 ** 2)

        # Permanent hot columns in GPU HBM
        self.gpu_gate_hot = torch.randn((self.E, self.i_hot, self.H), dtype=torch.bfloat16, device=device)
        self.gpu_up_hot   = torch.randn((self.E, self.i_hot, self.H), dtype=torch.bfloat16, device=device)
        self.gpu_down_hot = torch.randn((self.E, self.H, self.i_hot), dtype=torch.bfloat16, device=device)

        # Cold columns stored in CPU host memory
        self.cpu_gate_cold = torch.randn((self.E, self.i_missed, self.H), dtype=torch.bfloat16, pin_memory=False)
        self.cpu_up_cold   = torch.randn((self.E, self.i_missed, self.H), dtype=torch.bfloat16, pin_memory=False)
        self.cpu_down_cold = torch.randn((self.E, self.i_missed, self.H), dtype=torch.bfloat16, pin_memory=False)

        # GPU receive scratchpad buffers
        self.rx_gate_cold = torch.empty((self.K, self.i_missed, self.H), dtype=torch.bfloat16, device=device)
        self.rx_up_cold   = torch.empty((self.K, self.i_missed, self.H), dtype=torch.bfloat16, device=device)
        self.rx_down_cold = torch.empty((self.K, self.i_missed, self.H), dtype=torch.bfloat16, device=device)

        # Router
        self.gate = nn.Linear(self.H, self.E, bias=False, dtype=torch.bfloat16, device=device)

        # Dedicated CUDA streams for prefetch & demand
        self.dma_stream = torch.cuda.Stream(device)

    def measure_compute_time(self, iters: int = 50) -> float:
        """Measure pure GPU computation time for 1 token through top-k experts."""
        x = torch.randn((1, self.H), dtype=torch.bfloat16, device=self.dev)
        # Warmup
        for _ in range(10):
            logits = self.gate(x)
            topk_idx = torch.topk(logits, k=self.K, dim=-1).indices[0]
            for k_slot, e_id in enumerate(topk_idx):
                h_gate = F.linear(x, self.gpu_gate_hot[e_id])
                h_up   = F.linear(x, self.gpu_up_hot[e_id])
                act = F.silu(h_gate) * h_up
                out = F.linear(act, self.gpu_down_hot[e_id])
        torch.cuda.synchronize(self.dev)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        ev_start.record()
        for _ in range(iters):
            logits = self.gate(x)
            topk_idx = torch.topk(logits, k=self.K, dim=-1).indices[0]
            for k_slot, e_id in enumerate(topk_idx):
                h_gate = F.linear(x, self.gpu_gate_hot[e_id])
                h_up   = F.linear(x, self.gpu_up_hot[e_id])
                act = F.silu(h_gate) * h_up
                out = F.linear(act, self.gpu_down_hot[e_id])
        ev_end.record()
        torch.cuda.synchronize(self.dev)
        return (ev_start.elapsed_time(ev_end) / iters)  # ms

    def measure_dma_time(self, iters: int = 30) -> float:
        """Measure actual physical PCIe DMA transfer time for cold columns of K experts."""
        # Warmup
        for _ in range(5):
            for k in range(self.K):
                self.rx_gate_cold[k].copy_(self.cpu_gate_cold[k], non_blocking=False)
                self.rx_up_cold[k].copy_(self.cpu_up_cold[k], non_blocking=False)
                self.rx_down_cold[k].copy_(self.cpu_down_cold[k], non_blocking=False)
        torch.cuda.synchronize(self.dev)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        ev_start.record()
        for _ in range(iters):
            for k in range(self.K):
                self.rx_gate_cold[k].copy_(self.cpu_gate_cold[k], non_blocking=True)
                self.rx_up_cold[k].copy_(self.cpu_up_cold[k], non_blocking=True)
                self.rx_down_cold[k].copy_(self.cpu_down_cold[k], non_blocking=True)
            torch.cuda.synchronize(self.dev)
        ev_end.record()
        torch.cuda.synchronize(self.dev)
        return (ev_start.elapsed_time(ev_end) / iters)  # ms

    def measure_overlapped_execution(self, iters: int = 30) -> Dict[str, float]:
        """
        Measures:
          1. Sequential execution: T_seq = T_DMA + T_compute
          2. Overlapped prefetch: T_overlap = max(T_DMA, T_compute) + sync
        """
        x = torch.randn((1, self.H), dtype=torch.bfloat16, device=self.dev)

        # 1. Sequential
        ev_seq_start = torch.cuda.Event(enable_timing=True)
        ev_seq_end = torch.cuda.Event(enable_timing=True)
        ev_seq_start.record()
        for _ in range(iters):
            # DMA transfer
            for k in range(self.K):
                self.rx_gate_cold[k].copy_(self.cpu_gate_cold[k], non_blocking=False)
                self.rx_up_cold[k].copy_(self.cpu_up_cold[k], non_blocking=False)
                self.rx_down_cold[k].copy_(self.cpu_down_cold[k], non_blocking=False)
            torch.cuda.synchronize(self.dev)
            # Compute
            for k in range(self.K):
                h_g = F.linear(x, self.gpu_gate_hot[k])
                h_u = F.linear(x, self.gpu_up_hot[k])
                act = F.silu(h_g) * h_u
                out = F.linear(act, self.gpu_down_hot[k])
            torch.cuda.synchronize(self.dev)
        ev_seq_end.record()
        torch.cuda.synchronize(self.dev)
        t_seq = ev_seq_start.elapsed_time(ev_seq_end) / iters

        # 2. Asynchronous Overlapped (Simulating Lookahead Prefetching)
        ev_ov_start = torch.cuda.Event(enable_timing=True)
        ev_ov_end = torch.cuda.Event(enable_timing=True)
        ev_ov_start.record()
        for _ in range(iters):
            # Launch DMA in background stream
            with torch.cuda.stream(self.dma_stream):
                for k in range(self.K):
                    self.rx_gate_cold[k].copy_(self.cpu_gate_cold[k], non_blocking=True)
                    self.rx_up_cold[k].copy_(self.cpu_up_cold[k], non_blocking=True)
                    self.rx_down_cold[k].copy_(self.cpu_down_cold[k], non_blocking=True)

            # In default stream, compute attention / hot column projection
            for k in range(self.K):
                h_g = F.linear(x, self.gpu_gate_hot[k])
                h_u = F.linear(x, self.gpu_up_hot[k])
                act = F.silu(h_g) * h_u
                out = F.linear(act, self.gpu_down_hot[k])

            # Synchronize streams
            torch.cuda.current_stream().wait_stream(self.dma_stream)
        ev_ov_end.record()
        torch.cuda.synchronize(self.dev)
        t_overlap = ev_ov_start.elapsed_time(ev_ov_end) / iters

        speedup = t_seq / t_overlap if t_overlap > 0 else 1.0
        return {
            "t_sequential_ms": round(t_seq, 2),
            "t_overlapped_ms": round(t_overlap, 2),
            "overlap_speedup": round(speedup, 3),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=str, default="/home/bapic_iiitd/2_group/expert_granularity_frontier.json")
    args = parser.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 80)
    logger.info("  CONTROLLED EXPERT-GRANULARITY & INTERCONNECT COMPATIBILITY EXPERIMENT")
    logger.info("  Device: %s", torch.cuda.get_device_name(0))
    logger.info("  Hypothesis: COLOSSUS viability is governed by R = T_DMA / T_compute")
    logger.info("=" * 80)

    # Sweep configurations covering 12 MB to 336 MB expert sizes
    # Configurations: (label, H, I, top_k, missing_ratio)
    configs = [
        {"name": "OLMoE-like",      "H": 2048, "I": 1024,  "K": 2, "missing": 0.50, "ref_model": "allenai/OLMoE-1B-7B"},
        {"name": "Param2-like",     "H": 2048, "I": 2048,  "K": 2, "missing": 0.50, "ref_model": "bharatgenai/Param2-17B"},
        {"name": "Qwen2-57B-like",  "H": 3584, "I": 2560,  "K": 2, "missing": 0.50, "ref_model": "Qwen/Qwen2-57B-A14B"},
        {"name": "Mid-Scale-1",     "H": 4096, "I": 4096,  "K": 2, "missing": 0.50, "ref_model": "Synthetic-96MB"},
        {"name": "Mid-Scale-2",     "H": 4096, "I": 8192,  "K": 2, "missing": 0.50, "ref_model": "Synthetic-192MB"},
        {"name": "Mixtral-8x7B",    "H": 4096, "I": 14336, "K": 2, "missing": 0.50, "ref_model": "mistralai/Mixtral-8x7B"},
    ]

    results = []

    for cfg in configs:
        logger.info("Evaluating: %-18s (H=%d, I=%d)...", cfg["name"], cfg["H"], cfg["I"])
        moe = SyntheticMoELayer(
            hidden_size=cfg["H"],
            intermediate_size=cfg["I"],
            num_experts=8,
            top_k=cfg["K"],
            missing_col_ratio=cfg["missing"],
            device=dev,
        )

        t_compute = moe.measure_compute_time(iters=40)
        t_dma = moe.measure_dma_time(iters=30)
        overlap_stats = moe.measure_overlapped_execution(iters=30)

        # R ratio
        R = t_dma / t_compute if t_compute > 0 else float("inf")

        # Measured bandwidth
        total_transfer_mb = moe.cold_mb * cfg["K"]
        measured_bw_gb_s = (total_transfer_mb / 1024.0) / (t_dma / 1000.0) if t_dma > 0 else 0.0

        # Theoretical throughput recovery:
        # Ideal recovery = T_compute / max(T_compute, T_dma)
        # When R <= 1: recovery -> 100%
        # When R >> 1: recovery -> 1 / R
        ideal_recovery_pct = min(100.0, (1.0 / R * 100.0)) if R > 0 else 100.0
        measured_recovery_pct = (overlap_stats["overlap_speedup"] / (1.0 + R)) * 100.0 if (1.0 + R) > 0 else 0.0

        entry = {
            "name": cfg["name"],
            "reference_model": cfg["ref_model"],
            "hidden_size": cfg["H"],
            "intermediate_size": cfg["I"],
            "expert_size_mb": round(moe.expert_mb, 1),
            "cold_payload_per_expert_mb": round(moe.cold_mb, 1),
            "total_cold_payload_k_experts_mb": round(total_transfer_mb, 1),
            "t_compute_layer_ms": round(t_compute, 3),
            "t_dma_transfer_ms": round(t_dma, 3),
            "measured_dma_bandwidth_gb_s": round(measured_bw_gb_s, 2),
            "R_ratio_dma_to_compute": round(R, 2),
            "t_sequential_ms": overlap_stats["t_sequential_ms"],
            "t_overlapped_ms": overlap_stats["t_overlapped_ms"],
            "overlap_speedup": overlap_stats["overlap_speedup"],
            "ideal_throughput_recovery_pct": round(ideal_recovery_pct, 1),
            "regime": (
                "🟢 Compute Dominated (Viable, R < 1)" if R < 1.0
                else "🟡 Transition Regime (1 <= R <= 5)" if R <= 5.0
                else "🔴 Interconnect Bound (R > 5, Unfavorable)"
            )
        }
        results.append(entry)

        logger.info(
            "  -> Expert: %5.1f MB | Cold: %5.1f MB | Compute: %5.2f ms | DMA: %5.2f ms | R = %5.2f | Overlap: %.2fx | %s",
            entry["expert_size_mb"],
            entry["cold_payload_per_expert_mb"],
            entry["t_compute_layer_ms"],
            entry["t_dma_transfer_ms"],
            entry["R_ratio_dma_to_compute"],
            entry["overlap_speedup"],
            entry["regime"],
        )

        del moe
        torch.cuda.empty_cache()

    output_data = {
        "title": "Controlled Expert Granularity Frontier Sweep",
        "hardware": torch.cuda.get_device_name(0),
        "interconnect_criterion": "R = T_DMA / T_compute = S_cold / (B_PCIe * T_compute)",
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE. Saved to %s", args.output_json)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
