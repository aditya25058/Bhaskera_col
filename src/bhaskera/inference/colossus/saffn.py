"""
bhaskera.inference.colossus.saffn
=================================
Sparse-Aware Feed-Forward Network (SA-FFN) and
Adaptive Dynamic Expert Tensor Reorganization (ADETR) Layout.

As specified in COLOSSUS v3 (Section 5.3, Equations 30-34, and Section 7.1):
1. ADETR Memory Layout: Partitions SwiGLU intermediate dimension I into:
   - Hot columns (I_hot): Permanently resident in accelerator memory.
   - Cold columns (I_cold): Stored in pinned host memory, transferred on demand/prefetch.
2. SA-FFN Decomposed Execution:
   y_cached = W_down_c @ (act_fn(x @ W_gate_c.T) * (x @ W_up_c.T))
   y_missed = W_down_m @ (act_fn(x @ W_gate_m.T) * (x @ W_up_m.T))
   y = y_cached + y_missed
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ADETRBuffer:
    """Manages contiguous hot and cold column partitions for an expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hot_ratio: float = 0.25,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hot_ratio = hot_ratio
        self.dtype = dtype
        self.device = device

        self.i_hot = int(intermediate_size * hot_ratio)
        self.i_cold = intermediate_size - self.i_hot

        # 1. Permanent hot GPU residency (ADETR Partition 1)
        self.gpu_gate_hot = torch.empty((self.i_hot, hidden_size), dtype=dtype, device=device)
        self.gpu_up_hot   = torch.empty((self.i_hot, hidden_size), dtype=dtype, device=device)
        self.gpu_down_hot = torch.empty((hidden_size, self.i_hot), dtype=dtype, device=device)

        # 2. Host cold partition (ADETR Partition 2)
        pin = torch.cuda.is_available()
        self.cpu_gate_cold = torch.empty((self.i_cold, hidden_size), dtype=dtype, pin_memory=pin)
        self.cpu_up_cold   = torch.empty((self.i_cold, hidden_size), dtype=dtype, pin_memory=pin)
        self.cpu_down_cold = torch.empty((hidden_size, self.i_cold), dtype=dtype, pin_memory=pin)

        # 3. Pre-allocated contiguous receive buffer on GPU for cold columns
        self.gpu_gate_cold_rx = torch.empty((self.i_cold, hidden_size), dtype=dtype, device=device)
        self.gpu_up_cold_rx   = torch.empty((self.i_cold, hidden_size), dtype=dtype, device=device)
        self.gpu_down_cold_rx = torch.empty((hidden_size, self.i_cold), dtype=dtype, device=device)

        self.cold_resident = False

    def init_from_weights(self, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor):
        """Partition master expert weights into hot GPU and cold CPU buffers."""
        with torch.no_grad():
            # Hot partition -> GPU
            self.gpu_gate_hot.copy_(gate_w[:self.i_hot, :], non_blocking=True)
            self.gpu_up_hot.copy_(up_w[:self.i_hot, :], non_blocking=True)
            self.gpu_down_hot.copy_(down_w[:, :self.i_hot], non_blocking=True)

            # Cold partition -> CPU Pinned
            self.cpu_gate_cold.copy_(gate_w[self.i_hot:, :], non_blocking=True)
            self.cpu_up_cold.copy_(up_w[self.i_hot:, :], non_blocking=True)
            self.cpu_down_cold.copy_(down_w[:, self.i_hot:], non_blocking=True)

    def load_cold_columns(self, non_blocking: bool = True, stream: Optional[torch.cuda.Stream] = None):
        """Transfer cold columns into pre-allocated contiguous GPU receive buffer."""
        if torch.cuda.is_available():
            stream_ctx = stream if stream else torch.cuda.current_stream()
            with torch.cuda.stream(stream_ctx):
                self.gpu_gate_cold_rx.copy_(self.cpu_gate_cold, non_blocking=non_blocking)
                self.gpu_up_cold_rx.copy_(self.cpu_up_cold, non_blocking=non_blocking)
                self.gpu_down_cold_rx.copy_(self.cpu_down_cold, non_blocking=non_blocking)
        else:
            self.gpu_gate_cold_rx.copy_(self.cpu_gate_cold)
            self.gpu_up_cold_rx.copy_(self.cpu_up_cold)
            self.gpu_down_cold_rx.copy_(self.cpu_down_cold)
        self.cold_resident = True

    @property
    def cold_bytes(self) -> int:
        """Total bytes required to transfer cold column partition."""
        return (
            self.cpu_gate_cold.numel() * self.cpu_gate_cold.element_size() +
            self.cpu_up_cold.numel() * self.cpu_up_cold.element_size() +
            self.cpu_down_cold.numel() * self.cpu_down_cold.element_size()
        )


class SA_FFN_Expert(nn.Module):
    """Sparse-Aware Feed-Forward Network executing decomposed SwiGLU forward pass."""

    def __init__(self, adetr_buffer: ADETRBuffer, act_fn=F.silu):
        super().__init__()
        self.buf = adetr_buffer
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decomposed SwiGLU forward: y = y_cached + y_missed (Eq 30-34)."""
        # Phase 1: Cached Hot Columns
        g_c = F.linear(x, self.buf.gpu_gate_hot)
        u_c = F.linear(x, self.buf.gpu_up_hot)
        y_cached = F.linear(self.act_fn(g_c) * u_c, self.buf.gpu_down_hot)

        # Phase 2: Transferred Cold Columns
        if self.buf.cold_resident:
            g_m = F.linear(x, self.buf.gpu_gate_cold_rx)
            u_m = F.linear(x, self.buf.gpu_up_cold_rx)
            y_missed = F.linear(self.act_fn(g_m) * u_m, self.buf.gpu_down_cold_rx)
            return y_cached + y_missed

        return y_cached
