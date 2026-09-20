"""Pre-allocated Slot-Based Dynamic MoE Expert Cache for COLOSSUS.

Architecture:
  - Exactly C expert slots pre-allocated on GPU per layer (zero cudaMalloc in forward).
  - 64 experts initialized on GPU pointing to pre-allocated slot buffers or dummy tensors.
  - Master expert copies reside permanently in pinned CPU host memory.
  - Streaming CPU -> GPU DMA transfers via non-blocking CUDA streams.
  - Zero modifications to model router (lossless contract).
  - Exact native moe_infer execution.
"""
from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DynamicMoELayerWrapper(nn.Module):
    """Wraps SparseMoeBlock with pre-allocated GPU slot cache and async DMA streaming."""

    _shared_dummy_expert: Optional[Any] = None
    _shared_down_cold_rx: Optional[torch.Tensor] = None
    _pinned_staging: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def get_pinned_staging(
        cls, key: str, intermediate_size: int, hidden_size: int, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate reusable pinned host staging buffers for direct DMA (Opt 1b).
        One buffer is ~352 MB. We maintain one for demand ('demand') and one for prefetch ('prefetch').
        Total host pinned RAM: ~704 MB (instantaneous 0.05s allocation vs 213s for 90GB).
        """
        if key not in cls._pinned_staging:
            g = torch.empty((intermediate_size, hidden_size), dtype=dtype, device="cpu", pin_memory=True)
            u = torch.empty((intermediate_size, hidden_size), dtype=dtype, device="cpu", pin_memory=True)
            d = torch.empty((hidden_size, intermediate_size), dtype=dtype, device="cpu", pin_memory=True)
            cls._pinned_staging[key] = (g, u, d)
        return cls._pinned_staging[key]

    def _exp_proj(self, expert, proj_type: str):
        """proj_type in ['gate', 'down', 'up']"""
        if proj_type == "gate":
            return getattr(expert, "gate_proj", getattr(expert, "w1", None))
        elif proj_type == "down":
            return getattr(expert, "down_proj", getattr(expert, "w2", None))
        elif proj_type == "up":
            return getattr(expert, "up_proj", getattr(expert, "w3", None))
        raise ValueError(f"Unknown proj_type {proj_type}")

    def __init__(
        self,
        layer_idx: int,
        moe_block: nn.Module,
        capacity: int = 16,
        device: torch.device = torch.device("cuda"),
        missing_col_ratio: float = 1.0,
        config: Optional[Any] = None,
        warmup_slots: int = 0,
        lookahead_enabled: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block = moe_block
        self.capacity = capacity
        self.warmup_slots = warmup_slots
        self.lookahead_enabled = lookahead_enabled
        self.device = device
        self.num_experts = len(moe_block.experts)
        self.top_k = getattr(moe_block, "num_experts_per_tok", getattr(moe_block, "top_k", 2))
        self.is_mixtral = hasattr(moe_block, "gate") and not hasattr(moe_block, "moe_infer")

        # Router weight for ZSSR speculative projection
        gate = getattr(moe_block, "gate", None)
        if hasattr(gate, "weight"):
            self.router_weight = gate.weight.detach().to(device).float()
        elif hasattr(gate, "gate") and hasattr(gate.gate, "weight"):
            self.router_weight = gate.gate.weight.detach().to(device).float()
        else:
            self.router_weight = None

        sample_exp = self.block.experts[0]
        if config is not None:
            self.config = config
        else:
            self.config = getattr(sample_exp, "config", getattr(moe_block, "config", None))
        g_proj = self._exp_proj(sample_exp, "gate")
        u_proj = self._exp_proj(sample_exp, "up")
        d_proj = self._exp_proj(sample_exp, "down")

        self.intermediate_size = getattr(sample_exp, "intermediate_size", g_proj.weight.shape[0])
        self.hidden_size = g_proj.weight.shape[1]
        self.dtype = g_proj.weight.dtype

        self.missing_col_ratio = float(missing_col_ratio)
        self.i_missed = int(self.intermediate_size * self.missing_col_ratio)
        self.i_hot = self.intermediate_size - self.i_missed
        self.missing_col_stats: List[float] = []

        # Empirical DMA microbenchmark on Rudra rdgpu01 proved:
        # Pageable transfer is 111.69 ms vs Pinned 110.67 ms (only 0.9% delta).
        # Pinning 67-90 GB requires hundreds of mlock() syscalls taking 200+ seconds.
        # Pageable allocation is instantaneous (0.01s), saving 3+ minutes per job.
        pin = False

        if self.missing_col_ratio < 1.0:
            # ADETR Memory Layout:
            # Hot columns reside permanently on GPU; cold columns in pinned host memory
            self.gpu_gate_hot = torch.empty((self.num_experts, self.i_hot, self.hidden_size), dtype=self.dtype, device=device)
            self.gpu_up_hot   = torch.empty((self.num_experts, self.i_hot, self.hidden_size), dtype=self.dtype, device=device)
            self.gpu_down_hot = torch.empty((self.num_experts, self.hidden_size, self.i_hot), dtype=self.dtype, device=device)

            self.cpu_gate_cold = torch.empty((self.num_experts, self.i_missed, self.hidden_size), dtype=self.dtype, pin_memory=pin)
            self.cpu_up_cold   = torch.empty((self.num_experts, self.i_missed, self.hidden_size), dtype=self.dtype, pin_memory=pin)
            # ADETR down_proj: stored as [I_cold, H] in pinned memory for contiguous DMA
            self.cpu_down_cold = torch.empty((self.num_experts, self.i_missed, self.hidden_size), dtype=self.dtype, pin_memory=pin)
            # Per-slot receive buffer to eliminate cross-slot and prefetch/demand stream race hazards
            # Shared across all layers to save ~3.76 GB GPU HBM (layers execute sequentially in decode)
            if (
                DynamicMoELayerWrapper._shared_down_cold_rx is None
                or DynamicMoELayerWrapper._shared_down_cold_rx.shape != (capacity, self.i_missed, self.hidden_size)
                or DynamicMoELayerWrapper._shared_down_cold_rx.device != device
            ):
                DynamicMoELayerWrapper._shared_down_cold_rx = torch.empty(
                    (capacity, self.i_missed, self.hidden_size), dtype=self.dtype, device=device
                )
            self.slot_down_cold_rx = DynamicMoELayerWrapper._shared_down_cold_rx

            for i, e in enumerate(self.block.experts):
                eg = self._exp_proj(e, "gate")
                eu = self._exp_proj(e, "up")
                ed = self._exp_proj(e, "down")
                self.gpu_gate_hot[i].copy_(eg.weight.data[:self.i_hot, :], non_blocking=True)
                self.gpu_up_hot[i].copy_(eu.weight.data[:self.i_hot, :], non_blocking=True)
                self.gpu_down_hot[i].copy_(ed.weight.data[:, :self.i_hot], non_blocking=True)

                self.cpu_gate_cold[i].copy_(eg.weight.data[self.i_hot:, :], non_blocking=True)
                self.cpu_up_cold[i].copy_(eu.weight.data[self.i_hot:, :], non_blocking=True)
                self.cpu_down_cold[i].copy_(ed.weight.data[:, self.i_hot:].t().contiguous(), non_blocking=True)
        else:
            # 1. Master CPU expert weights: reference existing CPU tensors directly (instantaneous 0.01s, 0 extra RAM)
            self.cpu_gate_buffer = [self._exp_proj(e, "gate").weight.data.detach().cpu() for e in self.block.experts]
            self.cpu_up_buffer = [self._exp_proj(e, "up").weight.data.detach().cpu() for e in self.block.experts]
            self.cpu_down_buffer = [self._exp_proj(e, "down").weight.data.detach().cpu() for e in self.block.experts]

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # 2. Pre-allocate exactly C GPU slots (constant VRAM buffer)
        def _make_slot_expert():
            try:
                with torch.device("meta"):
                    try:
                        mod = sample_exp.__class__(self.config, intermediate_size=self.intermediate_size)
                    except TypeError:
                        mod = sample_exp.__class__(self.config)
                mod = mod.to_empty(device=device)
            except Exception:
                try:
                    mod = sample_exp.__class__(self.config, intermediate_size=self.intermediate_size)
                except TypeError:
                    mod = sample_exp.__class__(self.config)
                mod = mod.to(device=device)
            return mod.to(dtype=self.dtype).requires_grad_(False)

        self.slots = [_make_slot_expert() for _ in range(capacity)]
        if DynamicMoELayerWrapper._shared_dummy_expert is None or getattr(DynamicMoELayerWrapper._shared_dummy_expert, "device", None) != device:
            DynamicMoELayerWrapper._shared_dummy_expert = _make_slot_expert()
            self._exp_proj(DynamicMoELayerWrapper._shared_dummy_expert, "gate").weight.data.zero_()
            self._exp_proj(DynamicMoELayerWrapper._shared_dummy_expert, "up").weight.data.zero_()
            self._exp_proj(DynamicMoELayerWrapper._shared_dummy_expert, "down").weight.data.zero_()
        self.dummy_expert = DynamicMoELayerWrapper._shared_dummy_expert

        if hasattr(self.block, "gate"):
            self.block.gate.to(device)

        for e in self.block.experts:
            e.requires_grad_(False)
            self._exp_proj(e, "gate").weight.data = self._exp_proj(self.dummy_expert, "gate").weight.data
            self._exp_proj(e, "up").weight.data = self._exp_proj(self.dummy_expert, "up").weight.data
            self._exp_proj(e, "down").weight.data = self._exp_proj(self.dummy_expert, "down").weight.data

        # Ensure wrapper is in eval mode
        self.eval()

        # 4. Slot & LRU tracking
        self.expert_to_slot: Dict[int, int] = {}
        self.slot_to_expert: Dict[int, int] = {}
        self.free_slots: List[int] = list(range(capacity))
        self.slot_lru: List[int] = []  # most recently used at end

        # 5. Dedicated prefetch CUDA stream
        self.prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        # ─── 7. Heterogeneous Compute-to-Data Engine (Fiddler + MoE-Gen) ───
        self.hetero_enabled = bool(getattr(config, "hetero_enabled", True)) if config is not None else True
        self.cpu_token_threshold = int(getattr(config, "cpu_token_threshold", 4)) if config is not None else 4
        self.max_cpu_batch = 64
        if device.type == "cuda":
            self.act_dma_stream = torch.cuda.Stream(device=device)
            self.cpu_act_in = torch.empty((self.max_cpu_batch, self.hidden_size), dtype=self.dtype, pin_memory=True)
            self.cpu_act_out = torch.empty((self.max_cpu_batch, self.hidden_size), dtype=self.dtype, pin_memory=True)
        else:
            self.act_dma_stream = None
            self.cpu_act_in = None
            self.cpu_act_out = None
        self.cpu_dispatches = 0

        # 6. Performance & stall metrics
        self.hits = 0
        self.misses = 0
        self.prefetch_bytes = 0
        self.demand_bytes = 0
        self.prediction_matches = 0
        self.total_routing_decisions = 0
        self.total_tokens_processed = 0

        self.demand_stall_s = 0.0
        self.prefetch_stall_s = 0.0
        self.pcie_time_s = 0.0
        self.pending_prefetch_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.pending_demand_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.last_predicted_topk: Optional[List[int]] = None

        # Warmup cache with initial experts if requested
        if self.warmup_slots > 0:
            self.warmup(list(range(min(self.warmup_slots, self.capacity, self.num_experts))))

    def warmup(self, initial_ids: List[int]):
        """Warm up slots with initial experts."""
        for e_id in initial_ids[:self.capacity]:
            self._load_to_slot(e_id, non_blocking=False)
        self.prefetch_bytes = 0
        self.demand_bytes = 0
        self.demand_stall_s = 0.0
        self.prefetch_stall_s = 0.0
        self.pcie_time_s = 0.0
        self.pending_prefetch_events.clear()
        self.pending_demand_events.clear()

    def warmup_from_prefill(self, hidden_states: torch.Tensor) -> List[int]:
        """Opt 6: Analyze prefill routing decisions to seed cache with most-activated experts.

        Runs the router on all prefill token hidden states, counts expert activations,
        and pre-loads the top-C most frequently routed experts into GPU slots.
        This eliminates cold-start misses for the first few autoregressive tokens.
        """
        if self.router_weight is None:
            return []
        with torch.no_grad():
            h2d = hidden_states.view(-1, hidden_states.shape[-1]).to(
                dtype=self.router_weight.dtype, device=self.router_weight.device
            )
            logits = torch.matmul(h2d, self.router_weight.t())  # [seq_len, num_experts]
            top_k_indices = torch.topk(logits, k=self.top_k, dim=-1).indices  # [seq_len, top_k]
            counts = torch.bincount(top_k_indices.view(-1), minlength=self.num_experts)
            top_c = counts.topk(min(self.capacity, self.num_experts)).indices.tolist()
        logger.info(f"[Colossus] Layer {self.layer_idx}: prefill warmup → top-{len(top_c)} experts {top_c}")
        self.warmup(top_c)
        return top_c

    def reset_state(self):
        """Reset slot bindings and speculative state without clearing cumulative benchmark metrics."""
        self.last_predicted_topk = None
        self.expert_to_slot.clear()
        self.slot_to_expert.clear()
        self.free_slots = list(range(self.capacity))
        self.slot_lru.clear()
        for e in self.block.experts:
            self._exp_proj(e, "gate").weight.data = self._exp_proj(self.dummy_expert, "gate").weight.data
            self._exp_proj(e, "up").weight.data = self._exp_proj(self.dummy_expert, "up").weight.data
            self._exp_proj(e, "down").weight.data = self._exp_proj(self.dummy_expert, "down").weight.data
        if getattr(self, "warmup_slots", 0) > 0:
            for e_id in range(min(self.warmup_slots, self.capacity, self.num_experts)):
                self._load_to_slot(e_id, non_blocking=False)
        self.pending_prefetch_events.clear()
        self.pending_demand_events.clear()

    def _drain_demand_events(self, sync_first: bool = False):
        """Drain completed demand CUDA timing events without CPU stall."""
        if not self.pending_demand_events:
            return
        if self.device.type != "cuda":
            self.pending_demand_events.clear()
            return
        if sync_first:
            torch.cuda.synchronize(self.device)
        remaining = []
        for ev_start, ev_end in self.pending_demand_events:
            if sync_first or ev_end.query():
                dma_s = ev_start.elapsed_time(ev_end) / 1000.0
                self.demand_stall_s += dma_s
                self.pcie_time_s += dma_s
            else:
                remaining.append((ev_start, ev_end))
        self.pending_demand_events = remaining

    def _load_to_slot(self, expert_id: int, non_blocking: bool = True, stream=None,
                      locked_slots: Optional[Set[int]] = None) -> int:
        """Stream expert weights from pinned CPU into assigned GPU slot via DMA."""
        if expert_id in self.expert_to_slot:
            slot_idx = self.expert_to_slot[expert_id]
            self.slot_lru.remove(slot_idx)
            self.slot_lru.append(slot_idx)
            return slot_idx

        # Acquire a slot: free slot if available, else LRU slot not in locked_slots
        if self.free_slots:
            slot_idx = self.free_slots.pop(0)
        else:
            candidates = [s for s in self.slot_lru if locked_slots is None or s not in locked_slots]
            if not candidates:
                candidates = self.slot_lru  # fallback
            slot_idx = candidates[0]
            self.slot_lru.remove(slot_idx)
            old_expert = self.slot_to_expert.pop(slot_idx)
            del self.expert_to_slot[old_expert]
            # Unbind old expert weights back to dummy
            old_mod = self.block.experts[old_expert]
            self._exp_proj(old_mod, "gate").weight.data = self._exp_proj(self.dummy_expert, "gate").weight.data
            self._exp_proj(old_mod, "up").weight.data = self._exp_proj(self.dummy_expert, "up").weight.data
            self._exp_proj(old_mod, "down").weight.data = self._exp_proj(self.dummy_expert, "down").weight.data

        self.expert_to_slot[expert_id] = slot_idx
        self.slot_to_expert[slot_idx] = expert_id
        self.slot_lru.append(slot_idx)

        # DMA transfer into target slot
        slot = self.slots[slot_idx]
        slot_g = self._exp_proj(slot, "gate")
        slot_u = self._exp_proj(slot, "up")
        slot_d = self._exp_proj(slot, "down")

        if self.device.type == "cuda":
            stream_ctx = stream if stream else torch.cuda.current_stream(self.device)
            stream_scope = torch.cuda.stream(stream_ctx)
        else:
            stream_scope = contextlib.nullcontext()

        ev_start = None
        ev_end = None
        if self.device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record(stream_ctx)

        dev_scope = torch.cuda.device(self.device) if self.device.type == "cuda" else contextlib.nullcontext()
        with dev_scope, stream_scope:
            if self.missing_col_ratio < 1.0:
                # 1. Hot columns: instantaneous intra-GPU copy from resident table
                slot_g.weight.data[:self.i_hot].copy_(self.gpu_gate_hot[expert_id], non_blocking=True)
                slot_u.weight.data[:self.i_hot].copy_(self.gpu_up_hot[expert_id], non_blocking=True)
                slot_d.weight.data[:, :self.i_hot].copy_(self.gpu_down_hot[expert_id], non_blocking=True)

                # 2. Cold columns: contiguous PCIe DMA transfer
                slot_g.weight.data[self.i_hot:].copy_(self.cpu_gate_cold[expert_id], non_blocking=non_blocking)
                slot_u.weight.data[self.i_hot:].copy_(self.cpu_up_cold[expert_id], non_blocking=non_blocking)
                # ADETR: DMA into contiguous per-slot buffer, then transpose in GPU HBM (1.5 TB/s)
                rx_buf = self.slot_down_cold_rx[slot_idx]
                rx_buf.copy_(self.cpu_down_cold[expert_id], non_blocking=non_blocking)
                slot_d.weight.data[:, self.i_hot:].copy_(rx_buf.t(), non_blocking=True)

                b = (
                    self.cpu_gate_cold[expert_id].numel() * self.cpu_gate_cold[expert_id].element_size() * 3
                )
                self.missing_col_stats.append(self.missing_col_ratio * 100.0)
            else:
                if self.device.type == "cuda" and torch.cuda.is_available():
                    st_key = "prefetch" if stream is not None else "demand"
                    st_g, st_u, st_d = self.get_pinned_staging(
                        st_key, self.intermediate_size, self.hidden_size, self.dtype
                    )
                    # 1. Fast CPU-to-CPU memcpy into pinned staging buffer (~5ms)
                    st_g.copy_(self.cpu_gate_buffer[expert_id])
                    st_u.copy_(self.cpu_up_buffer[expert_id])
                    st_d.copy_(self.cpu_down_buffer[expert_id])
                    # 2. Direct hardware DMA from pinned memory to GPU slot at 25.6 GB/s (~14ms)
                    slot_g.weight.data.copy_(st_g, non_blocking=non_blocking)
                    slot_u.weight.data.copy_(st_u, non_blocking=non_blocking)
                    slot_d.weight.data.copy_(st_d, non_blocking=non_blocking)
                else:
                    slot_g.weight.data.copy_(self.cpu_gate_buffer[expert_id], non_blocking=non_blocking)
                    slot_u.weight.data.copy_(self.cpu_up_buffer[expert_id], non_blocking=non_blocking)
                    slot_d.weight.data.copy_(self.cpu_down_buffer[expert_id], non_blocking=non_blocking)

                b = (
                    self.cpu_gate_buffer[expert_id].numel() * self.cpu_gate_buffer[expert_id].element_size() * 3
                )
                self.missing_col_stats.append(100.0)

            if stream is not None:
                self.prefetch_bytes += b
            else:
                self.demand_bytes += b

            if self.device.type == "cuda" and ev_end is not None:
                ev_end.record(stream_ctx)
                if stream is not None:
                    self.pending_prefetch_events.append((ev_start, ev_end))
                else:
                    self.pending_demand_events.append((ev_start, ev_end))

        # Bind active expert to this slot's tensors
        exp_mod = self.block.experts[expert_id]
        self._exp_proj(exp_mod, "gate").weight.data = slot_g.weight.data
        self._exp_proj(exp_mod, "up").weight.data = slot_u.weight.data
        self._exp_proj(exp_mod, "down").weight.data = slot_d.weight.data

        return slot_idx

    def _cpu_expert_exec(self, expert_id: int, toks_gpu: torch.Tensor) -> torch.Tensor:
        """Executes cold expert on CPU in native precision (AVX-512/AMX).
        Streams activations (10 KB) rather than full expert weights (45 MB) over PCIe.
        """
        M = toks_gpu.shape[0]
        if self.cpu_act_in is None or self.act_dma_stream is None:
            toks_cpu = toks_gpu.to("cpu", non_blocking=False)
        else:
            if M > self.max_cpu_batch:
                self.max_cpu_batch = M
                self.cpu_act_in = torch.empty((self.max_cpu_batch, self.hidden_size), dtype=self.dtype, pin_memory=True)
                self.cpu_act_out = torch.empty((self.max_cpu_batch, self.hidden_size), dtype=self.dtype, pin_memory=True)
            with torch.cuda.stream(self.act_dma_stream):
                self.cpu_act_in[:M].copy_(toks_gpu, non_blocking=True)
            self.act_dma_stream.synchronize()
            toks_cpu = self.cpu_act_in[:M]

        if hasattr(self, "cpu_gate_buffer"):
            Wg = self.cpu_gate_buffer[expert_id]
            Wu = self.cpu_up_buffer[expert_id]
            Wd = self.cpu_down_buffer[expert_id]
        else:
            # ADETR layout fallback: reconstruct full column slices
            Wg = torch.cat([self.gpu_gate_hot[expert_id].cpu(), self.cpu_gate_cold[expert_id]], dim=0)
            Wu = torch.cat([self.gpu_up_hot[expert_id].cpu(), self.cpu_up_cold[expert_id]], dim=0)
            Wd = torch.cat([self.gpu_down_hot[expert_id].cpu(), self.cpu_down_cold[expert_id].t().cpu()], dim=1)

        with torch.no_grad():
            h_g = F.linear(toks_cpu, Wg)
            h_u = F.linear(toks_cpu, Wu)
            act = F.silu(h_g) * h_u
            y = F.linear(act, Wd)

        out = torch.empty_like(toks_gpu)
        if self.cpu_act_out is None or self.act_dma_stream is None:
            out.copy_(y, non_blocking=False)
        else:
            self.cpu_act_out[:M].copy_(y)
            with torch.cuda.stream(self.act_dma_stream):
                out.copy_(self.cpu_act_out[:M], non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(self.act_dma_stream)
        return out

    def async_prefetch(self, expert_ids: List[int], locked_slots: Optional[Set[int]] = None) -> None:
        """Stream predicted experts to GPU via dedicated non-blocking CUDA stream."""
        miss_pref = [e_id for e_id in expert_ids if e_id not in self.expert_to_slot]
        if not miss_pref:
            return
        if self.device.type != "cuda":
            for e_id in miss_pref:
                self._load_to_slot(e_id, non_blocking=False, locked_slots=locked_slots)
            return
        with torch.cuda.device(self.device):
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.prefetch_stream):
                ev_start.record(self.prefetch_stream)
                for e_id in miss_pref:
                    self._load_to_slot(e_id, non_blocking=True, stream=self.prefetch_stream, locked_slots=locked_slots)
                ev_end.record(self.prefetch_stream)
            self.pending_prefetch_events.append((ev_start, ev_end))

    def synchronize_prefetch(self) -> None:
        """Synchronize prefetch stream before expert execution, recording stall & transfer time."""
        if not self.pending_prefetch_events:
            return
        if self.device.type != "cuda":
            self.pending_prefetch_events.clear()
            return
        t0 = time.perf_counter()
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).wait_stream(self.prefetch_stream)
            self.prefetch_stream.synchronize()
        wait_s = time.perf_counter() - t0
        self.prefetch_stall_s += wait_s

        for ev_start, ev_end in self.pending_prefetch_events:
            dma_s = ev_start.elapsed_time(ev_end) / 1000.0
            self.pcie_time_s += dma_s
        self.pending_prefetch_events.clear()

    def pre_attention_prefetch(self, hidden_states: torch.Tensor, confidence_threshold: float = 0.0) -> List[int]:
        """Trigger speculative prefetch at layer entrance (before MHA) so DMA overlaps attention computation."""
        predicted_topk: List[int] = []
        if self.router_weight is not None:
            with torch.no_grad():
                h_rep = hidden_states[:, -1, :].to(dtype=self.router_weight.dtype, device=self.router_weight.device)
                projected_logits = torch.matmul(h_rep, self.router_weight.t())
                if confidence_threshold > 0.0:
                    probs = torch.softmax(projected_logits, dim=-1)
                    top_probs, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
                    mask = top_probs >= confidence_threshold
                    filtered_idx = top_idx[mask].view(-1).tolist()
                    predicted_topk = list(dict.fromkeys(filtered_idx))
                else:
                    predicted_topk = torch.topk(projected_logits, k=self.top_k, dim=-1).indices.view(-1).tolist()

                if predicted_topk:
                    self.async_prefetch(predicted_topk)
        self.last_predicted_topk = predicted_topk
        return predicted_topk

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        self.total_tokens_processed += (bsz * seq_len)

        # ── 1. Speculative Prefetch (Pre-Attention or Fallback) ─────────────
        if self.last_predicted_topk is not None:
            predicted_topk = self.last_predicted_topk
            self.last_predicted_topk = None
        elif self.lookahead_enabled and seq_len == 1:
            predicted_topk = self.pre_attention_prefetch(hidden_states)
        else:
            predicted_topk = []

        # ── 2. Native Model Router (UNTOUCHED - Exact Execution) ─────────────
        if self.is_mixtral:
            hidden_states_2d = hidden_states.view(-1, h)
            router_logits = self.block.gate(hidden_states_2d)

            routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

            actual_flat = selected_experts.unique().tolist()
            pred_set = set(predicted_topk)
            actual_set = set(actual_flat)
            self.prediction_matches += len(pred_set.intersection(actual_set))
            self.total_routing_decisions += len(actual_set)

            self.synchronize_prefetch()

            final_hidden_states = torch.zeros(
                (bsz * seq_len, h), dtype=hidden_states.dtype, device=hidden_states.device
            )
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            expert_list = [ei.item() for ei in expert_hit]
            for e_idx in expert_list:
                idx, top_x = torch.where(expert_mask[e_idx])
                current_state = hidden_states_2d[top_x]
                if e_idx in self.expert_to_slot:
                    self.hits += 1
                    self.missing_col_stats.append(0.0)
                    slot_idx = self.expert_to_slot[e_idx]
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)
                    slot_mod = self.slots[slot_idx]
                    current_hidden_states = slot_mod(current_state) * routing_weights[top_x, idx, None]
                elif self.hetero_enabled and current_state.shape[0] <= self.cpu_token_threshold:
                    self.cpu_dispatches += 1
                    cpu_out = self._cpu_expert_exec(e_idx, current_state)
                    current_hidden_states = cpu_out * routing_weights[top_x, idx, None]
                else:
                    self.misses += 1
                    slot_idx = self._load_to_slot(e_idx, non_blocking=False)
                    slot_mod = self.slots[slot_idx]
                    current_hidden_states = slot_mod(current_state) * routing_weights[top_x, idx, None]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

            self._drain_demand_events(sync_first=False)
            final_hidden_states = final_hidden_states.reshape(bsz, seq_len, h)
            return final_hidden_states, router_logits

        topk_idx, topk_weight, router_logits = self.block.gate(hidden_states)

        actual_flat = topk_idx.unique().tolist()
        pred_set = set(predicted_topk)
        actual_set = set(actual_flat)
        self.prediction_matches += len(pred_set.intersection(actual_set))
        self.total_routing_decisions += len(actual_set)

        # ── 3. Expert Execution (Lossless & Robust to Prefill vs Generation) ───
        hidden_states_2d = hidden_states.view(-1, h)

        if len(actual_flat) <= self.capacity:
            # Standard generation step: all required experts fit in GPU slots simultaneously
            self.synchronize_prefetch()

            locked_slots: Set[int] = set()
            for e_id in actual_flat:
                if e_id in self.expert_to_slot:
                    self.hits += 1
                    self.missing_col_stats.append(0.0)
                    slot_idx = self.expert_to_slot[e_id]
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)
                    locked_slots.add(slot_idx)

            miss_ids = [e_id for e_id in actual_flat if e_id not in self.expert_to_slot]
            cpu_ids = set()
            gpu_misses = []
            if self.hetero_enabled:
                flat_topk = topk_idx.view(-1)
                for e_id in miss_ids:
                    tok_count = (flat_topk == e_id).sum().item()
                    if tok_count <= self.cpu_token_threshold:
                        cpu_ids.add(e_id)
                    else:
                        gpu_misses.append(e_id)
            else:
                gpu_misses = miss_ids

            if not cpu_ids:
                if gpu_misses:
                    if self.device.type == "cuda":
                        with torch.cuda.device(self.device):
                            ev_d_start = torch.cuda.Event(enable_timing=True)
                            ev_d_end = torch.cuda.Event(enable_timing=True)
                            cur_stream = torch.cuda.current_stream(self.device)
                            ev_d_start.record(cur_stream)
                            for e_id in gpu_misses:
                                self.misses += 1
                                slot_idx = self._load_to_slot(e_id, non_blocking=True, locked_slots=locked_slots)
                                locked_slots.add(slot_idx)
                            ev_d_end.record(cur_stream)
                            self.pending_demand_events.append((ev_d_start, ev_d_end))
                    else:
                        for e_id in gpu_misses:
                            self.misses += 1
                            slot_idx = self._load_to_slot(e_id, non_blocking=False, locked_slots=locked_slots)
                            locked_slots.add(slot_idx)

                y = self.block.moe_infer(hidden_states_2d, topk_idx, topk_weight).view(bsz, seq_len, h)
                if len(self.pending_demand_events) >= 32:
                    self._drain_demand_events()
            else:
                for e_id in gpu_misses:
                    self.misses += 1
                    slot_idx = self._load_to_slot(e_id, non_blocking=True, locked_slots=locked_slots)
                    locked_slots.add(slot_idx)
                if self.device.type == "cuda" and gpu_misses:
                    torch.cuda.current_stream(self.device).wait_stream(self.prefetch_stream or torch.cuda.current_stream(self.device))

                cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.block.experts)))
                cnts.scatter_(1, topk_idx, 1)
                tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
                idxs = topk_idx.view(-1).argsort()
                sorted_tokens = hidden_states_2d[idxs // topk_idx.shape[1]]
                outputs = []
                start_idx = 0
                for i, num_tokens in enumerate(tokens_per_expert):
                    end_idx = start_idx + num_tokens
                    if num_tokens == 0:
                        continue
                    toks_this = sorted_tokens[start_idx:end_idx]
                    if i in cpu_ids:
                        self.cpu_dispatches += 1
                        out_this = self._cpu_expert_exec(i, toks_this)
                    else:
                        slot_idx = self.expert_to_slot[i]
                        out_this = self.slots[slot_idx](toks_this)
                    outputs.append(out_this.to(hidden_states.device))
                    start_idx = end_idx

                outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
                new_x = torch.empty_like(outs)
                new_x[idxs] = outs
                y = (
                    new_x.view(*topk_idx.shape, -1)
                    .type(topk_weight.dtype)
                    .mul_(topk_weight.unsqueeze(dim=-1))
                    .sum(dim=1)
                    .type(new_x.dtype)
                ).view(bsz, seq_len, h)
        else:
            # Prefill step with many tokens: unique experts exceed slot capacity
            # Execute sequentially per expert so slots can be reused without memory errors
            cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.block.experts)))
            cnts.scatter_(1, topk_idx, 1)
            tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
            idxs = topk_idx.view(-1).argsort()
            sorted_tokens = hidden_states_2d[idxs // topk_idx.shape[1]]
            outputs = []
            start_idx = 0
            for i, num_tokens in enumerate(tokens_per_expert):
                end_idx = start_idx + num_tokens
                if num_tokens == 0:
                    continue
                tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
                if i in self.expert_to_slot:
                    self.hits += 1
                    self.missing_col_stats.append(0.0)
                    slot_idx = self.expert_to_slot[i]
                    self.slot_lru.remove(slot_idx)
                    self.slot_lru.append(slot_idx)
                    expert = self.slots[slot_idx]
                    expert_out = expert(tokens_for_this_expert)
                elif self.hetero_enabled and num_tokens <= self.cpu_token_threshold:
                    self.cpu_dispatches += 1
                    expert_out = self._cpu_expert_exec(i, tokens_for_this_expert)
                else:
                    self.misses += 1
                    slot_idx = self._load_to_slot(i, non_blocking=False)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    expert = self.slots[slot_idx]
                    expert_out = expert(tokens_for_this_expert)
                outputs.append(expert_out.to(hidden_states.device))
                start_idx = end_idx

            outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
            new_x = torch.empty_like(outs)
            new_x[idxs] = outs
            y = (
                new_x.view(*topk_idx.shape, -1)
                .type(topk_weight.dtype)
                .mul_(topk_weight.unsqueeze(dim=-1))
                .sum(dim=1)
                .type(new_x.dtype)
            ).view(bsz, seq_len, h)

        if self.block.config.num_shared_experts is not None:
            y = y + self.block.shared_experts(identity)

        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    def get_stats(self) -> Dict[str, Any]:
        """Return layer cache metrics."""
        self._drain_demand_events(sync_first=True)
        total_accesses = self.hits + self.misses
        hit_rate = (self.hits / total_accesses * 100.0) if total_accesses > 0 else 0.0
        recall = (self.prediction_matches / self.total_routing_decisions * 100.0) if self.total_routing_decisions > 0 else 0.0

        if self.missing_col_stats:
            arr = sorted(self.missing_col_stats)
            n = len(arr)
            mean_m = sum(arr) / n
            p50_m = arr[int(n * 0.50)]
            p90_m = arr[min(n - 1, int(n * 0.90))]
            p95_m = arr[min(n - 1, int(n * 0.95))]
            max_m = arr[-1]
        else:
            mean_m = p50_m = p90_m = p95_m = max_m = 0.0

        dma_b_tok = (self.prefetch_bytes + self.demand_bytes) / max(1, self.total_tokens_processed)

        return {
            "layer_idx": self.layer_idx,
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "cpu_dispatches": self.cpu_dispatches,
            "hit_rate_pct": hit_rate,
            "recall_pct": recall,
            "prefetch_mb": self.prefetch_bytes / (1024 * 1024),
            "demand_mb": self.demand_bytes / (1024 * 1024),
            "demand_stall_s": self.demand_stall_s,
            "prefetch_stall_s": self.prefetch_stall_s,
            "pcie_time_s": self.pcie_time_s,
            "missing_col_mean": mean_m,
            "missing_col_p50": p50_m,
            "missing_col_p90": p90_m,
            "missing_col_p95": p95_m,
            "missing_col_max": max_m,
            "dma_bytes_per_tok": dma_b_tok,
        }
