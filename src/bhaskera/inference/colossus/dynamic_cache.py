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

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DynamicMoELayerWrapper(nn.Module):
    """Wraps SparseMoeBlock with pre-allocated GPU slot cache and async DMA streaming."""

    def __init__(
        self,
        layer_idx: int,
        moe_block: nn.Module,
        capacity: int = 16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block = moe_block
        self.capacity = capacity
        self.device = device
        self.num_experts = len(moe_block.experts)
        self.top_k = getattr(moe_block, "num_experts_per_tok", 6)

        # Router weight for ZSSR speculative projection
        gate = getattr(moe_block, "gate", None)
        if hasattr(gate, "weight"):
            self.router_weight = gate.weight.detach().to(device).float()
        elif hasattr(gate, "gate") and hasattr(gate.gate, "weight"):
            self.router_weight = gate.gate.weight.detach().to(device).float()
        else:
            self.router_weight = None

        sample_exp = self.block.experts[0]
        self.config = sample_exp.config
        self.intermediate_size = sample_exp.intermediate_size
        self.dtype = sample_exp.gate_proj.weight.dtype

        # 1. Master CPU expert copies in pinned memory
        self.cpu_experts = []
        for e in self.block.experts:
            e.to("cpu")
            for p in e.parameters():
                p.requires_grad_(False)
                if not p.data.is_pinned():
                    p.data = p.data.pin_memory()
            self.cpu_experts.append(e)

        # 2. Pre-allocate exactly C GPU slots (constant VRAM buffer)
        self.slots = [
            sample_exp.__class__(self.config, intermediate_size=self.intermediate_size).to(
                device=device, dtype=self.dtype
            ).requires_grad_(False)
            for _ in range(capacity)
        ]

        # 3. Create a shared 1-element dummy tensor on GPU for inactive experts
        self.dummy = torch.zeros(1, dtype=self.dtype, device=device)
        self.block.to(device)
        for e in self.block.experts:
            e.requires_grad_(False)
            e.gate_proj.weight.data = self.dummy
            e.up_proj.weight.data = self.dummy
            e.down_proj.weight.data = self.dummy

        # 4. Slot & LRU tracking
        self.expert_to_slot: Dict[int, int] = {}
        self.slot_to_expert: Dict[int, int] = {}
        self.free_slots: List[int] = list(range(capacity))
        self.slot_lru: List[int] = []  # most recently used at end

        # 5. Dedicated prefetch CUDA stream
        self.prefetch_stream = torch.cuda.Stream(device=device)

        # 6. Performance metrics
        self.hits = 0
        self.misses = 0
        self.prefetch_bytes = 0
        self.demand_bytes = 0
        self.prediction_matches = 0
        self.total_routing_decisions = 0
        self.total_tokens_processed = 0

        # Warmup cache with first C experts
        self.warmup(list(range(min(self.capacity, self.num_experts))))

    def warmup(self, initial_ids: List[int]):
        """Warm up slots with initial experts."""
        for e_id in initial_ids[:self.capacity]:
            self._load_to_slot(e_id, non_blocking=False)
        self.prefetch_bytes = 0
        self.demand_bytes = 0

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
            # Unbind old expert weights
            old_mod = self.block.experts[old_expert]
            old_mod.gate_proj.weight.data = self.dummy
            old_mod.up_proj.weight.data = self.dummy
            old_mod.down_proj.weight.data = self.dummy

        self.expert_to_slot[expert_id] = slot_idx
        self.slot_to_expert[slot_idx] = expert_id
        self.slot_lru.append(slot_idx)

        # DMA transfer into target slot
        slot = self.slots[slot_idx]
        src = self.cpu_experts[expert_id]

        stream_ctx = torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream())
        with stream_ctx:
            slot.gate_proj.weight.data.copy_(src.gate_proj.weight.data, non_blocking=non_blocking)
            slot.up_proj.weight.data.copy_(src.up_proj.weight.data, non_blocking=non_blocking)
            slot.down_proj.weight.data.copy_(src.down_proj.weight.data, non_blocking=non_blocking)
            b = (
                src.gate_proj.weight.numel() * src.gate_proj.weight.element_size() +
                src.up_proj.weight.numel() * src.up_proj.weight.element_size() +
                src.down_proj.weight.numel() * src.down_proj.weight.element_size()
            )
            if stream is not None:
                self.prefetch_bytes += b
            else:
                self.demand_bytes += b

        # Bind active expert to this slot's tensors
        exp_mod = self.block.experts[expert_id]
        exp_mod.gate_proj.weight.data = slot.gate_proj.weight.data
        exp_mod.up_proj.weight.data = slot.up_proj.weight.data
        exp_mod.down_proj.weight.data = slot.down_proj.weight.data

        return slot_idx

    def async_prefetch(self, expert_ids: List[int], locked_slots: Optional[Set[int]] = None) -> None:
        """Stream predicted experts to GPU via dedicated non-blocking CUDA stream."""
        for e_id in expert_ids:
            if e_id not in self.expert_to_slot:
                self._load_to_slot(e_id, non_blocking=True, stream=self.prefetch_stream, locked_slots=locked_slots)

    def synchronize_prefetch(self) -> None:
        """Synchronize prefetch stream before expert execution."""
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        self.total_tokens_processed += (bsz * seq_len)

        # ── 1. ZSSR Speculative Prediction ──────────────────────────────────
        predicted_topk = []
        if self.router_weight is not None:
            with torch.no_grad():
                h_flat = hidden_states.reshape(-1, self.router_weight.shape[1])[-1].float()
                spec_logits = h_flat @ self.router_weight.T
                predicted_topk = torch.topk(spec_logits, k=self.top_k).indices.tolist()
                self.async_prefetch(predicted_topk)

        # ── 2. Native Model Router (UNTOUCHED - Exact Execution) ─────────────
        topk_idx, topk_weight, router_logits = self.block.gate(hidden_states)

        actual_flat = topk_idx.unique().tolist()
        pred_set = set(predicted_topk)
        actual_set = set(actual_flat)
        self.prediction_matches += len(pred_set.intersection(actual_set))
        self.total_routing_decisions += len(actual_set)

        # ── 3. Ensure All Actual Experts Are Resident in GPU Slots ───────────
        locked_slots: Set[int] = {self.expert_to_slot[e] for e in actual_flat if e in self.expert_to_slot}
        for e_id in actual_flat:
            if e_id in self.expert_to_slot:
                self.hits += 1
                slot_idx = self.expert_to_slot[e_id]
                self.slot_lru.remove(slot_idx)
                self.slot_lru.append(slot_idx)
            else:
                self.misses += 1
                slot_idx = self._load_to_slot(e_id, non_blocking=False, locked_slots=locked_slots)
                locked_slots.add(slot_idx)

        # Synchronize prefetch stream
        self.synchronize_prefetch()

        # ── 4. Native moe_infer (100% Mathematically Identical) ───────────────
        hidden_states_2d = hidden_states.view(-1, h)
        if self.training:
            flat_topk_idx = topk_idx.view(-1)
            hidden_states_rep = hidden_states_2d.repeat_interleave(self.top_k, dim=0)
            y = torch.empty_like(hidden_states_rep)
            for i, expert in enumerate(self.block.experts):
                mask = (flat_topk_idx == i)
                if mask.any():
                    y[mask] = expert(hidden_states_rep[mask])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.block.moe_infer(hidden_states_2d, topk_idx, topk_weight).view(bsz, seq_len, h)

        if self.block.config.num_shared_experts is not None:
            y = y + self.block.shared_experts(identity)

        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    def get_stats(self) -> Dict[str, Any]:
        """Return layer cache metrics."""
        total_accesses = self.hits + self.misses
        hit_rate = (self.hits / total_accesses * 100.0) if total_accesses > 0 else 0.0
        recall = (self.prediction_matches / self.total_routing_decisions * 100.0) if self.total_routing_decisions > 0 else 0.0
        return {
            "layer_idx": self.layer_idx,
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": hit_rate,
            "recall_pct": recall,
            "prefetch_mb": self.prefetch_bytes / (1024 * 1024),
            "demand_mb": self.demand_bytes / (1024 * 1024),
        }
