"""Dynamic MoE Expert Cache for COLOSSUS.

Implements dynamic GPU residency with ZSSR speculative prefetch and exact router execution.
Contract:
  - Router is NEVER modified (lossless guarantee).
  - Selected experts compute native forward pass.
  - Active GPU working set bounded to capacity C << E_total.
  - ZSSR initiates non-blocking CUDA stream prefetch ahead of gate/inference.
  - Misses fall back to demand-paging into LRU slots.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DynamicMoELayerWrapper(nn.Module):
    """Wraps a SparseMoeBlock with dynamic GPU expert residency and ZSSR prefetch."""

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

        # Pin all expert weights to CPU host memory
        self.pinned_cpu_experts: List[nn.Module] = []
        for e in self.block.experts:
            e.to("cpu")
            for p in e.parameters():
                p.requires_grad_(False)
                if not p.data.is_pinned():
                    p.data = p.data.pin_memory()
            self.pinned_cpu_experts.append(e)

        # Gate and shared experts stay permanently on GPU
        if hasattr(self.block, "gate"):
            self.block.gate.to(device)
        if hasattr(self.block, "shared_experts") and self.block.shared_experts is not None:
            self.block.shared_experts.to(device)

        # Residency tracking
        self.resident_set: Set[int] = set()
        self.lru_order: List[int] = []  # most recent at end

        # Prefetch stream
        self.prefetch_stream = torch.cuda.Stream(device=device)

        # Performance & accuracy statistics
        self.hits = 0
        self.misses = 0
        self.prefetch_bytes = 0
        self.demand_bytes = 0
        self.prediction_matches = 0
        self.total_routing_decisions = 0
        self.total_tokens_processed = 0

        # Warm up initial cache up to capacity
        self.warmup(list(range(min(self.capacity, self.num_experts))))

    def warmup(self, initial_ids: List[int]):
        """Warm up GPU cache with initial expert weights."""
        for e_id in initial_ids[:self.capacity]:
            self._bring_to_gpu(e_id, non_blocking=False)
        self.prefetch_bytes = 0
        self.demand_bytes = 0

    def _bring_to_gpu(self, expert_id: int, non_blocking: bool = True, stream=None,
                      locked_set: Optional[Set[int]] = None) -> None:
        """Transfer an expert module to GPU and update LRU tracking."""
        if expert_id in self.resident_set:
            self.lru_order.remove(expert_id)
            self.lru_order.append(expert_id)
            return

        # If cache is at or above capacity, evict LRU expert that is NOT in locked_set
        if len(self.resident_set) >= self.capacity:
            evictable = [e for e in self.lru_order if locked_set is None or e not in locked_set]
            if evictable:
                evict_id = evictable[0]
                self.lru_order.remove(evict_id)
                evict_exp = self.block.experts[evict_id]
                evict_exp.to("cpu")
                for p in evict_exp.parameters():
                    if not p.data.is_pinned():
                        p.data = p.data.pin_memory()
                self.resident_set.remove(evict_id)

        # Transfer expert to GPU
        expert = self.block.experts[expert_id]
        stream_ctx = torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream())
        with stream_ctx:
            expert.to(self.device, non_blocking=non_blocking)
            b = sum(p.numel() * p.element_size() for p in expert.parameters())
            if stream is not None:
                self.prefetch_bytes += b
            else:
                self.demand_bytes += b

        self.resident_set.add(expert_id)
        self.lru_order.append(expert_id)

    def async_prefetch(self, expert_ids: List[int], locked_set: Optional[Set[int]] = None) -> None:
        """Stream predicted experts to GPU via dedicated non-blocking CUDA stream."""
        for e_id in expert_ids:
            if e_id not in self.resident_set:
                self._bring_to_gpu(e_id, non_blocking=True, stream=self.prefetch_stream, locked_set=locked_set)

    def synchronize_prefetch(self) -> None:
        """Synchronize prefetch stream before execution."""
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

        # Track prediction recall vs actual routing
        actual_flat = topk_idx.unique().tolist()
        pred_set = set(predicted_topk)
        actual_set = set(actual_flat)
        self.prediction_matches += len(pred_set.intersection(actual_set))
        self.total_routing_decisions += len(actual_set)

        # ── 3. Ensure All Actual Experts Are Resident on GPU ─────────────────
        for e_id in actual_flat:
            if e_id in self.resident_set:
                self.hits += 1
                self.lru_order.remove(e_id)
                self.lru_order.append(e_id)
            else:
                self.misses += 1
                # Demand fetch fallback: lock the active set from being evicted
                self._bring_to_gpu(e_id, non_blocking=False, locked_set=actual_set)

        # Wait for prefetch stream to ensure prefetched experts are ready
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

        # Prune excess experts back to capacity if prefill temporarily expanded cache
        while len(self.resident_set) > self.capacity:
            evictable = [e for e in self.lru_order if e not in actual_set]
            if not evictable:
                break
            evict_id = evictable[0]
            self.lru_order.remove(evict_id)
            evict_exp = self.block.experts[evict_id]
            evict_exp.to("cpu")
            for p in evict_exp.parameters():
                if not p.data.is_pinned():
                    p.data = p.data.pin_memory()
            self.resident_set.remove(evict_id)

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
