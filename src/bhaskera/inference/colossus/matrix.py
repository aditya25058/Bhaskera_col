"""Matrix-granular MoE tiering (FIRM-like control for gate 3).

FIRM-MoE (AAAI-26) decomposes experts into independently loadable
linear-layer components (gate/up/down matrices) with LRU + prefetch.
This module reimplements the RESIDENCY half faithfully (per-matrix LRU
pools; no predictor — we compare residency granularity, not predictors,
and state that scope): each role (gate/up/down) gets its own slot pool,
demand-fetched per use, native op order, router untouched.

Honest prior, stated before measurement: an expert's three matrices are
used in perfect correlation (all three, every selected expert, every
time), so per-matrix LRU should behave ~identically to whole-expert LRU
— same traffic, more bookkeeping. If confirmed, it frames the column
result properly: expert == matrix (measured null) < columns (measured
win on HBM). If matrices differ, that itself is worth knowing.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .placement import apply_gate


class MatrixTieredMoEWrapper(nn.Module):
    """One LRU pool per matrix role; demand-fetch missing matrices.

    Constructor mirrors TieredMoEWrapper (spec, expert_keys, capacity per
    ROLE pool in whole matrices; HBM == capacity full-expert equivalents).
    Forward uses native op order (ascending experts, index_add).
    """

    ROLES = ("gate", "up", "down")

    def __init__(self, layer_idx: int, spec: Any,
                 expert_keys: list[dict[str, str]], device: torch.device,
                 capacity: int, handles: Any, dma_stream: Any = None,
                 tupled: bool = False, **opts):
        super().__init__()
        assert len(expert_keys) == spec.n_routed, "keys/experts misaligned"
        if opts.get("cpu_exec"):
            raise ValueError("matrix slots need GPU placement (cpu_exec off)")
        self.layer_idx = layer_idx
        self.spec = spec
        self.expert_keys = expert_keys
        self.device = device
        self.handles = handles
        self.dma_stream = dma_stream
        self.tupled = bool(tupled)
        self.streaming = (capacity == 0)
        if self.streaming:
            capacity = 1
        self.capacity = capacity
        inter, hidden = spec.inter, spec.hidden
        self.pools: dict[str, dict] = {}
        for role in self.ROLES:
            if role == "down":
                mods = [nn.Linear(inter, hidden, bias=False, device=device,
                                  dtype=spec.dtype) for _ in range(capacity)]
            else:
                mods = [nn.Linear(hidden, inter, bias=False, device=device,
                                  dtype=spec.dtype) for _ in range(capacity)]
            for m in mods:
                m.requires_grad_(False)
            self.pools[role] = {"slots": nn.ModuleList(mods),
                                "slot_to_expert": {},
                                "expert_to_slot": {},
                                "slot_lru": list(range(capacity))}
        self.gate = spec.gate
        self.shared_experts = spec.shared
        self.hits = 0
        self.misses = 0
        self.role_hits = {r: 0 for r in self.ROLES}
        self.role_misses = {r: 0 for r in self.ROLES}
        self.dma_bytes = 0
        self.zssr_predictions = 0
        self.zssr_correct = 0
        self.zssr_suppressed = 0
        self.routing_log = None
        self.cpu_ms = 0.0
        self.cpu_n = 0

    # -- loading ------------------------------------------------------
    def _sync_dma(self) -> None:
        if self.dma_stream is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)

    def _ensure(self, role: str, expert_id: int) -> int:
        pool = self.pools[role]
        if expert_id in pool["expert_to_slot"]:
            self.role_hits[role] += 1
            slot_idx = pool["expert_to_slot"][expert_id]
            pool["slot_lru"].remove(slot_idx)
            pool["slot_lru"].append(slot_idx)
            return slot_idx
        self.role_misses[role] += 1
        slot_idx = pool["slot_lru"].pop(0)
        t = self.handles.get_tensor(self.expert_keys[expert_id][role])
        nbytes = int(t.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        with torch.no_grad():
            if use_stream:
                with torch.cuda.stream(self.dma_stream):
                    pool["slots"][slot_idx].weight.copy_(t, non_blocking=True)
            else:
                pool["slots"][slot_idx].weight.copy_(t)
        if slot_idx in pool["slot_to_expert"]:
            pool["expert_to_slot"].pop(pool["slot_to_expert"].pop(slot_idx), None)
        pool["slot_to_expert"][slot_idx] = expert_id
        pool["expert_to_slot"][expert_id] = slot_idx
        self.dma_bytes += nbytes
        pool["slot_lru"].append(slot_idx)
        self._sync_dma()
        return slot_idx

    # -- forward -------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.spec.route(hidden_states)
        K = topk_indices.shape[-1]
        TI = topk_indices.reshape(-1, K).long()
        TW = topk_weights.reshape(-1, K)
        needed = sorted({int(e) for e in TI.unique().tolist()}
                        & set(range(self.spec.n_routed)))
        if self.routing_log is not None:
            self.routing_log.append((self.layer_idx, needed))
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = torch.zeros_like(flat)
        mask = F.one_hot(TI, num_classes=self.spec.n_routed + 1).permute(2, 1, 0)
        act = self.spec.activation
        for e in needed:
            m0 = (self.role_misses["gate"] + self.role_misses["up"]
                  + self.role_misses["down"])
            gi = self._ensure("gate", e)
            ui = self._ensure("up", e)
            di = self._ensure("down", e)
            # Expert-level hit = all three matrices already resident.
            m1 = (self.role_misses["gate"] + self.role_misses["up"]
                  + self.role_misses["down"])
            if m1 == m0:
                self.hits += 1
            else:
                self.misses += 1
            kpos, rows = torch.where(mask[e])
            xe = flat[rows]
            w = TW[rows, kpos].unsqueeze(-1)
            with torch.no_grad():
                g = F.linear(xe, self.pools["gate"]["slots"][gi].weight)
                u = F.linear(xe, self.pools["up"]["slots"][ui].weight)
                ye = F.linear(apply_gate(g, u, act),
                              self.pools["down"]["slots"][di].weight)
                ye = ye * w
            out.index_add_(0, rows, ye.to(out.dtype))
        shared_out = self.shared_experts(identity) if self.shared_experts is not None \
            else torch.zeros_like(identity)
        out = shared_out + out.reshape(orig_shape)
        if self.streaming:
            for pool in self.pools.values():
                pool["expert_to_slot"].clear()
                pool["slot_to_expert"].clear()
                pool["slot_lru"] = list(range(len(pool["slots"])))
        if self.tupled:
            return out, None
        return out
