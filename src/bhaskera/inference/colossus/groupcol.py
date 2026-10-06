"""Column-granular residency for grouped-weight MoE (thesis, representation 2).

Same summation-partition principle as colres.py, adapted to fused grouped
tensors. Native grouped math per expert e (gate and up fused as
gate_up [2I, H], chunked gate-first):

    g, u = chunk(linear(x, GU[e]), 2); y = linear(act(g)*up, D[e])

Partition the intermediate dim into hot Hn + cold Cn columns. Hot fused
halves stay resident per slot ([gate_h | up_h] = GU[e,:Hn] ++ GU[e,I:I+Hn],
down_h = D[e][:,:Hn]); cold halves stream per use into shared per-layer
scratch. y = y_hot + y_cold exactly; router untouched (native upstream);
native op order (ascending, index_add). hot_frac=1.0 reproduces the
whole-slice GroupedTieredMoEWrapper path (single GEMM set).

No architecture names: fused layout (gate-first chunk) and shapes drive
everything, exactly like grouped.py.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .grouped import GroupedSlot, GroupedTieredMoEWrapper


class GroupedColumnSlot(GroupedSlot):
    """Slot holding one grouped expert's HOT fused halves only."""

    def __init__(self, hidden: int, inter: int, hot_n: int,
                 device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 activation: str = "gelu_tanh"):
        assert 0 < hot_n <= inter
        super().__init__(hidden, hot_n, device, dtype, activation)
        # GroupedSlot(hidden, inter=hot_n): gate_up [2*hot_n, H],
        # down [H, hot_n]. inter attr now means hot width.
        self.hot_n = int(hot_n)
        self.cold_n = int(inter - hot_n)


class GroupedColumnWrapper(GroupedTieredMoEWrapper):
    """Grouped experts with column-granular residency.

    Constructor mirrors GroupedTieredMoEWrapper plus hot_frac.
    Cold halves stream per use into shared scratch (no cold cache here).
    Telemetry adds hot_dma_bytes/cold_dma_bytes; dma_bytes stays total
    so existing ledgers just work.
    """

    def __init__(self, *args, hot_frac: float = 1.0, **kwargs):
        # Peek inter from kwargs (fused shape known only at wrap time, so
        # wrap_grouped_layers passes inter explicitly as today).
        super().__init__(*args, **kwargs)
        assert 0.0 < hot_frac <= 1.0, "hot_frac in (0, 1]"
        self.hot_frac = float(hot_frac)
        hn = max(1, int(round(self.inter * self.hot_frac)))
        self.hot_n = self.inter if hn >= self.inter else hn
        self.cold_n = self.inter - self.hot_n
        # Rebuild slots at hot width (parent built full-width ones).
        dev, act = self.device, self.activation
        dt0 = self.slots[0].gate_up_proj.weight.dtype
        self.slots = nn.ModuleList([
            GroupedColumnSlot(self.hidden, self.inter, self.hot_n, dev,
                              dt0, act)
            for _ in range(self.capacity)
        ])
        self.slot_to_expert = {}
        self.expert_to_slot = {}
        self.slot_lru = list(range(self.capacity))
        if self.cold_n > 0:
            self._scratch_fu = torch.empty(2 * self.cold_n, self.hidden,
                                           device=dev, dtype=dt0)
            self._scratch_dc = torch.empty(self.hidden, self.cold_n,
                                           device=dev, dtype=dt0)
        else:
            self._scratch_fu = self._scratch_dc = None
        self.hot_dma_bytes = 0
        self.cold_dma_bytes = 0

    # -- loading ------------------------------------------------------
    def _load_expert_to_slot(self, expert_id: int, slot_idx: int) -> int:
        slot = self.slots[slot_idx]
        hn, I = self.hot_n, self.inter
        t_fused, t_down = self._expert_slice(expert_id)
        gh, uh, dh = t_fused[:hn], t_fused[I:I + hn], t_down[:, :hn]
        nbytes = int(gh.nbytes + uh.nbytes + dh.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        with torch.no_grad():
            if use_stream:
                with torch.cuda.stream(self.dma_stream):
                    slot.gate_up_proj.weight[:hn].copy_(gh, non_blocking=True)
                    slot.gate_up_proj.weight[hn:].copy_(uh, non_blocking=True)
                    slot.down_proj.weight.copy_(dh, non_blocking=True)
            else:
                slot.gate_up_proj.weight[:hn].copy_(gh)
                slot.gate_up_proj.weight[hn:].copy_(uh)
                slot.down_proj.weight.copy_(dh)
        if slot_idx in self.slot_to_expert:
            self.expert_to_slot.pop(self.slot_to_expert.pop(slot_idx), None)
        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.hot_dma_bytes += nbytes
        self.dma_bytes += nbytes
        return nbytes

    def _fetch_cold(self, expert_id: int):
        if self.cold_n == 0:
            return None, None
        hn, I, cn = self.hot_n, self.inter, self.cold_n
        t_fused, t_down = self._expert_slice(expert_id)
        g_c, u_c, d_c = t_fused[hn:I], t_fused[I + hn:], t_down[:, hn:]
        nbytes = int(g_c.nbytes + u_c.nbytes + d_c.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        fu, dc = self._scratch_fu, self._scratch_dc
        with torch.no_grad():
            if use_stream:
                with torch.cuda.stream(self.dma_stream):
                    fu[:cn].copy_(g_c, non_blocking=True)
                    fu[cn:].copy_(u_c, non_blocking=True)
                    dc.copy_(d_c, non_blocking=True)
            else:
                fu[:cn].copy_(g_c)
                fu[cn:].copy_(u_c)
                dc.copy_(d_c)
        self.cold_dma_bytes += nbytes
        self.dma_bytes += nbytes
        return fu, dc

    # -- forward -------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor,
                topk_indices: torch.Tensor,
                topk_weights: torch.Tensor) -> torch.Tensor:
        from .placement import apply_gate
        lead = hidden_states.shape[:-1]
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        K = topk_indices.shape[-1]
        TI = topk_indices.reshape(-1, K).long()
        TW = topk_weights.reshape(-1, K)
        needed = sorted({int(e) for e in TI.unique().tolist()} & set(range(self.n_experts)))
        if self.routing_log is not None:
            self.routing_log.append((self.layer_idx, needed))
        out = torch.zeros_like(flat)
        mask = F.one_hot(TI, num_classes=self.n_experts + 1).permute(2, 1, 0)
        for e in needed:
            slot_idx = self._ensure(e)
            slot = self.slots[slot_idx]
            fu_c, d_c = self._fetch_cold(e)
            self._sync_dma()
            kpos, rows = torch.where(mask[e])
            xe = flat[rows]
            w = TW[rows, kpos].unsqueeze(-1)
            with torch.no_grad():
                g, u = F.linear(xe, slot.gate_up_proj.weight).chunk(2, dim=-1)
                yh = F.linear(apply_gate(g, u, self.activation),
                              slot.down_proj.weight)
                if fu_c is not None:
                    gc, uc = F.linear(xe, fu_c).chunk(2, dim=-1)
                    yc = F.linear(apply_gate(gc, uc, self.activation), d_c)
                    yh = yh + yc.to(yh.dtype)
                ye = yh * w
            out.index_add_(0, rows, ye.to(out.dtype))
        if self.streaming:
            self.expert_to_slot.clear()
            self.slot_to_expert.clear()
            self.slot_lru = list(range(len(self.slots)))
        return out.reshape(*lead, flat.shape[-1])
