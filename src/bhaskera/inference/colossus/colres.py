"""Column-granular MoE residency (the column thesis).

Research question: can an MoE expert be treated as independently-resident
computational columns rather than an indivisible parameter object, while
preserving exact inference?

Mechanism: SwiGLU factors over the intermediate dim, so for any hot/cold
column split H+C of [0, I):
    y = down_h(act(gate_h(x)) * up_h(x)) + down_c(act(gate_c(x)) * up_c(x))
exactly (summation partition — no approximation introduced by
construction; the flip audit verifies the construction).

Residency unit: the COLUMN, not the expert. A slot holds one expert's HOT
columns resident (LRU-managed across experts, as before); COLD columns are
demand-fetched per use into transient staging (no cold cache in this
prototype — honest DMA accounting). Router untouched. Native op order
(ascending experts, index_add accumulation) so f=1.0 reproduces native
bitwise, exactly like the grouped adapter.

The honest trade, stated upfront: at equal slot COUNT, column-tiering moves
MORE bytes/step (cold columns fetch on every use, including hot hits) for
LESS HBM. The fair comparison is at EQUAL HBM: expert C=X vs column
C=X/f experts at fraction f (wider-but-thinner coverage). If the union fits
the wider coverage, total DMA can fall; if not, columns buy HBM headroom
for HBM-bound operating points (bigger B, longer L) that whole-expert
tiering OOMs. Both outcomes are reported — a tradeoff is still a result.

Hot-set policy in this prototype: static contiguous first-fraction
(documented limitation; frequency-based selection is follow-up work —
correctness never depends on WHICH columns are hot).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .placement import apply_gate


class ColumnSlot(nn.Module):
    """HBM slot holding one expert's HOT columns ONLY.

    Resident: gate_hot/up_hot [Hn, H], down_hot [H, Hn]. Cold columns live
    in ONE shared per-layer scratch (wrapper-owned, reused across experts
    and steps) — never per-slot, or staging becomes shadow residency.
    """

    def __init__(self, hidden: int, inter: int, hot_n: int,
                 device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 activation: str = "silu"):
        super().__init__()
        assert 0 < hot_n <= inter
        self.hot_n = int(hot_n)
        self.cold_n = int(inter - hot_n)
        self.gate_hot = nn.Linear(hidden, hot_n, bias=False,
                                  device=device, dtype=dtype)
        self.up_hot = nn.Linear(hidden, hot_n, bias=False,
                                device=device, dtype=dtype)
        self.down_hot = nn.Linear(hot_n, hidden, bias=False,
                                  device=device, dtype=dtype)
        self.activation = activation
        self.requires_grad_(False)

    def resident_bytes(self) -> int:
        n = 0
        for m in (self.gate_hot, self.up_hot, self.down_hot):
            n += m.weight.nbytes
        return n


class ColumnTieredMoEWrapper(nn.Module):
    """Column-granular tiered execution for one module-based MoE layer.

    Constructor mirrors TieredMoEWrapper (drop-in for wrap_moe_layers via
    wrapper_cls hook) plus hot_frac. expert_keys: per-expert-position
    {gate|up|down: weight key} over FULL expert tensors (sliced after
    fetch — safetensors rows are contiguous, slices are views).
    """

    def __init__(
        self,
        layer_idx: int,
        spec: Any,
        expert_keys: list[dict[str, str]],
        device: torch.device,
        capacity: int,
        handles: Any,
        dma_stream: Any = None,
        tupled: bool = False,
        hot_frac: float = 1.0,
        **opts,
    ):
        super().__init__()
        assert len(expert_keys) == spec.n_routed, "keys/experts misaligned"
        assert 0.0 < hot_frac <= 1.0, "hot_frac in (0, 1]"
        self.layer_idx = layer_idx
        self.spec = spec
        self.expert_keys = expert_keys
        self.device = device
        self.handles = handles
        self.dma_stream = dma_stream
        self.tupled = bool(tupled)
        self.hot_frac = float(hot_frac)
        self.hot_n = max(1, int(round(spec.inter * self.hot_frac)))
        if self.hot_n >= spec.inter:
            self.hot_n = spec.inter
        if opts.get("cpu_exec"):
            raise ValueError("column slots need GPU placement (cpu_exec off)")
        self.streaming = (capacity == 0)
        if self.streaming:
            capacity = 1
        self.capacity = capacity
        self.slots: list[nn.Module] = nn.ModuleList([
            ColumnSlot(spec.hidden, spec.inter, self.hot_n, device,
                       spec.dtype, spec.activation)
            for _ in range(capacity)
        ])
        # Shared per-layer cold scratch: ONE cold-half buffer set, reused
        # across experts and steps (plain tensors, not modules).
        self.cold_n = int(spec.inter - self.hot_n)
        if self.cold_n > 0:
            self._scratch_gc = torch.empty(self.cold_n, spec.hidden,
                                           device=device, dtype=spec.dtype)
            self._scratch_uc = torch.empty(self.cold_n, spec.hidden,
                                           device=device, dtype=spec.dtype)
            self._scratch_dc = torch.empty(spec.hidden, self.cold_n,
                                           device=device, dtype=spec.dtype)
        else:
            self._scratch_gc = self._scratch_uc = self._scratch_dc = None
        self.slot_to_expert: dict[int, int] = {}
        self.expert_to_slot: dict[int, int] = {}
        self.slot_lru: list[int] = list(range(capacity))
        self.gate = spec.gate
        self.shared_experts = spec.shared
        self.hits = 0
        self.misses = 0
        self.dma_bytes = 0
        self.hot_dma_bytes = 0
        self.cold_dma_bytes = 0
        self.zssr_predictions = 0
        self.zssr_correct = 0
        self.zssr_suppressed = 0
        self.routing_log = None
        self.cpu_ms = 0.0
        self.cpu_n = 0

    # -- loading ------------------------------------------------------
    def _expert_tensors(self, expert_id: int):
        ks = self.expert_keys[expert_id]
        return (self.handles.get_tensor(ks["gate"]),
                self.handles.get_tensor(ks["up"]),
                self.handles.get_tensor(ks["down"]))

    def _load_hot_to_slot(self, expert_id: int, slot_idx: int) -> int:
        slot = self.slots[slot_idx]
        hn = self.hot_n
        t_gate, t_up, t_down = self._expert_tensors(expert_id)
        g_h, u_h, d_h = t_gate[:hn], t_up[:hn], t_down[:, :hn]
        nbytes = int(g_h.nbytes + u_h.nbytes + d_h.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        with torch.no_grad():
            if use_stream:
                with torch.cuda.stream(self.dma_stream):
                    slot.gate_hot.weight.copy_(g_h, non_blocking=True)
                    slot.up_hot.weight.copy_(u_h, non_blocking=True)
                    slot.down_hot.weight.copy_(d_h, non_blocking=True)
            else:
                slot.gate_hot.weight.copy_(g_h)
                slot.up_hot.weight.copy_(u_h)
                slot.down_hot.weight.copy_(d_h)
        if slot_idx in self.slot_to_expert:
            self.expert_to_slot.pop(self.slot_to_expert.pop(slot_idx), None)
        self.slot_to_expert[slot_idx] = expert_id
        self.expert_to_slot[expert_id] = slot_idx
        self.hot_dma_bytes += nbytes
        self.dma_bytes += nbytes
        return nbytes

    def _fetch_cold(self, expert_id: int) -> int:
        if self.cold_n == 0:
            return 0
        hn = self.hot_n
        t_gate, t_up, t_down = self._expert_tensors(expert_id)
        g_c, u_c, d_c = t_gate[hn:], t_up[hn:], t_down[:, hn:]
        nbytes = int(g_c.nbytes + u_c.nbytes + d_c.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        with torch.no_grad():
            if use_stream:
                with torch.cuda.stream(self.dma_stream):
                    self._scratch_gc.copy_(g_c, non_blocking=True)
                    self._scratch_uc.copy_(u_c, non_blocking=True)
                    self._scratch_dc.copy_(d_c, non_blocking=True)
            else:
                self._scratch_gc.copy_(g_c)
                self._scratch_uc.copy_(u_c)
                self._scratch_dc.copy_(d_c)
        self.cold_dma_bytes += nbytes
        self.dma_bytes += nbytes
        return nbytes

    def _sync_dma(self) -> None:
        if self.dma_stream is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)

    def _ensure_hot(self, expert_id: int) -> int:
        if expert_id in self.expert_to_slot:
            self.hits += 1
            slot_idx = self.expert_to_slot[expert_id]
            self.slot_lru.remove(slot_idx)
            self.slot_lru.append(slot_idx)
            return slot_idx
        self.misses += 1
        slot_idx = self.slot_lru.pop(0)
        self._load_hot_to_slot(expert_id, slot_idx)
        self.slot_lru.append(slot_idx)
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
        # Weights keep incoming dtype (promoted-precision accumulation —
        # same bitwise-parity rule as the grouped adapter).
        needed = sorted({int(e) for e in TI.unique().tolist()}
                        & set(range(self.spec.n_routed)))
        if self.routing_log is not None:
            self.routing_log.append((self.layer_idx, needed))
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = torch.zeros_like(flat)
        # Native op order, literally: ascending experts, torch.where
        # positions, index_add accumulation. Hot partial + cold partial
        # summed per expert (summation partition = exact).
        mask = F.one_hot(TI, num_classes=self.spec.n_routed + 1).permute(2, 1, 0)
        act = self.spec.activation
        for e in needed:
            slot_idx = self._ensure_hot(e)
            slot = self.slots[slot_idx]
            self._fetch_cold(e)
            self._sync_dma()
            kpos, rows = torch.where(mask[e])
            xe = flat[rows]
            w = TW[rows, kpos].unsqueeze(-1)
            with torch.no_grad():
                gh = F.linear(xe, slot.gate_hot.weight)
                uh = F.linear(xe, slot.up_hot.weight)
                yh = F.linear(apply_gate(gh, uh, act), slot.down_hot.weight)
                if self.cold_n > 0:
                    gc = F.linear(xe, self._scratch_gc)
                    uc = F.linear(xe, self._scratch_uc)
                    yc = F.linear(apply_gate(gc, uc, act), self._scratch_dc)
                    ye = yh + yc.to(yh.dtype)
                else:
                    ye = yh
                ye = ye * w
            out.index_add_(0, rows, ye.to(out.dtype))
        shared_out = self.shared_experts(identity) if self.shared_experts is not None \
            else torch.zeros_like(identity)
        out = shared_out + out.reshape(orig_shape)
        if self.streaming:
            self.expert_to_slot.clear()
            self.slot_to_expert.clear()
            self.slot_lru = list(range(len(self.slots)))
        if self.tupled:
            return out, None
        return out
