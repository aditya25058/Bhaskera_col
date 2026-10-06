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

import time
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
        hot_fracs: list[float] | None = None,
        tier_pools: dict[float, int] | None = None,
        cold_cache_cap: int = 0,
        grouped_gemm: bool = False,
        **opts,
    ):
        super().__init__()
        assert len(expert_keys) == spec.n_routed, "keys/experts misaligned"
        if hot_fracs is None:
            hot_fracs = [float(hot_frac)] * spec.n_routed
        assert len(hot_fracs) == spec.n_routed, "fracs/experts misaligned"
        assert all(0.0 < f <= 1.0 for f in hot_fracs), "fracs in (0, 1]"
        self.layer_idx = layer_idx
        self.spec = spec
        self.expert_keys = expert_keys
        self.device = device
        self.handles = handles
        self.dma_stream = dma_stream
        self.tupled = bool(tupled)
        self.hot_fracs = [float(f) for f in hot_fracs]
        # Batched dispatch: one bmm trio per tier per layer instead of
        # per-expert-per-half launches. Same math, fewer launches; bmm
        # kernel selection may differ by ~1 ulp from per-expert GEMMs —
        # audit-gated, same discipline as split-summation.
        self.grouped_gemm = bool(grouped_gemm)
        if opts.get("cpu_exec"):
            raise ValueError("column slots need GPU placement (cpu_exec off)")
        self.streaming = (capacity == 0)
        if self.streaming:
            capacity = 1
        self.capacity = capacity
        # Tier pools: one LRU pool per distinct fraction, so hot experts
        # never share eviction pressure with thin ones. tier_pools pins
        # exact slot counts per tier (pool >= tier members = pinned);
        # otherwise capacity splits evenly, remainder to thickest tier.
        self.tiers = sorted(set(self.hot_fracs), reverse=True)
        if tier_pools is not None:
            pool_sizes = {t: int(tier_pools.get(t, tier_pools.get(str(t), 0)))
                          for t in self.tiers}
            missing = [t for t, n in pool_sizes.items() if n <= 0]
            if missing:
                raise ValueError(f"tier_pools lacks slots for tiers {missing}")
        else:
            per, rem = divmod(capacity, len(self.tiers))
            pool_sizes = {t: max(1, per + (1 if i < rem else 0))
                          for i, t in enumerate(self.tiers)}
        self.pools: dict[float, dict] = {}
        for t in self.tiers:
            nslots = pool_sizes[t]
            hot_n = self._hot_n(spec.inter, t)
            self.pools[t] = {
                "slots": nn.ModuleList([
                    ColumnSlot(spec.hidden, spec.inter, hot_n, device,
                               spec.dtype, spec.activation)
                    for _ in range(nslots)]),
                "slot_to_expert": {},
                "expert_to_slot": {},
                "slot_lru": list(range(nslots)),
            }
        self.tier_of = list(self.hot_fracs)
        # Cold-column second tier: LRU over cold halves (HBM-resident).
        # A hit skips host DMA AND the staging copy (compute reads the
        # cached tensors directly — same bytes, bitwise identical).
        # Cap 0 = off (transient scratch only, current behavior).
        self.cold_cache_cap = int(cold_cache_cap)
        self._cold_cache: dict[int, tuple] = {}
        self._cold_lru: list[int] = []
        self.cold_hits = 0
        self.cold_misses = 0
        # Shared per-layer cold scratch at max width; narrowed per use.
        max_cold = max(spec.inter - self._hot_n(spec.inter, t)
                       for t in self.tiers)
        if max_cold > 0:
            self._scratch_gc = torch.empty(max_cold, spec.hidden,
                                           device=device, dtype=spec.dtype)
            self._scratch_uc = torch.empty(max_cold, spec.hidden,
                                           device=device, dtype=spec.dtype)
            self._scratch_dc = torch.empty(spec.hidden, max_cold,
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
        self.stall_ms = 0.0

    # -- loading ------------------------------------------------------
    @staticmethod
    def _hot_n(inter: int, frac: float) -> int:
        hn = max(1, int(round(inter * frac)))
        return inter if hn >= inter else hn

    def _expert_tensors(self, expert_id: int):
        ks = self.expert_keys[expert_id]
        return (self.handles.get_tensor(ks["gate"]),
                self.handles.get_tensor(ks["up"]),
                self.handles.get_tensor(ks["down"]))

    def _pool(self, expert_id: int) -> dict:
        return self.pools[self.tier_of[expert_id]]

    def _load_hot_to_slot(self, expert_id: int, pool: dict,
                          slot_idx: int) -> int:
        slot = pool["slots"][slot_idx]
        hn = slot.hot_n
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
        if slot_idx in pool["slot_to_expert"]:
            pool["expert_to_slot"].pop(pool["slot_to_expert"].pop(slot_idx), None)
        pool["slot_to_expert"][slot_idx] = expert_id
        pool["expert_to_slot"][expert_id] = slot_idx
        self.hot_dma_bytes += nbytes
        self.dma_bytes += nbytes
        return nbytes

    def _fetch_cold(self, expert_id: int):
        """-> (gc, uc, dc) cold weight tensors (scratch views or cache).

        Hit: cached tensors, zero DMA, zero copy. Miss: host -> staging,
        then clone into the cold LRU (HBM->HBM, off the PCIe path).
        """
        hn = self._hot_n(self.spec.inter, self.tier_of[expert_id])
        cn = self.spec.inter - hn
        if cn == 0:
            return None, None, None
        hit = self._cold_cache.get(expert_id)
        if hit is not None:
            self.cold_hits += 1
            if expert_id in self._cold_lru:
                self._cold_lru.remove(expert_id)
            self._cold_lru.append(expert_id)
            return hit
        self.cold_misses += 1
        t_gate, t_up, t_down = self._expert_tensors(expert_id)
        g_c, u_c, d_c = t_gate[hn:], t_up[hn:], t_down[:, hn:]
        nbytes = int(g_c.nbytes + u_c.nbytes + d_c.nbytes)
        use_stream = (self.dma_stream is not None and self.device.type == "cuda")
        with torch.no_grad():
            if use_stream:
                with torch.cuda.stream(self.dma_stream):
                    self._scratch_gc[:cn].copy_(g_c, non_blocking=True)
                    self._scratch_uc[:cn].copy_(u_c, non_blocking=True)
                    self._scratch_dc[:, :cn].copy_(d_c, non_blocking=True)
            else:
                self._scratch_gc[:cn].copy_(g_c)
                self._scratch_uc[:cn].copy_(u_c)
                self._scratch_dc[:, :cn].copy_(d_c)
        self.cold_dma_bytes += nbytes
        self.dma_bytes += nbytes
        out = (self._scratch_gc[:cn], self._scratch_uc[:cn],
               self._scratch_dc[:, :cn])
        if self.cold_cache_cap > 0:
            with torch.no_grad():
                self._cold_cache[expert_id] = (
                    out[0].clone(), out[1].clone(), out[2].clone())
            self._cold_lru.append(expert_id)
            while len(self._cold_lru) > self.cold_cache_cap:
                self._cold_cache.pop(self._cold_lru.pop(0), None)
        return out

    def _sync_dma(self) -> None:
        if self.dma_stream is not None and self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(self.dma_stream)

    def _ensure_hot(self, expert_id: int):
        pool = self._pool(expert_id)
        if expert_id in pool["expert_to_slot"]:
            self.hits += 1
            slot_idx = pool["expert_to_slot"][expert_id]
            pool["slot_lru"].remove(slot_idx)
            pool["slot_lru"].append(slot_idx)
            return pool, slot_idx
        self.misses += 1
        slot_idx = pool["slot_lru"].pop(0)
        self._load_hot_to_slot(expert_id, pool, slot_idx)
        pool["slot_lru"].append(slot_idx)
        self._sync_dma()
        return pool, slot_idx

    def _clear_all(self) -> None:
        for pool in self.pools.values():
            pool["expert_to_slot"].clear()
            pool["slot_to_expert"].clear()
            pool["slot_lru"] = list(range(len(pool["slots"])))
        self._cold_cache.clear()
        self._cold_lru.clear()

    def _forward_grouped(self, flat, TI, TW, needed, mask, act, out):
        """Batched dispatch: one bmm trio per tier (MoEShard-style fusion).

        Per tier: ensure hot for all needed experts, stack hot weights +
        padded token rows, single bmm trio for hot, one coalesced cold
        fetch + single bmm trio for cold, single index_add scatter.
        Zero padding never scatters (only real rows land in out).
        """
        K = TI.shape[-1]
        dev, dt = flat.device, flat.dtype
        for t in self.tiers:
            es = [e for e in needed if self.tier_of[e] == t]
            if not es:
                continue
            pool = self.pools[t]
            hn = pool["slots"][0].hot_n
            inter = self.spec.inter
            cn = inter - hn
            per_rows, per_w = [], []
            for e in es:
                slot_idx = self._ensure_hot(e)
                kpos, rows = torch.where(mask[e])
                per_rows.append(rows)
                per_w.append(TW[rows, kpos])
            nes = [r.numel() for r in per_rows]
            nmax = max(nes)
            E = len(es)
            H = flat.shape[-1]
            xb = flat.new_zeros(E, nmax, H)
            wb = flat.new_zeros(E, nmax, 1)
            wg = flat.new_empty(E, hn, H)
            wu = flat.new_empty(E, hn, H)
            wd = flat.new_empty(E, H, hn)
            for i, e in enumerate(es):
                n = nes[i]
                if n == 0:
                    continue
                xb[i, :n] = flat[per_rows[i]]
                wb[i, :n, 0] = per_w[i].to(dt)
                slot = pool["slots"][pool["expert_to_slot"][e]]
                wg[i].copy_(slot.gate_hot.weight)
                wu[i].copy_(slot.up_hot.weight)
                wd[i].copy_(slot.down_hot.weight)
            with torch.no_grad():
                gh = torch.bmm(xb, wg.transpose(1, 2))
                uh = torch.bmm(xb, wu.transpose(1, 2))
                yh = torch.bmm(apply_gate(gh, uh, act), wd.transpose(1, 2))
                if cn > 0:
                    fu = flat.new_empty(E, 2 * cn, H)
                    dc = flat.new_empty(E, H, cn)
                    for i, e in enumerate(es):
                        gc, uc, dcc = self._fetch_cold_slices(e, hn, cn)
                        fu[i, :cn].copy_(gc)
                        fu[i, cn:].copy_(uc)
                        dc[i].copy_(dcc)
                        # NOTE: grouped path bypasses the cold-cache tier
                        # (values identical; hits simply not harvested).
                        nbytes = int(gc.nbytes + uc.nbytes + dcc.nbytes)
                        self.cold_dma_bytes += nbytes
                        self.dma_bytes += nbytes
                    self._sync_dma()
                    g, u = torch.bmm(xb, fu.transpose(1, 2)).chunk(2, dim=-1)
                    yc = torch.bmm(apply_gate(g, u, act), dc.transpose(1, 2))
                    yh = yh + yc.to(yh.dtype)
                ye = yh * wb.to(yh.dtype)
            rows_all, ye_all = [], []
            for i in range(E):
                if nes[i]:
                    rows_all.append(per_rows[i])
                    ye_all.append(ye[i, :nes[i]].to(out.dtype))
            if rows_all:
                out.index_add_(0, torch.cat(rows_all),
                               torch.cat(ye_all))
        if self.streaming:
            self._clear_all()
        return out

    def _fetch_cold_slices(self, expert_id: int, hn: int, cn: int):
        """Cold slices as views (caller copies + accounts)."""
        t_gate, t_up, t_down = self._expert_tensors(expert_id)
        return t_gate[hn:], t_up[hn:], t_down[:, hn:]

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
        if self.grouped_gemm:
            out = self._forward_grouped(flat, TI, TW, needed, mask, act, out)
            shared_out = self.shared_experts(identity) \
                if self.shared_experts is not None \
                else torch.zeros_like(identity)
            out = shared_out + out.reshape(orig_shape)
            if self.streaming:
                self._clear_all()
            if self.tupled:
                return out, None
            return out
        for e in needed:
            pool, slot_idx = self._ensure_hot(e)
            slot = pool["slots"][slot_idx]
            hn = slot.hot_n
            cn = self.spec.inter - hn
            # Overlap: issue cold fetch async, compute the HOT partial
            # immediately (hot weights are resident — no dependency), and
            # only then wait for cold. Cold-fetch latency hides behind
            # hot GEMMs; the wait left over is the exposed stall.
            gc, uc, dc = self._fetch_cold(e)
            kpos, rows = torch.where(mask[e])
            xe = flat[rows]
            w = TW[rows, kpos].unsqueeze(-1)
            with torch.no_grad():
                gh = F.linear(xe, slot.gate_hot.weight)
                uh = F.linear(xe, slot.up_hot.weight)
                yh = F.linear(apply_gate(gh, uh, act), slot.down_hot.weight)
                if cn > 0:
                    t0 = time.perf_counter()
                    self._sync_dma()
                    self.stall_ms += (time.perf_counter() - t0) * 1000.0
                    gc = F.linear(xe, gc)
                    uc = F.linear(xe, uc)
                    yc = F.linear(apply_gate(gc, uc, act), dc)
                    ye = yh + yc.to(yh.dtype)
                else:
                    ye = yh
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
            self._cold_cache.clear()
            self._cold_lru.clear()
        if self.tupled:
            return out, None
        return out
