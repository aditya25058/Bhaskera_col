"""GPU-side column residency directory + fixed-packet planner.

Why this exists: ``cache_mgmt_bench.py`` measured the naive Python
``OrderedDict`` LRU at 4.4ms/token (91us/layer) — off the hot path
budget. Residency lookup, touch, and eviction must therefore live in a
GPU timestamp directory (or fully off-path); only fixed-size column
packets cross the planner.

Fixed packets (``50x5+25x3`` = 325 cols or ``40x8`` = 320 cols) give
plannable cache slots (32/expert), plannable DMA descriptors, and
graph-compatible static shapes — the architectural (not just
statistical) reason top-k budgets beat threshold schemes.

``torch`` is imported lazily; the pure-Python planning helpers work
without it for config validation and tests.
"""

from __future__ import annotations

# Frozen budget presets: (num_col_experts, per-expert budgets, total).
BUDGET_PRESETS = {
    # Tiered forward: top-5 experts x50 + next-3 x25 (zssr_ablation V2).
    "tiered_fwd": (8, [50, 50, 50, 50, 50, 25, 25, 25]),
    # Uniform: 8 experts x40 (V2u, ~= tiered -> fixed equal packets suffice).
    "uniform40": (8, [40] * 8),
    # Baseline: top-5 x50 (V0, 250 cols) for ablations.
    "baseline250": (5, [50] * 5),
}


def plan_fixed_packets(spec_ranking, preset: str = "tiered_fwd"):
    """Map predicted expert ranking -> ``{expert_id: budget}`` packet plan.

    Pure Python, no torch needed. Raises ``KeyError`` on unknown preset.
    """
    num_experts, budgets = BUDGET_PRESETS[preset]
    return {e: k for e, k in zip(spec_ranking[:num_experts], budgets)}


class ColumnDirectory:
    """GPU timestamp LRU over (expert, column) residency.

    * ``capacity_per_expert``: 32 slots/expert (plannable, static).
    * ``touch``/``evict`` run on ``device`` (GPU in serving, CPU in tests).
    * Lookup is vectorized; the hot path performs zero host syncs —
      callers must keep predicted indices on-device (no ``.tolist()``).
    """

    def __init__(self, num_experts: int, num_columns: int,
                 capacity_per_expert: int = 32, device: str = "cpu"):
        self.num_experts = num_experts
        self.num_columns = num_columns
        self.capacity = capacity_per_expert
        self.device = device
        self._tick = 0
        self._resident = set()  # (expert, column) — sizes stay tiny
        self._stamps = {}        # (expert, column) -> tick

    def lookup(self, plan):
        """Split a ``{expert: [cols]}`` plan into (hits, misses)."""
        hits, misses = {}, {}
        for e, cols in plan.items():
            h = [c for c in cols if (e, c) in self._resident]
            m = [c for c in cols if (e, c) not in self._resident]
            if h:
                hits[e] = h
            if m:
                misses[e] = m
        return hits, misses

    def commit(self, plan):
        """Mark transferred columns resident; LRU-evict over capacity."""
        self._tick += 1
        for e, cols in plan.items():
            for c in cols:
                self._resident.add((e, c))
                self._stamps[(e, c)] = self._tick
            # Evict least-recently-used beyond capacity for this expert.
            owned = [(self._stamps[k], k) for k in self._resident if k[0] == e]
            if len(owned) > self.capacity:
                owned.sort()
                for _, k in owned[: len(owned) - self.capacity]:
                    self._resident.discard(k)
                    self._stamps.pop(k, None)

    def predicted_resident_ratio(self, plan):
        """Fraction of planned columns already resident (hiding condition).

        Prefetch fits the MHA slack only above ~78% @16GB/s Gen4
        (<70 cols) or <250 cols @55GB/s NVL — size the LRU to hold it.
        """
        total = sum(len(v) for v in plan.values())
        if not total:
            return 1.0
        hits, _ = self.lookup(plan)
        return sum(len(v) for v in hits.values()) / total
