"""Unit tests for column-granular residency (the column thesis).

Synthetic module-based experts, CPU-only. Bars:
- f=1.0 reproduces the native reference bitwise (single-GEMM path);
- f<1 matches the native reference within split-summation noise
  (two GEMMs + add reorder rounding: NOT bitwise by construction —
  exactness at f<1 is flip-audited on GPU, same discipline as ulp1);
- cold/hot DMA accounting matches the (1-f)/f split on misses.
"""
import torch
import torch.nn.functional as F
from torch import nn

from bhaskera.inference.colossus.colres import ColumnTieredMoEWrapper
from bhaskera.inference.colossus.interface import MoELayerSpec
from bhaskera.inference.colossus.placement import apply_gate
from bhaskera.inference.colossus.tests.test_interface import FakeDeepSeekBlock
from bhaskera.inference.colossus.tests.test_placement import DictHandles


def _setup(n_exp=6, hidden=8, inter=10, top_k=2, seed=0):
    torch.manual_seed(seed)
    block = FakeDeepSeekBlock(n_exp=n_exp, hidden=hidden, top_k=top_k)
    for e in list(block.experts) + [block.shared_experts]:
        e.gate_proj = nn.Linear(hidden, inter, bias=False,
                                dtype=torch.bfloat16)
        e.up_proj = nn.Linear(hidden, inter, bias=False,
                              dtype=torch.bfloat16)
        e.down_proj = nn.Linear(inter, hidden, bias=False,
                                dtype=torch.bfloat16)
    spec = MoELayerSpec.from_block(block)
    assert spec.inter == inter
    table = {}
    keys = []
    for i, e in enumerate(block.experts):
        d = {}
        for role, mod in (("gate", e.gate_proj), ("up", e.up_proj),
                          ("down", e.down_proj)):
            k = f"experts.{i}.{role}_proj.weight"
            table[k] = mod.weight.detach().clone()
            d[role] = k
        keys.append(d)
    return spec, DictHandles(table), keys


def _native_ref(spec, handles, keys, x):
    """Full-expert math in native op order (ascending, index_add)."""
    flat = x.reshape(-1, x.shape[-1])
    K = spec.top_k
    idx, w = spec.route(x)
    TI = idx.reshape(-1, K).long()
    TW = w.reshape(-1, K)
    out = torch.zeros_like(flat)
    mask = F.one_hot(TI, num_classes=spec.n_routed + 1).permute(2, 1, 0)
    for e in range(spec.n_routed):
        kp, rows = torch.where(mask[e])
        if rows.numel() == 0:
            continue
        t = handles.get_tensor(keys[e]["gate"])
        tu = handles.get_tensor(keys[e]["up"])
        td = handles.get_tensor(keys[e]["down"])
        ye = F.linear(apply_gate(F.linear(flat[rows], t),
                                 F.linear(flat[rows], tu),
                                 spec.activation), td)
        out.index_add_(0, rows, (ye * TW[rows, kp].unsqueeze(-1)).to(out.dtype))
    shared = spec.shared
    return out.reshape(*x.shape)


def _inputs(hidden=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(5, hidden, dtype=torch.bfloat16, generator=g)


def _shared_forward(spec, x):
    # FakeDeepSeekBlock shared path: shared_experts(x) added by wrappers.
    return spec.shared(x)


def test_f10_matches_native_bitwise():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    w = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=4, handles=handles,
                               hot_frac=1.0)
    x = _inputs()
    got = w(x)
    ref = _native_ref(spec, handles, keys, x) + _shared_forward(spec, x)
    assert torch.equal(got, ref)
    assert w.cold_dma_bytes == 0
    assert w.pools[1.0]["slots"][0].resident_bytes() > 0


def test_fraction_matches_native_within_summation_noise():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    w = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=6, handles=handles,
                               hot_frac=0.5)
    x = _inputs()
    got = w(x)
    ref = _native_ref(spec, handles, keys, x) + _shared_forward(spec, x)
    assert got.shape == ref.shape
    # Split-summation reorders rounding: bounded noise, never bitwise.
    assert torch.allclose(got.float(), ref.float(), atol=0.05, rtol=0.05)


def test_dma_split_and_lru():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    w = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=6, handles=handles,
                               hot_frac=0.5)
    x = _inputs()
    w(x)
    assert w.misses > 0
    per_expert = sum(handles.get_tensor(keys[0][r]).nbytes for r in
                     ("gate", "up", "down"))
    hot_per = per_expert // 2
    assert w.hot_dma_bytes == w.misses * hot_per
    # Cold fetched on every use (no cold cache): grows past hot on reuse.
    assert w.cold_dma_bytes >= w.hot_dma_bytes
    assert w.dma_bytes == w.hot_dma_bytes + w.cold_dma_bytes
    m0 = w.misses
    w(x)
    assert w.misses == m0  # hot resident now; cold refetched silently
    assert w.hits > 0
    # Residency is genuinely halved vs whole-expert slots.
    from bhaskera.inference.colossus.placement import FastSlot
    full = FastSlot(spec.hidden, spec.inter, dev, spec.dtype,
                    spec.activation)
    full_bytes = sum(p.nbytes for p in full.parameters())
    assert w.pools[0.5]["slots"][0].resident_bytes() * 2 == full_bytes


def test_mixed_tiers_match_native():
    from bhaskera.inference.colossus.colprofile import assign_tiers, hbm_equiv
    from collections import Counter
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    freq = Counter({0: 50, 1: 30, 2: 10, 3: 5, 4: 2, 5: 1})
    fracs = assign_tiers(freq, 6, pin_top=2, thin_frac=0.25)
    assert fracs[0] == 1.0 and fracs[1] == 1.0  # top-2 pinned
    assert all(f == 0.25 for f in fracs[2:])
    assert hbm_equiv({1.0: 2, 0.25: 6}) == 2 + 1.5
    pools = {1.0: 2, 0.25: 4}
    w = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=6, handles=handles,
                               hot_fracs=fracs, tier_pools=pools)
    assert set(w.pools) == {1.0, 0.25}
    assert len(w.pools[1.0]["slots"]) == 2
    x = _inputs()
    got = w(x)
    ref = _native_ref(spec, handles, keys, x) + _shared_forward(spec, x)
    assert got.shape == ref.shape
    assert torch.allclose(got.float(), ref.float(), atol=0.05, rtol=0.05)


def test_assign_tiers_pin_and_hbm():
    from bhaskera.inference.colossus.colprofile import assign_tiers
    from collections import Counter
    fracs = assign_tiers(Counter(), 8, pin_top=0, thin_frac=0.1)
    assert all(f == 0.1 for f in fracs)  # no pinning: all thin
    fracs = assign_tiers(Counter({3: 9, 7: 1}), 8, pin_top=3,
                         thin_frac=0.5)
    assert fracs[3] == 1.0  # most frequent is pinned
    assert sum(1 for f in fracs if f == 1.0) == 3


def test_cold_cache_hits_skip_dma():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    w = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=6, handles=handles,
                               hot_frac=0.5, cold_cache_cap=8)
    x = _inputs()
    ref = _native_ref(spec, handles, keys, x) + _shared_forward(spec, x)
    w(x)
    assert w.cold_misses > 0 and w.cold_hits == 0
    d0 = w.cold_dma_bytes
    got = w(x)
    assert w.cold_hits > 0  # reuse turns refetch into hits
    assert w.cold_dma_bytes == d0  # hits move zero host bytes
    assert torch.allclose(got.float(), ref.float(), atol=0.05, rtol=0.05)


def test_cold_cache_evicts_lru():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    w = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=6, handles=handles,
                               hot_frac=0.5, cold_cache_cap=1)
    x = _inputs()
    w(x)
    n = len(w._cold_cache)
    assert n <= 1
    w(x)
    # cap-1 with several experts: evictions force refetch misses
    assert w.cold_misses > w.cold_hits


def test_grouped_gemm_matches_loop_path():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    loop = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                                  device=dev, capacity=6, handles=handles,
                                  hot_frac=0.5)
    batched = ColumnTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                                     device=dev, capacity=6, handles=handles,
                                     hot_frac=0.5, grouped_gemm=True)
    # Skewed token distribution across experts (uneven rows stress padding).
    g = torch.Generator().manual_seed(7)
    x = torch.randn(13, 8, dtype=torch.bfloat16, generator=g)
    a = loop(x)
    b = batched(x)
    assert a.shape == b.shape
    assert torch.allclose(a.float(), b.float(), atol=0.05, rtol=0.05)
    assert batched.dma_bytes == loop.dma_bytes  # same bytes, fewer launches
