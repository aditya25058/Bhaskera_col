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
        e.gate_proj = nn.Linear(hidden, inter, bias=False)
        e.up_proj = nn.Linear(hidden, inter, bias=False)
        e.down_proj = nn.Linear(inter, hidden, bias=False)
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
    assert w.slots[0].resident_bytes() > 0


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
    assert w.slots[0].resident_bytes() * 2 == full_bytes
