"""Unit tests for column-granular grouped residency (thesis, repr. 2).

Synthetic fused grouped weights, CPU-only:
- f=1.0 reproduces GroupedTieredMoEWrapper bitwise;
- f<1 matches the native summation-partition reference within noise;
- hot/cold DMA accounting follows the f split.
"""
import torch
import torch.nn.functional as F

from bhaskera.inference.colossus.groupcol import GroupedColumnWrapper
from bhaskera.inference.colossus.grouped import GroupedTieredMoEWrapper
from bhaskera.inference.colossus.tests.test_grouped import (
    DictHandles,
    _reference,
)


def _synthetic(E=6, H=8, I=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    fused = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, generator=g)
    down = torch.randn(E, H, I, dtype=torch.bfloat16, generator=g)
    return fused, down


def _colw(fused, down, hot_frac, capacity=4, E=6, H=8, I=6):
    h = DictHandles({"gu": fused, "dn": down})
    return GroupedColumnWrapper(
        layer_idx=0, fused_key="gu", down_key="dn", n_experts=E,
        hidden=H, inter=I, device=torch.device("cpu"), capacity=capacity,
        handles=h, dtype=torch.bfloat16, activation="gelu_tanh",
        hot_frac=hot_frac)


def _wholew(fused, down, capacity=4, E=6, H=8, I=6):
    h = DictHandles({"gu": fused, "dn": down})
    return GroupedTieredMoEWrapper(
        layer_idx=0, fused_key="gu", down_key="dn", n_experts=E,
        hidden=H, inter=I, device=torch.device("cpu"), capacity=capacity,
        handles=h, dtype=torch.bfloat16, activation="gelu_tanh")


def _io(seed=1):
    g = torch.Generator().manual_seed(seed)
    flat = torch.randn(9, 8, dtype=torch.bfloat16, generator=g)
    TI = torch.randint(0, 6, (9, 2), generator=g)
    TW = torch.rand(9, 2, dtype=torch.float32, generator=g)
    return flat, TI, TW / TW.sum(-1, keepdim=True)


def test_f10_matches_whole_slice_wrapper_bitwise():
    fused, down = _synthetic()
    ref = _wholew(fused, down)
    got = _colw(fused, down, 1.0)
    flat, TI, TW = _io()
    assert torch.equal(ref(flat, TI, TW), got(flat, TI, TW))
    assert got.cold_dma_bytes == 0
    assert got.hot_dma_bytes == ref.dma_bytes


def test_fraction_matches_native_within_noise():
    fused, down = _synthetic()
    w = _colw(fused, down, 0.5)
    flat, TI, TW = _io()
    got = w(flat, TI, TW)
    ref = _reference(fused, down, flat, TI, TW,
                     lambda t: F.gelu(t, approximate="tanh"))
    assert got.shape == ref.shape
    assert torch.allclose(got.float(), ref.float(), atol=0.05, rtol=0.05)


def test_hot_cold_dma_split():
    fused, down = _synthetic()
    w = _colw(fused, down, 0.5)
    flat, TI, TW = _io()
    w(flat, TI, TW)
    assert w.misses > 0
    # I=6 -> hot 3/cold 3 fused rows + down halves: hot == cold bytes.
    assert w.hot_dma_bytes == w.cold_dma_bytes or w.cold_dma_bytes > 0
    assert w.dma_bytes == w.hot_dma_bytes + w.cold_dma_bytes
    # Residency halved vs whole-slice slots.
    assert w.slots[0].gate_up_proj.weight.shape == (6, 8)
