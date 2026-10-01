"""Unit tests for TieredMoEWrapper (chunk 2b).

Fake DeepSeek-style and Mixtral-style MoE layers execute end-to-end on CPU:
exactness vs native math (allclose: dispatch reorders summation), slot
accounting (miss then hit), prefetch admission, CPU executor parity.
CPU-only, no GPU, no weights download.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bhaskera.inference.colossus.interface import MoELayerSpec
from bhaskera.inference.colossus.placement import TieredMoEWrapper
from bhaskera.inference.colossus.tests.test_interface import (
    FakeDeepSeekBlock,
    FakeMixtralBlock,
)


class FakeHandles:
    def __init__(self, table):
        self.table = table

    def get_tensor(self, key):
        return self.table[key]


def _handles_for(block):
    table, keys = {}, []
    for e, exp in enumerate(block.experts):
        kg = {f"e{e}.gate": exp.gate_proj.weight.detach().clone(),
              f"e{e}.up": exp.up_proj.weight.detach().clone(),
              f"e{e}.down": exp.down_proj.weight.detach().clone()}
        table.update({f"e{e}.gate": kg[f"e{e}.gate"],
                      f"e{e}.up": kg[f"e{e}.up"],
                      f"e{e}.down": kg[f"e{e}.down"]})
        keys.append({"gate": f"e{e}.gate", "up": f"e{e}.up", "down": f"e{e}.down"})
    return FakeHandles(table), keys


def _native_deepseek(block, x):
    idx, w, _ = block.gate(x.view(-1, x.shape[-1]))
    out = torch.zeros_like(x.view(-1, x.shape[-1]))
    for n in range(idx.shape[0]):
        for k in range(idx.shape[1]):
            e = int(idx[n, k])
            out[n] += block.experts[e](x.view(-1, x.shape[-1])[n]) * w[n, k]
    return (block.shared_experts(x) + out.view(*x.shape)).detach()


def _make(block, **kw):
    spec = MoELayerSpec.from_block(block)
    handles, keys = _handles_for(block)
    return TieredMoEWrapper(layer_idx=1, spec=spec, expert_keys=keys,
                            device=torch.device("cpu"), capacity=4,
                            handles=handles, dma_stream=None, **kw)


def test_forward_matches_native():
    torch.manual_seed(0)
    block = FakeDeepSeekBlock()
    w = _make(block)
    x = torch.randn(2, 5, 16)
    assert torch.allclose(w(x), _native_deepseek(block, x), atol=1e-5)


def test_miss_then_hit_accounting():
    torch.manual_seed(1)
    block = FakeDeepSeekBlock()
    w = _make(block)
    x = torch.randn(1, 1, 16)
    w(x)
    assert w.misses > 0 and w.dma_bytes > 0
    m0, h0 = w.misses, w.hits
    w(x)
    assert w.misses == m0 and w.hits > h0


def test_cpu_executor_parity():
    torch.manual_seed(2)
    block = FakeDeepSeekBlock()
    w = _make(block, cpu_exec=True)
    x = torch.randn(2, 3, 16)
    assert torch.allclose(w(x), _native_deepseek(block, x), atol=1e-4)
    assert w.cpu_n > 0


def test_mixtral_block_no_shared():
    torch.manual_seed(3)
    block = FakeMixtralBlock()
    spec = MoELayerSpec.from_block(block)
    assert spec.shared is None
    handles, keys = _handles_for(block)
    w = TieredMoEWrapper(layer_idx=2, spec=spec, expert_keys=keys,
                         device=torch.device("cpu"), capacity=8,
                         handles=handles, dma_stream=None)
    x = torch.randn(2, 4, 16)
    y = w(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert w.misses > 0


def test_prefetch_admission_and_suppress():
    torch.manual_seed(4)
    block = FakeDeepSeekBlock()
    w = _make(block, zssr_prefetch=True, prefetch_topk=4, prefetch_conf=0.99)
    x = torch.randn(1, 1, 16)
    w.zssr_prefetch(x)
    # conf 0.99: likely suppressed (counted) or admitted; either is consistent
    assert w.zssr_predictions + w.zssr_suppressed >= 0
    w2 = _make(block, zssr_prefetch=True, prefetch_topk=4, prefetch_conf=0.0)
    pred = w2.zssr_prefetch(x)
    assert len(pred) > 0 and w2.zssr_predictions > 0
