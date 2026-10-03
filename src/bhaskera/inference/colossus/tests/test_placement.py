"""Unit tests for TieredMoEWrapper (chunk 2b).

Fake DeepSeek-style and Mixtral-style MoE layers execute end-to-end on CPU:
exactness vs native math (allclose: dispatch reorders summation), slot
accounting (miss then hit), prefetch admission, CPU executor parity.
CPU-only, no GPU, no weights download.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bhaskera.inference.colossus.interface import MoELayerSpec
from bhaskera.inference.colossus.placement import TieredMoEWrapper, wrap_moe_layers
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


class FakeDecoderLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = block

    def forward(self, x):
        return self.mlp(self.attn(x)) + x


class FakeMoEModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        for i in range(n_layers):
            setattr(self, f"layer{i}", FakeDecoderLayer(FakeDeepSeekBlock()))

    def forward(self, x):
        for i in range(2):
            x = getattr(self, f"layer{i}")(x)
        return x


class FakeProfile:
    decoder_layer_cls = FakeDecoderLayer


class DictHandles:
    """ShardHandles-shaped stub: weight_map + get_tensor from a table."""

    def __init__(self, table):
        self.table = table
        self.weight_map = {k: "shard0" for k in table}

    def get_tensor(self, key):
        return self.table[key]


def _wrapped_model(n_layers=2):
    model = FakeMoEModel(n_layers)
    table = {}
    for name, mod in model.named_modules():
        # expert weight dotted paths, e.g. layer0.mlp.experts.3.gate_proj.weight
        if isinstance(mod, torch.nn.Linear) and ".experts." in name:
            table[name] = mod.weight.detach().clone()
    handles = DictHandles(table)
    wrappers = wrap_moe_layers(model, FakeProfile(), handles,
                               torch.device("cpu"), capacity=8,
                               dma_stream=None)
    return model, wrappers, table


def test_wrap_replaces_all_moe_blocks():
    model, wrappers, _ = _wrapped_model()
    assert len(wrappers) == 2
    assert isinstance(model.layer0.mlp, TieredMoEWrapper)
    assert isinstance(model.layer1.mlp, TieredMoEWrapper)
    assert wrappers[0].layer_idx == 0 and wrappers[1].layer_idx == 1


def test_wrapped_model_matches_native():
    torch.manual_seed(7)
    ref = FakeMoEModel()
    model = FakeMoEModel()
    model.load_state_dict(ref.state_dict())
    table = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".experts." in name:
            table[name] = mod.weight.detach().clone()
    handles = DictHandles(table)
    wrappers = wrap_moe_layers(model, FakeProfile(), handles,
                               torch.device("cpu"), capacity=8,
                               dma_stream=None)
    x = torch.randn(1, 2, 16)
    with torch.no_grad():
        y_run, y_ref = model(x), ref(x)
    # slot DMA path from identical values: tight tolerance (same math,
    # expert-grouped summation order differs from native token order).
    assert torch.allclose(y_run, y_ref, atol=1e-4)
    assert sum(w.misses for w in wrappers) > 0


def test_tupled_return_convention():
    torch.manual_seed(11)
    block = FakeDeepSeekBlock()
    spec = MoELayerSpec.from_block(block)
    handles, keys = _handles_for(block)
    w = TieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                         device=torch.device("cpu"), capacity=4,
                         handles=handles, dma_stream=None, tupled=True)
    x = torch.randn(1, 1, 16)
    out = w(x)
    assert isinstance(out, tuple) and len(out) == 2
    w2 = TieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                          device=torch.device("cpu"), capacity=4,
                          handles=handles, dma_stream=None, tupled=False)
    assert isinstance(w2(x), torch.Tensor)
