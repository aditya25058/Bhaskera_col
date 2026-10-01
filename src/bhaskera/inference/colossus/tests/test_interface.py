"""Unit tests for the model-agnostic MoE interface (chunk 1).

Fake DeepSeek-style and Mixtral-style MoE blocks verify that
MoELayerSpec removes arch-specific assumptions without changing routing math.
CPU-only, no GPU, no weights download.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bhaskera.inference.colossus.interface import (
    GATE_DEEPSEEK_3TUPLE,
    GATE_LOGITS,
    MoELayerSpec,
    find_proj,
)


class FakeExpert(nn.Module):
    def __init__(self, hidden=16, inter=24):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FakeDeepSeekGate(nn.Module):
    """Returns (idx, weight, aux) like DeepSeek-V2/V3 routers."""

    def __init__(self, hidden=16, n_exp=8, top_k=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_exp, hidden))
        self.top_k = top_k

    def forward(self, x):
        s = F.softmax(F.linear(x.float(), self.weight), dim=-1)
        w, idx = torch.topk(s, k=self.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        return idx, w.to(x.dtype), None


class FakeDeepSeekBlock(nn.Module):
    def __init__(self, n_exp=8, hidden=16, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([FakeExpert(hidden) for _ in range(n_exp)])
        self.gate = FakeDeepSeekGate(hidden, n_exp, top_k)
        self.shared_experts = FakeExpert(hidden)
        self.num_experts_per_tok = top_k

    def moe_infer(self, x, idx, w):
        return torch.stack([self.experts[int(i)](x[n]) * w[n, k]
                            for n, row in enumerate(idx)
                            for k, i in enumerate(row)]).sum(0)


class FakeMixtralGate(nn.Module):
    """Returns raw logits like Mixtral/Qwen routers."""

    def __init__(self, hidden=16, n_exp=8):
        super().__init__()
        self.linear = nn.Linear(hidden, n_exp, bias=False)
        self.weight = self.linear.weight

    def forward(self, x):
        return self.linear(x.float())


class FakeMixtralBlock(nn.Module):
    def __init__(self, n_exp=8, hidden=16, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([FakeExpert(hidden) for _ in range(n_exp)])
        self.gate = FakeMixtralGate(hidden, n_exp)
        self.top_k = top_k
        # NOTE: no moe_infer, no shared_experts (unlike DeepSeek).


def test_from_block_deepseek():
    b = FakeDeepSeekBlock()
    s = MoELayerSpec.from_block(b)
    assert s.gate_style == GATE_DEEPSEEK_3TUPLE
    assert s.n_routed == 8 and s.top_k == 2
    assert s.shared is b.shared_experts
    assert (s.hidden, s.inter) == (16, 24)
    assert s.router_weight() is not None


def test_from_block_mixtral():
    b = FakeMixtralBlock()
    s = MoELayerSpec.from_block(b)
    assert s.gate_style == GATE_LOGITS
    assert s.n_routed == 8 and s.top_k == 2
    assert s.shared is None
    assert s.router_weight() is not None


def test_route_deepseek_passthrough():
    torch.manual_seed(0)
    b = FakeDeepSeekBlock()
    s = MoELayerSpec.from_block(b)
    x = torch.randn(3, 16)
    idx, w = s.route(x)
    assert idx.dtype == torch.long and idx.shape == (3, 2)
    assert w.dtype == x.dtype and w.shape == (3, 2)
    # passthrough: identical to the gate's own decision
    gi, gw, _ = b.gate(x)
    assert torch.equal(idx, gi) and torch.equal(w, gw)


def test_route_logits_matches_manual_topk():
    torch.manual_seed(1)
    b = FakeMixtralBlock()
    s = MoELayerSpec.from_block(b)
    x = torch.randn(3, 16)
    idx, w = s.route(x)
    scores = F.softmax(b.gate.linear(x.float()), dim=-1)
    ew, eidx = torch.topk(scores, k=2, dim=-1)
    ew = ew / ew.sum(-1, keepdim=True)
    assert torch.equal(idx, eidx) and torch.allclose(w.float(), ew, atol=1e-5)


def test_find_proj_and_errors():
    e = FakeExpert()
    assert find_proj(e, "gate") is e.gate_proj
    assert find_proj(e, "down") is e.down_proj
    with pytest.raises(KeyError):
        find_proj(e, "nonexistent")


def test_missing_gate_raises():
    m = nn.Module()
    m.experts = nn.ModuleList([FakeExpert()])
    with pytest.raises(AttributeError):
        MoELayerSpec.from_block(m)
