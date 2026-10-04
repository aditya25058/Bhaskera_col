"""Unit tests for the grouped-weight MoE adapter (representation 2).

Synthetic fused tensors only — no model downloads, CPU-runnable.
"""
import torch
from torch import nn

from bhaskera.inference.colossus.grouped import (
    GROUPED_KEY_RES,
    GroupedTieredMoEWrapper,
    classify_grouped_pair,
    grouped_container_attr,
    is_grouped_expert_key,
)


class DictHandles:
    """Minimal handles stub: .get_tensor + .header from dicts."""

    def __init__(self, tensors: dict):
        self.tensors = tensors
        self.weight_map = {k: "shard0" for k in tensors}

    def get_tensor(self, key):
        return self.tensors[key]

    def header(self, key):
        t = self.tensors[key]
        return {"shape": list(t.shape)}


def _synthetic(E=6, H=8, I=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    fused = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, generator=g)
    down = torch.randn(E, H, I, dtype=torch.bfloat16, generator=g)
    return fused, down


def _reference(fused, down, flat, TI, TW, act):
    """Native-style grouped experts math (ascending experts, index_add)."""
    import torch.nn.functional as F

    E = fused.shape[0]
    K = TI.shape[-1]
    out = torch.zeros_like(flat)
    mask = F.one_hot(TI, num_classes=E + 1).permute(2, 1, 0)
    for e in range(E):
        kpos, rows = torch.where(mask[e])
        if rows.numel() == 0 or e >= E:
            continue
        xe = flat[rows]
        g, u = F.linear(xe, fused[e]).chunk(2, dim=-1)
        ye = F.linear(act(g) * u, down[e])
        ye = ye * TW[rows, kpos].unsqueeze(-1)
        out.index_add_(0, rows, ye.to(out.dtype))
    return out


def _wrapper(fused, down, capacity=3, E=6, H=8, I=5):
    h = DictHandles({"gu": fused, "dn": down})
    return GroupedTieredMoEWrapper(
        layer_idx=0, fused_key="gu", down_key="dn", n_experts=E,
        hidden=H, inter=I, device=torch.device("cpu"), capacity=capacity,
        handles=h, dtype=torch.bfloat16, activation="gelu_tanh")


def test_forward_matches_reference_bitwise():
    fused, down = _synthetic()
    w = _wrapper(fused, down)
    g = torch.Generator().manual_seed(1)
    flat = torch.randn(11, 8, dtype=torch.bfloat16, generator=g)
    TI = torch.randint(0, 6, (11, 2), generator=g)
    TW = torch.rand(11, 2, dtype=torch.float32, generator=g)
    TW = TW / TW.sum(-1, keepdim=True)
    import torch.nn.functional as F

    ref = _reference(fused, down, flat, TI, TW,
                     lambda t: F.gelu(t, approximate="tanh"))
    got = w(flat, TI, TW)
    assert torch.equal(got, ref)


def test_slots_hit_and_streaming():
    fused, down = _synthetic()
    w = _wrapper(fused, down, capacity=6)
    flat = torch.randn(4, 8, dtype=torch.bfloat16)
    TI = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
    TW = torch.full((4, 2), 0.5)
    w(flat, TI, TW)
    m1 = w.misses
    assert m1 > 0
    w(flat, TI, TW)
    assert w.misses == m1  # all resident now
    assert w.hits > 0
    assert w.dma_bytes > 0
    assert w.routing_log is None

    ws = _wrapper(fused, down, capacity=0)
    assert ws.streaming
    ws(flat, TI, TW)
    m = ws.misses
    ws(flat, TI, TW)
    assert ws.misses > m  # transient slot: misses every forward


def test_3d_leading_dims_and_routing_log():
    fused, down = _synthetic()
    w = _wrapper(fused, down)
    w.routing_log = []
    flat = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    TI = torch.randint(0, 6, (2, 3, 2))
    TW = torch.full((2, 3, 2), 0.5)
    got = w(flat, TI, TW)
    assert got.shape == flat.shape
    assert w.routing_log and w.routing_log[0][0] == 0
    assert sorted(w.routing_log[0][1]) == w.routing_log[0][1]


def test_classify_grouped_pair_orders():
    assert classify_grouped_pair({"a": (8, 10, 4), "b": (8, 4, 5)}) == \
        {"fused": "a", "down": "b"}
    assert classify_grouped_pair({"a": (8, 4, 5), "b": (8, 10, 4)}) == \
        {"fused": "b", "down": "a"}
    for bad in ({"a": (8, 10, 4)}, {"a": (8, 10, 4), "b": (8, 4, 5), "c": (8, 4, 5)},
                {"a": (8, 10, 4), "b": (7, 4, 5)}, {"a": (8, 9, 4), "b": (8, 4, 5)}):
        try:
            classify_grouped_pair(bad)
        except KeyError:
            pass
        else:
            raise AssertionError(f"should reject {bad}")


def test_key_classification():
    assert is_grouped_expert_key("model.layers.0.experts.gate_up_proj")
    assert is_grouped_expert_key("x.local_experts.fused")
    assert not is_grouped_expert_key("model.layers.0.experts.3.gate_proj.weight")
    assert not is_grouped_expert_key("model.layers.0.mlp.gate_proj.weight")
    assert not is_grouped_expert_key("model.layers.0.router.proj.weight")
    assert not is_grouped_expert_key("model.layers.0.shared_experts.gate_up_proj")
    assert GROUPED_KEY_RES  # single-sourced patterns exist


def test_grouped_container_attr():
    fused, _ = _synthetic()

    class Grouped(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(fused)
            self.down_proj = nn.Parameter(fused[..., :5].clone())
            self.act_fn = nn.GELU()

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Grouped()
            self.mlp = nn.Linear(4, 4)

    assert grouped_container_attr(Layer()) == "experts"

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(4, 4)

    assert grouped_container_attr(Plain()) is None

    class Listed(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    assert grouped_container_attr(Listed()) is None  # representation 1: not ours
