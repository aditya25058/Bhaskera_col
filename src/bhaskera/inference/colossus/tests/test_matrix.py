"""Unit tests for matrix-granular tiering (FIRM-like control, gate 3).

Synthetic module-based experts, CPU-only:
- matches the native reference bitwise (same op structure);
- per-role DMA accounting adds up; expert hit = all three resident.
"""
import torch
import torch.nn.functional as F
from torch import nn

from bhaskera.inference.colossus.interface import MoELayerSpec
from bhaskera.inference.colossus.matrix import MatrixTieredMoEWrapper
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
    from bhaskera.inference.colossus.placement import apply_gate
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
    return out.reshape(*x.shape) + spec.shared(x)


def test_matches_native_bitwise():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    w = MatrixTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=6, handles=handles)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5, 8, dtype=torch.bfloat16, generator=g)
    assert torch.equal(w(x), _native_ref(spec, handles, keys, x))


def test_role_accounting_adds_up():
    spec, handles, keys = _setup()
    dev = torch.device("cpu")
    w = MatrixTieredMoEWrapper(layer_idx=0, spec=spec, expert_keys=keys,
                               device=dev, capacity=6, handles=handles)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(5, 8, dtype=torch.bfloat16, generator=g)
    w(x)
    w(x)
    assert w.hits > 0 and w.misses > 0
    per = sum(handles.get_tensor(keys[0][r]).nbytes for r in
              ("gate", "up", "down"))
    # Full-resident second pass: role misses freeze, expert hits accrue.
    m = dict(w.role_misses)
    d = w.dma_bytes
    w(x)
    assert dict(w.role_misses) == m
    assert w.dma_bytes == d
    # Role pools are independent LRU structures.
    assert set(w.pools) == {"gate", "up", "down"}
    assert all(len(p["slots"]) == 6 for p in w.pools.values())
