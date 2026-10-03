"""Unit tests for huge-model loading + routed partition (chunk 2).

Builds a tiny fake sharded safetensors model on disk, then verifies:
mmap views are exact, partition classification matches DeepSeek-style and
Mixtral-style layouts, shared experts stay resident, conventions track
bhaskera.introspect. CPU-only.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from bhaskera.inference.colossus.loading import (
    ShardHandles,
    is_routed_expert_key,
    materialize,
    split_routed,
)


@pytest.fixture()
def fake_model_dir(tmp_path):
    from safetensors.torch import save_file
    d = tmp_path / "model"
    d.mkdir()
    g = torch.randn(4, 8, dtype=torch.bfloat16)
    wm, shard0, shard1 = {}, {}, {}
    # NOTE: distinct clones per entry (safetensors rejects shared storage).
    shard0["model.layers.0.self_attn.q_proj.weight"] = g.clone()
    shard0["model.layers.1.mlp.experts.0.gate_proj.weight"] = g.clone()
    shard0["model.layers.1.mlp.experts.3.up_proj.weight"] = g.clone()
    shard0["model.layers.1.mlp.shared_experts.gate_proj.weight"] = g.clone()
    shard1["model.layers.1.mlp.experts.1.down_proj.weight"] = g.clone()
    shard1["model.layers.2.mlp.local_experts.0.gate_proj.weight"] = g.clone()
    shard1["model.embed_tokens.weight"] = g.clone()
    shard1["a.weight"] = torch.randn(4, 8, dtype=torch.bfloat16)
    save_file(shard0, str(d / "model-00001-of-00002.safetensors"))
    save_file(shard1, str(d / "model-00002-of-00002.safetensors"))
    for k in list(shard0) + list(shard1):
        fn = "model-00001-of-00002.safetensors" if k in shard0 else "model-00002-of-00002.safetensors"
        wm[k] = fn
    with open(d / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": wm}, f)
    return str(d), g


def test_partition_classification():
    assert is_routed_expert_key("model.layers.1.mlp.experts.0.gate_proj.weight")
    assert is_routed_expert_key("model.layers.1.mlp.experts.159.down_proj.weight")
    assert is_routed_expert_key("model.layers.1.mlp.local_experts.0.gate_proj.weight")
    assert not is_routed_expert_key("model.layers.1.mlp.shared_experts.gate_proj.weight")
    assert not is_routed_expert_key("model.layers.0.self_attn.q_proj.weight")
    assert not is_routed_expert_key("model.embed_tokens.weight")
    assert not is_routed_expert_key("lm_head.weight")


def test_split_routed(fake_model_dir):
    d, _ = fake_model_dir
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    resident, routed = split_routed(wm)
    assert len(routed) == 4 and len(resident) == 4
    assert all("shared_experts" not in k for k in routed)


def test_shard_views_exact(fake_model_dir):
    from safetensors import safe_open
    d, g = fake_model_dir
    h = ShardHandles.open(d)
    assert len(h.shards()) == 2
    key = "model.layers.1.mlp.experts.0.gate_proj.weight"
    t = h.get_tensor(key)
    with safe_open(os.path.join(d, "model-00001-of-00002.safetensors"),
                   framework="pt", device="cpu") as fh:
        ref = fh.get_tensor(key)
    assert torch.equal(t.cpu(), ref.cpu())
    assert t.shape == (4, 8)


def test_materialize_into_empty_model(fake_model_dir):
    d, g = fake_model_dir
    h = ShardHandles.open(d)
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4, bias=False)

    m = Tiny().to("meta")
    nbytes = materialize(m, ["a.weight"], h, torch.device("cpu"))
    assert nbytes == 4 * 8 * 2
    got = m.a.weight.detach().cpu()
    want = h.get_tensor("a.weight")
    assert torch.equal(got, want)


def test_set_module_tensor_prefix_tolerant():
    from torch import nn

    from bhaskera.inference.colossus.loading import set_module_tensor

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    m = Outer().to("meta")
    v = torch.randn(4, 4)
    # index-style key without the top prefix resolves via .model container
    set_module_tensor(m, "layers.0.weight", v, device=torch.device("cpu"),
                      dtype=torch.float32)
    assert torch.equal(m.model.layers[0].weight.cpu(), v)


def test_conventions_in_sync():
    import bhaskera.introspect as intro
    from bhaskera.inference.colossus import loading as L
    assert set(intro._EXPERT_LEAF_NAMES) == {"experts", "local_experts", "routed_experts"}
    assert set(intro._SHARED_EXPERT_HINTS) == set(L.SHARED_HINTS)
