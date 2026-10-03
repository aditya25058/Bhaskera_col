"""Unit tests for the canonical model description (Phase 2 schema).

Fake sharded index + stub profile verify: byte accounting, hidden/inter
derivation, capability flags, and JSON round-trip. CPU-only.
"""
from __future__ import annotations

import json

import torch

from bhaskera.inference.colossus.inspector import (
    describe_model,
    inspect_weights,
    write_inspect_json,
)


class FakeProfile:
    is_moe = True
    model_type = "fake_moe"
    num_experts = 4
    num_shared_experts = 1
    experts_per_token = 2
    num_hidden_layers = 3
    has_aux_loss = True


def _fake_dir(tmp_path, tag="m"):
    from safetensors.torch import save_file
    d = tmp_path / tag
    d.mkdir(exist_ok=True)
    wm, sh = {}, {}
    # embed [vocab 16, H 8]; experts gate [I 6, H 8], down [H 8, I 6]
    sh["model.embed_tokens.weight"] = torch.randn(16, 8, dtype=torch.bfloat16)
    for e in range(2):
        sh[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = \
            torch.randn(6, 8, dtype=torch.bfloat16).clone()
        sh[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = \
            torch.randn(8, 6, dtype=torch.bfloat16).clone()
    sh["model.layers.0.mlp.shared_experts.gate_proj.weight"] = \
        torch.randn(6, 8, dtype=torch.bfloat16).clone()
    save_file({k: v.clone() for k, v in sh.items()}, str(d / "model-00001-of-00001.safetensors"))
    for k in sh:
        wm[k] = "model-00001-of-00001.safetensors"
    with open(d / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": wm}, f)
    return str(d)


def test_inspect_weights(tmp_path):
    w = inspect_weights(_fake_dir(tmp_path))
    # total = embed(256) + 2*(gate 96 + down 96) + shared 96 = 736 B
    assert w["total_gb"] == 736 / (1024 ** 3)
    # routed = 2 experts x (96+96) = 384
    assert w["routed_gb"] == 384 / (1024 ** 3)
    assert w["resident_gb"] == (736 - 384) / (1024 ** 3)
    assert w["dtype"] == "BF16" and w["shards"] == 1
    assert w["hidden"] == 8 and w["intermediate"] == 6
    assert w["expert_tensors"] == 4


def test_describe_model(tmp_path):
    d = describe_model(_fake_dir(tmp_path), FakeProfile(), name="fake")
    assert d["model"]["name"] == "fake"
    assert d["moe"]["routed_per_layer"] == 4
    assert d["moe"]["top_k"] == 2 and d["moe"]["hidden"] == 8
    assert d["capabilities"]["tiered_execution"] is True
    assert d["capabilities"]["cpu_placement"] is True
    assert d["capabilities"]["reasons"] == []


def test_describe_dense_and_noprofile(tmp_path):
    d = describe_model(_fake_dir(tmp_path, "a"), None)
    assert d["moe"] is None
    assert d["capabilities"]["tiered_execution"] is False

    class Dense:
        is_moe = False
        model_type = "dense"
        num_experts = 0
        num_shared_experts = 0
        experts_per_token = 0
        num_hidden_layers = 12
        has_aux_loss = False

    d2 = describe_model(_fake_dir(tmp_path, "b"), Dense())
    assert d2["capabilities"]["tiered_execution"] is False
    assert any("not MoE" in r for r in d2["capabilities"]["reasons"])


def test_write_roundtrip(tmp_path):
    p = str(tmp_path / "desc.json")
    d = write_inspect_json(_fake_dir(tmp_path, "c"), p, FakeProfile())
    assert json.load(open(p)) == d
