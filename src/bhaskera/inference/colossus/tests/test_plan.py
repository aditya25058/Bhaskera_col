"""Unit tests for plan rendering + handles-based description (plan CLI).

Fake dicts verify table output for feasible/infeasible cases; a stub
handles object verifies describe-without-directory. CPU-only.
"""
from __future__ import annotations

from bhaskera.inference.colossus.feasibility import plan, render_table
from bhaskera.inference.colossus.inspector import describe_model


def _hw(hbm=93.1):
    return {"gpus": [{"name": "H100", "total_gb": hbm}],
            "ram_gb": {"available_gb": 400.0}, "pcie_gbs": 51.0,
            "dram_gbs": 92.0, "cuda_available": True, "cpu": {"bf16": True}}


def _model():
    return {"model": {"name": "fake"},
            "moe": {"routed_per_layer": 160, "shared": 2, "top_k": 6,
                    "moe_layers": 60, "hidden": 5120, "intermediate": 1536},
            "weights": {"total_gb": 439.0, "resident_gb": 24.0,
                        "routed_gb": 415.0, "dtype": "BF16"},
            "capabilities": {}}


def test_render_feasible_table():
    r = plan(_model(), _hw(), {"batch": 1, "gen_tokens": 16, "shared": True})
    t = render_table(r, _model(), _hw())
    assert "Model" in t and "Hardware" in t and "Plans" in t
    assert "Recommended:" in t and "--offload-tier" in t


def test_render_infeasible_explains():
    m = _model()
    r = plan(m, _hw(hbm=4.0), {"batch": 1, "gen_tokens": 16})
    t = render_table(r, m, _hw(hbm=4.0))
    assert "No feasible" in t and "Reason" in t


class StubHandles:
    def __init__(self, keys, shapes):
        self.weight_map = {k: "s0" for k in keys}
        self._shapes = shapes

    def shards(self):
        return ["s0"]

    def header(self, key):
        n = 1
        for d in self._shapes[key]:
            n *= d
        return {"data_offsets": [0, n * 2], "dtype": "BF16",
                "shape": list(self._shapes[key])}


def test_describe_handles_without_directory(tmp_path):
    keys = ["model.embed_tokens.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight"]
    shapes = {"model.embed_tokens.weight": (16, 8),
              "model.layers.0.mlp.experts.0.gate_proj.weight": (6, 8),
              "model.layers.0.mlp.shared_experts.gate_proj.weight": (6, 8)}
    h = StubHandles(keys, shapes)

    class P:
        is_moe = True
        model_type = "fake"
        num_experts = 1
        num_shared_experts = 1
        experts_per_token = 1
        num_hidden_layers = 1
        has_aux_loss = False

    d = describe_model("", P(), name="stub", handles=h)
    assert d["weights"]["tensors"] == 3
    assert d["weights"]["hidden"] == 8
    assert d["weights"]["expert_tensors"] == 1
    assert d["moe"]["routed_per_layer"] == 1
