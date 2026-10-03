"""Unit tests for the feasibility engine (pure arithmetic, CPU-only).

Fake model/hardware dicts verify: small-fits ranking, huge-model fallback,
streaming floor always present, unknown arch refused with reasons.
"""
from __future__ import annotations

from bhaskera.inference.colossus.feasibility import plan


def _moe(routed=160, topk=6, layers=60, resident=24.0, routed_gb=415.0):
    return {"model": {"name": "fake"},
            "moe": {"routed_per_layer": routed, "shared": 2, "top_k": topk,
                    "moe_layers": layers, "hidden": 5120, "intermediate": 1536},
            "weights": {"total_gb": resident + routed_gb, "resident_gb": resident,
                        "routed_gb": routed_gb, "dtype": "BF16"},
            "capabilities": {}}


def _hw(hbm=93.1, ram=400.0, pcie=51.0, dram=92.0, cuda=True):
    return {"gpus": [{"total_gb": hbm}] if cuda else [],
            "ram_gb": {"available_gb": ram}, "pcie_gbs": pcie,
            "dram_gbs": dram, "cuda_available": cuda,
            "cpu": {"bf16": True}}


def test_small_model_slots_recommended():
    m = _moe(routed=8, topk=2, layers=12, resident=2.0, routed_gb=4.0)
    r = plan(m, _hw(), {"batch": 1, "gen_tokens": 16, "shared": True})
    assert r["feasible"] and r["recommended"]["placement"] == "slots"
    assert r["recommended"]["fits"]


def test_huge_model_cpu_or_streaming():
    m = _moe()
    r = plan(m, _hw(hbm=24.0, ram=400.0), {"batch": 1, "gen_tokens": 16},
             fidelity="ulp1")
    assert r["feasible"]
    assert r["recommended"]["placement"] in ("slots", "cpu")
    # 24GB HBM cannot hold 24GB resident + slots: slots must fail, cpu wins
    slots = [c for c in r["candidates"] if c["placement"] == "slots" and c["capacity"] > 0]
    assert all(not c["fits"] for c in slots)


def test_streaming_floor_always_fits():
    m = _moe()
    r = plan(m, _hw(hbm=4.0, ram=400.0), {"batch": 1, "gen_tokens": 16})
    caps = [c for c in r["candidates"] if c["capacity"] == 0]
    assert caps and all(c["fits"] for c in caps)


def test_unknown_arch_refused_with_reasons():
    r = plan({"moe": None, "weights": {}}, _hw(), {"batch": 1})
    assert not r["feasible"] and r["reasons"]


def test_bitwise_excludes_cpu():
    m = _moe()
    r = plan(m, _hw(), {"batch": 1})
    assert all(c["placement"] == "slots" for c in r["candidates"])


def test_batch_scales_estimate():
    m = _moe()
    hw = _hw()
    r1 = plan(m, hw, {"batch": 1, "gen_tokens": 16, "shared": True})
    r8 = plan(m, hw, {"batch": 8, "gen_tokens": 16, "shared": True})
    assert r8["recommended"]["steady_tps"] > r1["recommended"]["steady_tps"]
