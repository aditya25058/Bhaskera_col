"""Config schema tests for tiered serving (chunk 3).

load_config parses the example YAML; as_dict/from_dict round-trips the new
fields (upstream invariant: Ray serializes configs to every worker).
CPU-only, no GPU, no model.
"""
from __future__ import annotations

import os

from bhaskera.config import load_config

YAML = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                    "configs", "inference_colossus_tiered.yaml")


def _norm(p):
    return os.path.normpath(p)


def test_tiered_yaml_parses():
    cfg = load_config(_norm(YAML))
    col = cfg.inference.colossus
    assert col.placement == "slots"
    assert col.capacity == 12
    assert col.exactness_mode == "bitwise"
    assert col.prefill_chunk == 0
    assert col.prefetch_topk == 8
    assert col.cpu_threads == 6


def test_colossus_round_trip():
    cfg = load_config(_norm(YAML))
    d = cfg.as_dict()
    assert d["inference"]["colossus"]["placement"] == "slots"
    cfg2 = cfg.from_dict(d)
    assert cfg2.inference.colossus.capacity == 12
    assert cfg2.inference.colossus.exactness_mode == "bitwise"


def test_defaults_off():
    from bhaskera.config import Config
    cfg = Config()
    assert cfg.inference.colossus.placement == "off"
    assert cfg.inference.colossus.exactness_mode == "bitwise"
