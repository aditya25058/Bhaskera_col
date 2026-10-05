"""End-to-end serve test on a tiny fake causal MoE model (chunk 2c).

Exercises prepare_model + serve_huge_moe (prefill, lockstep decode, ledger)
with zero GPU, zero downloads: fake weights stay in-module (materialize is a
no-op over empty resident keys), expert weights come from a stub table.
"""
from __future__ import annotations

import types

import torch
from torch import nn

from bhaskera.inference.colossus.placement import TieredMoEWrapper
from bhaskera.inference.colossus.serve import prepare_model, serve_huge_moe
from bhaskera.inference.colossus.tests.test_interface import FakeDeepSeekBlock
from bhaskera.inference.colossus.tests.test_placement import DictHandles


class FakeDecoderLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block

    def forward(self, x):
        return self.mlp(x) + x


class FakeCausalMoE(nn.Module):
    def __init__(self, n_layers=2, vocab=32, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [FakeDecoderLayer(FakeDeepSeekBlock()) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.ln(x))
        return types.SimpleNamespace(logits=logits, past_key_values=None)


class FakeProfile:
    decoder_layer_cls = FakeDecoderLayer


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, padding=True, return_tensors="pt"):
        ids = [[max(2, (ord(c) % 30) + 2) for c in t[:8]] for t in texts]
        n = max(len(r) for r in ids)
        ids = [r + [0] * (n - len(r)) for r in ids]
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor([[1] * len(r) for r in ids])}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(97 + (int(i) % 26)) for i in row) for row in ids]


def _stub_handles(model):
    table = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".experts." in name:
            table[name + ".weight"] = mod.weight.detach().clone()
    stubs = DictHandles(table)
    stubs.weight_map = dict(table)
    return stubs


def test_serve_fake_model_slots():
    torch.manual_seed(0)
    model = FakeCausalMoE()
    tok = FakeTokenizer()
    handles = _stub_handles(model)
    res = serve_huge_moe(model, tok, FakeProfile(), handles,
                         torch.device("cpu"), ["hi", "yo"],
                         max_new_tokens=3, capacity=8, placement="slots")
    assert len(res["texts"]) == 2 and all(isinstance(t, str) for t in res["texts"])
    assert res["total_misses"] > 0
    assert res["batch_decode_tps"] >= 0.0
    assert res["peak_vram_gb"] == 0.0


def test_serve_fake_model_cpu():
    torch.manual_seed(0)
    model = FakeCausalMoE()
    tok = FakeTokenizer()
    handles = _stub_handles(model)
    res = serve_huge_moe(model, tok, FakeProfile(), handles,
                         torch.device("cpu"), ["hi"],
                         max_new_tokens=2, capacity=8, placement="cpu")
    assert len(res["texts"]) == 1


def test_prepare_wraps_all_layers():
    model = FakeCausalMoE()
    handles = _stub_handles(model)
    wrappers = prepare_model(model, FakeProfile(), handles,
                             torch.device("cpu"), capacity=4)
    assert len(wrappers) == 2
    assert all(isinstance(model.layers[i].mlp, TieredMoEWrapper) for i in (0, 1))


def test_prefill_token_appended_not_dropped(tmp_path):
    """Regression: prefill's next_token is generated token #1.

    The loop used to overwrite it before appending, dropping position 1
    from every continuation (and shifting all teacher comparisons).
    """
    import json

    torch.manual_seed(0)
    model = FakeCausalMoE()
    tok = FakeTokenizer()
    tok.eos_token_id = -1  # no early stop: expect exactly max_new_tokens
    handles = _stub_handles(model)
    ids_path = str(tmp_path / "ids.json")
    res = serve_huge_moe(model, tok, FakeProfile(), handles,
                         torch.device("cpu"), ["hi"],
                         max_new_tokens=4, capacity=8,
                         dump_ids=ids_path)
    assert res["generated_tokens"] == 4
    cont = json.load(open(ids_path))["continuation_ids"]
    assert isinstance(cont, list) and len(cont) == 1  # batched
    assert len(cont[0]) == 4, f"prefill token dropped: {cont}"
    assert len(res["texts"]) == 1


def test_teacher_audit_records(tmp_path):
    model = FakeCausalMoE()
    tok = FakeTokenizer()
    handles = _stub_handles(model)
    audit_path = str(tmp_path / "audit.json")
    teacher = [7] * 8
    res = serve_huge_moe(model, tok, FakeProfile(), handles,
                         torch.device("cpu"), ["hi"],
                         max_new_tokens=6, capacity=8,
                         teacher_tokens=teacher, audit_logits=audit_path)
    import json
    audit = json.load(open(audit_path))["audit"]
    assert len(audit) == 6
    for e in audit:
        assert set(e) == {"pos", "own", "ref", "match", "margin", "own_bf16"}
        assert e["ref"] == 7 and e["match"] == (e["own"] == 7)
    assert res["flip_audit"] == audit


def test_cache_compat_mask_patch_gated_by_family():
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter as AMC
    from bhaskera.inference.colossus.serve import install_cache_compat
    orig = AMC.to_causal_4d
    try:
        install_cache_compat("deepseek_v2")
        assert getattr(AMC.to_causal_4d, "_colossus_patched", False) is True
        install_cache_compat("deepseek_v2")  # re-entrant: no stacking
        install_cache_compat("param2moe")
        assert getattr(AMC.to_causal_4d, "_colossus_patched", False) is False
        install_cache_compat(None)
        assert getattr(AMC.to_causal_4d, "_colossus_patched", False) is False
    finally:
        AMC.to_causal_4d = orig
