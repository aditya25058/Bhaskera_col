"""Unit tests for COLOSSUS MoE offload, ZSSR predictor, and lossless SA-FFN."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bhaskera.inference.colossus import (
    ColumnDirectory,
    ZSSRPredictor,
    dense_expert_forward,
    plan_fixed_packets,
    sa_expert_forward,
    verify_lossless,
)
from bhaskera.inference.colossus.hook import ColossusMoEHook


class SyntheticSwiGLUExpert(nn.Module):
    """Synthetic SwiGLU expert matching Param2/DeepSeek expert architecture."""

    def __init__(self, hidden_size: int = 128, intermediate_size: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SyntheticMoEBlock(nn.Module):
    """Synthetic MoE block matching Param2MoESparseMoeBlock layout."""

    def __init__(self, num_experts: int = 8, hidden_size: int = 128, intermediate_size: int = 64, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SyntheticSwiGLUExpert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H] or [H]
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        B, H = x.shape
        logits = self.gate(x.float())
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        y = torch.zeros_like(x)
        for b in range(B):
            for k in range(self.top_k):
                exp_idx = topk_indices[b, k].item()
                weight = topk_weights[b, k]
                y[b] += weight * self.experts[exp_idx](x[b:b+1]).squeeze(0)

        return y.squeeze(0) if squeeze else y


def test_zssr_predictor_ranking_and_columns():
    H, I, E, K = 128, 64, 8, 4
    g = torch.Generator().manual_seed(42)
    router = torch.randn(E, H, generator=g)
    gate = {e: torch.randn(I, H, generator=g) for e in range(E)}
    up = {e: torch.randn(I, H, generator=g) for e in range(E)}

    pred = ZSSRPredictor(top_k_experts=K, top_cols=16, num_col_experts=K)
    pred.router[1] = router
    pred.gate_w[1] = gate
    pred.up_w[1] = up

    h = torch.randn(H, generator=g)
    out = pred.predict(h, 1)

    assert len(out["experts_top8"]) == K
    assert len(out["ranking"]) == E
    assert len(out["columns"]) == K
    for exp_id in out["columns"]:
        assert len(out["columns"][exp_id]) == 16


def test_column_directory_lru_and_packets():
    num_experts = 8
    num_columns = 64
    cd = ColumnDirectory(num_experts=num_experts, num_columns=num_columns, capacity_per_expert=16)

    ranking = list(range(num_experts))
    plan = plan_fixed_packets(ranking, preset="uniform40")
    assert sum(plan.values()) == 320

    # Cold start: all lookups should be misses
    plan_cols = {e: list(range(10)) for e in range(4)}
    hits, misses = cd.lookup(plan_cols)
    assert sum(len(v) for v in hits.values()) == 0
    assert sum(len(v) for v in misses.values()) == 40

    # Commit to LRU
    cd.commit(plan_cols)
    hits_after, misses_after = cd.lookup(plan_cols)
    assert sum(len(v) for v in hits_after.values()) == 40
    assert sum(len(v) for v in misses_after.values()) == 0
    assert cd.predicted_resident_ratio(plan_cols) == 1.0


def test_sa_expert_forward_lossless():
    H, I = 128, 64
    g = torch.Generator().manual_seed(123)
    Wg = torch.randn(I, H, generator=g)
    Wu = torch.randn(I, H, generator=g)
    Wd = torch.randn(H, I, generator=g)
    x = torch.randn(H, generator=g)

    y_dense = dense_expert_forward(x, Wg, Wu, Wd)

    # 1. Half cached, half missed
    cached = list(range(0, I // 2))
    missed = list(range(I // 2, I))
    _, _, y_sa = sa_expert_forward(x, cached, missed, Wg, Wu, Wd)
    linf, l1, cos = verify_lossless(y_dense, y_sa)
    rel = linf / (y_dense.abs().max().item() + 1e-12)
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert rel < 1e-5

    # 2. All cached
    _, _, y_all_c = sa_expert_forward(x, list(range(I)), [], Wg, Wu, Wd)
    linf, _, cos = verify_lossless(y_dense, y_all_c)
    assert linf == 0.0
    assert cos == pytest.approx(1.0, abs=1e-6)

    # 3. Batched [B, H]
    xb = torch.randn(4, H, generator=g)
    yd_batch = dense_expert_forward(xb, Wg, Wu, Wd)
    _, _, ysa_batch = sa_expert_forward(xb, cached, missed, Wg, Wu, Wd)
    linf_b, _, cos_b = verify_lossless(yd_batch, ysa_batch)
    rel_b = linf_b / (yd_batch.abs().max().item() + 1e-12)
    assert cos_b == pytest.approx(1.0, abs=1e-6)
    assert rel_b < 1e-5


def test_moe_scaffolding_dense_vs_colossus():
    """Verify that routing tokens through MoE via execute_expert_sa is numerically equivalent."""
    H, I, E, K = 128, 64, 8, 2
    moe = SyntheticMoEBlock(num_experts=E, hidden_size=H, intermediate_size=I, top_k=K)
    moe.eval()

    # Build dummy predictor and directory
    pred = ZSSRPredictor(top_k_experts=K, top_cols=16, num_col_experts=K)
    directory = ColumnDirectory(num_experts=E, num_columns=I, capacity_per_expert=32)
    hook = ColossusMoEHook(predictor=pred, directory=directory)

    x = torch.randn(4, H)

    # 1. Forward dense
    y_dense = moe(x)

    # 2. Forward with COLOSSUS SA-FFN expert execution
    logits = moe.gate(x.float())
    scores = F.softmax(logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    y_colossus = torch.zeros_like(x)
    for b in range(x.shape[0]):
        for k in range(K):
            exp_idx = topk_indices[b, k].item()
            weight = topk_weights[b, k]
            # Execute through COLOSSUS split execution
            y_exp = hook.execute_expert_sa(x[b:b+1], exp_idx, moe.experts[exp_idx])
            y_colossus[b] += weight * y_exp.squeeze(0)

    linf, l1, cos = verify_lossless(y_dense, y_colossus)
    rel = linf / (y_dense.abs().max().item() + 1e-12)

    assert cos == pytest.approx(1.0, abs=1e-6)
    assert rel < 1e-5
