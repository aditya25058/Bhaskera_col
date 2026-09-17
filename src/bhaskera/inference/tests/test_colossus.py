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

    def __init__(self, config=None, hidden_size: int = 128, intermediate_size: int = 64):
        if isinstance(config, int):
            hidden_size = config
        elif hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        super().__init__()
        self.config = type("Config", (), {"num_shared_experts": None, "hidden_size": hidden_size})()
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SyntheticRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(hidden_size, num_experts, bias=False)
        self.weight = self.linear.weight

    def forward(self, x: torch.Tensor):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        logits = self.linear(x_2d.float())
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_indices, topk_weights, logits


class SyntheticMoEBlock(nn.Module):
    """Synthetic MoE block matching Param2MoESparseMoeBlock layout."""

    def __init__(self, num_experts: int = 8, hidden_size: int = 128, intermediate_size: int = 64, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_experts_per_tok = top_k
        self.config = type("Config", (), {"num_shared_experts": None})()
        self.gate = SyntheticRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList([
            SyntheticSwiGLUExpert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def moe_infer(self, x_2d: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        B_tot, _ = x_2d.shape
        y = torch.zeros_like(x_2d)
        for b in range(B_tot):
            for k in range(self.top_k):
                exp_idx = topk_idx[b, k].item()
                weight = topk_weight[b, k]
                y[b] += weight * self.experts[exp_idx](x_2d[b:b+1]).squeeze(0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        topk_idx, topk_weights, _ = self.gate(x_2d)
        y = self.moe_infer(x_2d, topk_idx, topk_weights)
        return y.reshape(orig_shape)


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
    topk_indices, topk_weights, _ = moe.gate(x.float())

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


def test_expert_offload_manager_cpu_migration():
    """Verify ExpertOffloadManager moves cold experts to CPU and restores them."""
    from bhaskera.inference.colossus.offload import ExpertOffloadManager

    H, I, E = 128, 64, 8

    # Build a minimal model structure: model.layers[L].mlp.experts
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            layers = nn.ModuleList()
            for _ in range(3):
                layer = nn.Module()
                mlp = nn.Module()
                experts = nn.ModuleList([
                    SyntheticSwiGLUExpert(H, I) for _ in range(E)
                ])
                mlp.experts = experts
                mlp.gate = nn.Linear(H, E, bias=False)
                layer.mlp = mlp
                layers.append(layer)
            self.model.layers = layers

    fake_model = FakeModel()
    # All experts start on CPU (since we don't have CUDA in tests)
    device = "cpu"

    mgr = ExpertOffloadManager(hot_expert_topk=2, device=device)

    # Define hot_map: layers 1 and 2 have hot experts [0, 1]
    hot_map = {
        1: [0, 1],
        2: [3, 5],
    }

    # Offload cold experts
    result = mgr.offload_cold_experts(fake_model, hot_map)

    assert result["status"] == "offloaded"
    assert result["experts_offloaded"] == (E - 2) * 2  # 6 cold per layer × 2 layers
    assert result["experts_kept"] == 2 * 2  # 2 hot per layer × 2 layers
    assert result["bytes_offloaded"] > 0
    assert result["vram_saved_gb"] >= 0

    # Verify VRAM savings report
    savings = mgr.vram_savings()
    assert savings["mode"] == "active"
    assert savings["offload_ratio"] > 0.5  # Most experts should be offloaded
    assert savings["layers_affected"] == 2

    # Restore all
    mgr.restore_all_experts(fake_model)
    savings_after = mgr.vram_savings()
    assert savings_after["bytes_offloaded"] == 0


def test_hook_active_mode_stats():
    """Verify that hook.stats() reports active mode when offload_mgr is set."""
    from bhaskera.inference.colossus.offload import ExpertOffloadManager

    pred = ZSSRPredictor(top_k_experts=4, top_cols=16, num_col_experts=4)
    directory = ColumnDirectory(num_experts=8, num_columns=64, capacity_per_expert=32)
    hook = ColossusMoEHook(predictor=pred, directory=directory)

    # Without offload manager -> shadow mode
    assert hook.stats()["mode"] == "shadow"

    # With offload manager -> active mode
    hook._offload_mgr = ExpertOffloadManager(hot_expert_topk=4, device="cpu")
    stats = hook.stats()
    assert stats["mode"] == "active"
    assert "offload" in stats


def test_predict_lookahead_and_confidence_gating():
    """Verify multi-layer lookahead (L+1...L+4) and confidence gating."""
    H, E, K = 128, 16, 4
    g = torch.Generator().manual_seed(101)
    pred = ZSSRPredictor(top_k_experts=K, top_cols=16, num_col_experts=K)
    for layer_idx in range(1, 6):
        pred.router[layer_idx] = torch.randn(E, H, generator=g)

    h = torch.randn(1, 1, H, generator=g)

    # 1. Standard lookahead up to depth 4
    plan = pred.predict_lookahead(h, current_layer=1, max_depth=4, confidence_threshold=0.0)
    assert set(plan.keys()) == {2, 3, 4, 5}
    for l_idx, experts in plan.items():
        assert len(experts) == K

    # 2. Confidence gating: very high threshold suppresses speculation
    suppressed_plan = pred.predict_lookahead(h, current_layer=1, max_depth=4, confidence_threshold=0.99)
    assert len(suppressed_plan) == 0

    # 3. Confidence gating: threshold below uniform (1/16 = 0.0625) lets predictions through
    gated_plan = pred.predict_lookahead(h, current_layer=1, max_depth=4, confidence_threshold=0.05)
    assert len(gated_plan) > 0


def test_adetr_buffer_and_saffn_expert_lossless():
    """Verify column-level SA-FFN decomposition y = y_cached + y_missed with ADETR buffer."""
    from bhaskera.inference.colossus.saffn import ADETRBuffer, SA_FFN_Expert

    H, I = 128, 64
    g = torch.Generator().manual_seed(202)
    Wg = torch.randn(I, H, generator=g, dtype=torch.float32)
    Wu = torch.randn(I, H, generator=g, dtype=torch.float32)
    Wd = torch.randn(H, I, generator=g, dtype=torch.float32)

    buf = ADETRBuffer(hidden_size=H, intermediate_size=I, hot_ratio=0.25, dtype=torch.float32, device=torch.device("cpu"))
    buf.init_from_weights(Wg, Wu, Wd)
    buf.load_cold_columns(non_blocking=False)

    expert = SA_FFN_Expert(buf)
    x = torch.randn(2, 4, H, generator=g, dtype=torch.float32)

    y_decomposed = expert(x)
    y_monolithic = F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)

    linf, l1, cos = verify_lossless(y_monolithic, y_decomposed)
    rel = linf / (y_monolithic.abs().max().item() + 1e-12)
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert rel < 1e-5


def test_dynamic_moe_wrapper_column_partitioning_lossless():
    """Verify DynamicMoELayerWrapper with column partitioning produces exact output."""
    from bhaskera.inference.colossus.dynamic_cache import DynamicMoELayerWrapper

    H, I, E, K = 128, 64, 8, 2
    moe = SyntheticMoEBlock(num_experts=E, hidden_size=H, intermediate_size=I, top_k=K)
    moe.eval()

    x_dec = torch.randn(1, 1, H)
    y_ref_dec = moe(x_dec)

    x_pref = torch.randn(2, 4, H)
    y_ref_pref = moe(x_pref)

    wrapper = DynamicMoELayerWrapper(
        layer_idx=1,
        moe_block=moe,
        capacity=4,
        device=torch.device("cpu"),
        missing_col_ratio=0.25,
    )

    # 1. Decoding step (seq_len=1): bitwise exact through moe_infer
    y_wrap_dec, _ = wrapper(x_dec)
    assert torch.equal(y_ref_dec, y_wrap_dec)

    # 2. Prefill step (seq_len > 1): allclose
    y_wrap_pref, _ = wrapper(x_pref)
    assert torch.allclose(y_ref_pref, y_wrap_pref, atol=1e-5)

    stats = wrapper.get_stats()
    assert "missing_col_mean" in stats



