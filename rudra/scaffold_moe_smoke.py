"""Scaffolding test: Dense vs COLOSSUS SA-FFN on full-scale synthetic MoE.

Runs on Rudra A100 via: sbatch rudra/scaffold_moe_smoke.sbatch
Asserts: cos == 1.0, relative error < 1e-5.
Geometry: matches Param2-17B MoE (H=2048, I=2048, E=64, K=6).
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from bhaskera.inference.colossus import (
        ColumnDirectory,
        ZSSRPredictor,
        dense_expert_forward,
        sa_expert_forward,
        verify_lossless,
    )
    from bhaskera.inference.colossus.hook import ColossusMoEHook

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s torch=%s" % (device, torch.__version__))
    if device == "cuda":
        print("gpu=%s" % torch.cuda.get_device_name(0))

    # Param2 MoE layer dimensions: H=2048, I=2048, E=64, top_k=6
    H, I, E, K = 2048, 2048, 64, 6
    g = torch.Generator().manual_seed(42)

    class SwiGLUExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(H, I, bias=False)
            self.up_proj = nn.Linear(H, I, bias=False)
            self.down_proj = nn.Linear(I, H, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class SyntheticMoELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(H, E, bias=False)
            self.experts = nn.ModuleList([SwiGLUExpert() for _ in range(E)])

        def forward_dense(self, x):
            B, dim = x.shape
            logits = self.gate(x.float())
            scores = F.softmax(logits, dim=-1)
            weights, indices = torch.topk(scores, K, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)

            out = torch.zeros_like(x)
            for b in range(B):
                for k in range(K):
                    e = indices[b, k].item()
                    w = weights[b, k]
                    out[b] += w * self.experts[e](x[b:b+1]).squeeze(0)
            return out, indices, weights

    print("Building synthetic MoE layer (H=%d, I=%d, E=%d, K=%d)..." % (H, I, E, K))
    moe = SyntheticMoELayer().to(device)
    moe.eval()

    # Build COLOSSUS hook & directory
    pred = ZSSRPredictor(top_k_experts=K, top_cols=50, num_col_experts=K)
    directory = ColumnDirectory(num_experts=E, num_columns=I, capacity_per_expert=64, device=device)
    hook = ColossusMoEHook(predictor=pred, directory=directory)

    B = 4
    x = torch.randn(B, H, generator=g).to(device)

    # 1. Dense MoE forward
    t0 = time.perf_counter()
    y_dense, indices, weights = moe.forward_dense(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t_dense = (time.perf_counter() - t0) * 1e3

    # 2. COLOSSUS split execution forward
    t1 = time.perf_counter()
    y_col = torch.zeros_like(x)
    for b in range(B):
        for k in range(K):
            e = indices[b, k].item()
            w = weights[b, k]
            y_exp = hook.execute_expert_sa(x[b:b+1], e, moe.experts[e])
            y_col[b] += w * y_exp.squeeze(0)
    if device == "cuda":
        torch.cuda.synchronize()
    t_col = (time.perf_counter() - t1) * 1e3

    linf, l1, cos = verify_lossless(y_dense, y_col)
    rel = linf / (y_dense.abs().max().item() + 1e-12)

    print("Timing: dense=%.2fms colossus=%.2fms" % (t_dense, t_col))
    print("Numerics: Linf=%g L1=%g rel=%g cos=%.12f" % (linf, l1, rel, cos))
    print("Directory resident ratio: %.2f%%" % (len(directory._resident) / (E * 64) * 100))

    assert cos == 1.0 or (1.0 - cos) < 1e-6, "Cosine similarity failed: %g" % cos
    assert rel < 1e-5, "Relative error exceeds threshold: %g" % rel

    print("SCAFFOLD_MOE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
