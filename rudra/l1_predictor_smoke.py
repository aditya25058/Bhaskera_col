"""L1 predictor smoke test for Rudra A100 (no model download needed).

Uses seeded synthetic MoE weights (Qwen3-30B-A3B geometry: H=2048, I=768,
E=128) to verify, on GPU where available:

1. ZSSRPredictor expert ranking + SwiGLU-energy column plans run.
2. columns_batched == columns_loop (bit-identical, CPU and CUDA).
3. Batched scorer latency vs the 136us MHA overlap window (H100 number;
   re-measured here on A100).
4. ColumnDirectory lookup/commit/resident-ratio semantics.
5. Fixed-packet budgets sum to 325 (tiered_fwd) / 320 (uniform40).

Run: sbatch rudra/l1_smoke.sbatch   (logs to 2_group/results/)
"""
from __future__ import annotations

import json
import sys
import time


def main() -> int:
    import torch

    from bhaskera.inference.colossus import (
        ColumnDirectory,
        ZSSRPredictor,
        columns_batched,
        columns_loop,
        plan_fixed_packets,
        stack_expert_weights,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}")
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    H, I, E, K = 2048, 768, 128, 8
    g = torch.Generator().manual_seed(0)

    # Synthetic router + gate/up weights for layer 1.
    router = torch.randn(E, H, generator=g)
    gate = {e: torch.randn(I, H, generator=g) for e in range(K)}
    up = {e: torch.randn(I, H, generator=g) for e in range(K)}

    pred = ZSSRPredictor(top_k_experts=8, top_cols=50, num_col_experts=8)
    pred.router[1] = router
    pred.gate_w[1] = gate
    pred.up_w[1] = up

    h = torch.randn(H, generator=g)
    out = pred.predict(h, 1)
    assert len(out["experts_top8"]) == 8, out["experts_top8"]
    assert len(out["columns"]) == 8, out["columns"]
    print(f"experts_top8={out['experts_top8']}")

    # Batched == loop (CPU).
    experts = out["ranking"][:8]
    budgets = [50] * 5 + [25] * 3
    ref = columns_loop(h, gate, up, experts, budgets)
    Wg8, Wu8 = stack_expert_weights(gate, up, experts)
    got = columns_batched(h, Wg8, Wu8, budgets)
    assert got == ref, "batched/loop mismatch on CPU"
    print("batched==loop on CPU")

    result = {"device": device, "experts_top8": out["experts_top8"]}
    if device == "cuda":
        h_c = h.cuda()
        Wg_c = Wg8.cuda()
        Wu_c = Wu8.cuda()
        got_c = columns_batched(h_c, Wg_c, Wu_c, budgets)
        assert got_c == ref, "batched/loop mismatch on CUDA"
        print("batched==loop on CUDA")
        # Latency of the column stage (batched GEMM + topk).
        for _ in range(10):
            columns_batched(h_c, Wg_c, Wu_c, budgets)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            columns_batched(h_c, Wg_c, Wu_c, budgets)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / reps * 1e6
        result["columns_batched_us"] = round(us, 1)
        print(f"columns_batched={us:.1f}us (MHA window ref 136us)")

    # Directory semantics.
    plan = plan_fixed_packets(out["ranking"], preset="tiered_fwd")
    assert sum(plan.values()) == 325, plan
    assert sum(plan_fixed_packets(out["ranking"], preset="uniform40").values()) == 320
    dd = ColumnDirectory(num_experts=E, num_columns=I)
    _, misses = dd.lookup(plan)
    assert sum(len(v) for v in misses.values()) == 325
    dd.commit(plan)
    assert dd.predicted_resident_ratio(plan) == 1.0
    print("directory + packets OK")

    print(json.dumps(result))
    print("L1_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
