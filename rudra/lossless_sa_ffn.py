"""L2 correctness gate: dense vs SA-FFN must be bit-identical (synthetic MoE).

Runs on Rudra via:  sbatch rudra/lossless_sa_ffn.sbatch   (logs: 2_group/)
Asserts per expert split: L_inf == 0, cosine == 1.
Geometry mirrors Qwen3-30B-A3B FFN slices: H=2048, I=768.
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch

    from bhaskera.inference.colossus import (
        dense_expert_forward,
        sa_expert_forward,
        verify_lossless,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s torch=%s" % (device, torch.__version__))
    g = torch.Generator().manual_seed(7)
    H, I = 2048, 768
    Wg = torch.randn(I, H, generator=g, dtype=torch.float32).to(device)
    Wu = torch.randn(I, H, generator=g, dtype=torch.float32).to(device)
    Wd = torch.randn(H, I, generator=g, dtype=torch.float32).to(device)
    x = torch.randn(H, generator=g, dtype=torch.float32).to(device)

    # Half cached / half missed split.
    cached = list(range(0, I // 2))
    missed = list(range(I // 2, I))
    y_dense = dense_expert_forward(x, Wg, Wu, Wd)
    _, _, y_sa = sa_expert_forward(x, cached, missed, Wg, Wu, Wd)
    linf, l1, cos = verify_lossless(y_dense, y_sa)
    print("half-split: Linf=%g L1=%g cos=%.12f" % (linf, l1, cos))
    assert linf == 0.0, linf
    assert cos == 1.0, cos

    # Degenerate splits: all-cached and all-missed.
    _, _, y_all_c = sa_expert_forward(x, list(range(I)), [], Wg, Wu, Wd)
    _, _, y_all_m = sa_expert_forward(x, [], list(range(I)), Wg, Wu, Wd)
    for name, y in (("all-cached", y_all_c), ("all-missed", y_all_m)):
        linf, _, cos = verify_lossless(y_dense, y)
        print("%s: Linf=%g cos=%.12f" % (name, linf, cos))
        assert linf == 0.0 and cos == 1.0

    # Batched tokens.
    xb = torch.randn(4, H, generator=g, dtype=torch.float32).to(device)
    yd = dense_expert_forward(xb, Wg, Wu, Wd)
    _, _, ys = sa_expert_forward(xb, cached, missed, Wg, Wu, Wd)
    linf, _, cos = verify_lossless(yd, ys)
    print("batch4: Linf=%g cos=%.12f" % (linf, cos))
    assert linf == 0.0 and cos == 1.0

    print("LOSSLESS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
