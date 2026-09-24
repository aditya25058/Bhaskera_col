"""H1 test: exact sparsity. Top-K intermediate columns only vs full expert, bitwise.

For N random tokens x hidden states through expert L1E0 (SwiGLU):
  y_full = down(silu(gate(x)) * up(x))
  y_K    = down[:, topK] @ (silu(gate(x)[:, topK]) * up(x)[:, topK])
Report bitwise match rate + error magnitude per K in {100,75,50,25,10}%.

Ranking: intermediate-dim energy proxy = mean over calibration-style
gate/up row norms (offline, static). Any ranking suffices: H1 claims SOME
subset is exactly sufficient; we test the most favorable static one.
"""
import argparse
import torch

PROJS = ("gate_proj", "up_proj", "down_proj")


def load_expert(model_dir, layer=1, expert=0):
    from safetensors import safe_open
    target = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    for fn in sorted(__import__("os").listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = f"{model_dir}/{fn}"
        with safe_open(path, framework="pt", device="cpu") as f:
            if target in f.keys():
                pfx = f"model.layers.{layer}.mlp.experts.{expert}."
                return {k: f.get_tensor(pfx + k + ".weight") for k in PROJS}
    raise RuntimeError("expert not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(6)
    w = load_expert(args.model)
    I = w["gate_proj"].shape[0]  # intermediate dim
    H = w["gate_proj"].shape[1]
    print(f"[H1] intermediate={I} hidden={H}")

    # static energy ranking over intermediate dim (row norms of gate/up)
    with torch.no_grad():
        e = (w["gate_proj"].float().pow(2).sum(1) + w["up_proj"].float().pow(2).sum(1))
        order = torch.argsort(e, descending=True)

    g = torch.Generator().manual_seed(args.seed)
    xs = [torch.randn(1, H, dtype=torch.bfloat16, generator=g) for _ in range(args.n)]
    with torch.no_grad():
        refs = [torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, w["gate_proj"])) *
            torch.nn.functional.linear(x, w["up_proj"]), w["down_proj"]) for x in xs]

    for frac in (1.00, 0.75, 0.50, 0.25, 0.10):
        k = max(1, int(I * frac))
        topk = order[:k]
        wg = w["gate_proj"][topk]
        wu = w["up_proj"][topk]
        wd = w["down_proj"][:, topk]
        n_eq, maxdiff = 0, 0.0
        with torch.no_grad():
            for x, r in zip(xs, refs):
                y = torch.nn.functional.linear(
                    torch.nn.functional.silu(torch.nn.functional.linear(x, wg)) *
                    torch.nn.functional.linear(x, wu), wd)
                n_eq += bool(torch.equal(y, r))
                maxdiff = max(maxdiff, float((y.float() - r.float()).abs().max()))
        print(f"[H1] K={frac * 100:5.1f}% ({k:4d}/{I}): bitwise {n_eq}/{args.n}, max|diff|={maxdiff:.3e}")


if __name__ == "__main__":
    main()
