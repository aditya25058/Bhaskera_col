"""Phase 3-0: CPU expert-compute microbenchmarks (measure, don't build).

3-0.1  RAM bandwidth triad (run under numactl --membind=0 / =1 / --interleave=all).
3-0.2  Single-expert SwiGLU GEMV: single-thread BW utilization, then scaled;
       torch.equal (CPU vs GPU) over 100 random inputs -- gate for 3-1.
3-0.3  6-expert parallel layer prototype (ThreadPool, NUMA-pinned).
3-0.4  Affinity info collected at runtime (numa node of process, GPU topo left to CLI).

Usage (server):
  numactl --membind=0 $VENV/bin/python rudra/bench_cpu_expert_30.py --mode bw
  gpurun -g 1 $VENV/bin/python rudra/bench_cpu_expert_30.py --mode gemv --model models/DeepSeek-Coder-V2-Instruct
"""
import argparse
import os
import time

import torch


def bench_bw(n_gb=8, iters=6):
    """Simple triad a=b+c over n_gb; reports GB/s (counts 3x traffic)."""
    n = (n_gb * 1024**3) // 8  # float64 elements
    a = torch.empty(n, dtype=torch.float64)
    b = torch.rand(n, dtype=torch.float64)
    c = torch.rand(n, dtype=torch.float64)
    for _ in range(2):  # warmup (also first-touch under current numactl)
        a.copy_(b).add_(c)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        a.copy_(b).add_(c)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    gbps = (3 * n * 8) / best / 1e9
    print(f"  [bw] size={n_gb}GB iters={iters}: {gbps:.1f} GB/s (best of {iters})")
    return gbps


def load_expert(model_dir, layer=1, expert=0):
    from safetensors import safe_open
    # find shard holding this expert's gate_proj
    target = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, fn)
        with safe_open(path, framework="pt", device="cpu") as f:
            if target in f.keys():
                pfx = f"model.layers.{layer}.mlp.experts.{expert}."
                w = {k: f.get_tensor(pfx + k + ".weight") for k in ("gate_proj", "up_proj", "down_proj")}
                print(f"  [load] expert L{layer}E{expert} from {fn}: " +
                      " ".join(f"{k}={tuple(v.shape)}" for k, v in w.items()) +
                      f" bytes={sum(v.numel() * 2 for v in w.values()) / 1e6:.1f}MB")
                return w
    raise RuntimeError("expert not found")


@torch.no_grad()
def expert_fwd(x, w):
    return torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, w["gate_proj"])) *
        torch.nn.functional.linear(x, w["up_proj"]), w["down_proj"])


def bench_gemv(model_dir, n_inputs=100, seed=0):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  [gemv] device for reference: {dev}")
    w = load_expert(model_dir)
    H = w["gate_proj"].shape[1]
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(1, H, dtype=torch.bfloat16, generator=g) for _ in range(n_inputs)]

    # GPU reference (bf16, same op order as serve path)
    wg = {k: v.to(torch.bfloat16).to(dev) for k, v in w.items()}
    with torch.no_grad():
        refs = [expert_fwd(x.to(dev), wg).to("cpu") for x in xs]
    torch.cuda.synchronize() if dev.type == "cuda" else None

    for threads in (1, 6):
        torch.set_num_threads(threads)
        # warmup
        with torch.no_grad():
            for x in xs[:3]:
                expert_fwd(x, w)
        t0 = time.perf_counter()
        with torch.no_grad():
            outs = [expert_fwd(x, w) for x in xs]
        dt = (time.perf_counter() - t0) / n_inputs
        wbytes = sum(v.numel() * 2 for v in w.values())
        print(f"  [gemv] threads={threads}: {dt * 1000:.2f} ms/expert "
              f"(BW util {wbytes / dt / 1e9:.1f} GB/s vs ~45MB weights)")

    # torch.equal gate (threads=1, deterministic order) + bound on failure
    torch.set_num_threads(1)
    with torch.no_grad():
        outs = [expert_fwd(x, w) for x in xs]
    n_eq = sum(bool(torch.equal(o, r)) for o, r in zip(outs, refs))
    maxdiff = max(float((o.float() - r.float()).abs().max()) for o, r in zip(outs, refs))
    print(f"  [equal] torch.equal CPU-vs-GPU: {n_eq}/{n_inputs} pass, max|diff|={maxdiff:.3e}")
    return dt, n_eq, maxdiff


def bench_compiled(model_dir, repeats=100):
    """3-2 Option A: torch.compile SwiGLU (static [1,H] shape, pre-warmed)."""
    import torch.nn.functional as F
    w = load_expert(model_dir)
    H = w["gate_proj"].shape[1]
    torch.set_num_threads(6)
    try:
        @torch.compile(mode="max-autotune")
        def compiled_expert(x, wg, wu, wd):
            return F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)
        x = torch.randn(1, H, dtype=torch.bfloat16)
        for _ in range(3):  # warmup (compile + autotune)
            compiled_expert(x, w["gate_proj"], w["up_proj"], w["down_proj"])
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                compiled_expert(x, w["gate_proj"], w["up_proj"], w["down_proj"])
        dt = (time.perf_counter() - t0) / repeats
        print(f"  [compiled] max-autotune: {dt * 1000:.2f} ms/expert")
    except Exception as e:
        print(f"  [compiled] FAILED: {type(e).__name__}: {str(e)[:200]}")
        return None
    # eager reference, same conditions
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            expert_fwd(x, w)
    dt_eager = (time.perf_counter() - t0) / repeats
    print(f"  [compiled] eager baseline : {dt_eager * 1000:.2f} ms/expert")
    return dt


def bench_parallel(model_dir, n_experts=6, layer=1, repeats=10):
    from concurrent.futures import ThreadPoolExecutor
    torch.set_num_threads(1)  # one thread per expert worker; 6 workers
    experts = [load_expert(model_dir, layer, e) for e in range(n_experts)]
    H = experts[0]["gate_proj"].shape[1]
    xs = [torch.randn(1, H, dtype=torch.bfloat16) for _ in range(n_experts)]

    def run_one(args):
        x, w = args
        with torch.no_grad():
            return expert_fwd(x, w)

    with ThreadPoolExecutor(max_workers=n_experts) as pool:
        list(pool.map(run_one, zip(xs, experts)))  # warmup
        t0 = time.perf_counter()
        for _ in range(repeats):
            list(pool.map(run_one, zip(xs, experts)))
        dt = (time.perf_counter() - t0) / repeats
    wbytes = sum(v.numel() * 2 for v in experts[0].values())
    print(f"  [parallel] {n_experts} experts x {repeats}: {dt * 1000:.1f} ms/layer "
          f"(agg BW {n_experts * wbytes / dt / 1e9:.1f} GB/s)")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bw", "gemv", "parallel", "compiled", "all"], default="all")
    ap.add_argument("--model", default="models/DeepSeek-Coder-V2-Instruct")
    ap.add_argument("--n_inputs", type=int, default=100)
    args = ap.parse_args()
    print(f"[3-0] pid={os.getpid()} threads_default={torch.get_num_threads()} "
          f"mkl={torch.backends.mkl.is_available()} mkldnn={torch.backends.mkldnn.is_available()}")
    try:
        with open(f"/proc/{os.getpid()}/numa_maps") as f:
            nodes = {}
            for line in f:
                if "heap" in line or "stack" in line:
                    continue
                for tok in line.split():
                    if tok.startswith("N"):
                        n, pg = tok.split("=")
                        nodes[n] = nodes.get(n, 0) + int(pg)
            print(f"  [numa] heap-adjacent pages by node: {nodes}")
    except Exception as e:
        print(f"  [numa] unavailable: {e}")
    if args.mode in ("bw", "all"):
        bench_bw()
    if args.mode in ("gemv", "all"):
        bench_gemv(args.model, args.n_inputs)
    if args.mode in ("parallel", "all"):
        bench_parallel(args.model)
    if args.mode in ("compiled",):
        bench_compiled(args.model)


if __name__ == "__main__":
    main()
