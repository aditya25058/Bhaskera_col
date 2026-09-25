# COLOSSUS — Comprehensive Report

**Date:** 2026-09-20 · **Branch:** `colossus-zssr` · **Hardware:** 1–2× NVIDIA H100 NVL (93.1 GB), Rudra A100 80GB (Gen3 x4)
**Models:** DeepSeek-Coder-V2-Instruct 236B MoE (160 routed + 2 shared, top-6, 471.5 GB BF16) · Param2-17B-A2.4B (64 experts) · Mixtral-8x7B (8×336 MB)
**Latest:** §9 (Exp 39) added 2026-09-24.

---

## 1. Problem Statement

Very large MoE models (236B–671B params, 400+ GB weights) nominally need 8×H100 clusters. Only a tiny fraction of experts (top-k) is active per token, so most GPU residency is waste. Goal: serve such models on **one single GPU** at low VRAM **without quantizing weights** (bitwise-exact BF16) — then recover the lost throughput.

## 2. Core Idea (unchanged throughout)

> **COLOSSUS changes the unit of MoE memory management from the entire expert to dynamically resident portions of experts, trading GPU memory/throughput for interconnect traffic while preserving exact computation.**

Column-level, not expert-level:

```text
Expert E ──► hot columns (GPU HBM) ──► y_cached ──┐
            cold columns (DMA on demand) ──► y_missed ──├── + ──► y = y_cached + y_missed == native
```

`f_native(x) = f_COLOSSUS(x)` — router untouched, SwiGLU math identical, only weight residency changes.

## 3. Architecture

| Component | Location | Function |
|---|---|---|
| Dynamic slot cache | `inference/colossus/dynamic_cache.py` (`DynamicMoELayerWrapper`, 654L) | C pre-alloc GPU slots/layer, LRU, demand DMA + `prefetch_stream`, `pre_attention_prefetch`, prefill warmup |
| ZSSR predictor | `colossus/predictor.py` | Zero-param: L0 frequency, L≥1 `h@Wᵀ` Top-8, SwiGLU-energy column plans |
| Column machinery | `colossus/columns.py`, `directory.py`, `sa_ffn.py`/`saffn.py` | Fixed packets (`tiered_fwd` 50×5+25×3=325 cols), LRU 32/expert, lossless split accumulate |
| TurboQuant KV | `inference/kv_cache.py` | K4/V2, O(1) incremental (667 s → 18–25 s fix); KV-cache gave 7.2× (13761→1908 ms/tok) |
| Inference engine | `inference/engine.py` | HF `generate()` + Cache, COLOSSUS hook, vLLM auto-select |
| DeepSeek E2E scaffold | `rudra/serve_deepseek_colossus.py` | mmap 55 shards, non-routed split, `DeepSeekColossusMoEWrapper`, NVLink bridge, JSON telemetry |
| Config | `configs/inference_colossus.yaml` | `replica: int4_row`, `budget: tiered_fwd`, `missing_col_ratio: 0.50` → ~19.6 GB vs ~34 GB dense (Param2) |

Research extensions (all **opt-in flags**, defaults = original behavior): hetero CPU fallback (`--enable_hetero`), ADETR per-slot split (`--adetr_ratio`), `GlobalColumnPool` (`--col_pool_gb`, `--col_block`), offline batching (`--batch_size`, `--prompt_file`).

## 4. Results

### 4.1 Feasibility (the headline — all exact, BF16, no quant)

| Run | Setup | Peak VRAM | Decode | Notes |
|---|---|---|---|---|
| Mixtral 2×A100 dense | 2 GPU | 43.51 GB | **14.50 tok/s** | reference (`aggregate_run3.py`) |
| Mixtral 1×A100 COLOSSUS | C=4 | 55.51 GB (23.6 free) | 0.09 tok/s, 45.7% hits, `torch_equal True` | 1×A100 holds 87 GB; bus-bound (336 MB/24 ms) |
| Param2 1×A100 dense | 1 GPU | 34.2 GB | 38.2 tok/s | baseline |
| Param2 1×A100 COLOSSUS | offload 60–80% | **19.6 GB** | 34.8 tok/s (91% retained) | fits 24 GB 4090; 6.3 ms fits MHA window |
| DeepSeek layer exact (Job 1780) | H100, C=12 | — | `torch.equal True`, diff `0.0` | PCIe Gen5 51.56 GB/s, R=0.55<1 |
| DeepSeek 2-GPU (1790→1804) | C=12/6, KV+batched DMA | 29.9/22.3 GB | 1774 ms/tok (7.8×) | 502 hits, peak 19.8% |
| **DeepSeek 1-GPU (1805)** | C=12, all 60 layers | **60.09/93.1 GB** | 1585 ms/tok (0.63) | 8×H100 → 1×H100, 666 hits (9.5%), peak 28% |

### 4.2 Throughput experiments (all preserved exact output)

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 1813 | CPU-expert exactness (Layer-1, EPYC vs Hopper BF16) | `torch.equal True`, 30.7 KB vs 135 MB (4400×) | mechanism proven |
| 1814 | Hetero full-serve (thr=4) | 10.82 s/tok, slots starved (0.25%) | Fiddler flips: 0.85 ms DMA < 30 ms CPU for 45 MB experts |
| 1815 | Hetero fix (thr=1, freq retain, prefill guard) | 5.73 s/tok, 5.1% hits | better; still 3× slower than DMA-only |
| ADETR-50% | per-slot hot/cold split | **22.5 MB/miss exact**, but 8.9 s/tok | transfer fine-grained, execution not — parked |
| Pool v1→v2→v3 | column-LFU keyed (layer,expert,block), GPU-only | v1 23 s → v2 2.05 s → v3 parity 1.87 s/tok | fixed 70M-op eviction scan, 0-hit fast path, admission filter |
| Pool sweep | 8/16/24 GB × 16/64 tok | hits 0.26→1.04→2.47%, eff. 45→44.5→43.9 MB/miss | monotonic, shallow; slots capture easy reuse |
| B=8 shared | same prompt ×8 + pool16 | **3.51 tok/s (6.6×)**, DMA flat 282 GB, 2.35 GB/token | MoE-Gen amortization proven |
| B=3 diverse | mixed prompts | 0.33→0.34 tok/s (C=12→18: hits +54%, latency −4%) | diversity ceiling B×6≤C; DMA-bound |
| B=3 ×64 tok | longer gen | flat 0.34 (prefill amortized, decode dominates) | amortization exhausted |
| v4 | block256/admit3 | 0.31 (no gain) | pool tuning exhausted |

### 4.3 Envelope (1×H100 NVL, exact, no quant)

| Workload | Throughput | Dominant factor |
|---|---|---|
| B=1 | ~0.53 tok/s | Expert DMA (45 MB/miss) |
| B=8 shared routing | **3.51 tok/s** | Cross-request weight reuse |
| B=3 diverse routing | ~0.34 tok/s | Routing-induced unique DMA (~56 GB/step) |

## 5. Issues Found (each with measurement)

1. **Whole-expert drift:** DeepSeek E2E used 45 MB full DMA (`14355/319`), not columns — scaffold built faster than column port. Fixed by reunification work; defaults restored.
2. **CPU-fallback trap:** 30 ms CPU vs 0.85 ms DMA per 45 MB expert at B=1/Gen5. Fiddler wins only for huge experts / starved buses.
3. **Slot starvation:** thr=4 sent all misses to CPU (0.25% hits). Fixed with DMA-first + freq retention.
4. **Pool v1 thrash:** O(cap) eviction scan (70M ops) + block staging → 122 s prefill. Fixed: sampling, fast path, admission.
5. **Diversity ceiling:** unique experts/layer ≤ C required for fast path; diverse B=3 ≈ 18 > 12. C=18 helped hits (+54%) not latency (−4%).
6. **Transfer fragmentation:** 1.3 MB blocks run at ~8 GB/s vs 51 GB/s for 45 MB contiguous.

## 6. Gaps / Open Work (idea intact)

- **Fused/coalesced assembly kernel** (nsys-profiled): one contiguous staging DMA per miss instead of block scatter; predicted ~3× on diverse steps. Biggest remaining lever.
- **Longer-horizon pool value:** pool pays only on slot-eviction recall; needs 64+ tok + ≥16 GB to show. Unexplored: 24 GB + B=8 shared (combined best-config run).
- **Diverse large-B:** B=16–32 mixed prompts (prefill amortization + overlap statistics unknown).
- **Dynamic_cache reunification:** E2E scaffold and `DynamicMoELayerWrapper` diverged; column-pool exists only in serve path.
- **Lossless compression over PCIe** (e.g. nvComp on BF16 mantissa): halves bytes with zero numerical change — untouched.
- **CPU attention** (MoE-Lightning lesson): saves I/O bandwidth for weights — not tried.

## 7. Standing Conclusion

COLOSSUS provably consolidates 236B-MoE serving onto one H100 at 60 GB VRAM with bitwise-exact outputs and no quantization. Throughput today: 0.5 (single) / 3.5 (shared batch) / 0.34 (diverse) tok/s. Controlling variable everywhere is **unique expert bytes per step** — the optimization mandate going forward, with the column-level exact idea frozen.

## 8. Codebase Status: Experiments Removed, Idea Intact (commit `4a359ba`)

No throughput experiment produced a significant increase, so all experiment code paths were **removed** from the codebase and COLOSSUS was restored to its pre-experiment state:

| Removed path | Commits (preserved in git history) | Result it produced |
|---|---|---|
| Hetero CPU engine (Fiddler-style) | `399009b`, `cf434bd`, `a2aa338` | 10.8 s → 5.7 s vs 1.9 s baseline — strictly slower at B=1/Gen5 |
| ADETR per-slot hot/cold split | `5410ae7` | 22.5 MB/miss exact, but 8.9 s/tok (CPU hot-half trap) |
| GlobalColumnPool v1–v4 | `f25eb4a`, `1d7e5b6`, `4cce63b`, `4b5ea3a` | parity at best (1.87 s), 0–2.5% hits, tuning exhausted |
| Offline batching (`--batch_size`) | `3647594` | only real gain (3.51 tok/s shared) — removed with the rest per directive |
| Hetero test harness | `17b703f`–`02c2e44`, deleted | Layer-1 `torch.equal True` record kept here |

Net diff: **−1016 lines** across `rudra/serve_deepseek_colossus.py`, `src/bhaskera/inference/colossus/dynamic_cache.py` (both reverted to `02c2e44`), `rudra/test_cpu_expert_hybrid.py` deleted. Zero experiment references remain; both files compile clean. Research can be re-enabled from history, but the standing rule is: **optimize throughput without changing the COLOSSUS idea** — dynamic residency of expert portions, exact computation, no quantization.

## 9. Exp 39 — Speculative family closed by routing-structure measurement (2026-09-24, commits `725fb50`, `0c5647d`)

**Question:** can speculative verification batching (SVB) convert the B=8 shared gain into B=1 interactive gain? Requires K consecutive tokens to share experts the way K same-position sequences do.

**Method** (measurement only, defaults OFF, baseline paths untouched): `--log_routing` dumps per-layer per-step expert unions; `shared_only` wrapper mode (resident attention + 2 shared experts, zero DMA, early return after `shared_out`) drafts K tokens from each Pass-A prefix; greedy prefix-match vs Pass-A truth (`--svb_probe_positions`, `--svb_K`).

| Measurement | Result |
|---|---|
| B=1 single-step union/layer | 6.00 experts (top-6; 59 MoE layers — layer 0 dense) |
| K=8 consecutive-token union/layer | **39.92 (6.65× single)** |
| Consecutive-step routing Jaccard | **0.041** — essentially independent expert sets per position |
| B=8 shared union/layer/step (`--batch_size` revived, re-measured) | **6.00, min=max=6, all steps** — bit-identical routing; **5.36 tok/s agg** |
| Shared-only draft cost | 66–69 ms/step (~25× cheaper than full step) |
| Shared-only draft acceptance (K=8, 5 probes) | matches 0/4/1/3/2, mean 2.0; offsets 0–3 agree 4/5, 3/5, 3/5, 2/5, offsets 4+ collapse |
| External-drafter vocab check (Path 1) | DeepSeek-V2 100k vs deepseek-coder-1.3b 32k — incompatible, blocked before measurement |

**Conclusion:** batching correlates on position, speculation correlates on sequence; DeepSeek-V2 routing correlates on neither across positions. Oracle ceiling for K=8 SVB = 8/6.65 ≈ **1.2×**, below break-even with draft overhead; realistic acceptance (2–3 tokens) → 0.3–0.45×, a loss. This binds the **entire speculative family** (EAGLE, Medusa, MoE-SpeQ, SP-MoE) on offloaded MoE with this routing pattern — verification always pays the consecutive-token union. SP-MoE/MoE-SpeQ tested coarser models (Phi-MoE/Mixtral) with stickier routing; the incompatibility is structural to fine-grained top-6/160 routing, not draft quality. No integration pursued — clean kill.

**Artifacts (server):** `deepseek_svb0_{b1,b8}.json`, `routing_{b1,b8}.json`. Harness retained as opt-in flags (`--log_routing`, `--svb_probe_positions`, `--svb_K`) for future draft-policy tests. Standing envelope (1×H100, exact): B=1 GPU 0.42–0.63 tok/s · B=1 CPU (ulp1) 1.36–1.55 · B=8 GPU shared 5.36 agg · B=8 CPU shared 10.10 agg · **B=16 CPU shared 16.84 agg** · **B=16 CPU diverse 3.22 agg** (`deepseek_cpu_b16.json`, `deepseek_cpu_div16.json` + `rudra/diverse16.txt` — CPU has no slots so no eviction death, but pays union bytes linearly: 16.84/3.22 = 5.2× ≈ union ratio; corrects the "bytes/token constant" thesis). **B=32 CPU shared 23.97 agg** (`deepseek_cpu_b32.json`; efficiency 81/68/48% at 8/16/32×). **B=64 CPU shared 34.96 agg; B=64 GPU shared 38.60 agg** (`deepseek_cpu_b64.json`, `deepseek_gpu_b64.json` — GPU wins shared-batch at scale: constant ~14 GB DMA/step, no CPU tax; paths complementary: GPU/shared vs CPU/single+diverse).

**Prefill autopsy (healthy):** 4.6/5.8/7.7 s at B=8/16/32 (10.5→24.8 tok/s) — scales with batch, not a bottleneck.

**Fused-kernel sizing (not recommended):** coalesced staging ≈ 2–5%/token; full step capture blocked by dynamic hit/miss control flow. CPU gap fully accounted; GPU gap is launch/sync only.

**Host bandwidth (measured, quiet window load 0.02):** OpenMP STREAM-class triad **92 GB/s/socket**, symmetric — not 320 (hardware reality, no latent bandwidth; GEMV util 67–92 GB/s ≈ 100% of achievable). The CPU path is bandwidth-optimal for this box.

## 11. C-1 — Hot-column prefetch: columns as the risk unit (2026-09-24, commits `4d127a8`, `ae2ed3a`)

**Design:** ZSSR-predicted experts stage calib-hot blocks only (raw 128-row-block DMA from safetensors mmap, ~14.9 MB/expert ≈ 1/3 of whole; no ANS, no pool). Demand verifies → completes cold blocks into the staged slot; unverified staging auto-counts as waste. Tests whether misprediction cost scales with granularity at matched recall.

| Run | Recall | Prefetch | Useful | Wasted | Waste/demand | tps | Exact |
|---|---|---|---|---|---|---|---|
| conf 0.5 | 52.3% (524/1001) | 14,890 MB | 7,795 MB | 7,095 MB | **2.6%** | 0.543 | True |
| conf 0.7 | 47.4% (309/652) | 9,699 MB | 4,596 MB | 5,102 MB | **1.8%** | 0.535 | True |
| whole-prefetch (Exp 17) | ~62% | — | — | +47% coeff | — | 0.31 | True |

**Verdict — mechanism PASS, wall parity:** waste coefficient 3× better than whole-expert prefetch (14.9 vs 45 MB/prediction) at comparable recall; waste negligible against demand (≤2.6% vs 15% gate); assembly bit-exact. But cold completion costs 3.9 ms/expert vs 0.85 ms whole load (block fragmentation, as predicted) → net wall parity with baseline. Columns proven as the risk-control unit; B=1 wall needs coverage (C-2), not just cheaper speculation.

**Artifacts (server):** `deepseek_c1_smoke.json`, `deepseek_c1_conf07.json`. Flags: `--col_prefetch/--block_index/--calib` (all default OFF; asserts probe+geometry+energy present, rejects ANS/CPU combos).

## 12. H1 (dead) + H2a (viable): exact sparsity vs exact partitioning (2026-09-24)

**H1 — top-K-only vs full expert, bitwise (`rudra/test_h1_sparsity.py`):** 0/50 matches at every K<100% (75/50/25/10%); error scales with omitted fraction. Exact sparsity is FALSE for SiLU dense experts — "hot" ≠ "zero." Bonus kill: even K=100% in a different summation order matches only 13/50 (1-ulp reassociation breaks bitwise before any approximation enters). H1 closed in one CPU-only run.

**H2a — intermediate-dim recalibration + split bound (`rudra/calib_col_intermediate.py`):** per-expert energy E[i] = ‖gate[i]‖²+‖up[i]‖²+‖down[:,i]‖² over all 9440 experts (668 s, CPU-only) → `models/DeepSeek-Coder-V2-CALIBCOL/calib_col.json` (top-512/1536 hot sets). Split protocol (y_hot + y_cold in native suborders vs full): 1/200 bitwise, worst diff 7.8e-3 (4 ulp — regrouping bound, wider than Path 3's 1 ulp; needs end-to-end flip audit, same ulp1 contract).

**Standing:** H2 (y_hot GPU + y_cold CPU, exact accumulation) is viable pending H2b flip audit. Both halves already built (C-1 partial residency, 3-1 CPU executor); remaining work is the split protocol + combine, plus down-column geometry (done here).

## 13. H2b — true column executor: architecture proven, economics negative (2026-09-24)

**Built:** hot-third col slots (`FastColSlot`, uniform 512/1536 from `calib_col.json`), raw hot-row/col DMA on miss, CPU cold-complement (cached views), single-worker overlap (GPU-hot ∥ CPU-cold), ulp1 combine. All behind `--h2b` (asserts ulp1, rejects ANS/CPU/prefetch/warm combos).

| Config | tps | hits/miss (16tok) | DMA | VRAM | Exact |
|---|---|---|---|---|---|
| H2b B=1 cap12 | 0.395 | 673/6367 | 93 GB | 39.4 GB | True |
| H2b B=1 cap36 | 0.415 | 1448/5592 | 82 GB | 60.1 GB | True |
| H2b B=8 cap36 | 3.21 agg | 1071/4550 | 67 GB | 60.1 GB | (coherent) |
| whole-expert CPU B=1 / B=8 | 1.55 / 10.10 | — | — | 60 GB | ulp1 |

**Why it loses:** per-miss fixed costs (3-copy hot DMA dance + view management + cold dispatch + combine ≈ 2.9 s/step) exceed the 3× byte savings. Coverage tripling doesn't settle misses because the per-step union-6 **churns across steps** (GPU B=8 reference also misses ~300/354 sustained — my "near-100% hits" prediction was wrong; the B=8 win was always constant-bytes-per-step, never residency). Same fragmentation moral as C-1, one level up: columns save bytes, interconnect charges per transfer.

**Verdict:** mechanism PROVEN (exact split compute, both halves, overlap — the architecture the doc describes, running), performance NEGATIVE at B=1 and B=8 on this host. H2b stays opt-in; whole-expert paths stand. The 500-tok flip audit is moot for deployment (perf kills it first); 16-tok exact-True establishes mechanism exactness.

**Artifacts (server):** `deepseek_h2b_smoke.json`, `deepseek_h2b_cov.json`, `deepseek_h2b_b8.json`.

## 10. Path 3-0/3-1 — CPU expert compute + ulp1 exactness gate (2026-09-24, commits `d0fcc91`, `cc57cc9`)

**Ceiling:** 6 experts × 47.2 MB × 59 layers ≈ 16.7 GB RAM-read/token; box measured 57–102 GB/s achievable (below 320 GB/s STREAM hope — cause undetermined, no resctrl cap; 4 GB swap in use). Only ~5% of wire speed needed to beat the 1585 ms/token baseline.

| Measurement | Result |
|---|---|
| Single-expert SwiGLU GEMV (oneDNN BF16) | 2.81 ms (1 thread) → **0.70 ms** (6 threads, 67 GB/s util) |
| 6-expert parallel layer (ThreadPool) | **3.1 ms/layer** vs 30 ms kill criterion — PASS 10× |
| `torch.equal` CPU-vs-GPU ×100 inputs | **34/100 pass**, max\|diff\| = 1.95e-3 (1 BF16 ulp — reduction-order reassociation, both sides FP32-accumulate) |
| Flip protocol, 500-token greedy (teacher-forced both paths) | **independent flips 1/500 = 0.20%** (< 1% gate); the single flip at pos 141 had cpu_margin = 0.0000 (exact tie) vs gpu_margin = 0.1250 — a tie-break, the most benign kind |
| Free-run divergence | first div pos 141, **499/500 match**, no cascade |
| CPU free-run throughput (eager Python loop, `--exactness_mode ulp1 --cpu_layers all`) | **1.36–1.55 tok/s** (9.5 ms/layer-step) — first B=1 gain in 40 experiments, 2.4–2.7× baseline |

**Ruling:** ulp1 mode accepted — reassociation is not approximation, bound measured (≤1 ulp), flip rate 0.2% at a tie position. `--exactness_mode` flag preserves the contract: `bitwise` default (GPU-only, untouched) vs `ulp1` (CPU path, flip metrics logged). FP32-weights fallback not needed. Teacher harness validated (teacher-forced GPU reproduces baseline 500/500).

**Artifacts (server):** `deepseek_flip_{A,B,Cgpu,Ccpu}.json`, `flip_audit_{gpu,cpu,Cgpu,Ccpu}.json`, `flip_teacher.json`. Harness: `rudra/bench_cpu_expert_30.py`, serve flags `--exactness_mode/--cpu_layers/--cpu_threads/--cpu_cache/--audit_logits/--teacher_tokens`.

**Open threads:** step-latency decay within long runs (fast start → 3–6 s steps; page-cache churn of the 471 GB model on the 503 GB box + swap use — affects both paths); CPU per-layer 9.5 ms vs 3.1 ms bench (Python-loop overhead → torch.compile candidate for 3-2).

**3-2 probe results (DNNL_VERBOSE, oneDNN v3.4.2, brg:avx512_core_bf16):** compile SLOWER than eager (1.34 vs 0.73 ms — dead); **zero primitive creates** (cache healthy, same descriptor everywhere — no capacity fix); sequential exec-sum ≈ wall (no fork/join gaps — 9.5 ms fully accounted: 18 matmuls × ~0.35 ms + pointwise + shuttle); pool serializes (18.9 ms wait vs 5.1 ms standalone — working-set churn over 354 distinct tensors/layer-token, not a tunable). C++ extension would save ~1 ms Python, not memory traffic — not pursued. Standing: sequential-6-thread CPU path, 9.5 ms/layer, 1.55 tok/s B=1.
