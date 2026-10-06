# COLOSSUS User Manual — exact MoE serving on minimal hardware

Serve Mixture-of-Experts models far larger than your GPU memory, with
unquantized weights and a measured exactness contract. This manual covers
installation, every serving mode, every flag, output interpretation,
exactness grades, and troubleshooting. Branch: `colossus-mainline`.

---

## 1. Concepts (60 seconds)

A 236B MoE needs 471 GB; your GPU has 93 GB. COLOSSUS keeps a **working
set** resident and streams or computes the rest. The residency unit is
your choice:

- **Expert slots (bitwise):** whole experts in LRU slots, missing experts
  DMA from host RAM. Exact (`torch.equal`).
- **Column slots:** each expert split into hot columns (resident, LRU) +
  cold columns (streamed per use into shared scratch). Partial-GEMM + sum
  is mathematically identical to whole-expert compute. Same exactness,
  less HBM — and HBM-bound operating points whole-expert tiering cannot
  reach (B=512 below). See §4.
- **Matrix pools (control):** per-matrix (gate/up/down) LRU pools.
  Behaves like expert slots (±0.5%) — included so you can verify that
  whole-matrix splitting adds nothing; only sub-matrix fractions move
  the frontier.
- **CPU (ulp1):** experts computed on CPU from RAM (oneDNN BF16). Tiny
  measured noise (see §7).
- **Batching:** same-position sequences route identically → bytes/step flat.
- **Remote:** weights fetched on demand from Hub + persistent cache.
- **Planner:** inspects model + hardware, recommends flags, can apply them.

The native router is never touched: predictions move data only.
Exactness is a hard invariant with a verification protocol (§7), not a hope.

## 2. Installation

```bash
git clone <Bhaskera_col> && cd Bhaskera_col && git checkout colossus-mainline
export PYTHONPATH=$PWD/src
pip install torch transformers accelerate safetensors peft datasets
python -m pytest src/bhaskera/inference/colossus/tests/  # CPU-only, ~90s
```

GPU: any CUDA card (HBM sets capacity choices). CPU: x86-64 with BF16
preferred (falls back slower without it). Note: transformers ≥5.x is
required for Gemma-4 (`model_type="gemma4"`); everything else runs on 4.x.

## 3. Quickstart

**Fit check (no weights moved):**
```bash
bhaskera-infer --plan --model /models/DeepSeek-Coder-V2-Instruct
```

**Serve 236B on one GPU (bitwise exact):**
```bash
bhaskera-infer --model /models/DeepSeek-Coder-V2-Instruct \
  --prompt 'def quicksort(arr):' --max-new-tokens 16 --no-sample \
  --offload-tier slots --capacity 12
# expect ~0.6 tok/s, ~60 GB VRAM
```

**Faster single-stream (ulp1):**
```bash
bhaskera-infer --model <dir> --prompt '...' \
  --offload-tier cpu --exactness-mode ulp1
# expect ~1.5 tok/s
```

**Column slots (less HBM, same exactness):**
```bash
bhaskera-infer --model <dir> --prompt '...' --max-new-tokens 16 \
  --offload-tier slots --capacity 12 --hot-col-frac 0.5
# half-resident experts; B=1 audit 16/16; B=512-batch capable (see §4)
```

**One command (plan → apply → serve):**
```bash
bhaskera-infer --plan --plan-apply --model <dir> \
  --prompt '...' --max-new-tokens 16 --no-sample
```

## 4. Batch serving (throughput)

One prompt per line; **identical lines share routing** (fast), mixed lines
pay union bytes (slower, still exact):

```bash
bhaskera-infer --model <dir> --prompt-file batch64.txt \
  --max-new-tokens 64 --offload-tier slots --capacity 12 --prefill-chunk 64
```

Measured envelope (1×H100, DeepSeek-Coder-V2-236B, decode-agg tok/s):

| B | expert C=12 | column f=0.5 | VRAM (expert/col) |
|---|---|---|---|
| 64 | 32.9 | 31.8 | 60.5 / 61.8 GB |
| 128 | 64.7 | — | 65.5 GB |
| 256 | 133.2 | — | 75.7 GB |
| 384 | 194.7 | — | 85.8 GB |
| 448 | 218.4 | 195.8 | 90.9 / 76.6 GB |
| 512 | OOM | **249.2** | — / 81.7 GB |

Shared batches scale ~linearly (union converges, DMA/step flat ~20 GB)
until the KV wall. B=512 fits only with column slots (expert-tier OOMs).
B=64×200 OOMs for both — the remaining wall is KV, not weights.

Adaptive tiers (`colprofile.py` from `--log-routing` → `--hot-tier-file`):
pin top-K experts at f=1.0, rest thin. B=64: 29.6 tok/s at 50.4 GB VRAM
(90% speed, less DMA than expert-tier). Tier shape is a tuning dimension
matched to union size, not a universal win.

## 5. Remote weights (no full download)

```bash
# public repo: fetch working set only (~20/60 GB for short runs)
bhaskera-infer --remote-repo Qwen/Qwen3-30B-A3B --remote-cache ./cache \
  --prompt '...' --offload-tier slots
# gated repo: add --remote-token $HF_TOKEN (never in YAML)
# partial local + remote fill:
bhaskera-infer --remote-repo <id> --remote-cache ./cache \
  --local-mirror ./partial --prompt '...' --offload-tier slots
```

Cold runs pay network per first touch; warm runs approach local speed
(cache hit rate printed every run). Cold/warm numbers are never blended.

## 6. Supported models (five families, two representations)

Same CLI, zero per-model code: DeepSeek-Coder-V2-236B + V2-Lite
(160-expert top-6, 3-tuple gate), Qwen3-30B-A3B (logits gate, top-8),
Mixtral-8x7B (w1/w3/w2, tupled return), Param2-17B (tied head handled),
Gemma-4-26B-A4B (grouped weights, scaled routing, GELUTanh — needs
transformers ≥5.x). Representations: per-expert modules AND fused grouped
tensors (column-split works on both).

## 7. Exactness grades + verification protocol

- **bitwise** (default): GPU-only paths. Verifiable with `torch.equal`.
- **ulp1**: any CPU/split compute. Bound ≤1 ulp typical (≤4 split);
  measured 0.2% independent flips, all at exact-tie positions.
- **Column splits**: exact by construction (summation partition); verified
  by audit, never assumed. f=1.0→0.1 all 16/16 on DeepSeek; 1.0→0.25 all
  16/16 on Gemma. Margins (min 1.6+) and bf16==fp32 agreement printed.

**Audit your own workload (B=1, valid protocol):**
```bash
# 1. dump reference IDs (continuation INCLUDES the prefill token)
bhaskera-infer --model <dir> --prompt '...' --max-new-tokens 16 \
  --offload-tier slots --capacity 12 --dump-ids teacher.json
# 2. teacher-force and compare per position
bhaskera-infer --model <dir> --prompt '...' --max-new-tokens 16 \
  --offload-tier slots --capacity 12 --teacher-tokens teacher.json \
  --audit-logits audit.json
# gate: 16/16 match, min margin comfortably > 0, own_bf16 == own everywhere
```

## 8. Measurement & telemetry

- `--log-routing PATH`: per-layer per-step expert unions (JSON).
  Analyze stickiness: consecutive-step Jaccard; K-window union growth.
  Feeds `colprofile.py` for adaptive tiers.
- Run summary lines: hits/misses, DMA MB (+hot/cold split for column
  runs), prefill s, peak VRAM, tok/s.
- Remote runs add: net fetched GB, disk reread, requests, cache hit %.
- Planner: `--plan [--plan-batch N --plan-gen M --plan-diverse
  --plan-fidelity ulp1]` prints model/hardware/ranked-flag table.

## 9. YAML config (reproducible runs)

```bash
bhaskera-infer --config configs/inference_colossus_tiered.yaml \
  --model <dir> --prompt '...'
```

See `configs/inference_colossus_tiered.yaml` — fully commented, including
the decision guide. CLI flags override YAML values.

## 10. Flag reference

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--offload-tier` | off/slots/cpu | off | slots=bitwise GPU; cpu=ulp1 oneDNN |
| `--capacity` | int | 12 | expert slots/layer (0 = streaming floor) |
| `--exactness-mode` | bitwise/ulp1 | bitwise | cpu tier requires ulp1 |
| `--prefill-chunk` | int | 0 | 0=single shot; 64 bounds activation memory (exact) |
| `--hot-col-frac` | float (0,1] | — | column slots: hot fraction (module + grouped) |
| `--hot-tier-file` | path | — | adaptive-f tier JSON (overrides frac) |
| `--cold-cache` | int | 0 | cold-column LRU/layer; falsified — leave 0 |
| `--grouped-gemm` | — | off | batched dispatch; falsified as implemented — leave off |
| `--matrix-tiers` | — | off | FIRM-like per-matrix control |
| `--matrix-tiers` | — | off | FIRM-like per-matrix control |
| `--prefault` | — | off | page-in shards first; helps cold big-RAM boxes only |
| `--dump-ids` | path | — | B=1 continuation IDs (teacher artifacts) |
| `--teacher-tokens` | path | — | reference IDs (dump format accepted) |
| `--audit-logits` | path | — | per-position flip audit (B=1) |
| `--log-routing` | path | — | routing-union dump |
| `--remote-repo` | Hub id | — | needs `--remote-cache` |
| `--remote-cache` | dir | — | persistent tensor cache |
| `--remote-cap-gb` | float | 200 | LRU byte cap |
| `--local-mirror` | dir | — | partial checkout, no index needed |
| `--remote-token` | token | `$HF_TOKEN` | flag/env only, never YAML |
| `--plan/--plan-apply` | — | — | inspect→probe→table→(serve) |
| `--plan-batch/--plan-gen` | ints | 1/16 | workload assumption |
| `--plan-diverse` | — | shared | diverse-batch assumption |
| `--plan-fidelity` | bitwise/ulp1 | bitwise | fidelity floor |

Batching is via `--prompt-file` (one prompt per line, no `--batch-size`
flag): identical lines share routing (fast); mixed lines pay union bytes.

## 11. Decision guide

- Fits in VRAM → plain `bhaskera-infer` (never offload what fits).
- 200B+, exact → `slots`.
- HBM-bound batch/length → `slots` + `--hot-col-frac 0.5` (halves slot
  HBM; audit first at B=1).
- Skewed routing + fixed HBM → `--hot-tier-file` from colprofile.
- 200B+, B=1 speed → `cpu` + ulp1.
- Shared batch/eval → slots + big batch-file (push B to the KV wall).
- Diverse batch → cpu tier.
- No disk → `--remote-repo`.
- Quantization OK → use KTransformers/llama.cpp instead.
- Unknown arch → `--plan` first; refusal names the reason.

## 12. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| CUDA OOM at start | Lower `--capacity`; halve `--hot-col-frac`; shorten context; `--prefill-chunk 64` |
| OOM in long decode | KV wall (not weights): smaller B×L product; column slots only move the weight wall |
| Slow first run, fast reruns | Page-cache warming (471 GB model, 503 GB RAM). `--prefault` helps cold big-RAM boxes; hurts marginal-cache ones |
| `--cold-cache 64` OOMs | Per-layer caches ×60 layers; keep ≤8/layer — and expect little (falsified, §12) |
| `get_usable_length` / mask errors | Old modeling file + new transformers: handled by built-in compat shims; update transformers if persists |
| Gemma `model_type="gemma4"` unrecognized | transformers ≥5.x required |
| Gated Hub 401/404 | Accept license + valid read token (`whoami` must pass locally first) |
| SSL verify failed (uv python) | Install `certifi`, export `SSL_CERT_FILE=$(python -m certifi)` |
| Trust remote code prompt | Pass needs `trust_remote_code=True`: supported via CLI automatically |
| Suspect outputs | Rerun §7 audit; check flip rate + margins + bf16 agreement |
| Multi-GPU | Single-GPU only in this release |

## 13. Tests & verification

```bash
pytest src/bhaskera/inference/colossus/tests/  # CPU-only, 77 green
```

Covers: interface parity (all gate styles), wrap/replace, slot accounting,
CPU parity, loading/partition, config round-trip, planner math, remote
gates (mock Range server), streaming floor, grouped adapter + parity,
column slots (f=1.0 bitwise, fractional noise bounds, DMA accounting,
tiers, cold-cache), grouped columns, matrix control. GPU proofs (236B
parity + f-curve, Gemma + f-curve, batch envelope) run on hardware; see
COLOSSUS_REPORT.md §§20–33.
