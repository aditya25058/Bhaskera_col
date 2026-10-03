# COLOSSUS User Manual — exact MoE serving on minimal hardware

Serve Mixture-of-Experts models far larger than your GPU memory, with
unquantized weights and a measured exactness contract. This manual covers
installation, every serving mode, every flag, output interpretation,
exactness grades, and troubleshooting.

---

## 1. Concepts (60 seconds)

A 236B MoE needs 471 GB; your GPU has 93 GB. COLOSSUS keeps a **working
set** resident (attention, shared experts, C routed experts in slots) and
streams or computes the rest:

- **Slots (bitwise):** missing experts DMA from host RAM. Exact (`torch.equal`).
- **CPU (ulp1):** experts computed on CPU from RAM (oneDNN BF16). Tiny
  measured noise (see §7).
- **Batching:** same-position sequences route identically → bytes/step flat.
- **Remote:** weights fetched on demand from Hub + persistent cache.
- **Planner:** inspects model + hardware, recommends flags, can apply them.

The native router is never touched: predictions move data only.

## 2. Installation

```bash
git clone <Bhaskera_col> && cd Bhaskera_col && git checkout colossus-zssr
export PYTHONPATH=$PWD/src
pip install torch transformers accelerate safetensors peft datasets
python -m pytest src/bhaskera/inference/colossus/tests/  # CPU-only, ~90s
```

GPU: any CUDA card (HBM sets capacity choices). CPU: x86-64 with BF16
preferred (falls back slower without it).

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
# shared B=64: ~39 tok/s agg (236B) · diverse B=16 cpu: ~3 tok/s
```

Measured envelope (1×H100, DeepSeek-Coder-V2): B=1 0.6 · B=8 10.1 ·
B=16 16.8 · B=32 24.0 · B=64 38.6 (GPU) / 35.0 (CPU) · diverse-16 3.2.

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

## 6. Exactness grades

- **bitwise** (default): GPU-only paths. Verifiable with `torch.equal`.
- **ulp1**: any CPU/split compute. Bound ≤1 ulp typical (≤4 split);
  measured 0.2% independent flips, all at exact-tie positions.

**Audit your own workload (B=1):**
```bash
bhaskera-infer --model <dir> --prompt '...' --offload-tier cpu \
  --exactness-mode ulp1 --audit-logits audit.json --teacher-tokens teacher.json
# teacher.json: reference token ids; gate: <1% independent flips
```

## 7. Measurement & telemetry

- `--log-routing PATH`: per-layer per-step expert unions (JSON).
  Analyze stickiness: consecutive-step Jaccard; K-window union growth.
- Run summary lines: hits/misses, DMA MB, prefill s, peak VRAM, tok/s.
- Remote runs add: net fetched GB, disk reread, requests, cache hit %.
- Planner: `--plan [--plan-batch N --plan-gen M --plan-diverse
  --plan-fidelity ulp1]` prints model/hardware/ranked-flag table.

## 8. YAML config (reproducible runs)

```bash
bhaskera-infer --config configs/inference_colossus_tiered.yaml \
  --model <dir> --prompt '...'
```

See `configs/inference_colossus_tiered.yaml` — fully commented, including
the decision guide. CLI flags override YAML values.

## 9. Flag reference

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--offload-tier` | off/slots/cpu | off | slots=bitwise GPU; cpu=ulp1 oneDNN |
| `--capacity` | int | 12 | expert slots/layer (0 = streaming floor) |
| `--exactness-mode` | bitwise/ulp1 | bitwise | cpu tier requires ulp1 |
| `--prefill-chunk` | int | 0 | 0=single shot; 64 bounds activation memory (exact) |
Batching is via `--prompt-file` (one prompt per line, no `--batch-size`
flag): identical lines share routing (fast); mixed lines pay union bytes.
| `--remote-repo` | Hub id | — | needs `--remote-cache` |
| `--remote-cache` | dir | — | persistent tensor cache |
| `--remote-cap-gb` | float | 200 | LRU byte cap |
| `--local-mirror` | dir | — | partial checkout, no index needed |
| `--remote-token` | token | `$HF_TOKEN` | flag/env only, never YAML |
| `--log-routing` | path | — | routing-union dump |
| `--audit-logits` | path | — | per-position flip audit (B=1) |
| `--plan/--plan-apply` | — | — | inspect→probe→table→(serve) |
| `--plan-batch/--plan-gen` | ints | 1/16 | workload assumption |
| `--plan-diverse` | — | shared | diverse-batch assumption |
| `--plan-fidelity` | bitwise/ulp1 | bitwise | fidelity floor |

## 10. Decision guide

- Fits in VRAM → plain `bhaskera-infer` (never offload what fits).
- 200B+, exact → `slots`.
- 200B+, B=1 speed → `cpu` + ulp1.
- Shared batch/eval → slots + big batch-file.
- Diverse batch → cpu tier.
- No disk → `--remote-repo`.
- Quantization OK → use KTransformers/llama.cpp instead.
- Unknown arch → `--plan` first; refusal names the reason.

## 11. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| CUDA OOM at start | Lower `--capacity`; shorten context; `--prefill-chunk 64` |
| Slow first run, fast reruns | Page-cache warming (471 GB model, 503 GB RAM). Expected. |
| `get_usable_length` / mask errors | Old modeling file + new transformers: handled by built-in compat shims; update transformers if persists |
| Gated Hub 401/404 | Accept license + valid read token (`whoami` must pass locally first) |
| SSL verify failed (uv python) | Install `certifi`, export `SSL_CERT_FILE=$(python -m certifi)` |
| Trust remote code prompt | Pass needs `trust_remote_code=True`: supported via CLI automatically |
| Suspect outputs | Rerun with `--audit-logits` + teacher tokens; check flip rate |
| Multi-GPU | Single-GPU only in this release (2-GPU bridge parked) |

## 12. Tests & verification

```bash
pytest src/bhaskera/inference/colossus/tests/ \
       src/bhaskera/inference/tests/test_inference.py  # CPU-only
```

Covers: interface parity (both gate styles), wrap/replace, slot accounting,
CPU parity, loading/partition, config round-trip, planner math, remote
gates (mock Range server), streaming floor. GPU proofs (236B parity,
Qwen/Mixtral generality) run on hardware; see COLOSSUS_REPORT.md.
