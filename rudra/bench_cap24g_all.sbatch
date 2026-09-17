#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -J cap24g_all
#SBATCH -o /home/bapic_iiitd/2_group/bench_cap24g_all.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=========================================================================="
echo "  BENCHMARK: 24-GB ENFORCED MEMORY CEILING EVALUATION (GPU-COUNT REDUCTION)"
echo "=========================================================================="
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
echo "=========================================================================="

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

echo ""
echo ">>> STEP 1: 1-GPU DENSE SERVING UNDER 24-GB CAP (EXPECTED: OOM) <<<"
python rudra/bench_cap24g_dense_1gpu.py \
  --model-dir "$MODEL_DIR" \
  --prompts-file rudra/prompts.txt \
  --cap-gb 24.0 \
  --output-json /home/bapic_iiitd/2_group/bench_cap24g_dense_1gpu_results.json

echo ""
echo ">>> STEP 2: 2-GPU DENSE LAYER-SHARDED SERVING UNDER 24-GB/GPU CAP (EXPECTED: PASS) <<<"
python rudra/bench_cap24g_dense_2gpu.py \
  --model-dir "$MODEL_DIR" \
  --prompts-file rudra/prompts.txt \
  --cap-gb 24.0 \
  --output-json /home/bapic_iiitd/2_group/bench_cap24g_dense_2gpu_results.json

echo ""
echo ">>> STEP 3: 1-GPU COLOSSUS RESIDENCY SEARCH UNDER 24-GB CAP (EXPECTED: DISCOVER FEASIBLE POINT) <<<"
python rudra/bench_cap24g_colossus_1gpu.py \
  --model-dir "$MODEL_DIR" \
  --prompts-file rudra/prompts.txt \
  --cap-gb 24.0 \
  --output-json /home/bapic_iiitd/2_group/bench_cap24g_colossus_1gpu_results.json

echo ""
echo "=========================================================================="
echo "  CONSOLIDATED 24-GB BUDGET PROOF SUMMARY"
echo "=========================================================================="
python -c '
import json

try:
    with open("/home/bapic_iiitd/2_group/bench_cap24g_dense_1gpu_results.json") as f:
        d1 = json.load(f)
except Exception as e:
    d1 = {"error": str(e)}

try:
    with open("/home/bapic_iiitd/2_group/bench_cap24g_dense_2gpu_results.json") as f:
        d2 = json.load(f)
except Exception as e:
    d2 = {"error": str(e)}

try:
    with open("/home/bapic_iiitd/2_group/bench_cap24g_colossus_1gpu_results.json") as f:
        c1 = json.load(f)
except Exception as e:
    c1 = {"error": str(e)}

print(f"{'Regime':<25} | {'GPUs':<5} | {'24GB Cap':<10} | {'Status':<12} | {'Peak HBM':<12} | {'TPS':<8}")
print("-" * 80)
d1_status = "OOM (PASS)" if d1.get("oom_triggered") else ("FEASIBLE" if d1.get("feasible") else "ERROR")
d1_peak = f"{d1.get(\"peak_hbm_gb\", 0):.2f} GB"
print(f"{\"Dense 1-GPU\":<25} | {1:<5} | {\"24.0 GiB\":<10} | {d1_status:<12} | {d1_peak:<12} | {0.0:<8}")

d2_status = "FEASIBLE" if d2.get("feasible") else "ERROR"
d2_peak = f"{d2.get(\"max_peak_hbm_gb\", 0):.2f} GB"
d2_tps = f"{d2.get(\"tps\", 0):.2f}"
print(f"{\"Dense 2-GPU (Sharded)\":<25} | {2:<5} | {\"24.0 GiB/GPU\":<10} | {d2_status:<12} | {d2_peak:<12} | {d2_tps:<8}")

for r in c1.get("results", []):
    cfg_name = f"COLOSSUS 1-GPU (C={r.get(\"capacity\")}, m={r.get(\"missing_col_ratio\"):.2f})"
    c_status = "FEASIBLE" if r.get("feasible") and r.get("fits_under_cap") else "OOM/INCOMPAT"
    c_peak = f"{r.get(\"peak_hbm_gb\", 0):.2f} GB"
    c_tps = f"{r.get(\"tps\", 0):.2f}" if r.get("tps") else "---"
    print(f"{cfg_name:<25} | {1:<5} | {\"24.0 GiB\":<10} | {c_status:<12} | {c_peak:<12} | {c_tps:<8}")

print("=" * 80)
'
