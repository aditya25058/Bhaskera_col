#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J sweep_missing
#SBATCH -o /home/bapic_iiitd/2_group/bench_2gpu_sweep_missing.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

echo "=========================================================================="
echo "  SWEEP PART 1: Missing Column Ratio Sweep (C=32, m=0.25 and m=0.50)"
echo "=========================================================================="

echo ""
echo ">>> RUNNING CONFIG 1: C=32, missing=0.25 (25% cold columns, 75% hot resident) <<<"
python rudra/bench_2gpu_colossus.py \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-tokens 64 \
  --output-dir /home/bapic_iiitd/2_group \
  --ref-tokens /home/bapic_iiitd/2_group/tokens_1gpu.pt \
  --ref-texts /home/bapic_iiitd/2_group/outputs_1gpu.txt \
  --hot-expert-topk 32 \
  --missing-col-ratio 0.25 \
  --bhaskera-json /home/bapic_iiitd/2_group/bench_2gpu_bhaskera_results.json \
  --json-out /home/bapic_iiitd/2_group/bench_2gpu_c32_m25_results.json

echo ""
echo ">>> RUNNING CONFIG 2: C=32, missing=0.50 (50% cold columns, 50% hot resident) <<<"
python rudra/bench_2gpu_colossus.py \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-tokens 64 \
  --output-dir /home/bapic_iiitd/2_group \
  --ref-tokens /home/bapic_iiitd/2_group/tokens_1gpu.pt \
  --ref-texts /home/bapic_iiitd/2_group/outputs_1gpu.txt \
  --hot-expert-topk 32 \
  --missing-col-ratio 0.50 \
  --bhaskera-json /home/bapic_iiitd/2_group/bench_2gpu_bhaskera_results.json \
  --json-out /home/bapic_iiitd/2_group/bench_2gpu_c32_m50_results.json
