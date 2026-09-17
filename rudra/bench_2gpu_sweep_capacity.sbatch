#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J sweep_capacity
#SBATCH -o /home/bapic_iiitd/2_group/bench_2gpu_sweep_capacity.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

echo "=========================================================================="
echo "  SWEEP PART 2: Slot Capacity Sweep (C=16 and C=24, missing=0.10)"
echo "=========================================================================="

echo ""
echo ">>> RUNNING CONFIG 3: C=16, missing=0.10 (16 slots, 10% cold columns) <<<"
python rudra/bench_2gpu_colossus.py \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-tokens 64 \
  --output-dir /home/bapic_iiitd/2_group \
  --ref-tokens /home/bapic_iiitd/2_group/tokens_1gpu.pt \
  --ref-texts /home/bapic_iiitd/2_group/outputs_1gpu.txt \
  --hot-expert-topk 16 \
  --missing-col-ratio 0.10 \
  --bhaskera-json /home/bapic_iiitd/2_group/bench_2gpu_bhaskera_results.json \
  --json-out /home/bapic_iiitd/2_group/bench_2gpu_c16_m10_results.json

echo ""
echo ">>> RUNNING CONFIG 4: C=24, missing=0.10 (24 slots, 10% cold columns) <<<"
python rudra/bench_2gpu_colossus.py \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-tokens 64 \
  --output-dir /home/bapic_iiitd/2_group \
  --ref-tokens /home/bapic_iiitd/2_group/tokens_1gpu.pt \
  --ref-texts /home/bapic_iiitd/2_group/outputs_1gpu.txt \
  --hot-expert-topk 24 \
  --missing-col-ratio 0.10 \
  --bhaskera-json /home/bapic_iiitd/2_group/bench_2gpu_bhaskera_results.json \
  --json-out /home/bapic_iiitd/2_group/bench_2gpu_c24_m10_results.json
