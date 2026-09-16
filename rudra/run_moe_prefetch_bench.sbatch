#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J moe_bench
#SBATCH -o /home/bapic_iiitd/2_group/bench_moe.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

python tests/benchmarks/bench_moe_prefetch.py \
  --model-dir "$MODEL_DIR" \
  --layer-idx 1 \
  --num-tokens 128 \
  --cache-sizes 8 16 24 32

