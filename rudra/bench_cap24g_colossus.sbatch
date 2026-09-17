#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J cap24g_col
#SBATCH -o /home/bapic_iiitd/2_group/bench_cap24g_colossus.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=========================================================================="
echo "  PART 2: 1-GPU COLOSSUS RESIDENCY SEARCH UNDER 24-GB MEMORY CEILING"
echo "=========================================================================="
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
echo "=========================================================================="

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

python rudra/bench_cap24g_colossus_1gpu.py \
  --model-dir "$MODEL_DIR" \
  --prompts-file rudra/prompts.txt \
  --cap-gb 24.0 \
  --output-json /home/bapic_iiitd/2_group/bench_cap24g_colossus_1gpu_results.json
