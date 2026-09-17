#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J mixtral_col1
#SBATCH -o /home/bapic_iiitd/2_group/bench_mixtral_colossus_1gpu.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================================================="
echo "  RUN 3: 1-GPU COLOSSUS MIXTRAL-8x7B (RESIDENCY SWEEP & 1 < 2 PROOF)"
echo "=========================================================================="
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
echo "=========================================================================="

python rudra/bench_mixtral_colossus_1gpu.py \
  --model-dir /home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1 \
  --prompts-file rudra/prompts.txt \
  --output-json /home/bapic_iiitd/2_group/bench_mixtral_colossus_1gpu_results.json \
  --ref-tokens /home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt
