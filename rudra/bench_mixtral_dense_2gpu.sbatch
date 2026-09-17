#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J mixtral_d2
#SBATCH -o /home/bapic_iiitd/2_group/bench_mixtral_dense_2gpu.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================================================="
echo "  RUN 2: 2-GPU DENSE LAYER-SHARDED MIXTRAL-8x7B (REFERENCE SERVING)"
echo "=========================================================================="
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
echo "=========================================================================="

python rudra/bench_mixtral_dense_2gpu.py \
  --model-dir /home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1 \
  --prompts-file rudra/prompts.txt \
  --max-new-tokens 64 \
  --output-json /home/bapic_iiitd/2_group/bench_mixtral_dense_2gpu_results.json \
  --output-tokens /home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt \
  --output-texts /home/bapic_iiitd/2_group/outputs_mixtral_2gpu.txt
