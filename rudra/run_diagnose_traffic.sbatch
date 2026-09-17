#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 00:05:00
#SBATCH -J diag_traffic
#SBATCH -o /home/bapic_iiitd/2_group/bench_diagnose_traffic.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u rudra/diagnose_unavoidable_traffic.py \
  --max-new-tokens 16 \
  --output-json /home/bapic_iiitd/2_group/diagnostic_traffic_mixtral.json
