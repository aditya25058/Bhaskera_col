#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH -t 00:05:00
#SBATCH -J gran_frontier
#SBATCH -o /home/bapic_iiitd/2_group/bench_granularity_frontier.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u rudra/bench_granularity_frontier.py \
  --output-json /home/bapic_iiitd/2_group/expert_granularity_frontier.json
