#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 00:05:00
#SBATCH -J dma_gate
#SBATCH -o /home/bapic_iiitd/2_group/bench_dma_micro.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH

# Run DMA microbenchmark on the physical A100 GPU
python -u rudra/bench_dma_micro.py \
  --out-json /home/bapic_iiitd/2_group/dma_microbench_results.json \
  --iters 30 \
  --warmup 10

