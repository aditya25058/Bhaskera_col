#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J mem_tiers
#SBATCH -o /home/bapic_iiitd/2_group/bench_memory_tiers.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1

echo "=== TOPOLOGY PROBE ==="
nvidia-smi topo -m || nvidia-smi
echo "======================"

python rudra/bench_memory_tiers.py
