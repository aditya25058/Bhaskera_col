#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J bench_adetr
#SBATCH -o /home/bapic_iiitd/2_group/bench_adetr_stage4.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== GPU LINK TOPOLOGY ==="
nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,memory.total --format=csv || lspci -v | grep -i nvidia
echo "========================="

python rudra/bench_adetr_stage4.py
