#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 00:05:00
#SBATCH -J run3_col
#SBATCH -o /home/bapic_iiitd/2_group/run3_colossus_1gpu.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u rudra/test_mixtral_exactness.py --capacity 4 --max-new-tokens 16 --prompt-idx 1
