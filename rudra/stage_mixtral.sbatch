#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -J stage_mixtral
#SBATCH -o /home/bapic_iiitd/2_group/stage_mixtral.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

python rudra/stage_mixtral.py
