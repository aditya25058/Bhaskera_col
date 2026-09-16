#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH -J col_scaffold
#SBATCH -o /home/bapic_iiitd/2_group/scaffold_moe_smoke.out
source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col
python rudra/scaffold_moe_smoke.py
