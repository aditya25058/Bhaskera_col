#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -J col_lossless
#SBATCH -o /home/bapic_iiitd/2_group/lossless_sa_ffn.out
source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col
python rudra/lossless_sa_ffn.py
