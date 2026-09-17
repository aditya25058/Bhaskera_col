#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 00:05:00
#SBATCH -J adetr_c2_25pct
#SBATCH -o /home/bapic_iiitd/2_group/bench_adetr_c2_25.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run ADETR sweep point: C=2 slots, 25% hot columns resident in GPU HBM, 75% cold streaming (264 MB cold payload)
python -u rudra/bench_adetr_sweep.py \
  --capacity 2 \
  --missing-col-ratio 0.75 \
  --max-new-tokens 16 \
  --prompt-idx 0 \
  --out-tokens /home/bapic_iiitd/2_group/tokens_adetr_c2_25.pt \
  --out-metrics /home/bapic_iiitd/2_group/metrics_adetr_c2_25.json
