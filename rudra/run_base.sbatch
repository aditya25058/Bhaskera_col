#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH -J infer_base
#SBATCH -o /home/bapic_iiitd/2_group/infer_base.out
source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col
export PYTHONPATH=src:$PYTHONPATH
python -m bhaskera.launcher.infer \
  --config configs/inference_param2.yaml \
  --prompt-file rudra/prompts.txt \
  --max-new-tokens 128 \
  --no-sample \
  --output-file /home/bapic_iiitd/2_group/outputs_base.txt
