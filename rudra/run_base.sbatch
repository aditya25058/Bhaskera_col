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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

python -m bhaskera.launcher.infer \
  --config configs/inference_param2.yaml \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-new-tokens 128 \
  --no-sample \
  --output-file /home/bapic_iiitd/2_group/outputs_base.txt
