#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J col_ablation

MISSING_RATIO=${1:-0.50}
OUT_SUFFIX=${2:-m50}

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== GPU LINK TOPOLOGY ==="
nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,memory.total --format=csv || lspci -v | grep -i nvidia
echo "========================="
echo "Starting COLOSSUS Run with C=32, L+4, Gating, missing_col_ratio=${MISSING_RATIO}"

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

python -m bhaskera.launcher.infer \
  --config configs/inference_colossus.yaml \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-new-tokens 64 \
  --no-sample \
  --missing-col-ratio "${MISSING_RATIO}" \
  --output-file "/home/bapic_iiitd/2_group/outputs_c32_l4_${OUT_SUFFIX}.txt"
