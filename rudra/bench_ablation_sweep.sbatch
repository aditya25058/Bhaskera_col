#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -J bench_ablation
#SBATCH -o /home/bapic_iiitd/2_group/bench_ablation_sweep.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export BHASKERA_BACKEND=hf

echo "=== GPU LINK TOPOLOGY ==="
nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,memory.total --format=csv
echo "========================="

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

python rudra/bench_ablation_sweep.py \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-tokens 64 \
  --output-dir /home/bapic_iiitd/2_group \
  --json-out /home/bapic_iiitd/2_group/bench_ablation_results.json

echo ""
echo "=== ABLATION COMPLETE ==="
