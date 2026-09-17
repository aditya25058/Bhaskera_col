#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J bench_2gpu
#SBATCH -o /home/bapic_iiitd/2_group/bench_2gpu_dense.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== GPU LINK TOPOLOGY ==="
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
echo "========================="

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

python rudra/bench_2gpu_dense.py \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-tokens 64 \
  --output-dir /home/bapic_iiitd/2_group \
  --json-out /home/bapic_iiitd/2_group/bench_2gpu_dense_results.json

echo ""
echo "=== OUTPUT EXACTNESS DIFF ==="
diff -u /home/bapic_iiitd/2_group/outputs_1gpu.txt \
        /home/bapic_iiitd/2_group/outputs_2gpu.txt
echo "diff exit code: $?"
