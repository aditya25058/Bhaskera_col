#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J bench_ab
#SBATCH -o /home/bapic_iiitd/2_group/bench_controlled_ab.out

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

python rudra/bench_controlled_ab.py \
  --model "$MODEL_DIR" \
  --prompt-file rudra/prompts.txt \
  --max-tokens 64 \
  --output-dir /home/bapic_iiitd/2_group \
  --hot-expert-topk 32 \
  --missing-col-ratio 0.10 \
  --json-out /home/bapic_iiitd/2_group/bench_ab_results.json

echo ""
echo "=== OUTPUT CORRECTNESS DIFF ==="
diff -u /home/bapic_iiitd/2_group/bench_ab_baseline.txt \
        /home/bapic_iiitd/2_group/bench_ab_colossus.txt
echo "diff exit code: $?"
