#!/bin/bash
#SBATCH -A nsmext
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:05:00
#SBATCH -J cap24g_dense
#SBATCH -o /home/bapic_iiitd/2_group/bench_cap24g_dense.out

source /home/bapic_iiitd/2_group/bhaskera-venv/bin/activate
cd /home/bapic_iiitd/2_group/Bhaskera_col

export PYTHONPATH=src:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=========================================================================="
echo "  PART 1: 1-GPU vs 2-GPU DENSE SERVING UNDER 24-GB MEMORY CEILING"
echo "=========================================================================="
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
echo "=========================================================================="

MODEL_DIR=/home/bapic_iiitd/.cache/huggingface/hub/models--bharatgenai--Param2-17B-A2.4B-Thinking/snapshots/1d5b7897cf1eec5ad5159f216e757e71b751c979

echo ""
echo ">>> STEP 1: 1-GPU DENSE SERVING UNDER 24-GB CAP (EXPECTED: OOM) <<<"
python rudra/bench_cap24g_dense_1gpu.py \
  --model-dir "$MODEL_DIR" \
  --prompts-file rudra/prompts.txt \
  --cap-gb 24.0 \
  --output-json /home/bapic_iiitd/2_group/bench_cap24g_dense_1gpu_results.json

echo ""
echo ">>> STEP 2: 2-GPU DENSE LAYER-SHARDED SERVING UNDER 24-GB/GPU CAP (EXPECTED: PASS) <<<"
python rudra/bench_cap24g_dense_2gpu.py \
  --model-dir "$MODEL_DIR" \
  --prompts-file rudra/prompts.txt \
  --cap-gb 24.0 \
  --output-json /home/bapic_iiitd/2_group/bench_cap24g_dense_2gpu_results.json
