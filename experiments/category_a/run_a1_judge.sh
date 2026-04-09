#!/bin/bash
# A1 Judge: Run Qwen3Guard evaluation (requires transformers>=4.51 env)
# Usage: conda activate <your_env_with_transformers_451> && bash run_a1_judge.sh
set -e

cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a1_judge.py"
JUDGE="qwen3guard"
DATASET="saladbench"

echo "=== A1 Judge: ${JUDGE} on ${DATASET} ==="
for MODEL in llava_7b llava_13b qwen2vl_7b qwen2vl_32b internvl2_8b; do
    echo "--- Judging $MODEL ---"
    python $SCRIPT --model $MODEL --judge $JUDGE --dataset $DATASET --device cuda:0
done

echo "=== A1 Judge complete ==="
