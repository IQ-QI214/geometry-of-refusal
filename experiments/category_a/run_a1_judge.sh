#!/bin/bash
# A1 Judge: Qwen3Guard evaluation (qwen3-vl env, transformers>=4.51)
# 用法: bash experiments/category_a/run_a1_judge.sh [dataset]
# dataset 默认 saladbench，也可传 harmbench (Config-2)

set -e

ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
cd $ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a1_judge.py"
JUDGE="qwen3guard"
DATASET=${1:-saladbench}

echo "=== A1 Judge: ${JUDGE} on ${DATASET} (qwen3-vl env) ==="
for MODEL in llava_7b llava_13b qwen2vl_7b qwen2vl_32b internvl2_8b; do
    echo "--- Judging $MODEL ---"
    conda run --no-capture-output -n qwen3-vl \
        python $SCRIPT --model $MODEL --judge $JUDGE --dataset $DATASET --device cuda:0
done

echo "=== A1 Judge complete ==="
