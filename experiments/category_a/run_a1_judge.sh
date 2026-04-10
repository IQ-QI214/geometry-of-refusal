#!/bin/bash
# A1 Judge: Qwen3Guard evaluation (qwen3-vl env, transformers>=4.51)
# 用法: bash experiments/category_a/run_a1_judge.sh [dataset]
# dataset 默认 saladbench，也可传 harmbench (Config-2)
#
# 并行模式: 5 个模型分配到 4 张 GPU (32B 独占 GPU 3，其余共享)

set -e

ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
cd $ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a1_judge.py"
JUDGE="qwen3guard"
DATASET=${1:-saladbench}

echo "=== A1 Judge: ${JUDGE} on ${DATASET} (qwen3-vl env) ==="
echo "=== Parallel mode: GPU0=llava_7b, GPU1=llava_13b, GPU2=qwen2vl_7b+internvl2_8b, GPU3=qwen2vl_32b ==="

mkdir -p results/category_a/{llava_7b,llava_13b,qwen2vl_7b,qwen2vl_32b,internvl2_8b}

# GPU 0: LLaVA-7B
nohup conda run --no-capture-output -n qwen3-vl \
    python $SCRIPT --model llava_7b --judge $JUDGE --dataset $DATASET --device cuda:0 \
    > results/category_a/llava_7b/a1_judge.log 2>&1 &
echo "LLaVA-7B judge PID: $!"

# GPU 1: LLaVA-13B
nohup conda run --no-capture-output -n qwen3-vl \
    python $SCRIPT --model llava_13b --judge $JUDGE --dataset $DATASET --device cuda:1 \
    > results/category_a/llava_13b/a1_judge.log 2>&1 &
echo "LLaVA-13B judge PID: $!"

# GPU 2: Qwen-7B
nohup conda run --no-capture-output -n qwen3-vl \
    python $SCRIPT --model qwen2vl_7b --judge $JUDGE --dataset $DATASET --device cuda:2 \
    > results/category_a/qwen2vl_7b/a1_judge.log 2>&1 &
echo "Qwen-7B judge PID: $!"

# GPU 3: Qwen-32B (needs most VRAM for judge)
nohup conda run --no-capture-output -n qwen3-vl \
    python $SCRIPT --model qwen2vl_32b --judge $JUDGE --dataset $DATASET --device cuda:3 \
    > results/category_a/qwen2vl_32b/a1_judge.log 2>&1 &
echo "Qwen-32B judge PID: $!"

# Wait for GPU 2 to free up, then run InternVL2
echo ""
echo "=== Waiting for Qwen-7B judge to finish before starting InternVL2 ==="
wait
echo ""
echo "=== Starting InternVL2-8B on GPU 2 ==="
nohup conda run --no-capture-output -n qwen3-vl \
    python $SCRIPT --model internvl2_8b --judge $JUDGE --dataset $DATASET --device cuda:2 \
    > results/category_a/internvl2_8b/a1_judge.log 2>&1 &
echo "InternVL2 judge PID: $!"

wait
echo ""
echo "=== All A1 Judge evaluations complete ==="
echo "Results at: results/category_a/*/a1_judged_${JUDGE}_${DATASET}.json"
echo ""
echo "Monitor progress:"
echo "  tail -f results/category_a/llava_7b/a1_judge.log"
echo "  tail -f results/category_a/llava_13b/a1_judge.log"
echo "  tail -f results/category_a/qwen2vl_7b/a1_judge.log"
echo "  tail -f results/category_a/qwen2vl_32b/a1_judge.log"
