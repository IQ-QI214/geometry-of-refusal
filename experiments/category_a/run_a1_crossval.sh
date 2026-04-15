#!/bin/bash
# Run A1 cross-validation (LLaMA-Guard on disputed subsets) — 4 GPU parallel
#
# Usage: bash experiments/category_a/run_a1_crossval.sh
# Requires: qwen3-vl env (transformers>=4.51), HF offline mode, GPU node

set -e
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

mkdir -p results/category_a/qwen2vl_7b
mkdir -p results/category_a/qwen2vl_32b
mkdir -p results/category_a/llava_7b
mkdir -p results/category_a/llava_13b

echo "[$(date)] Starting A1 cross-validation (4 GPU parallel)..."

# GPU 0: Qwen-7B (~1095 disputed cases, ~15min)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/category_a/exp_a1_cross_validate.py \
    --model qwen2vl_7b --device cuda:0 \
    > results/category_a/qwen2vl_7b/a1_crossval.log 2>&1 &
echo "  [GPU 0] qwen2vl_7b started (PID $!)"

# GPU 1: Qwen-32B (~1510 disputed cases, ~20min)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/category_a/exp_a1_cross_validate.py \
    --model qwen2vl_32b --device cuda:1 \
    > results/category_a/qwen2vl_32b/a1_crossval.log 2>&1 &
echo "  [GPU 1] qwen2vl_32b started (PID $!)"

# GPU 2: LLaVA-7B (~253 disputed cases, ~5min, negative control)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/category_a/exp_a1_cross_validate.py \
    --model llava_7b --device cuda:2 \
    > results/category_a/llava_7b/a1_crossval.log 2>&1 &
echo "  [GPU 2] llava_7b started (PID $!)"

# GPU 3: LLaVA-13B (~349 disputed cases, ~5min, negative control)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/category_a/exp_a1_cross_validate.py \
    --model llava_13b --device cuda:3 \
    > results/category_a/llava_13b/a1_crossval.log 2>&1 &
echo "  [GPU 3] llava_13b started (PID $!)"

echo ""
echo "All 4 jobs launched. Monitor with:"
echo "  tail -f results/category_a/qwen2vl_7b/a1_crossval.log"
echo "  tail -f results/category_a/qwen2vl_32b/a1_crossval.log"
echo "  tail -f results/category_a/llava_7b/a1_crossval.log"
echo "  tail -f results/category_a/llava_13b/a1_crossval.log"
