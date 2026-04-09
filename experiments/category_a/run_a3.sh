#!/bin/bash
# A3 Norm Prediction: LLaVA-7B (primary)
# 用 rdo env 运行 (LLaVA-7B)
# 用法: bash experiments/category_a/run_a3.sh [gpu_id]
# 默认 GPU 0；如 GPU 0 被 A1 占用，传 cuda:1 等

set -e

ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
cd $ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DEVICE=${1:-cuda:0}
SCRIPT="experiments/category_a/exp_a3_norm_prediction.py"

echo "=== A3 Norm Prediction — LLaVA-7B on $DEVICE ==="
mkdir -p results/category_a/llava_7b

nohup conda run --no-capture-output -n rdo \
    python $SCRIPT --model llava_7b --device $DEVICE \
    > results/category_a/llava_7b/a3.log 2>&1 &
echo "A3 LLaVA-7B PID: $!"
echo "Monitor: tail -f results/category_a/llava_7b/a3.log"
