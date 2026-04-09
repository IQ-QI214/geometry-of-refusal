#!/bin/bash
# A3 Norm Prediction: LLaVA-7B primary
set -e

cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a3_norm_prediction.py"

echo "=== A3 Norm Prediction ==="
mkdir -p results/category_a/llava_7b
python $SCRIPT --model llava_7b --device cuda:0

echo "=== A3 complete ==="
