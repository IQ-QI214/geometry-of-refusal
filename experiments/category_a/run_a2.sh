#!/bin/bash
# A2 Causality: 3 models on 3 GPUs in parallel
# Requires A1 to be complete first (needs harmful prefixes)
set -e

cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a2_dsa_causality.py"

echo "=== A2 Causality: 3 models in parallel ==="

nohup python $SCRIPT --model llava_7b --device cuda:0 \
    > results/category_a/llava_7b/a2.log 2>&1 &
echo "GPU 0: LLaVA-7B"

nohup python $SCRIPT --model qwen2vl_7b --device cuda:2 \
    > results/category_a/qwen2vl_7b/a2.log 2>&1 &
echo "GPU 2: Qwen-7B"

nohup python $SCRIPT --model internvl2_8b --device cuda:3 \
    > results/category_a/internvl2_8b/a2.log 2>&1 &
echo "GPU 3: InternVL2-8B"

wait
echo "=== A2 complete ==="
