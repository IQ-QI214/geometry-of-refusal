#!/bin/bash
# A2 Causality: 3 models across 3 GPUs
# Requires A1 to complete first (needs harmful prefixes from A1 results)
# Env 分配:
#   GPU 0: LLaVA-7B     (rdo env, Type I — NW strategy)
#   GPU 1: Qwen-7B      (qwen3-vl env, Type II — all-layer strategy)
#   GPU 2: InternVL2-8B (rdo env, Type III — all-layer strategy)
#
# 用法: bash experiments/category_a/run_a2.sh

set -e

ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
cd $ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a2_dsa_causality.py"
RDO="conda run --no-capture-output -n rdo"
QWEN="conda run --no-capture-output -n qwen3-vl"

echo "=== A2 Causality: 3 models in parallel ==="

# GPU 0: LLaVA-7B (rdo)
nohup $RDO python $SCRIPT --model llava_7b --device cuda:0 \
    > results/category_a/llava_7b/a2.log 2>&1 &
P0=$!
echo "GPU 0: LLaVA-7B     (PID=$P0, env=rdo)"

# GPU 1: Qwen-7B (qwen3-vl)
nohup $QWEN python $SCRIPT --model qwen2vl_7b --device cuda:1 \
    > results/category_a/qwen2vl_7b/a2.log 2>&1 &
P1=$!
echo "GPU 1: Qwen-7B      (PID=$P1, env=qwen3-vl)"

# GPU 2: InternVL2-8B (rdo)
nohup $RDO python $SCRIPT --model internvl2_8b --device cuda:2 \
    > results/category_a/internvl2_8b/a2.log 2>&1 &
P2=$!
echo "GPU 2: InternVL2-8B (PID=$P2, env=rdo)"

echo ""
echo "Monitor: tail -f results/category_a/{llava_7b,qwen2vl_7b,internvl2_8b}/a2.log"
wait $P0 $P1 $P2
echo "=== A2 complete ==="
