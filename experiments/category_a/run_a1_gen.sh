#!/bin/bash
# A1 Generation: 5 models across 4×H100
# Env 分配:
#   GPU 0: LLaVA-7B     (rdo env)
#   GPU 1: LLaVA-13B    (rdo env)
#   GPU 2: InternVL2-8B (rdo env) → 完成后 Qwen-7B (qwen3-vl env, 串行)
#   GPU 3: Qwen-32B     (qwen3-vl env)
#
# 无需提前 conda activate，脚本用 conda run -n <env> 自动切换。
# 用法: bash experiments/category_a/run_a1_gen.sh

set -e

ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
cd $ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a1_dsa_validation.py"
RDO="conda run --no-capture-output -n rdo"
QWEN="conda run --no-capture-output -n qwen3-vl"

echo "=== A1 Generation: 5 models across 4 GPUs ==="
mkdir -p results/category_a/{llava_7b,llava_13b,qwen2vl_7b,qwen2vl_32b,internvl2_8b}

# GPU 0: LLaVA-7B (rdo)
nohup $RDO python $SCRIPT --model llava_7b --device cuda:0 --resume \
    > results/category_a/llava_7b/a1_gen.log 2>&1 &
P0=$!
echo "GPU 0: LLaVA-7B     (PID=$P0, env=rdo)"

# GPU 1: LLaVA-13B (rdo)
nohup $RDO python $SCRIPT --model llava_13b --device cuda:1 --resume \
    > results/category_a/llava_13b/a1_gen.log 2>&1 &
P1=$!
echo "GPU 1: LLaVA-13B    (PID=$P1, env=rdo)"

# GPU 2: InternVL2 (rdo) 完成后 → Qwen-7B (qwen3-vl) 串行
nohup bash -c "
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd $ROOT
    conda run --no-capture-output -n rdo \
        python $SCRIPT --model internvl2_8b --device cuda:2 --resume \
        >> results/category_a/internvl2_8b/a1_gen.log 2>&1
    conda run --no-capture-output -n qwen3-vl \
        python $SCRIPT --model qwen2vl_7b --device cuda:2 --resume \
        >> results/category_a/qwen2vl_7b/a1_gen.log 2>&1
" > results/category_a/gpu2_a1_gen.log 2>&1 &
P2=$!
echo "GPU 2: InternVL2→Qwen-7B (PID=$P2, env=rdo→qwen3-vl)"

# GPU 3: Qwen-32B (qwen3-vl, ~64GB)
nohup $QWEN python $SCRIPT --model qwen2vl_32b --device cuda:3 --resume \
    > results/category_a/qwen2vl_32b/a1_gen.log 2>&1 &
P3=$!
echo "GPU 3: Qwen-32B     (PID=$P3, env=qwen3-vl)"

echo ""
echo "Monitor: tail -f results/category_a/llava_7b/a1_gen.log"
echo "         tail -f results/category_a/llava_13b/a1_gen.log"
echo "         tail -f results/category_a/internvl2_8b/a1_gen.log"
echo "         tail -f results/category_a/qwen2vl_7b/a1_gen.log"
echo "         tail -f results/category_a/qwen2vl_32b/a1_gen.log"
echo "         tail -f results/category_a/gpu2_a1_gen.log"
echo ""
echo "Wait for all: wait $P0 $P1 $P2 $P3"
wait $P0 $P1 $P2 $P3
echo "=== A1 Generation complete ==="
