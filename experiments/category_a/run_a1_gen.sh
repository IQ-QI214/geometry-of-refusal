#!/bin/bash
# A1 Generation: 4×H100 parallel execution
# Phase 0 (direction extraction) must be done first for llava_13b and qwen2vl_32b
set -e

cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT="experiments/category_a/exp_a1_dsa_validation.py"

echo "=== A1 Generation: Launching 4 models in parallel ==="

# GPU 0: LLaVA-7B
mkdir -p results/category_a/llava_7b
nohup python $SCRIPT --model llava_7b --device cuda:0 --resume \
    > results/category_a/llava_7b/a1_gen.log 2>&1 &
PID0=$!
echo "GPU 0: LLaVA-7B (PID=$PID0)"

# GPU 1: LLaVA-13B
mkdir -p results/category_a/llava_13b
nohup python $SCRIPT --model llava_13b --device cuda:1 --resume \
    > results/category_a/llava_13b/a1_gen.log 2>&1 &
PID1=$!
echo "GPU 1: LLaVA-13B (PID=$PID1)"

# GPU 2: Qwen-7B (then InternVL2 after)
mkdir -p results/category_a/qwen2vl_7b results/category_a/internvl2_8b
nohup bash -c "
    cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python $SCRIPT --model qwen2vl_7b --device cuda:2 --resume && \
    python $SCRIPT --model internvl2_8b --device cuda:2 --resume
" > results/category_a/gpu2_a1_gen.log 2>&1 &
PID2=$!
echo "GPU 2: Qwen-7B → InternVL2 (PID=$PID2)"

# GPU 3: Qwen-32B (largest, slowest)
mkdir -p results/category_a/qwen2vl_32b
nohup python $SCRIPT --model qwen2vl_32b --device cuda:3 --resume \
    > results/category_a/qwen2vl_32b/a1_gen.log 2>&1 &
PID3=$!
echo "GPU 3: Qwen-32B (PID=$PID3)"

echo ""
echo "All launched. Monitor with: tail -f results/category_a/*/a1_gen.log"
echo "Wait with: wait $PID0 $PID1 $PID2 $PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "=== A1 Generation complete ==="
