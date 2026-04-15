#!/bin/bash
# Phase 2 并行运行脚本 — Exp 2A 和 2B 在不同 GPU 上并行
# 用法: bash experiments/phase2/run_phase2_parallel.sh [GPU_2A] [GPU_2B]
#   bash experiments/phase2/run_phase2_parallel.sh 0 1     # GPU 0 跑 2A, GPU 1 跑 2B
#   USE_4BIT=1 bash experiments/phase2/run_phase2_parallel.sh 0 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

GPU_2A=${1:-0}
GPU_2B=${2:-1}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=""
if [ "${USE_4BIT:-0}" = "1" ]; then
    EXTRA_ARGS="--use_4bit"
fi

echo "Phase 2 Parallel Execution"
echo "  Exp 2A on GPU $GPU_2A"
echo "  Exp 2B on GPU $GPU_2B"
echo ""

# 并行运行 Exp 2A 和 2B
CUDA_VISIBLE_DEVICES=$GPU_2A python experiments/phase2/exp_2a/exp_2a_confound_resolution.py \
    $EXTRA_ARGS 2>&1 | tee results/exp_2a.log &
PID_2A=$!

CUDA_VISIBLE_DEVICES=$GPU_2B python experiments/phase2/exp_2b/exp_2b_ablation_attack.py \
    $EXTRA_ARGS 2>&1 | tee results/exp_2b_phase2.log &
PID_2B=$!

echo "Waiting for Exp 2A (PID $PID_2A) and Exp 2B (PID $PID_2B)..."
wait $PID_2A
echo "Exp 2A completed"
wait $PID_2B
echo "Exp 2B completed"

echo ""
echo "Both done. Check results/exp_2a_results.json and results/exp_2b_results.json"
echo "Next: run Exp 2C (depends on both 2A and 2B results)"
echo "  python experiments/phase2/exp_2c/exp_2c_visual_perturbation.py $EXTRA_ARGS"
