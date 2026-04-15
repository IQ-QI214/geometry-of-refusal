#!/bin/bash
# Phase 2 实验运行脚本
# 用法: bash experiments/phase2/run_phase2.sh [exp_name...]
#   bash experiments/phase2/run_phase2.sh          # 运行全部
#   bash experiments/phase2/run_phase2.sh 2a 2b    # 只运行 Exp 2A 和 2B
#   USE_4BIT=1 bash experiments/phase2/run_phase2.sh  # 4-bit 量化模式

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# 环境变量
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=""
if [ "${USE_4BIT:-0}" = "1" ]; then
    EXTRA_ARGS="--use_4bit"
    echo "Using 4-bit quantization mode"
fi

# 确定要运行的实验
EXPERIMENTS="${@:-2a 2b 2c 2d}"

run_exp() {
    local exp=$1
    local start_time=$(date +%s)
    echo ""
    echo "============================================================"
    echo "  Running Exp $exp  $(date)"
    echo "============================================================"

    case $exp in
        2a)
            python experiments/phase2/exp_2a/exp_2a_confound_resolution.py $EXTRA_ARGS \
                2>&1 | tee results/exp_2a.log
            ;;
        2b)
            python experiments/phase2/exp_2b/exp_2b_ablation_attack.py $EXTRA_ARGS \
                2>&1 | tee results/exp_2b_phase2.log
            ;;
        2c)
            python experiments/phase2/exp_2c/exp_2c_visual_perturbation.py $EXTRA_ARGS \
                2>&1 | tee results/exp_2c_phase2.log
            ;;
        2d)
            python experiments/phase2/exp_2d/exp_2d_ablation_study.py $EXTRA_ARGS \
                2>&1 | tee results/exp_2d.log
            ;;
        *)
            echo "Unknown experiment: $exp"
            return 1
            ;;
    esac

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "  Exp $exp completed in ${elapsed}s"
}

# 运行
for exp in $EXPERIMENTS; do
    run_exp $exp
done

echo ""
echo "============================================================"
echo "  All Phase 2 experiments completed  $(date)"
echo "============================================================"
