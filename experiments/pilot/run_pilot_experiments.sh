#!/bin/bash
# Gap C Pilot Experiments 运行脚本
# 使用方法：
#   bash run_pilot_experiments.sh           # 运行全部 3 个实验
#   bash run_pilot_experiments.sh a         # 只运行 Exp A
#   bash run_pilot_experiments.sh b         # 只运行 Exp B
#   bash run_pilot_experiments.sh c         # 只运行 Exp C
#   bash run_pilot_experiments.sh a b       # 运行 Exp A 和 B
#
# 可选环境变量：
#   USE_4BIT=1 bash run_pilot_experiments.sh   # 启用 4-bit 量化（节省显存）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate rdo

# 强制离线模式（GPU 机器无网络）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 结果目录
RESULTS_DIR="${SAVE_DIR:-./results}"
mkdir -p "$RESULTS_DIR"

# 4-bit 量化开关
EXTRA_ARGS=""
if [ "${USE_4BIT}" = "1" ]; then
    EXTRA_ARGS="--use_4bit"
    echo "[INFO] 4-bit quantization enabled"
fi

# 确定要运行的实验
if [ $# -eq 0 ]; then
    EXPS="a b c"
else
    EXPS="$*"
fi

echo "========================================"
echo "Gap C Pilot Experiments"
echo "Running: $EXPS"
echo "Results dir: $RESULTS_DIR"
echo "========================================"
echo ""

for exp in $EXPS; do
    case $exp in
        a|A)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Exp A: Modality Stability"
            python exp_a/exp_a_modality_stability.py $EXTRA_ARGS 2>&1 | tee "$RESULTS_DIR/exp_a.log"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp A completed"
            echo ""
            ;;
        b|B)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Exp B: Timestep Consistency"
            python exp_b/exp_b_timestep_consistency.py $EXTRA_ARGS 2>&1 | tee "$RESULTS_DIR/exp_b.log"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp B completed"
            echo ""
            ;;
        c|C)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Exp C: Delayed Safety Reactivation"
            python exp_c/exp_c_delayed_reactivation.py $EXTRA_ARGS 2>&1 | tee "$RESULTS_DIR/exp_c.log"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp C completed"
            echo ""
            ;;
        *)
            echo "[WARN] Unknown experiment: $exp (use a, b, or c)"
            ;;
    esac
done

echo "========================================"
echo "All requested experiments completed."
echo "Check results in: $RESULTS_DIR/"
echo "  exp_a_results.json  - Modality stability"
echo "  exp_b_results.json  - Timestep consistency"
echo "  exp_c_results.json  - Delayed reactivation"
echo "========================================"
