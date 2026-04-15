#!/bin/bash
# Gap C Pilot Experiments - 并行运行脚本
# 在 3 张 GPU 上同时运行 Exp A/B/C，第 4 张空闲备用
#
# 使用方法：
#   bash run_pilot_parallel.sh              # 默认 GPU 0,1,2
#   bash run_pilot_parallel.sh 0 1 2        # 指定 GPU ID
#   USE_4BIT=1 bash run_pilot_parallel.sh   # 启用 4-bit 量化
#
# LLaVA-1.5-7B fp16 约需 14GB 显存，H100 80GB 完全够用

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate rdo

# 强制离线模式（GPU 机器无网络）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# GPU 分配（默认 0, 1, 2）
GPU_A=${1:-0}
GPU_B=${2:-1}
GPU_C=${3:-2}

# 结果目录
RESULTS_DIR="${SAVE_DIR:-./results}"
mkdir -p "$RESULTS_DIR"

# 4-bit 量化开关
EXTRA_ARGS=""
if [ "${USE_4BIT}" = "1" ]; then
    EXTRA_ARGS="--use_4bit"
    echo "[INFO] 4-bit quantization enabled"
fi

echo "========================================"
echo "Gap C Pilot Experiments (PARALLEL)"
echo "  Exp A -> GPU ${GPU_A}"
echo "  Exp B -> GPU ${GPU_B}"
echo "  Exp C -> GPU ${GPU_C}"
echo "  Results: ${RESULTS_DIR}/"
echo "========================================"
echo ""

# 启动三个实验，各自绑定不同 GPU
echo "[$(date '+%H:%M:%S')] Launching Exp A on GPU ${GPU_A} ..."
CUDA_VISIBLE_DEVICES=${GPU_A} python exp_a/exp_a_modality_stability.py ${EXTRA_ARGS} \
    2>&1 | tee "${RESULTS_DIR}/exp_a.log" &
PID_A=$!

echo "[$(date '+%H:%M:%S')] Launching Exp B on GPU ${GPU_B} ..."
CUDA_VISIBLE_DEVICES=${GPU_B} python exp_b/exp_b_timestep_consistency.py ${EXTRA_ARGS} \
    2>&1 | tee "${RESULTS_DIR}/exp_b.log" &
PID_B=$!

echo "[$(date '+%H:%M:%S')] Launching Exp C on GPU ${GPU_C} ..."
CUDA_VISIBLE_DEVICES=${GPU_C} python exp_c/exp_c_delayed_reactivation.py ${EXTRA_ARGS} \
    2>&1 | tee "${RESULTS_DIR}/exp_c.log" &
PID_C=$!

echo ""
echo "PIDs: A=${PID_A}, B=${PID_B}, C=${PID_C}"
echo "Waiting for all experiments to finish ..."
echo ""

# 等待各实验完成，记录退出状态
FAIL=0

wait $PID_A
STATUS_A=$?
echo "[$(date '+%H:%M:%S')] Exp A finished (exit code: ${STATUS_A})"
[ $STATUS_A -ne 0 ] && FAIL=1

wait $PID_B
STATUS_B=$?
echo "[$(date '+%H:%M:%S')] Exp B finished (exit code: ${STATUS_B})"
[ $STATUS_B -ne 0 ] && FAIL=1

wait $PID_C
STATUS_C=$?
echo "[$(date '+%H:%M:%S')] Exp C finished (exit code: ${STATUS_C})"
[ $STATUS_C -ne 0 ] && FAIL=1

echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"
echo "  Exp A (GPU ${GPU_A}): $([ $STATUS_A -eq 0 ] && echo 'OK' || echo 'FAILED')"
echo "  Exp B (GPU ${GPU_B}): $([ $STATUS_B -eq 0 ] && echo 'OK' || echo 'FAILED')"
echo "  Exp C (GPU ${GPU_C}): $([ $STATUS_C -eq 0 ] && echo 'OK' || echo 'FAILED')"
echo ""
echo "Result files:"
[ -f "${RESULTS_DIR}/exp_a_results.json" ] && echo "  exp_a_results.json  OK" || echo "  exp_a_results.json  MISSING"
[ -f "${RESULTS_DIR}/exp_b_results.json" ] && echo "  exp_b_results.json  OK" || echo "  exp_b_results.json  MISSING"
[ -f "${RESULTS_DIR}/exp_c_results.json" ] && echo "  exp_c_results.json  OK" || echo "  exp_c_results.json  MISSING"
echo "========================================"

exit $FAIL
