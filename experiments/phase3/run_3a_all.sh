#!/bin/bash
# run_3a_all.sh
# 4 个模型并行，每个 GPU 一个
# 用法: bash run_3a_all.sh [--skip MODEL_NAME]
#
# CUDA_VISIBLE_DEVICES=X + --device cuda:0: 映射后每个进程只看到 1 张卡

set -e

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE3_DIR="$PROJ_DIR/experiments/phase3"
LOG_DIR="$PHASE3_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT="$PHASE3_DIR/exp_3a_amplitude_reversal.py"

# 激活 conda rdo 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rdo
PYTHON=$(conda run -n rdo which python)
echo "Python: $PYTHON"

echo "============================================================"
echo "Phase 3 Exp 3A: Amplitude Reversal (4 models parallel)"
echo "Project dir: $PROJ_DIR"
echo "Logs dir: $LOG_DIR"
echo "============================================================"

echo "[$(date '+%H:%M:%S')] Starting llava_7b on GPU 0..."
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT" \
    --model llava_7b --device cuda:0 \
    > "$LOG_DIR/llava7b.log" 2>&1 &
PID_LLAVA=$!

echo "[$(date '+%H:%M:%S')] Starting qwen2vl_7b on GPU 1..."
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPT" \
    --model qwen2vl_7b --device cuda:0 \
    > "$LOG_DIR/qwen2vl.log" 2>&1 &
PID_QWEN=$!

echo "[$(date '+%H:%M:%S')] Starting internvl2_8b on GPU 2..."
CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$SCRIPT" \
    --model internvl2_8b --device cuda:0 \
    > "$LOG_DIR/internvl2.log" 2>&1 &
PID_INTERN=$!

echo "[$(date '+%H:%M:%S')] Starting instructblip_7b on GPU 3..."
CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$SCRIPT" \
    --model instructblip_7b --device cuda:0 \
    > "$LOG_DIR/instructblip.log" 2>&1 &
PID_BLIP=$!

echo ""
echo "All 4 jobs started. PIDs: llava=$PID_LLAVA, qwen=$PID_QWEN, intern=$PID_INTERN, blip=$PID_BLIP"
echo "Logs: $LOG_DIR/{llava7b,qwen2vl,internvl2,instructblip}.log"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/llava7b.log"
echo "  tail -f $LOG_DIR/qwen2vl.log"
echo ""

wait $PID_LLAVA $PID_QWEN $PID_INTERN $PID_BLIP
echo ""
echo "[$(date '+%H:%M:%S')] All 4 Exp 3A jobs complete."

# 汇总结果
echo ""
echo "Results:"
for model in llava_7b qwen2vl_7b internvl2_8b instructblip_7b; do
    RESULT_FILE="$PROJ_DIR/results/phase3/$model/exp_3a_results.json"
    if [ -f "$RESULT_FILE" ]; then
        NW=$("$PYTHON" -c "import json; r=json.load(open('$RESULT_FILE')); print(f\"narrow_waist=L{r['narrow_waist_layer']}(depth={r['narrow_waist_relative_depth']:.2f},cos={r['narrow_waist_cos']:.3f}), reversal={r['amplitude_reversal_exists']}\")")
        echo "  $model: $NW"
    else
        echo "  $model: FAILED (no result file)"
    fi
done
