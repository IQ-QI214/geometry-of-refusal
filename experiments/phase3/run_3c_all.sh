#!/bin/bash
# run_3c_all.sh — 4 GPU 并行运行 Exp 3C (Ablation Attack)
# 注意: Qwen2.5-VL 需要 qwen3-vl 环境 (transformers >= 4.52)
#       其他 3 个模型使用 rdo 环境
#
# 用法: bash run_3c_all.sh

set -e

PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE3_DIR="$PROJ_DIR/experiments/phase3"
LOG_DIR="$PHASE3_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPT="$PHASE3_DIR/exp_3c_ablation_attack.py"

source "$(conda info --base)/etc/profile.d/conda.sh"

RDO_PYTHON=$(conda run -n rdo which python)
QWEN_PYTHON=$(conda run -n qwen3-vl which python)

echo "============================================================"
echo "Phase 3 Exp 3C: Ablation Attack (4 models parallel)"
echo "Project dir: $PROJ_DIR"
echo "Logs dir: $LOG_DIR"
echo "rdo python: $RDO_PYTHON"
echo "qwen3-vl python: $QWEN_PYTHON"
echo "============================================================"

echo "[$(date '+%H:%M:%S')] Starting llava_7b on GPU 0..."
CUDA_VISIBLE_DEVICES=0 "$RDO_PYTHON" "$SCRIPT" --model llava_7b --device cuda:0 > "$LOG_DIR/3c_llava.log" 2>&1 &
PID_LLAVA=$!

echo "[$(date '+%H:%M:%S')] Starting qwen2vl_7b on GPU 1 (qwen3-vl env)..."
CUDA_VISIBLE_DEVICES=1 "$QWEN_PYTHON" "$SCRIPT" --model qwen2vl_7b --device cuda:0 > "$LOG_DIR/3c_qwen2vl.log" 2>&1 &
PID_QWEN=$!

echo "[$(date '+%H:%M:%S')] Starting internvl2_8b on GPU 2..."
CUDA_VISIBLE_DEVICES=2 "$RDO_PYTHON" "$SCRIPT" --model internvl2_8b --device cuda:0 > "$LOG_DIR/3c_internvl2.log" 2>&1 &
PID_INTERN=$!

echo "[$(date '+%H:%M:%S')] Starting instructblip_7b on GPU 3..."
CUDA_VISIBLE_DEVICES=3 "$RDO_PYTHON" "$SCRIPT" --model instructblip_7b --device cuda:0 > "$LOG_DIR/3c_instructblip.log" 2>&1 &
PID_BLIP=$!

echo ""
echo "All 4 jobs started. PIDs: llava=$PID_LLAVA, qwen=$PID_QWEN, intern=$PID_INTERN, blip=$PID_BLIP"
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/3c_llava.log"
echo "  tail -f $LOG_DIR/3c_qwen2vl.log"
echo "  tail -f $LOG_DIR/3c_internvl2.log"
echo "  tail -f $LOG_DIR/3c_instructblip.log"
echo ""

wait $PID_LLAVA $PID_QWEN $PID_INTERN $PID_BLIP
echo ""
echo "[$(date '+%H:%M:%S')] All 4 Exp 3C jobs complete."

echo ""
echo "Results:"
for model in llava_7b qwen2vl_7b internvl2_8b instructblip_7b; do
    RESULT_FILE="$PROJ_DIR/results/phase3/$model/exp_3c_results.json"
    if [ -f "$RESULT_FILE" ]; then
        SUMMARY=$("$RDO_PYTHON" -c "
import json
r = json.load(open('$RESULT_FILE'))
configs = r['configs']
lines = []
for name, data in configs.items():
    m = data['metrics']
    lines.append(f\"{name}: bypass={m['initial_bypass_rate']:.3f} full_harm={m['full_harmful_completion_rate']:.3f}\")
print(' | '.join(lines))
")
        echo "  $model (NW=L${nw_layer}): $SUMMARY"
    else
        echo "  $model: FAILED (no result file)"
    fi
done
