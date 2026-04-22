#!/usr/bin/env bash
# RDO k=1 run: Qwen on GPU0, Llama on GPU1 (parallel).
set -e

REPO_ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
LOG_DIR=$REPO_ROOT/experiments/repro_arditi_wollschlager/logs
SAVE_DIR=$REPO_ROOT/results/repro_arditi_wollschlager
QWEN_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct
LLAMA_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Llama-3.1-8B-Instruct

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "[run_rdo.sh] Starting Qwen RDO k=1 on GPU0..."
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u rdo.py \
    --model "$QWEN_PATH" \
    --train_direction \
    --splits saladbench \
    --save_dir "$SAVE_DIR" \
    --dim_dir "dim" \
    > "$LOG_DIR/rdo_qwen_${TS}.log" 2>&1 &
PID_QWEN=$!
echo "  Qwen PID=$PID_QWEN, log: $LOG_DIR/rdo_qwen_${TS}.log"

echo "[run_rdo.sh] Starting Llama RDO k=1 on GPU1..."
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u rdo.py \
    --model "$LLAMA_PATH" \
    --train_direction \
    --splits saladbench \
    --save_dir "$SAVE_DIR" \
    --dim_dir "dim" \
    > "$LOG_DIR/rdo_llama_${TS}.log" 2>&1 &
PID_LLAMA=$!
echo "  Llama PID=$PID_LLAMA, log: $LOG_DIR/rdo_llama_${TS}.log"

echo "[run_rdo.sh] Waiting for both jobs..."
wait $PID_QWEN  && echo "[run_rdo.sh] Qwen RDO DONE"  || { echo "[run_rdo.sh] Qwen RDO FAILED"; exit 1; }
wait $PID_LLAMA && echo "[run_rdo.sh] Llama RDO DONE" || { echo "[run_rdo.sh] Llama RDO FAILED"; exit 1; }

echo ""
echo "=== RDO k=1 SUMMARY ==="
ls -lh "$SAVE_DIR/rdo/" 2>/dev/null || echo "  (rdo/ 目录待创建)"
