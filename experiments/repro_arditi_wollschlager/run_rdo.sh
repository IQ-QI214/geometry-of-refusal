#!/usr/bin/env bash
# RDO k=1 run: Qwen on GPU0, Llama on GPU1 (parallel).
set -e

REPO_ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
LOG_DIR=$REPO_ROOT/experiments/repro_arditi_wollschlager/logs
mkdir -p "$LOG_DIR"

echo "[run_rdo.sh] Starting Qwen RDO k=1 on GPU0..."
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u experiments/repro_arditi_wollschlager/run_rdo_wrapper.py --model qwen2.5_7b \
    > "$LOG_DIR/rdo_qwen.log" 2>&1 &
PID_QWEN=$!
echo "  Qwen PID=$PID_QWEN, log: $LOG_DIR/rdo_qwen.log"

echo "[run_rdo.sh] Starting Llama RDO k=1 on GPU1..."
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u experiments/repro_arditi_wollschlager/run_rdo_wrapper.py --model llama3.1_8b \
    > "$LOG_DIR/rdo_llama.log" 2>&1 &
PID_LLAMA=$!
echo "  Llama PID=$PID_LLAMA, log: $LOG_DIR/rdo_llama.log"

echo "[run_rdo.sh] Waiting for both jobs..."
wait $PID_QWEN  && echo "[run_rdo.sh] Qwen RDO DONE"  || { echo "[run_rdo.sh] Qwen RDO FAILED";  exit 1; }
wait $PID_LLAMA && echo "[run_rdo.sh] Llama RDO DONE" || { echo "[run_rdo.sh] Llama RDO FAILED"; exit 1; }

echo ""
echo "=== RDO k=1 SUMMARY ==="
ls -lh results/repro_arditi_wollschlager/rdo/ 2>/dev/null || echo "  (rdo/ 目录待创建)"
