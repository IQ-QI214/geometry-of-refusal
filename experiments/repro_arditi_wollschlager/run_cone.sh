#!/usr/bin/env bash
# Cone k=2→5: Qwen (GPU0) and Llama (GPU1) in parallel, k serial within each.
set -e

REPO_ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
LOG_DIR=$REPO_ROOT/experiments/repro_arditi_wollschlager/logs
mkdir -p "$LOG_DIR"

echo "[run_cone.sh] Starting Qwen Cone k=2→5 on GPU0..."
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u experiments/repro_arditi_wollschlager/run_rdo_wrapper.py \
    --model qwen2.5_7b --train_cone --min_cone_dim 2 --max_cone_dim 5 \
    > "$LOG_DIR/cone_qwen.log" 2>&1 &
PID_QWEN=$!
echo "  Qwen PID=$PID_QWEN, log: $LOG_DIR/cone_qwen.log"

echo "[run_cone.sh] Starting Llama Cone k=2→5 on GPU1..."
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u experiments/repro_arditi_wollschlager/run_rdo_wrapper.py \
    --model llama3.1_8b --train_cone --min_cone_dim 2 --max_cone_dim 5 \
    > "$LOG_DIR/cone_llama.log" 2>&1 &
PID_LLAMA=$!
echo "  Llama PID=$PID_LLAMA, log: $LOG_DIR/cone_llama.log"

echo "[run_cone.sh] Waiting (expected ~4-6h total)..."
wait $PID_QWEN  && echo "[run_cone.sh] Qwen Cone DONE"  || { echo "[run_cone.sh] Qwen Cone FAILED";  exit 1; }
wait $PID_LLAMA && echo "[run_cone.sh] Llama Cone DONE" || { echo "[run_cone.sh] Llama Cone FAILED"; exit 1; }

echo ""
echo "=== Cone k=2→5 完成 ==="
ls -lh results/repro_arditi_wollschlager/rdo/ 2>/dev/null || echo "  (rdo/ 目录待创建)"
