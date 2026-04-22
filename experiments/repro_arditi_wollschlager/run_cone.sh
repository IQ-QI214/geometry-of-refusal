#!/usr/bin/env bash
# Cone k=2→5: Qwen (GPU0) and Llama (GPU1) in parallel, k serial within each.
set -e

REPO_ROOT=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
LOG_DIR=$REPO_ROOT/experiments/repro_arditi_wollschlager/logs
SAVE_DIR=$REPO_ROOT/results/repro_arditi_wollschlager
QWEN_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct
LLAMA_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Llama-3.1-8B-Instruct

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "[run_cone.sh] Starting Qwen Cone k=2→5 on GPU0..."
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u rdo.py \
    --model "$QWEN_PATH" \
    --train_cone \
    --min_cone_dim 2 --max_cone_dim 5 \
    --splits saladbench \
    --save_dir "$SAVE_DIR" \
    --dim_dir "dim" \
    > "$LOG_DIR/cone_qwen_${TS}.log" 2>&1 &
PID_QWEN=$!
echo "  Qwen PID=$PID_QWEN, log: $LOG_DIR/cone_qwen_${TS}.log"

echo "[run_cone.sh] Starting Llama Cone k=2→5 on GPU1..."
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output --cwd "$REPO_ROOT" -n rdo \
    python -u rdo.py \
    --model "$LLAMA_PATH" \
    --train_cone \
    --min_cone_dim 2 --max_cone_dim 5 \
    --splits saladbench \
    --save_dir "$SAVE_DIR" \
    --dim_dir "dim" \
    > "$LOG_DIR/cone_llama_${TS}.log" 2>&1 &
PID_LLAMA=$!
echo "  Llama PID=$PID_LLAMA, log: $LOG_DIR/cone_llama_${TS}.log"

echo "[run_cone.sh] Waiting (expected ~4-6h total)..."
wait $PID_QWEN  && echo "[run_cone.sh] Qwen Cone DONE"  || { echo "[run_cone.sh] Qwen Cone FAILED"; exit 1; }
wait $PID_LLAMA && echo "[run_cone.sh] Llama Cone DONE" || { echo "[run_cone.sh] Llama Cone FAILED"; exit 1; }

echo ""
echo "=== Cone k=2→5 完成 ==="
ls -lh "$SAVE_DIR/rdo/" 2>/dev/null
