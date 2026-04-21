#!/usr/bin/env bash
# Full DIM run: Qwen on GPU0, Llama on GPU1 (parallel).
# Run from repo root after Gate 1 (T6) passes.
set -e

LOG_DIR=experiments/repro_arditi_wollschlager/logs
mkdir -p "$LOG_DIR"

source activate rdo

echo "[run_dim.sh] Starting Qwen DIM on GPU0..."
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python -u experiments/repro_arditi_wollschlager/run_dim.py \
    --model qwen2.5_7b \
    > "$LOG_DIR/dim_qwen.log" 2>&1 &
PID_QWEN=$!
echo "  Qwen PID=$PID_QWEN, log: $LOG_DIR/dim_qwen.log"

echo "[run_dim.sh] Starting Llama DIM on GPU1..."
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    python -u experiments/repro_arditi_wollschlager/run_dim.py \
    --model llama3.1_8b \
    > "$LOG_DIR/dim_llama.log" 2>&1 &
PID_LLAMA=$!
echo "  Llama PID=$PID_LLAMA, log: $LOG_DIR/dim_llama.log"

echo "[run_dim.sh] Waiting for both jobs..."
wait $PID_QWEN  && echo "[run_dim.sh] Qwen DIM DONE"  || { echo "[run_dim.sh] Qwen DIM FAILED"; exit 1; }
wait $PID_LLAMA && echo "[run_dim.sh] Llama DIM DONE" || { echo "[run_dim.sh] Llama DIM FAILED"; exit 1; }

echo ""
echo "=== DIM SUMMARY ==="
grep -E "ASR_kw|Delta|direction.pt" "$LOG_DIR/dim_qwen.log"  | sed 's/^/  [Qwen]  /'
grep -E "ASR_kw|Delta|direction.pt" "$LOG_DIR/dim_llama.log" | sed 's/^/  [Llama] /'
echo ""
echo "Gate 2 check: verify both models show ablation ASR > baseline + 5%"
echo "Show qi these numbers for Gate 2 approval before starting T8/T9."
