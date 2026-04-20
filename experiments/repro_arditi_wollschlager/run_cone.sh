#!/usr/bin/env bash
# Cone k=2→5: each model runs k=2→5 serially on its own GPU.
# Qwen (GPU0) and Llama (GPU1) run in parallel with each other.
# Run from repo root after T8 (RDO k=1) is complete.
set -e

LOG_DIR=experiments/repro_arditi_wollschlager/logs
mkdir -p "$LOG_DIR"

QWEN_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct
LLAMA_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Llama-3.1-8B-Instruct

echo "[run_cone.sh] Starting Qwen Cone k=2→5 on GPU0..."
CUDA_VISIBLE_DEVICES=0 \
    SAVE_DIR=results/repro_arditi_wollschlager \
    DIM_DIR=. \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
    conda run -n rdo \
    python rdo.py \
        --model "$QWEN_PATH" \
        --train_cone \
        --min_cone_dim 2 \
        --max_cone_dim 5 \
        --splits saladbench \
    > "$LOG_DIR/cone_qwen.log" 2>&1 &
PID_QWEN=$!
echo "  Qwen PID=$PID_QWEN, log: $LOG_DIR/cone_qwen.log"

echo "[run_cone.sh] Starting Llama Cone k=2→5 on GPU1..."
CUDA_VISIBLE_DEVICES=1 \
    SAVE_DIR=results/repro_arditi_wollschlager \
    DIM_DIR=. \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
    conda run -n rdo \
    python rdo.py \
        --model "$LLAMA_PATH" \
        --train_cone \
        --min_cone_dim 2 \
        --max_cone_dim 5 \
        --splits saladbench \
    > "$LOG_DIR/cone_llama.log" 2>&1 &
PID_LLAMA=$!
echo "  Llama PID=$PID_LLAMA, log: $LOG_DIR/cone_llama.log"

echo "[run_cone.sh] Waiting (expected ~4-6h total)..."
wait $PID_QWEN  && echo "[run_cone.sh] Qwen Cone DONE"  || { echo "[run_cone.sh] Qwen Cone FAILED";  exit 1; }
wait $PID_LLAMA && echo "[run_cone.sh] Llama Cone DONE" || { echo "[run_cone.sh] Llama Cone FAILED"; exit 1; }

echo ""
echo "=== Cone k=2→5 完成 ==="
echo "下一步：提取 k=5 的 lowest_loss_vector.pt，然后跑 run_rdo_evaluate.py --config cone_k5"
