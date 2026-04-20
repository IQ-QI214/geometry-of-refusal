#!/usr/bin/env bash
# RDO k=1 run: Qwen on GPU0, Llama on GPU1 (parallel).
# Run from repo root after Gate 2 (T7) passes.
set -e

LOG_DIR=experiments/repro_arditi_wollschlager/logs
mkdir -p "$LOG_DIR"

QWEN_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct
LLAMA_PATH=/inspire/hdd/global_user/wenming-253108090054/models/Llama-3.1-8B-Instruct

echo "[run_rdo.sh] Starting Qwen RDO k=1 on GPU0..."
CUDA_VISIBLE_DEVICES=0 \
    SAVE_DIR=results/repro_arditi_wollschlager \
    DIM_DIR=. \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
    conda run -n rdo \
    python rdo.py \
        --model "$QWEN_PATH" \
        --train_direction \
        --splits saladbench \
    > "$LOG_DIR/rdo_qwen.log" 2>&1 &
PID_QWEN=$!
echo "  Qwen PID=$PID_QWEN, log: $LOG_DIR/rdo_qwen.log"

echo "[run_rdo.sh] Starting Llama RDO k=1 on GPU1..."
CUDA_VISIBLE_DEVICES=1 \
    SAVE_DIR=results/repro_arditi_wollschlager \
    DIM_DIR=. \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
    conda run -n rdo \
    python rdo.py \
        --model "$LLAMA_PATH" \
        --train_direction \
        --splits saladbench \
    > "$LOG_DIR/rdo_llama.log" 2>&1 &
PID_LLAMA=$!
echo "  Llama PID=$PID_LLAMA, log: $LOG_DIR/rdo_llama.log"

echo "[run_rdo.sh] Waiting for both jobs..."
wait $PID_QWEN  && echo "[run_rdo.sh] Qwen RDO DONE"  || { echo "[run_rdo.sh] Qwen RDO FAILED";  exit 1; }
wait $PID_LLAMA && echo "[run_rdo.sh] Llama RDO DONE" || { echo "[run_rdo.sh] Llama RDO FAILED"; exit 1; }

echo ""
echo "=== RDO k=1 完成 ==="
echo "下一步：提取 wandb artifact 中的 lowest_loss_vector.pt，然后跑 run_rdo_evaluate.py"
echo "  find wandb/offline-run-* -name 'lowest_loss_vector.pt' | sort"
ls -lh results/repro_arditi_wollschlager/rdo/ 2>/dev/null || echo "  (rdo/ 目录待创建)"
