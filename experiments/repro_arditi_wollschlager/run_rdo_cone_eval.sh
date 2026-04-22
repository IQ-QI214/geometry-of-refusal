#!/usr/bin/env bash
# RDO/Cone eval completions — 4×H100
#
# Phase A (4-way parallel): rdo_k1 + cone_k3 for both models
#   GPU0: Qwen  rdo_k1
#   GPU1: Llama rdo_k1
#   GPU2: Qwen  cone_k3
#   GPU3: Llama cone_k3
#
# Phase B (2-way parallel): cone_k5 for both models
#   GPU0: Qwen  cone_k5
#   GPU1: Llama cone_k5
#
# Run from repo root:
#   bash experiments/repro_arditi_wollschlager/run_rdo_cone_eval.sh \
#       | tee experiments/repro_arditi_wollschlager/logs/rdo_cone_eval_all.log
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
LOG_DIR=experiments/repro_arditi_wollschlager/logs
mkdir -p "$LOG_DIR"

RUN="conda run -n rdo python experiments/repro_arditi_wollschlager/run_rdo_evaluate.py"
QWEN_DIR=results/repro_arditi_wollschlager/rdo/Qwen2.5-7B-Instruct
LLAMA_DIR=results/repro_arditi_wollschlager/rdo/Llama-3.1-8B-Instruct

# ── Phase A ────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "[Phase A] rdo_k1 + cone_k3 — 4 jobs in parallel"
echo "======================================================================"

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $RUN --model qwen2.5_7b \
         --direction "$QWEN_DIR/rdo_direction.pt" \
         --config rdo_k1 --device cuda:0 \
    > "$LOG_DIR/rdo_eval_qwen.log" 2>&1 &
PID_A=$!; echo "  GPU0: Qwen  rdo_k1   PID=$PID_A  → $LOG_DIR/rdo_eval_qwen.log"

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $RUN --model llama3.1_8b \
         --direction "$LLAMA_DIR/rdo_direction.pt" \
         --config rdo_k1 --device cuda:0 \
    > "$LOG_DIR/rdo_eval_llama.log" 2>&1 &
PID_B=$!; echo "  GPU1: Llama rdo_k1   PID=$PID_B  → $LOG_DIR/rdo_eval_llama.log"

CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $RUN --model qwen2.5_7b \
         --direction "$QWEN_DIR/cone_k3_basis.pt" \
         --config cone_k3 --device cuda:0 \
    > "$LOG_DIR/cone_eval_qwen_k3.log" 2>&1 &
PID_C=$!; echo "  GPU2: Qwen  cone_k3  PID=$PID_C  → $LOG_DIR/cone_eval_qwen_k3.log"

CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $RUN --model llama3.1_8b \
         --direction "$LLAMA_DIR/cone_k3_basis.pt" \
         --config cone_k3 --device cuda:0 \
    > "$LOG_DIR/cone_eval_llama_k3.log" 2>&1 &
PID_D=$!; echo "  GPU3: Llama cone_k3  PID=$PID_D  → $LOG_DIR/cone_eval_llama_k3.log"

wait $PID_A && echo "[OK] Qwen  rdo_k1"  || { echo "[FAIL] Qwen rdo_k1 — see $LOG_DIR/rdo_eval_qwen.log";    exit 1; }
wait $PID_B && echo "[OK] Llama rdo_k1"  || { echo "[FAIL] Llama rdo_k1 — see $LOG_DIR/rdo_eval_llama.log";  exit 1; }
wait $PID_C && echo "[OK] Qwen  cone_k3" || { echo "[FAIL] Qwen cone_k3 — see $LOG_DIR/cone_eval_qwen_k3.log"; exit 1; }
wait $PID_D && echo "[OK] Llama cone_k3" || { echo "[FAIL] Llama cone_k3 — see $LOG_DIR/cone_eval_llama_k3.log"; exit 1; }

# ── Phase B ────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "[Phase B] cone_k5 — 2 jobs in parallel"
echo "======================================================================"

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $RUN --model qwen2.5_7b \
         --direction "$QWEN_DIR/cone_k5_basis.pt" \
         --config cone_k5 --device cuda:0 \
    > "$LOG_DIR/cone_eval_qwen_k5.log" 2>&1 &
PID_E=$!; echo "  GPU0: Qwen  cone_k5  PID=$PID_E  → $LOG_DIR/cone_eval_qwen_k5.log"

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $RUN --model llama3.1_8b \
         --direction "$LLAMA_DIR/cone_k5_basis.pt" \
         --config cone_k5 --device cuda:0 \
    > "$LOG_DIR/cone_eval_llama_k5.log" 2>&1 &
PID_F=$!; echo "  GPU1: Llama cone_k5  PID=$PID_F  → $LOG_DIR/cone_eval_llama_k5.log"

wait $PID_E && echo "[OK] Qwen  cone_k5" || { echo "[FAIL] Qwen cone_k5 — see $LOG_DIR/cone_eval_qwen_k5.log";  exit 1; }
wait $PID_F && echo "[OK] Llama cone_k5" || { echo "[FAIL] Llama cone_k5 — see $LOG_DIR/cone_eval_llama_k5.log"; exit 1; }

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "=== ASR_kw summary (eval completions) ==="
echo "======================================================================"
for log in rdo_eval_qwen rdo_eval_llama cone_eval_qwen_k3 cone_eval_llama_k3 cone_eval_qwen_k5 cone_eval_llama_k5; do
    label=$(echo "$log" | sed 's/_eval_/ /; s/_/ /g')
    grep -E "ASR_kw|Delta" "$LOG_DIR/${log}.log" 2>/dev/null | sed "s/^/  [$label] /" || true
done
echo ""
echo "Next: bash experiments/repro_arditi_wollschlager/run_full_eval.sh"
