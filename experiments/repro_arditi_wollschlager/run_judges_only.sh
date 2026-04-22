#!/usr/bin/env bash
# Judge evaluation: LG3 (GPU0+1) → SR (GPU0+1)
# 两阶段串行：LG3 全部跑完再跑 SR，避免 8 个大模型同时抢显存。
# LG3-8B 约 16GB × 2，SR gemma-2b 约 5GB × 2，每阶段最多用 2 卡。
#
# Usage (from repo root):
#   bash experiments/repro_arditi_wollschlager/run_judges_only.sh \
#       | tee experiments/repro_arditi_wollschlager/logs/judges_only.log
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
LOG="experiments/repro_arditi_wollschlager/logs"
mkdir -p "$LOG"
EVAL="experiments/repro_arditi_wollschlager/run_evaluate.py"
BASE_ENV="PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

# ── LG3: GPU0 + GPU1 parallel ─────────────────────────────────────────────
echo "======================================================================"
echo "[LG3] LlamaGuard3 — GPU0: Qwen  GPU1: Llama"
echo "======================================================================"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    conda run -n rdo python "$EVAL" \
    --judge llamaguard3 --model qwen2.5_7b --device cuda:0 \
    > "$LOG/eval_lg3_qwen.log" 2>&1 &
PID_A=$!
echo "  GPU0: LG3 Qwen  PID=$PID_A"

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    conda run -n rdo python "$EVAL" \
    --judge llamaguard3 --model llama3.1_8b --device cuda:0 \
    > "$LOG/eval_lg3_llama.log" 2>&1 &
PID_B=$!
echo "  GPU1: LG3 Llama PID=$PID_B"

wait $PID_A && echo "[OK] LG3 Qwen"  || { echo "[FAIL] LG3 Qwen  — see $LOG/eval_lg3_qwen.log";  exit 1; }
wait $PID_B && echo "[OK] LG3 Llama" || { echo "[FAIL] LG3 Llama — see $LOG/eval_lg3_llama.log"; exit 1; }

# ── SR: GPU0 + GPU1 parallel (LG3 models freed first) ────────────────────
echo ""
echo "======================================================================"
echo "[SR] StrongREJECT — GPU0: Qwen  GPU1: Llama"
echo "======================================================================"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    conda run -n rdo python "$EVAL" \
    --judge strongreject --model qwen2.5_7b --device cuda:0 \
    > "$LOG/eval_sr_qwen.log" 2>&1 &
PID_C=$!
echo "  GPU0: SR Qwen  PID=$PID_C"

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    conda run -n rdo python "$EVAL" \
    --judge strongreject --model llama3.1_8b --device cuda:0 \
    > "$LOG/eval_sr_llama.log" 2>&1 &
PID_D=$!
echo "  GPU1: SR Llama PID=$PID_D"

wait $PID_C && echo "[OK] SR Qwen"  || { echo "[FAIL] SR Qwen  — see $LOG/eval_sr_qwen.log";  exit 1; }
wait $PID_D && echo "[OK] SR Llama" || { echo "[FAIL] SR Llama — see $LOG/eval_sr_llama.log"; exit 1; }

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "[Summary] compute_summary.py"
echo "======================================================================"
python experiments/repro_arditi_wollschlager/compute_summary.py \
    | tee "$LOG/summary.log"

echo ""
echo "=== ALL DONE ==="
echo ""
cat results/repro_arditi_wollschlager/summary.md
