#!/usr/bin/env bash
# One-shot full pipeline: RDO/Cone eval → Keyword → LG3+SR → summary
#
# GPU assignment:
#   Phase A (eval completions round 1):
#     GPU0: Qwen  rdo_k1
#     GPU1: Llama rdo_k1
#     GPU2: Qwen  cone_k3
#     GPU3: Llama cone_k3
#   Phase B (eval completions round 2):
#     GPU0: Qwen  cone_k5
#     GPU1: Llama cone_k5
#   Phase C (judges, all 4 in parallel):
#     GPU0: LlamaGuard3 Qwen
#     GPU1: LlamaGuard3 Llama
#     GPU2: StrongREJECT Qwen
#     GPU3: StrongREJECT Llama
#   Phase D (CPU):
#     compute_summary.py
#
# Usage (from repo root):
#   bash experiments/repro_arditi_wollschlager/run_all.sh \
#       | tee experiments/repro_arditi_wollschlager/logs/run_all.log
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

LOG="experiments/repro_arditi_wollschlager/logs"
mkdir -p "$LOG"

EVAL_GEN="conda run -n rdo python experiments/repro_arditi_wollschlager/run_rdo_evaluate.py"
EVAL_JUD="conda run -n rdo python experiments/repro_arditi_wollschlager/run_evaluate.py"
OFFLINE="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

QWEN_RDO=results/repro_arditi_wollschlager/rdo/Qwen2.5-7B-Instruct
LLAMA_RDO=results/repro_arditi_wollschlager/rdo/Llama-3.1-8B-Instruct

banner() { echo ""; echo "======================================================================"; echo "$1"; echo "======================================================================"; }

# ── Phase A: rdo_k1 + cone_k3 (4-way parallel) ───────────────────────────
banner "[Phase A] eval completions: rdo_k1 + cone_k3 — 4×GPU"

CUDA_VISIBLE_DEVICES=0 eval $OFFLINE $EVAL_GEN \
    --model qwen2.5_7b \
    --direction "$QWEN_RDO/rdo_direction.pt" \
    --config rdo_k1 --device cuda:0 \
    > "$LOG/rdo_eval_qwen.log" 2>&1 &
PID_A=$!; echo "  GPU0: Qwen  rdo_k1   → $LOG/rdo_eval_qwen.log"

CUDA_VISIBLE_DEVICES=1 eval $OFFLINE $EVAL_GEN \
    --model llama3.1_8b \
    --direction "$LLAMA_RDO/rdo_direction.pt" \
    --config rdo_k1 --device cuda:0 \
    > "$LOG/rdo_eval_llama.log" 2>&1 &
PID_B=$!; echo "  GPU1: Llama rdo_k1   → $LOG/rdo_eval_llama.log"

CUDA_VISIBLE_DEVICES=2 eval $OFFLINE $EVAL_GEN \
    --model qwen2.5_7b \
    --direction "$QWEN_RDO/cone_k3_basis.pt" \
    --config cone_k3 --device cuda:0 \
    > "$LOG/cone_eval_qwen_k3.log" 2>&1 &
PID_C=$!; echo "  GPU2: Qwen  cone_k3  → $LOG/cone_eval_qwen_k3.log"

CUDA_VISIBLE_DEVICES=3 eval $OFFLINE $EVAL_GEN \
    --model llama3.1_8b \
    --direction "$LLAMA_RDO/cone_k3_basis.pt" \
    --config cone_k3 --device cuda:0 \
    > "$LOG/cone_eval_llama_k3.log" 2>&1 &
PID_D=$!; echo "  GPU3: Llama cone_k3  → $LOG/cone_eval_llama_k3.log"

wait $PID_A && echo "[OK] Qwen  rdo_k1"  || { echo "[FAIL] Qwen  rdo_k1  — see $LOG/rdo_eval_qwen.log";    exit 1; }
wait $PID_B && echo "[OK] Llama rdo_k1"  || { echo "[FAIL] Llama rdo_k1  — see $LOG/rdo_eval_llama.log";   exit 1; }
wait $PID_C && echo "[OK] Qwen  cone_k3" || { echo "[FAIL] Qwen  cone_k3 — see $LOG/cone_eval_qwen_k3.log"; exit 1; }
wait $PID_D && echo "[OK] Llama cone_k3" || { echo "[FAIL] Llama cone_k3 — see $LOG/cone_eval_llama_k3.log"; exit 1; }

# ── Phase B: cone_k5 (2-way parallel) ────────────────────────────────────
banner "[Phase B] eval completions: cone_k5 — 2×GPU"

CUDA_VISIBLE_DEVICES=0 eval $OFFLINE $EVAL_GEN \
    --model qwen2.5_7b \
    --direction "$QWEN_RDO/cone_k5_basis.pt" \
    --config cone_k5 --device cuda:0 \
    > "$LOG/cone_eval_qwen_k5.log" 2>&1 &
PID_E=$!; echo "  GPU0: Qwen  cone_k5  → $LOG/cone_eval_qwen_k5.log"

CUDA_VISIBLE_DEVICES=1 eval $OFFLINE $EVAL_GEN \
    --model llama3.1_8b \
    --direction "$LLAMA_RDO/cone_k5_basis.pt" \
    --config cone_k5 --device cuda:0 \
    > "$LOG/cone_eval_llama_k5.log" 2>&1 &
PID_F=$!; echo "  GPU1: Llama cone_k5  → $LOG/cone_eval_llama_k5.log"

wait $PID_E && echo "[OK] Qwen  cone_k5" || { echo "[FAIL] Qwen  cone_k5 — see $LOG/cone_eval_qwen_k5.log";  exit 1; }
wait $PID_F && echo "[OK] Llama cone_k5" || { echo "[FAIL] Llama cone_k5 — see $LOG/cone_eval_llama_k5.log"; exit 1; }

# ── Keyword (CPU, instant) ────────────────────────────────────────────────
banner "[Keyword] CPU evaluation"
python experiments/repro_arditi_wollschlager/run_evaluate.py \
    --judge keyword --model all \
    | tee "$LOG/eval_keyword.log"
echo "[OK] Keyword done"

# ── Phase C: LG3 + SR judges — all 4 GPUs in parallel ───────────────────
banner "[Phase C] LlamaGuard3 + StrongREJECT — 4×GPU"

CUDA_VISIBLE_DEVICES=0 eval $OFFLINE $EVAL_JUD \
    --judge llamaguard3 --model qwen2.5_7b --device cuda:0 \
    > "$LOG/eval_lg3_qwen.log" 2>&1 &
PID_G=$!; echo "  GPU0: LG3  Qwen  → $LOG/eval_lg3_qwen.log"

CUDA_VISIBLE_DEVICES=1 eval $OFFLINE $EVAL_JUD \
    --judge llamaguard3 --model llama3.1_8b --device cuda:0 \
    > "$LOG/eval_lg3_llama.log" 2>&1 &
PID_H=$!; echo "  GPU1: LG3  Llama → $LOG/eval_lg3_llama.log"

CUDA_VISIBLE_DEVICES=2 eval $OFFLINE $EVAL_JUD \
    --judge strongreject --model qwen2.5_7b --device cuda:0 \
    > "$LOG/eval_sr_qwen.log" 2>&1 &
PID_I=$!; echo "  GPU2: SR   Qwen  → $LOG/eval_sr_qwen.log"

CUDA_VISIBLE_DEVICES=3 eval $OFFLINE $EVAL_JUD \
    --judge strongreject --model llama3.1_8b --device cuda:0 \
    > "$LOG/eval_sr_llama.log" 2>&1 &
PID_J=$!; echo "  GPU3: SR   Llama → $LOG/eval_sr_llama.log"

wait $PID_G && echo "[OK] LG3 Qwen"  || echo "[WARN] LG3 Qwen  failed — check $LOG/eval_lg3_qwen.log"
wait $PID_H && echo "[OK] LG3 Llama" || echo "[WARN] LG3 Llama failed — check $LOG/eval_lg3_llama.log"
wait $PID_I && echo "[OK] SR  Qwen"  || echo "[WARN] SR  Qwen  failed — check $LOG/eval_sr_qwen.log"
wait $PID_J && echo "[OK] SR  Llama" || echo "[WARN] SR  Llama failed — check $LOG/eval_sr_llama.log"

# ── Phase D: SRR summary (CPU) ────────────────────────────────────────────
banner "[Phase D] compute_summary.py (CPU)"
python experiments/repro_arditi_wollschlager/compute_summary.py \
    | tee "$LOG/summary.log"

# ── Final printout ────────────────────────────────────────────────────────
banner "=== ALL DONE ==="
echo "evaluation.json  → results/repro_arditi_wollschlager/evaluation.json"
echo "summary.md       → results/repro_arditi_wollschlager/summary.md"
echo ""
cat results/repro_arditi_wollschlager/summary.md 2>/dev/null || true
