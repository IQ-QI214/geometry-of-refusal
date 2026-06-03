#!/bin/bash
# Launch Qwen3-VL Phase 1 experiment (foreground, with tee logging)
# Usage: bash scripts/launch_qwen3vl.sh [--gpu 0]

set -euo pipefail

ROOT="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal"
MMSAFETY="/inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench"
GPU="${GPU:-0}"

# Parse --gpu argument
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    *) shift ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT/logs/mibd_phase1/qwen3vl"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
RESULT_DIR="$ROOT/results/mibd/phase1_probe/qwen3_vl_8b"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

echo "============================================================"
echo "  MIBD Phase 1 — Qwen3-VL-8B"
echo "  GPU      : $GPU"
echo "  Log      : $LOG_FILE"
echo "  Results  : $RESULT_DIR"
echo "  Started  : $(date)"
echo "============================================================"

cd "$ROOT"

conda run -n qwen3-vl python -m experiments.mibd.run_phase1 \
  --model qwen3vl \
  --gpu "$GPU" \
  --config experiments/mibd/configs/phase1_probe.yaml \
  --data-dir data/saladbench_splits \
  --mmsafety-dir "$MMSAFETY" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "  Qwen3-VL FINISHED  exit=$EXIT_CODE  $(date)"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "  Report: $RESULT_DIR/phase1_report.md"
  if [[ -f "$RESULT_DIR/phase1_report.md" ]]; then
    echo ""
    cat "$RESULT_DIR/phase1_report.md"
  fi
fi
echo "============================================================"

exit $EXIT_CODE
