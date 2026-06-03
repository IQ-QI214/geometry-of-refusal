#!/bin/bash
# Launch InternVL3 Phase 1 experiment (foreground, with tee logging)
# Usage: bash scripts/launch_internvl3.sh [--gpu 1]

set -euo pipefail

ROOT="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal"
MMSAFETY="/inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench"
GPU="${GPU:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    *) shift ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT/logs/mibd_phase1/internvl3"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
RESULT_DIR="$ROOT/results/mibd/phase1_probe/internvl3_8b"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

echo "============================================================"
echo "  MIBD Phase 1 — InternVL3-8B"
echo "  GPU      : $GPU"
echo "  Log      : $LOG_FILE"
echo "  Results  : $RESULT_DIR"
echo "  Started  : $(date)"
echo "============================================================"

cd "$ROOT"

conda run --no-capture-output -n rdo python -u -m experiments.mibd.run_phase1 \
  --model internvl3 \
  --gpu "$GPU" \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
  --data-dir data/saladbench_splits \
  --mmsafety-dir "$MMSAFETY" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "  InternVL3 FINISHED  exit=$EXIT_CODE  $(date)"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "  Report: $RESULT_DIR/phase1_report.md"
  if [[ -f "$RESULT_DIR/phase1_report.md" ]]; then
    echo ""
    cat "$RESULT_DIR/phase1_report.md"
  fi
fi
echo "============================================================"

exit $EXIT_CODE
