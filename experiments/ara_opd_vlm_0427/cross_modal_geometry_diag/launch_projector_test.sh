#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=refusal_direction
LOGFILE="results/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.log"
mkdir -p results/ara_opd_vlm_0427/cross_modal_geometry_diag
nohup conda run -n qwen3-vl python3 \
    experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.py \
    > "$LOGFILE" 2>&1 &
echo $! > results/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.pid
echo "[launch] PID=$! -> $LOGFILE"
