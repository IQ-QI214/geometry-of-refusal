#!/usr/bin/env bash
# Qwen3-VL ablate + generate for 3 conditions
set -euo pipefail
export PYTHONPATH=refusal_direction

MODEL_NAME="qwen3vl_8b"
MODEL_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B"
OUTBASE="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl"

for COND in V-text V-blank V-noise; do
    SWEEP_DIR="$OUTBASE/$COND/sweep"
    OUTDIR="$OUTBASE/$COND"
    LOGFILE="$OUTBASE/$COND/ablate.log"
    mkdir -p "$OUTDIR"
    echo "[launch] Starting ablate for $COND"
    nohup conda run -n qwen3-vl python3 experiments/pcd/exp_pcd_ablate.py \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --condition "$COND" \
        --sweep_dir "$SWEEP_DIR" \
        --output_dir "$OUTDIR" \
        > "$LOGFILE" 2>&1 &
    echo $! > "$OUTBASE/$COND/ablate.pid"
    echo "[launch] PID=$! for $COND"
done
echo "[launch] All 3 ablations started."
