#!/usr/bin/env bash
# Qwen3-VL layer sweep for 3 conditions (V-text, V-blank, V-noise)
# Run from project root: bash experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/launch_qwen3vl_sweep.sh
set -euo pipefail
export PYTHONPATH=refusal_direction

MODEL_NAME="qwen3vl_8b"
MODEL_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B"
OUTBASE="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl"

for COND in V-text V-blank V-noise; do
    OUTDIR="$OUTBASE/$COND/sweep"
    LOGFILE="$OUTBASE/$COND/sweep.log"
    mkdir -p "$OUTDIR"
    echo "[launch] Starting sweep for $COND -> $LOGFILE"
    nohup conda run -n qwen3-vl python3 experiments/pcd/exp_pcd_layer_sweep.py \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --condition "$COND" \
        --output_dir "$OUTDIR" \
        --select_n_val 128 \
        > "$LOGFILE" 2>&1 &
    echo $! > "$OUTBASE/$COND/sweep.pid"
    echo "[launch] PID=$! for $COND"
done
echo "[launch] All 3 sweeps started. Monitor with: tail -f results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/V-text/sweep.log"
