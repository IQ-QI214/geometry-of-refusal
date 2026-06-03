#!/usr/bin/env bash
# Qwen3-VL 4-judge evaluation for 3 conditions
set -euo pipefail
export PYTHONPATH=refusal_direction

MODEL_NAME="qwen3vl_8b"
MODEL_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B"
OUTBASE="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl"
Q3G_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen3Guard-Gen-8B"
LG3_PATH="/inspire/hdd/global_user/wenming-253108090054/models/llama-guard-3-8b"
SR_BASE="/inspire/hdd/global_user/wenming-253108090054/models/gemma-2b"
SR_ADAPTER="/inspire/hdd/global_user/wenming-253108090054/models/strongreject-15k-v1"

for COND in V-text V-blank V-noise; do
    RESP="$OUTBASE/$COND/dim_responses.json"
    OUT="$OUTBASE/$COND/dim_eval.json"
    LOGFILE="$OUTBASE/$COND/evaluate.log"
    echo "[launch] Starting evaluate for $COND"
    nohup conda run -n qwen3-vl python3 experiments/pcd/exp_pcd_evaluate.py \
        --responses_json "$RESP" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --output_json "$OUT" \
        --layers kw sr q3g lg3 \
        --q3g_path "$Q3G_PATH" \
        --lg3_path "$LG3_PATH" \
        --sr_base "$SR_BASE" \
        --sr_adapter "$SR_ADAPTER" \
        > "$LOGFILE" 2>&1 &
    echo $! > "$OUTBASE/$COND/evaluate.pid"
done
echo "[launch] All 3 evaluations started."
