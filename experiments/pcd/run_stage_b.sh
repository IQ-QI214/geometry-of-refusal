#!/usr/bin/env bash
# =============================================================================
# PCD Stage B — Full Execution Script (4×H100)
#
# Phases:
#   sweep    — DIM layer×pos sweep on 6 conditions (2 batches × 4 GPUs)
#   ablate   — DIM weight ablation + generation on 6 conditions
#   evaluate — 4-judge eval (kw + lg3 + arditi) on all dim_responses.json
#   aggregate— Build 8×6 summary matrix from eval JSONs
#   all      — sweep → ablate → evaluate → aggregate (default)
#
# Log structure (one file per task, co-located with results):
#   results/pcd/qwen_family/V-text/sweep.log
#   results/pcd/qwen_family/V-text/ablate.log
#   results/pcd/qwen_family/V-text/eval.log
#   results/pcd/stage_b.log   ← master log (all phases, stderr + stdout)
#
# Monitor all logs live:
#   tail -f results/pcd/*/V-*/sweep.log            # all sweep jobs at once
#   tail -f results/pcd/qwen_family/V-text/*.log   # one condition
#   tail -f results/pcd/stage_b.log                # master orchestration log
#
# Usage:
#   bash experiments/pcd/run_stage_b.sh [sweep|ablate|evaluate|aggregate|all]
#   bash experiments/pcd/run_stage_b.sh all 2>&1 | tee results/pcd/stage_b.log
# =============================================================================
set -uo pipefail

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RD="$ROOT/refusal_direction"

QWEN_VLM_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct"
GEMMA_PATH="/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it"

STAGE="${1:-all}"

# --------------------------------------------------------------------------- #
# Helper: launch one GPU job in the background
#
#   run_gpu <gpu_id> <conda_env> <log_path> <command...>
#
# Log goes to <log_path>; progress also echoed with a [TAG] prefix via tee.
# --------------------------------------------------------------------------- #
run_gpu() {
    local gpu=$1
    local env=$2
    local log=$3
    shift 3
    mkdir -p "$(dirname "$log")"
    local tag
    tag="[GPU${gpu}|$(basename "$(dirname "$log")")/$(basename "$log" .log)]"
    CUDA_VISIBLE_DEVICES=$gpu \
        conda run --no-capture-output -n "$env" bash -c \
        "cd '$RD' && PYTHONPATH=. $*" \
        > >(while IFS= read -r line; do echo "$tag $line"; done | tee -a "$log") \
        2>&1 &
}

# Wait for a list of PIDs; exit on any failure and report which log to check.
wait_jobs() {
    local phase=$1
    shift
    local -a pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[ERROR] $phase — job pid=$pid failed"
            failed=1
        fi
    done
    if [ $failed -ne 0 ]; then
        echo "[ERROR] $phase had failures — see per-condition *.log files above"
        exit 1
    fi
    echo "✓ $phase done"
}

# Print a banner and the log paths for a set of jobs about to start.
log_banner() {
    local phase=$1; shift
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  $phase"
    echo "══════════════════════════════════════════════════"
    echo "  Logs:"
    for log in "$@"; do
        printf "    GPU%s  %s\n" "$(echo "$log" | grep -o 'GPU[0-9]')" \
               "$(echo "$log" | sed "s|$ROOT/||")" 2>/dev/null || echo "    $log"
    done
    echo ""
}

# --------------------------------------------------------------------------- #
# Phase 1: DIM Layer Sweep                                                    #
# --------------------------------------------------------------------------- #
phase_sweep() {
    mkdir -p "$ROOT/results/pcd/qwen_family/"{V-text,V-blank-resweep,V-noise}
    mkdir -p "$ROOT/results/pcd/gemma_family/"{V-text,V-blank,V-noise}

    local L=(
        "$ROOT/results/pcd/qwen_family/V-text/sweep.log"
        "$ROOT/results/pcd/qwen_family/V-blank-resweep/sweep.log"
        "$ROOT/results/pcd/qwen_family/V-noise/sweep.log"
        "$ROOT/results/pcd/gemma_family/V-text/sweep.log"
    )
    log_banner "Phase 1 — Sweep  Batch 1/2  (Qwen×3 + Gemma V-text)" "${L[@]}"

    local pids=()
    run_gpu 0 qwen3-vl "${L[0]}" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-text  --output_dir ../results/pcd/qwen_family/V-text"
    pids+=($!)

    run_gpu 1 qwen3-vl "${L[1]}" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank --output_dir ../results/pcd/qwen_family/V-blank-resweep"
    pids+=($!)

    run_gpu 2 qwen3-vl "${L[2]}" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise --output_dir ../results/pcd/qwen_family/V-noise"
    pids+=($!)

    run_gpu 3 qwen3-vl "${L[3]}" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-text  --output_dir ../results/pcd/gemma_family/V-text"
    pids+=($!)

    wait_jobs "sweep-batch1" "${pids[@]}"

    # Batch 2
    local L2=(
        "$ROOT/results/pcd/gemma_family/V-blank/sweep.log"
        "$ROOT/results/pcd/gemma_family/V-noise/sweep.log"
    )
    log_banner "Phase 1 — Sweep  Batch 2/2  (Gemma V-blank + V-noise)" "${L2[@]}"
    pids=()

    run_gpu 0 qwen3-vl "${L2[0]}" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank --output_dir ../results/pcd/gemma_family/V-blank"
    pids+=($!)

    run_gpu 1 qwen3-vl "${L2[1]}" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise --output_dir ../results/pcd/gemma_family/V-noise"
    pids+=($!)

    wait_jobs "sweep-batch2" "${pids[@]}"

    echo ""
    echo "── best_layer summary ──"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/best_layer.json; do
        [ -f "$f" ] || continue
        cond=$(dirname "$f" | sed "s|$ROOT/results/pcd/||")
        printf "  %-35s  %s\n" "$cond" "$(python3 -c "import json,sys; d=json.load(open('$f')); print(f\"layer={d['layer']} pos={d['pos']} filter={d.get('filter_passed','?')}\")")"
    done
}

# --------------------------------------------------------------------------- #
# Phase 2: DIM Ablate + Generate                                              #
# --------------------------------------------------------------------------- #
phase_ablate() {
    local L=(
        "$ROOT/results/pcd/qwen_family/V-text/ablate.log"
        "$ROOT/results/pcd/qwen_family/V-blank-resweep/ablate.log"
        "$ROOT/results/pcd/qwen_family/V-noise/ablate.log"
        "$ROOT/results/pcd/gemma_family/V-text/ablate.log"
    )
    log_banner "Phase 2 — Ablate  Batch 1/2" "${L[@]}"

    local pids=()
    run_gpu 0 qwen3-vl "${L[0]}" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/qwen_family/V-text \
         --output_dir ../results/pcd/qwen_family/V-text"
    pids+=($!)

    run_gpu 1 qwen3-vl "${L[1]}" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/qwen_family/V-blank-resweep \
         --output_dir ../results/pcd/qwen_family/V-blank-resweep"
    pids+=($!)

    run_gpu 2 qwen3-vl "${L[2]}" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/qwen_family/V-noise \
         --output_dir ../results/pcd/qwen_family/V-noise"
    pids+=($!)

    run_gpu 3 qwen3-vl "${L[3]}" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/gemma_family/V-text \
         --output_dir ../results/pcd/gemma_family/V-text"
    pids+=($!)

    wait_jobs "ablate-batch1" "${pids[@]}"

    local L2=(
        "$ROOT/results/pcd/gemma_family/V-blank/ablate.log"
        "$ROOT/results/pcd/gemma_family/V-noise/ablate.log"
    )
    log_banner "Phase 2 — Ablate  Batch 2/2" "${L2[@]}"
    pids=()

    run_gpu 0 qwen3-vl "${L2[0]}" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/gemma_family/V-blank \
         --output_dir ../results/pcd/gemma_family/V-blank"
    pids+=($!)

    run_gpu 1 qwen3-vl "${L2[1]}" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/gemma_family/V-noise \
         --output_dir ../results/pcd/gemma_family/V-noise"
    pids+=($!)

    wait_jobs "ablate-batch2" "${pids[@]}"
}

# --------------------------------------------------------------------------- #
# Phase 3: 4-Judge Evaluation                                                 #
# 6 jobs → round-robin across 4 GPUs (batch1: 4 jobs, batch2: 2 jobs)        #
# LlamaGuard-3-8B ~16 GB/job; H100 80 GB → safe to run 4 in parallel         #
# --------------------------------------------------------------------------- #
phase_evaluate() {
    # job spec: "result_subdir:model_name:model_path"
    local JOBS=(
        "qwen_family/V-text:qwen2.5-vl-7b:$QWEN_VLM_PATH"
        "qwen_family/V-blank-resweep:qwen2.5-vl-7b:$QWEN_VLM_PATH"
        "qwen_family/V-noise:qwen2.5-vl-7b:$QWEN_VLM_PATH"
        "gemma_family/V-text:gemma-3-4b-it-vlm:$GEMMA_PATH"
        "gemma_family/V-blank:gemma-3-4b-it-vlm:$GEMMA_PATH"
        "gemma_family/V-noise:gemma-3-4b-it-vlm:$GEMMA_PATH"
    )

    local total=${#JOBS[@]}
    local batch_size=4
    local batch=1
    local i=0
    local pids=()
    local batch_logs=()

    for entry in "${JOBS[@]}"; do
        IFS=':' read -r subdir model_name model_path <<< "$entry"
        local gpu=$((i % batch_size))
        local log="$ROOT/results/pcd/$subdir/eval.log"
        batch_logs+=("$log")

        run_gpu "$gpu" qwen3-vl "$log" \
            "python ../experiments/pcd/exp_pcd_evaluate.py \
             --responses_json ../results/pcd/$subdir/dim_responses.json \
             --model_name $model_name \
             --model_path '$model_path' \
             --output_json ../results/pcd/$subdir/dim_responses_eval.json \
             --layers kw lg3 arditi"
        pids+=($!)
        i=$((i + 1))

        if [ $((i % batch_size)) -eq 0 ] || [ $i -eq $total ]; then
            log_banner "Phase 3 — Evaluate  Batch ${batch}" "${batch_logs[@]}"
            wait_jobs "evaluate-batch${batch}" "${pids[@]}"
            pids=()
            batch_logs=()
            batch=$((batch + 1))
        fi
    done

    # Summary table
    echo ""
    echo "── eval summary ──"
    printf "  %-35s  %s  %s  %s\n" "condition" "asr_kw" "asr_lg3" "arditi"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/dim_responses_eval.json; do
        [ -f "$f" ] || continue
        cond=$(dirname "$f" | sed "s|$ROOT/results/pcd/||")
        python3 - "$f" "$cond" <<'PYEOF'
import json, sys
f, cond = sys.argv[1], sys.argv[2]
d = json.load(open(f))
print(f"  {cond:<35}  {d.get('asr_keyword',float('nan')):.3f}   "
      f"{d.get('asr_lg3',float('nan')):.3f}    {d.get('arditi_joint_asr',float('nan')):.3f}")
PYEOF
    done
}

# --------------------------------------------------------------------------- #
# Phase 4: Aggregate                                                          #
# --------------------------------------------------------------------------- #
phase_aggregate() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Phase 4 — Aggregate 8×6 Matrix"
    echo "══════════════════════════════════════════════════"
    cd "$ROOT"
    python3 experiments/pcd/aggregate.py \
        --root    results/pcd \
        --out_json results/pcd/pcd_8x6_matrix.json \
        --out_md   results/pcd/pcd_summary.md
    echo ""
    cat results/pcd/pcd_summary.md
}

# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
echo "╔══════════════════════════════════════════════════╗"
echo "║  PCD Stage B  |  stage=$STAGE"
echo "╠══════════════════════════════════════════════════╣"
echo "║  ROOT    : $ROOT"
echo "║  Qwen VLM: $QWEN_VLM_PATH"
echo "║  Gemma   : $GEMMA_PATH"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Log structure (per condition, per phase):       ║"
echo "║    results/pcd/<family>/<condition>/sweep.log    ║"
echo "║    results/pcd/<family>/<condition>/ablate.log   ║"
echo "║    results/pcd/<family>/<condition>/eval.log     ║"
echo "║  Master log: pipe stdout to results/pcd/stage_b.log ║"
echo "║    e.g.  bash run_stage_b.sh all |& tee results/pcd/stage_b.log"
echo "╚══════════════════════════════════════════════════╝"
echo ""

case "$STAGE" in
    sweep)     phase_sweep ;;
    ablate)    phase_ablate ;;
    evaluate)  phase_evaluate ;;
    aggregate) phase_aggregate ;;
    all)
        phase_sweep
        phase_ablate
        phase_evaluate
        phase_aggregate
        echo ""
        echo "╔══════════════════════════════════════════╗"
        echo "║  Stage B COMPLETE                        ║"
        echo "╚══════════════════════════════════════════╝"
        ;;
    *)
        echo "Usage: $0 [sweep|ablate|evaluate|aggregate|all]"
        exit 1
        ;;
esac
