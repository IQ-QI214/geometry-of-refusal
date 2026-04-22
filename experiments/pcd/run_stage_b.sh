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
# Usage:
#   bash experiments/pcd/run_stage_b.sh [sweep|ablate|evaluate|aggregate|all]
#
# Run from project root OR refusal_direction/; script auto-detects ROOT.
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
# Helper: launch one experiment on a specific GPU in background               #
# --------------------------------------------------------------------------- #
run_gpu() {
    local gpu=$1
    local env=$2
    local log=$3
    shift 3
    CUDA_VISIBLE_DEVICES=$gpu \
        conda run --no-capture-output -n "$env" bash -c \
        "cd '$RD' && PYTHONPATH=. $* 2>&1 | tee '$log'" &
}

wait_all() {
    local phase=$1
    local pids=("${@:2}")
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "[ERROR] A job in $phase failed (pid=$pid)"; failed=1; }
    done
    [ $failed -eq 0 ] || { echo "[ERROR] $phase had failures. Check logs in /tmp/pcd_*.log"; exit 1; }
    echo "=== $phase complete ==="
}

# --------------------------------------------------------------------------- #
# Phase 1: DIM Layer Sweep                                                    #
# 6 conditions split across two batches of 4 GPUs                            #
# --------------------------------------------------------------------------- #
phase_sweep() {
    echo ""
    echo "############################################################"
    echo "# Phase 1: DIM Layer Sweep × 6 conditions                  #"
    echo "############################################################"
    mkdir -p "$ROOT/results/pcd/qwen_family/"{V-text,V-blank-resweep,V-noise}
    mkdir -p "$ROOT/results/pcd/gemma_family/"{V-text,V-blank,V-noise}

    # Batch 1: 4 conditions on 4 GPUs
    echo "[sweep] Batch 1 starting (Qwen×3 + Gemma V-text) ..."
    local pids=()

    run_gpu 0 qwen3-vl /tmp/pcd_sweep_qwen_vtext.log \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-text  --output_dir ../results/pcd/qwen_family/V-text"
    pids+=($!)

    run_gpu 1 qwen3-vl /tmp/pcd_sweep_qwen_vblank.log \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank --output_dir ../results/pcd/qwen_family/V-blank-resweep"
    pids+=($!)

    run_gpu 2 qwen3-vl /tmp/pcd_sweep_qwen_vnoise.log \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise --output_dir ../results/pcd/qwen_family/V-noise"
    pids+=($!)

    run_gpu 3 qwen3-vl /tmp/pcd_sweep_gemma_vtext.log \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-text  --output_dir ../results/pcd/gemma_family/V-text"
    pids+=($!)

    wait_all "sweep-batch1" "${pids[@]}"

    # Batch 2: remaining 2 Gemma conditions
    echo "[sweep] Batch 2 starting (Gemma V-blank + V-noise) ..."
    pids=()

    run_gpu 0 qwen3-vl /tmp/pcd_sweep_gemma_vblank.log \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank --output_dir ../results/pcd/gemma_family/V-blank"
    pids+=($!)

    run_gpu 1 qwen3-vl /tmp/pcd_sweep_gemma_vnoise.log \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise --output_dir ../results/pcd/gemma_family/V-noise"
    pids+=($!)

    wait_all "sweep-batch2" "${pids[@]}"

    # Print summary
    echo ""
    echo "--- Sweep best_layer.json summary ---"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/best_layer.json; do
        echo "$(dirname "$f" | xargs basename) $(dirname "$f" | sed 's|.*/||'): $(cat "$f")"
    done
}

# --------------------------------------------------------------------------- #
# Phase 2: DIM Ablate + Generate                                              #
# --------------------------------------------------------------------------- #
phase_ablate() {
    echo ""
    echo "############################################################"
    echo "# Phase 2: DIM Ablate + Generate × 6 conditions            #"
    echo "############################################################"

    # Batch 1
    echo "[ablate] Batch 1 starting ..."
    local pids=()

    run_gpu 0 qwen3-vl /tmp/pcd_ablate_qwen_vtext.log \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/qwen_family/V-text \
         --output_dir ../results/pcd/qwen_family/V-text"
    pids+=($!)

    run_gpu 1 qwen3-vl /tmp/pcd_ablate_qwen_vblank.log \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/qwen_family/V-blank-resweep \
         --output_dir ../results/pcd/qwen_family/V-blank-resweep"
    pids+=($!)

    run_gpu 2 qwen3-vl /tmp/pcd_ablate_qwen_vnoise.log \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/qwen_family/V-noise \
         --output_dir ../results/pcd/qwen_family/V-noise"
    pids+=($!)

    run_gpu 3 qwen3-vl /tmp/pcd_ablate_gemma_vtext.log \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/gemma_family/V-text \
         --output_dir ../results/pcd/gemma_family/V-text"
    pids+=($!)

    wait_all "ablate-batch1" "${pids[@]}"

    # Batch 2
    echo "[ablate] Batch 2 starting ..."
    pids=()

    run_gpu 0 qwen3-vl /tmp/pcd_ablate_gemma_vblank.log \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/gemma_family/V-blank \
         --output_dir ../results/pcd/gemma_family/V-blank"
    pids+=($!)

    run_gpu 1 qwen3-vl /tmp/pcd_ablate_gemma_vnoise.log \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/gemma_family/V-noise \
         --output_dir ../results/pcd/gemma_family/V-noise"
    pids+=($!)

    wait_all "ablate-batch2" "${pids[@]}"
}

# --------------------------------------------------------------------------- #
# Phase 3: 4-Judge Evaluation                                                 #
# 6 jobs distributed round-robin across 4 GPUs (1.5 rounds)                  #
# Each job: kw (no GPU) + lg3 (LlamaGuard-3-8B ~16GB) + arditi               #
# --------------------------------------------------------------------------- #
phase_evaluate() {
    echo ""
    echo "############################################################"
    echo "# Phase 3: 4-Judge Evaluation × 6 conditions               #"
    echo "############################################################"

    declare -a JOBS=(
        "qwen_family/V-text:qwen2.5-vl-7b:$QWEN_VLM_PATH"
        "qwen_family/V-blank-resweep:qwen2.5-vl-7b:$QWEN_VLM_PATH"
        "qwen_family/V-noise:qwen2.5-vl-7b:$QWEN_VLM_PATH"
        "gemma_family/V-text:gemma-3-4b-it-vlm:$GEMMA_PATH"
        "gemma_family/V-blank:gemma-3-4b-it-vlm:$GEMMA_PATH"
        "gemma_family/V-noise:gemma-3-4b-it-vlm:$GEMMA_PATH"
    )

    local i=0
    local pids=()

    for entry in "${JOBS[@]}"; do
        IFS=':' read -r subdir model_name model_path <<< "$entry"
        local gpu=$((i % 4))
        local tag
        tag=$(echo "$subdir" | tr '/' '_')

        run_gpu "$gpu" qwen3-vl "/tmp/pcd_eval_${tag}.log" \
            "python ../experiments/pcd/exp_pcd_evaluate.py \
             --responses_json ../results/pcd/$subdir/dim_responses.json \
             --model_name $model_name \
             --model_path '$model_path' \
             --output_json ../results/pcd/$subdir/dim_responses_eval.json \
             --layers kw lg3 arditi"
        pids+=($!)
        i=$((i + 1))

        # Flush every 4 jobs (one full GPU round)
        if [ $((i % 4)) -eq 0 ]; then
            wait_all "evaluate-batch$((i / 4))" "${pids[@]}"
            pids=()
        fi
    done

    # Wait for any remaining jobs
    if [ ${#pids[@]} -gt 0 ]; then
        wait_all "evaluate-final" "${pids[@]}"
    fi

    # Print eval summary
    echo ""
    echo "--- Eval results summary ---"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/dim_responses_eval.json; do
        [ -f "$f" ] || continue
        subdir=$(dirname "$f" | sed "s|$ROOT/results/pcd/||")
        asr_kw=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d.get('asr_keyword','?'):.3f}\")" 2>/dev/null || echo "?")
        asr_lg3=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d.get('asr_lg3','?'):.3f}\")" 2>/dev/null || echo "?")
        arditi=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d.get('arditi_joint_asr','?'):.3f}\")" 2>/dev/null || echo "?")
        printf "  %-35s  asr_kw=%-6s  asr_lg3=%-6s  arditi=%s\n" "$subdir" "$asr_kw" "$asr_lg3" "$arditi"
    done
}

# --------------------------------------------------------------------------- #
# Phase 4: Aggregate matrix                                                   #
# --------------------------------------------------------------------------- #
phase_aggregate() {
    echo ""
    echo "############################################################"
    echo "# Phase 4: Aggregate 8x6 Matrix                            #"
    echo "############################################################"
    cd "$ROOT"
    python3 experiments/pcd/aggregate.py \
        --root results/pcd \
        --out_json results/pcd/pcd_8x6_matrix.json \
        --out_md results/pcd/pcd_summary.md
    echo ""
    cat results/pcd/pcd_summary.md
}

# --------------------------------------------------------------------------- #
# Dispatch                                                                    #
# --------------------------------------------------------------------------- #
echo "=== PCD Stage B  |  stage=$STAGE  |  ROOT=$ROOT ==="
echo "    Qwen VLM : $QWEN_VLM_PATH"
echo "    Gemma    : $GEMMA_PATH"
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
        echo "=== Stage B COMPLETE ==="
        ;;
    *)
        echo "Usage: $0 [sweep|ablate|evaluate|aggregate|all]"
        exit 1
        ;;
esac
