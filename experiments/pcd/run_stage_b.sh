#!/usr/bin/env bash
# =============================================================================
# PCD Stage B — Full Execution Script (4×H100)
#
# GPU 调度：动态池（named-pipe semaphore）
#   - 最多 4 个 job 同时运行
#   - 任意 job 结束（包括 --skip_if_done 快速退出）立刻释放 GPU 给下一个 job
#   - 不会出现 GPU 空置等待其他 batch 的情况
#
# Log 结构：
#   results/pcd/<family>/<cond>/sweep.log     ← 每条件每阶段独立日志
#   results/pcd/<family>/<cond>/ablate.log
#   results/pcd/<family>/<cond>/eval.log
#   results/pcd/stage_b.log                  ← 主编排日志（手动 tee）
#
# Usage:
#   bash experiments/pcd/run_stage_b.sh [sweep|ablate|evaluate|aggregate|all]
#   bash experiments/pcd/run_stage_b.sh all |& tee results/pcd/stage_b.log
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
N_GPU=4

# --------------------------------------------------------------------------- #
# Dynamic GPU pool (named-pipe semaphore)                                     #
#                                                                             #
# setup_pool  — opens FD 9 with N_GPU tokens (0..N_GPU-1)                   #
# teardown_pool — closes FD 9 and removes FIFO                               #
# enqueue ENV LOG CMD — blocking wrapper: acquires token → runs → releases   #
#   Call as:  enqueue env log "cmd ..." &   (the & is on the caller side)    #
# --------------------------------------------------------------------------- #
_POOL_FIFO=""

setup_pool() {
    _POOL_FIFO="$(mktemp -u /tmp/pcd_pool_XXXXXX)"
    mkfifo "$_POOL_FIFO"
    exec 9<>"$_POOL_FIFO"
    for gpu in $(seq 0 $((N_GPU - 1))); do
        printf '%d\n' "$gpu" >&9
    done
}

teardown_pool() {
    exec 9>&- 2>/dev/null || true
    rm -f "$_POOL_FIFO"
}

enqueue() {
    local env=$1 log=$2
    shift 2
    local cmd="$*"

    # Acquire a GPU token (blocks until one is free)
    local gpu
    read -r -u 9 gpu

    mkdir -p "$(dirname "$log")"
    local tag
    tag="[GPU${gpu}|$(basename "$(dirname "$log")")/$(basename "$log" .log)]"

    CUDA_VISIBLE_DEVICES=$gpu \
        conda run --no-capture-output -n "$env" bash -c \
        "cd '$RD' && PYTHONPATH=. $cmd" \
        > >(while IFS= read -r line; do echo "$tag $line"; done | tee -a "$log") \
        2>&1
    local ec=$?

    # Release GPU token
    printf '%d\n' "$gpu" >&9
    return $ec
}

wait_all() {
    local phase=$1; shift
    local failed=0
    for pid in "$@"; do
        wait "$pid" || { echo "[ERROR] $phase — pid=$pid failed"; failed=1; }
    done
    [ $failed -eq 0 ] || { echo "[ERROR] $phase had failures — check *.log files"; exit 1; }
    echo "✓ $phase done"
}

# --------------------------------------------------------------------------- #
# Phase 1: DIM Layer Sweep  (6 jobs → pool of 4 GPUs)                        #
# --------------------------------------------------------------------------- #
phase_sweep() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  Phase 1 — DIM Layer Sweep  (6 jobs, pool of $N_GPU GPUs)"
    echo "══════════════════════════════════════════════════════════"
    mkdir -p "$ROOT/results/pcd/qwen_family/"{V-text,V-blank-resweep,V-noise}
    mkdir -p "$ROOT/results/pcd/gemma_family/"{V-text,V-blank,V-noise}

    setup_pool

    local pids=()

    # Qwen: standard thresholds
    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-text/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-text --output_dir ../results/pcd/qwen_family/V-text \
         --skip_if_done" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-blank-resweep/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank --output_dir ../results/pcd/qwen_family/V-blank-resweep \
         --skip_if_done" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-noise/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise --output_dir ../results/pcd/qwen_family/V-noise \
         --skip_if_done" &
    pids+=($!)

    # Gemma: relaxed thresholds (Gemma3Config scores have different scale)
    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-text/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-text --output_dir ../results/pcd/gemma_family/V-text \
         --kl_threshold 100 --induce_refusal_threshold -999 --skip_if_done" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-blank/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank --output_dir ../results/pcd/gemma_family/V-blank \
         --kl_threshold 100 --induce_refusal_threshold -999 --skip_if_done" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-noise/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise --output_dir ../results/pcd/gemma_family/V-noise \
         --kl_threshold 100 --induce_refusal_threshold -999 --skip_if_done" &
    pids+=($!)

    wait_all "sweep" "${pids[@]}"
    teardown_pool

    echo ""
    echo "── best_layer summary ──"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/best_layer.json; do
        [ -f "$f" ] || continue
        cond=$(dirname "$f" | sed "s|$ROOT/results/pcd/||")
        info=$(python3 -c "
import json, sys
d = json.load(open('$f'))
print(f\"layer={d['layer']} pos={d['pos']} filter={d.get('filter_passed','?')}\")
" 2>/dev/null || cat "$f")
        printf "  %-35s  %s\n" "$cond" "$info"
    done
}

# --------------------------------------------------------------------------- #
# Phase 2: DIM Ablate + Generate  (6 jobs → pool of 4 GPUs)                  #
# --------------------------------------------------------------------------- #
phase_ablate() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  Phase 2 — DIM Ablate + Generate  (6 jobs, pool of $N_GPU GPUs)"
    echo "══════════════════════════════════════════════════════════"

    setup_pool

    local pids=()

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-text/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/qwen_family/V-text \
         --output_dir ../results/pcd/qwen_family/V-text" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-blank-resweep/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/qwen_family/V-blank-resweep \
         --output_dir ../results/pcd/qwen_family/V-blank-resweep" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-noise/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/qwen_family/V-noise \
         --output_dir ../results/pcd/qwen_family/V-noise" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-text/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/gemma_family/V-text \
         --output_dir ../results/pcd/gemma_family/V-text" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-blank/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/gemma_family/V-blank \
         --output_dir ../results/pcd/gemma_family/V-blank" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-noise/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/gemma_family/V-noise \
         --output_dir ../results/pcd/gemma_family/V-noise" &
    pids+=($!)

    wait_all "ablate" "${pids[@]}"
    teardown_pool
}

# --------------------------------------------------------------------------- #
# Phase 3: 4-Judge Evaluation  (6 jobs → pool of 4 GPUs)                     #
# LlamaGuard-3-8B ~16 GB/job; H100 80 GB → safe to run 4 in parallel         #
# --------------------------------------------------------------------------- #
phase_evaluate() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  Phase 3 — 4-Judge Evaluation  (6 jobs, pool of $N_GPU GPUs)"
    echo "══════════════════════════════════════════════════════════"

    setup_pool

    local pids=()

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-text/eval.log" \
        "python ../experiments/pcd/exp_pcd_evaluate.py \
         --responses_json ../results/pcd/qwen_family/V-text/dim_responses.json \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --output_json ../results/pcd/qwen_family/V-text/dim_responses_eval.json \
         --layers kw lg3 arditi" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-blank-resweep/eval.log" \
        "python ../experiments/pcd/exp_pcd_evaluate.py \
         --responses_json ../results/pcd/qwen_family/V-blank-resweep/dim_responses.json \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --output_json ../results/pcd/qwen_family/V-blank-resweep/dim_responses_eval.json \
         --layers kw lg3 arditi" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/qwen_family/V-noise/eval.log" \
        "python ../experiments/pcd/exp_pcd_evaluate.py \
         --responses_json ../results/pcd/qwen_family/V-noise/dim_responses.json \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --output_json ../results/pcd/qwen_family/V-noise/dim_responses_eval.json \
         --layers kw lg3 arditi" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-text/eval.log" \
        "python ../experiments/pcd/exp_pcd_evaluate.py \
         --responses_json ../results/pcd/gemma_family/V-text/dim_responses.json \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --output_json ../results/pcd/gemma_family/V-text/dim_responses_eval.json \
         --layers kw lg3 arditi" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-blank/eval.log" \
        "python ../experiments/pcd/exp_pcd_evaluate.py \
         --responses_json ../results/pcd/gemma_family/V-blank/dim_responses.json \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --output_json ../results/pcd/gemma_family/V-blank/dim_responses_eval.json \
         --layers kw lg3 arditi" &
    pids+=($!)

    enqueue qwen3-vl "$ROOT/results/pcd/gemma_family/V-noise/eval.log" \
        "python ../experiments/pcd/exp_pcd_evaluate.py \
         --responses_json ../results/pcd/gemma_family/V-noise/dim_responses.json \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --output_json ../results/pcd/gemma_family/V-noise/dim_responses_eval.json \
         --layers kw lg3 arditi" &
    pids+=($!)

    wait_all "evaluate" "${pids[@]}"
    teardown_pool

    echo ""
    echo "── eval summary ──"
    printf "  %-35s  %-8s  %-8s  %s\n" "condition" "asr_kw" "asr_lg3" "arditi"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/dim_responses_eval.json; do
        [ -f "$f" ] || continue
        cond=$(dirname "$f" | sed "s|$ROOT/results/pcd/||")
        python3 - "$f" "$cond" <<'PYEOF'
import json, sys, math
f, cond = sys.argv[1], sys.argv[2]
d = json.load(open(f))
def fmt(v): return f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else "-"
print(f"  {cond:<35}  {fmt(d.get('asr_keyword', float('nan')))}      "
      f"{fmt(d.get('asr_lg3', float('nan')))}      {fmt(d.get('arditi_joint_asr', float('nan')))}")
PYEOF
    done
}

# --------------------------------------------------------------------------- #
# Phase 4: Aggregate matrix  (CPU only)                                       #
# --------------------------------------------------------------------------- #
phase_aggregate() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  Phase 4 — Aggregate 8×6 Matrix"
    echo "══════════════════════════════════════════════════════════"
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
echo "╔══════════════════════════════════════════════════════════╗"
printf "║  PCD Stage B  |  stage=%-33s║\n" "$STAGE"
printf "║  ROOT    : %-44s║\n" "$ROOT"
printf "║  GPU pool: %d GPUs (named-pipe semaphore)%-14s║\n" "$N_GPU" ""
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Logs: results/pcd/<family>/<cond>/{sweep,ablate,eval}.log  ║"
echo "║  Master: bash run_stage_b.sh all |& tee results/pcd/stage_b.log ║"
echo "╚══════════════════════════════════════════════════════════╝"
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
        echo "╔══════════════════════════════════════╗"
        echo "║  Stage B COMPLETE                    ║"
        echo "╚══════════════════════════════════════╝"
        ;;
    *)
        echo "Usage: $0 [sweep|ablate|evaluate|aggregate|all]"
        exit 1
        ;;
esac
