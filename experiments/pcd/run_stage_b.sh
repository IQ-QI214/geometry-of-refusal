#!/usr/bin/env bash
# =============================================================================
# PCD Stage B — Fixed GPU Assignment Script (4×H100)
#
# GPU 分配策略：静态固定，每阶段明确哪张卡跑哪个任务
# 运行前提：无其他 exp_pcd 进程占用 GPU（用 ps aux|grep exp_pcd 确认）
#
# 用法：
#   bash experiments/pcd/run_stage_b.sh sweep
#   bash experiments/pcd/run_stage_b.sh ablate
#   bash experiments/pcd/run_stage_b.sh evaluate
#   bash experiments/pcd/run_stage_b.sh aggregate
#   bash experiments/pcd/run_stage_b.sh all
#   bash experiments/pcd/run_stage_b.sh all |& tee results/pcd/stage_b.log
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RD="$ROOT/refusal_direction"

QWEN_VLM_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct"
GEMMA_PATH="/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it"

STAGE="${1:-all}"

# --------------------------------------------------------------------------- #
# 基础工具                                                                    #
# --------------------------------------------------------------------------- #

# 在指定 GPU 上后台启动一个实验，日志写到 <log_file>
# 用法: launch <gpu_id> <conda_env> <log_file> <python args...>
launch() {
    local gpu=$1 env=$2 log=$3
    shift 3
    mkdir -p "$(dirname "$log")"
    echo "[GPU${gpu}] Starting: $*" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$gpu \
        conda run --no-capture-output -n "$env" bash -c \
        "cd '$RD' && PYTHONPATH=. $* 2>&1 | tee -a '$log'" &
    echo $!   # 返回 PID
}

# 等待一组 PID，有失败则报错退出
wait_batch() {
    local label=$1; shift
    local failed=0
    for pid in "$@"; do
        if ! wait "$pid"; then
            echo "[ERROR] $label — pid=$pid 失败，检查对应 *.log"
            failed=1
        fi
    done
    [ $failed -eq 0 ] && echo "✓ $label 完成" || exit 1
}

# --------------------------------------------------------------------------- #
# Phase 1: DIM Layer Sweep                                                    #
#                                                                             #
# 固定分配（4 GPU × 4 任务同时跑）：                                          #
#   GPU0 = Gemma  V-noise    （无缓存，最慢）                                 #
#   GPU1 = Qwen   V-blank    （有 mean_diffs 缓存，只跑 select_direction）    #
#   GPU2 = Qwen   V-noise    （同上）                                         #
#   GPU3 = Gemma  V-blank    （有 mean_diffs 缓存，只跑 select_direction）    #
#   Qwen V-text / Gemma V-text 已完成，--skip_if_done 秒退出                 #
# --------------------------------------------------------------------------- #
phase_sweep() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Phase 1 — Sweep"
    echo "  GPU0=Gemma/V-noise  GPU1=Qwen/V-blank"
    echo "  GPU2=Qwen/V-noise   GPU3=Gemma/V-blank"
    echo "══════════════════════════════════════════════════"
    mkdir -p "$ROOT/results/pcd/"{qwen_family,gemma_family}/{V-text,V-blank-resweep,V-noise,V-blank}

    local p0 p1 p2 p3

    p0=$(launch 0 qwen3-vl "$ROOT/results/pcd/gemma_family/V-noise/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise --output_dir ../results/pcd/gemma_family/V-noise \
         --kl_threshold 100 --induce_refusal_threshold -999")

    p1=$(launch 1 qwen3-vl "$ROOT/results/pcd/qwen_family/V-blank-resweep/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank --output_dir ../results/pcd/qwen_family/V-blank-resweep \
         --skip_if_done")

    p2=$(launch 2 qwen3-vl "$ROOT/results/pcd/qwen_family/V-noise/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise --output_dir ../results/pcd/qwen_family/V-noise \
         --skip_if_done")

    p3=$(launch 3 qwen3-vl "$ROOT/results/pcd/gemma_family/V-blank/sweep.log" \
        "python ../experiments/pcd/exp_pcd_layer_sweep.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank --output_dir ../results/pcd/gemma_family/V-blank \
         --kl_threshold 100 --induce_refusal_threshold -999 \
         --skip_if_done")

    wait_batch "sweep" "$p0" "$p1" "$p2" "$p3"

    echo ""
    echo "── best_layer 汇总 ──"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/best_layer.json; do
        [ -f "$f" ] || continue
        cond=$(dirname "$f" | sed "s|$ROOT/results/pcd/||")
        info=$(python3 -c "import json; d=json.load(open('$f')); \
            print(f\"L{d['layer']} pos={d['pos']} filter={d['filter_passed']}\")" 2>/dev/null)
        printf "  %-35s  %s\n" "$cond" "$info"
    done
}

# --------------------------------------------------------------------------- #
# Phase 2: DIM Ablate + Generate                                              #
#                                                                             #
# Batch 1 (4 并行): GPU0=Qwen/V-text  GPU1=Qwen/V-blank                      #
#                   GPU2=Qwen/V-noise  GPU3=Gemma/V-text                      #
# Batch 2 (2 并行): GPU0=Gemma/V-blank GPU1=Gemma/V-noise                    #
# --------------------------------------------------------------------------- #
phase_ablate() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Phase 2 — Ablate"
    echo "  Batch1: GPU0=Qwen/V-text   GPU1=Qwen/V-blank"
    echo "          GPU2=Qwen/V-noise  GPU3=Gemma/V-text"
    echo "  Batch2: GPU0=Gemma/V-blank GPU1=Gemma/V-noise"
    echo "══════════════════════════════════════════════════"

    # Batch 1
    local p0 p1 p2 p3
    p0=$(launch 0 qwen3-vl "$ROOT/results/pcd/qwen_family/V-text/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/qwen_family/V-text \
         --output_dir ../results/pcd/qwen_family/V-text")

    p1=$(launch 1 qwen3-vl "$ROOT/results/pcd/qwen_family/V-blank-resweep/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/qwen_family/V-blank-resweep \
         --output_dir ../results/pcd/qwen_family/V-blank-resweep")

    p2=$(launch 2 qwen3-vl "$ROOT/results/pcd/qwen_family/V-noise/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name qwen2.5-vl-7b --model_path '$QWEN_VLM_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/qwen_family/V-noise \
         --output_dir ../results/pcd/qwen_family/V-noise")

    p3=$(launch 3 qwen3-vl "$ROOT/results/pcd/gemma_family/V-text/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-text \
         --sweep_dir ../results/pcd/gemma_family/V-text \
         --output_dir ../results/pcd/gemma_family/V-text")

    wait_batch "ablate-batch1" "$p0" "$p1" "$p2" "$p3"

    # Batch 2
    local p0b p1b
    p0b=$(launch 0 qwen3-vl "$ROOT/results/pcd/gemma_family/V-blank/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-blank \
         --sweep_dir ../results/pcd/gemma_family/V-blank \
         --output_dir ../results/pcd/gemma_family/V-blank")

    p1b=$(launch 1 qwen3-vl "$ROOT/results/pcd/gemma_family/V-noise/ablate.log" \
        "python ../experiments/pcd/exp_pcd_ablate.py \
         --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
         --condition V-noise \
         --sweep_dir ../results/pcd/gemma_family/V-noise \
         --output_dir ../results/pcd/gemma_family/V-noise")

    wait_batch "ablate-batch2" "$p0b" "$p1b"
}

# --------------------------------------------------------------------------- #
# Phase 3: 4-Judge Evaluation                                                 #
#                                                                             #
# 同 ablate 分批策略（LlamaGuard 16GB/GPU，H100 80GB 足够）                   #
# --------------------------------------------------------------------------- #
phase_evaluate() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Phase 3 — Evaluate"
    echo "  Batch1: GPU0=Qwen/V-text   GPU1=Qwen/V-blank"
    echo "          GPU2=Qwen/V-noise  GPU3=Gemma/V-text"
    echo "  Batch2: GPU0=Gemma/V-blank GPU1=Gemma/V-noise"
    echo "══════════════════════════════════════════════════"

    _eval_job() {
        local gpu=$1 subdir=$2 model_name=$3 model_path=$4
        launch "$gpu" qwen3-vl "$ROOT/results/pcd/$subdir/eval.log" \
            "python ../experiments/pcd/exp_pcd_evaluate.py \
             --responses_json ../results/pcd/$subdir/dim_responses.json \
             --model_name $model_name --model_path '$model_path' \
             --output_json ../results/pcd/$subdir/dim_responses_eval.json \
             --layers kw lg3 arditi"
    }

    # Batch 1
    local p0 p1 p2 p3
    p0=$(_eval_job 0 qwen_family/V-text       qwen2.5-vl-7b       "$QWEN_VLM_PATH")
    p1=$(_eval_job 1 qwen_family/V-blank-resweep qwen2.5-vl-7b    "$QWEN_VLM_PATH")
    p2=$(_eval_job 2 qwen_family/V-noise      qwen2.5-vl-7b       "$QWEN_VLM_PATH")
    p3=$(_eval_job 3 gemma_family/V-text      gemma-3-4b-it-vlm   "$GEMMA_PATH")
    wait_batch "evaluate-batch1" "$p0" "$p1" "$p2" "$p3"

    # Batch 2
    local p0b p1b
    p0b=$(_eval_job 0 gemma_family/V-blank    gemma-3-4b-it-vlm   "$GEMMA_PATH")
    p1b=$(_eval_job 1 gemma_family/V-noise    gemma-3-4b-it-vlm   "$GEMMA_PATH")
    wait_batch "evaluate-batch2" "$p0b" "$p1b"

    echo ""
    echo "── eval 汇总 ──"
    printf "  %-35s  %-8s  %-8s  %s\n" "condition" "asr_kw" "asr_lg3" "arditi"
    for f in "$ROOT/results/pcd/"{qwen_family,gemma_family}/*/dim_responses_eval.json; do
        [ -f "$f" ] || continue
        cond=$(dirname "$f" | sed "s|$ROOT/results/pcd/||")
        python3 - "$f" "$cond" <<'PY'
import json, sys
f, cond = sys.argv[1], sys.argv[2]
d = json.load(open(f))
fmt = lambda v: f"{v:.3f}" if isinstance(v, (int,float)) else "-"
print(f"  {cond:<35}  {fmt(d.get('asr_keyword','-'))}      "
      f"{fmt(d.get('asr_lg3','-'))}      {fmt(d.get('arditi_joint_asr','-'))}")
PY
    done
}

# --------------------------------------------------------------------------- #
# Phase 4: Aggregate                                                          #
# --------------------------------------------------------------------------- #
phase_aggregate() {
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Phase 4 — Aggregate 8×6 matrix"
    echo "══════════════════════════════════════════════════"
    cd "$ROOT"
    python3 experiments/pcd/aggregate.py \
        --root results/pcd \
        --out_json results/pcd/pcd_8x6_matrix.json \
        --out_md   results/pcd/pcd_summary.md
    echo ""
    cat results/pcd/pcd_summary.md
}

# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
echo "╔══════════════════════════════════════════════════╗"
printf "║  PCD Stage B  stage=%-28s║\n" "$STAGE"
echo "╚══════════════════════════════════════════════════╝"

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
        echo ""; echo "=== Stage B COMPLETE ==="
        ;;
    *)  echo "Usage: $0 [sweep|ablate|evaluate|aggregate|all]"; exit 1 ;;
esac
