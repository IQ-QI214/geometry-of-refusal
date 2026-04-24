#!/usr/bin/env bash
# launch_ablate.sh — 启动 6 个 ablate 任务（分两批，nohup）
# 前提：所有 6 个条件的 sweep 已完成（best_layer.json 存在）
# 用法：bash experiments/pcd/launch_ablate.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RD="$ROOT/refusal_direction"
QWEN_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct"
GEMMA_PATH="/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it"

# 检查 sweep 是否全部完成
echo "=== 检查 sweep 前置条件 ==="
all_ok=1
for cond in qwen_family/V-text qwen_family/V-blank-resweep qwen_family/V-noise \
            gemma_family/V-text gemma_family/V-blank gemma_family/V-noise; do
    f="$ROOT/results/pcd/$cond/best_layer.json"
    if [ -f "$f" ]; then
        info=$(python3 -c "import json; d=json.load(open('$f')); \
            print(f\"L{d['layer']} pos={d['pos']}\")" 2>/dev/null)
        echo "  ✅ $cond → $info"
    else
        echo "  ❌ $cond → best_layer.json 缺失，请先完成 sweep"
        all_ok=0
    fi
done
[ $all_ok -eq 1 ] || { echo "Sweep 未全部完成，退出。"; exit 1; }

echo ""
echo "=== 批次 1：GPU0-3 并行（Qwen×3 + Gemma/V-text）==="

run() {
    local gpu=$1 log=$2; shift 2
    nohup bash -c "CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n qwen3-vl \
        bash -c \"cd '$RD' && PYTHONPATH=. $*\"" \
        >> "$log" 2>&1 &
    printf "  GPU%d  PID=%-7d  %s\n" "$gpu" "$!" "$(basename $(dirname $log))/$(basename $log)" >&2
    echo $!
}

pids=()
pids+=( $(run 0 "$ROOT/results/pcd/qwen_family/V-text/ablate.log" \
    "python ../experiments/pcd/exp_pcd_ablate.py \
     --model_name qwen2.5-vl-7b --model_path '$QWEN_PATH' \
     --condition V-text \
     --sweep_dir ../results/pcd/qwen_family/V-text \
     --output_dir ../results/pcd/qwen_family/V-text") )

pids+=( $(run 1 "$ROOT/results/pcd/qwen_family/V-blank-resweep/ablate.log" \
    "python ../experiments/pcd/exp_pcd_ablate.py \
     --model_name qwen2.5-vl-7b --model_path '$QWEN_PATH' \
     --condition V-blank \
     --sweep_dir ../results/pcd/qwen_family/V-blank-resweep \
     --output_dir ../results/pcd/qwen_family/V-blank-resweep") )

pids+=( $(run 2 "$ROOT/results/pcd/qwen_family/V-noise/ablate.log" \
    "python ../experiments/pcd/exp_pcd_ablate.py \
     --model_name qwen2.5-vl-7b --model_path '$QWEN_PATH' \
     --condition V-noise \
     --sweep_dir ../results/pcd/qwen_family/V-noise \
     --output_dir ../results/pcd/qwen_family/V-noise") )

pids+=( $(run 3 "$ROOT/results/pcd/gemma_family/V-text/ablate.log" \
    "python ../experiments/pcd/exp_pcd_ablate.py \
     --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
     --condition V-text \
     --sweep_dir ../results/pcd/gemma_family/V-text \
     --output_dir ../results/pcd/gemma_family/V-text") )

echo "等待批次 1 完成..."
for pid in "${pids[@]}"; do wait "$pid" || echo "  警告：pid=$pid 失败"; done
echo "✓ 批次 1 完成"

echo ""
echo "=== 批次 2：GPU0-1 并行（Gemma/V-blank + Gemma/V-noise）==="
pids=()
pids+=( $(run 0 "$ROOT/results/pcd/gemma_family/V-blank/ablate.log" \
    "python ../experiments/pcd/exp_pcd_ablate.py \
     --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
     --condition V-blank \
     --sweep_dir ../results/pcd/gemma_family/V-blank \
     --output_dir ../results/pcd/gemma_family/V-blank") )

pids+=( $(run 1 "$ROOT/results/pcd/gemma_family/V-noise/ablate.log" \
    "python ../experiments/pcd/exp_pcd_ablate.py \
     --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
     --condition V-noise \
     --sweep_dir ../results/pcd/gemma_family/V-noise \
     --output_dir ../results/pcd/gemma_family/V-noise") )

echo "等待批次 2 完成..."
for pid in "${pids[@]}"; do wait "$pid" || echo "  警告：pid=$pid 失败"; done
echo "✓ 批次 2 完成"

echo ""
echo "Ablate 全部完成，运行：bash experiments/pcd/launch_evaluate.sh"
