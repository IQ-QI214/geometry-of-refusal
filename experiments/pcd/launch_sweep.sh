#!/usr/bin/env bash
# launch_sweep.sh — 一键启动 4 个 sweep 实验（nohup，SSH 断开不影响）
# 用法：bash experiments/pcd/launch_sweep.sh
# 完成后用 bash experiments/pcd/launch_ablate.sh 继续

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RD="$ROOT/refusal_direction"
QWEN_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct"
GEMMA_PATH="/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it"
PID_FILE="$ROOT/results/pcd/sweep_pids.txt"

mkdir -p "$ROOT/results/pcd/"{qwen_family/{V-blank-resweep,V-noise},gemma_family/{V-blank,V-noise}}
: > "$PID_FILE"

echo "=== 启动 4 个 sweep 任务（nohup）==="

run() {
    local gpu=$1 log=$2; shift 2
    nohup bash -c "CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n qwen3-vl \
        bash -c \"cd '$RD' && PYTHONPATH=. $*\"" \
        >> "$log" 2>&1 &
    local pid=$!
    echo "$pid" >> "$PID_FILE"
    printf "  GPU%d  PID=%-7d  %s\n" "$gpu" "$pid" "$(basename $(dirname $log))/$(basename $log)"
}

run 1 "$ROOT/results/pcd/qwen_family/V-blank-resweep/sweep.log" \
    "python ../experiments/pcd/exp_pcd_layer_sweep.py \
     --model_name qwen2.5-vl-7b --model_path '$QWEN_PATH' \
     --condition V-blank --output_dir ../results/pcd/qwen_family/V-blank-resweep"

run 2 "$ROOT/results/pcd/qwen_family/V-noise/sweep.log" \
    "python ../experiments/pcd/exp_pcd_layer_sweep.py \
     --model_name qwen2.5-vl-7b --model_path '$QWEN_PATH' \
     --condition V-noise --output_dir ../results/pcd/qwen_family/V-noise"

run 3 "$ROOT/results/pcd/gemma_family/V-blank/sweep.log" \
    "python ../experiments/pcd/exp_pcd_layer_sweep.py \
     --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
     --condition V-blank --output_dir ../results/pcd/gemma_family/V-blank \
     --kl_threshold 100 --induce_refusal_threshold -999"

run 0 "$ROOT/results/pcd/gemma_family/V-noise/sweep.log" \
    "python ../experiments/pcd/exp_pcd_layer_sweep.py \
     --model_name gemma-3-4b-it-vlm --model_path '$GEMMA_PATH' \
     --condition V-noise --output_dir ../results/pcd/gemma_family/V-noise \
     --kl_threshold 100 --induce_refusal_threshold -999"

echo ""
echo "PIDs 保存到 $PID_FILE"
echo "监控：bash experiments/pcd/check_status.sh"
echo "完成后运行：bash experiments/pcd/launch_ablate.sh"
