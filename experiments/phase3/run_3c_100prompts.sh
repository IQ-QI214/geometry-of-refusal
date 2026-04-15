#!/bin/bash
# Exp 3C: 100 prompts 全模型并行跑
# 用法: bash run_3c_100prompts.sh

set -e
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$PROJ_DIR/experiments/phase3/logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] Starting Exp 3C (100 prompts) for all 4 models..."

CUDA_VISIBLE_DEVICES=0 conda run -n rdo --no-capture-output \
    python "$PROJ_DIR/experiments/phase3/exp_3c_ablation_attack.py" \
    --model llava_7b --device cuda:0 --n_prompts 100 \
    > "$LOG_DIR/3c_100_llava7b.log" 2>&1 &
PID_LLAVA=$!
echo "  [llava_7b]      PID=$PID_LLAVA  -> $LOG_DIR/3c_100_llava7b.log"

CUDA_VISIBLE_DEVICES=1 conda run -n qwen3-vl --no-capture-output \
    python "$PROJ_DIR/experiments/phase3/exp_3c_ablation_attack.py" \
    --model qwen2vl_7b --device cuda:0 --n_prompts 100 \
    > "$LOG_DIR/3c_100_qwen2vl.log" 2>&1 &
PID_QWEN=$!
echo "  [qwen2vl_7b]    PID=$PID_QWEN  -> $LOG_DIR/3c_100_qwen2vl.log"

CUDA_VISIBLE_DEVICES=2 conda run -n rdo --no-capture-output \
    python "$PROJ_DIR/experiments/phase3/exp_3c_ablation_attack.py" \
    --model internvl2_8b --device cuda:0 --n_prompts 100 \
    > "$LOG_DIR/3c_100_internvl2.log" 2>&1 &
PID_INTERN=$!
echo "  [internvl2_8b]  PID=$PID_INTERN  -> $LOG_DIR/3c_100_internvl2.log"

CUDA_VISIBLE_DEVICES=3 conda run -n rdo --no-capture-output \
    python "$PROJ_DIR/experiments/phase3/exp_3c_ablation_attack.py" \
    --model instructblip_7b --device cuda:0 --n_prompts 100 \
    > "$LOG_DIR/3c_100_instructblip.log" 2>&1 &
PID_BLIP=$!
echo "  [instructblip_7b] PID=$PID_BLIP  -> $LOG_DIR/3c_100_instructblip.log"

echo ""
echo "All 4 jobs launched. Waiting..."
wait $PID_LLAVA $PID_QWEN $PID_INTERN $PID_BLIP
EXIT_CODES=($? )

echo ""
echo "[$(date '+%H:%M:%S')] All done. Verifying results..."
python -c "
import json
for model in ['llava_7b', 'qwen2vl_7b', 'internvl2_8b', 'instructblip_7b']:
    try:
        with open('$PROJ_DIR/results/phase3/' + model + '/exp_3c_results.json') as f:
            r = json.load(f)
        n = r['n_prompts']
        asr = r['configs']['ablation_all_vmm']['metrics']['full_harmful_completion_rate']
        print(f'  {model}: n={n}, all-layer ASR={asr:.1%}')
    except Exception as e:
        print(f'  {model}: ERROR - {e}')
"
