#!/bin/bash
# Run Exp 3D: Layer-wise Ablation Curve on all 4 models in parallel
# Usage: bash run_3d_layerwise.sh [n_prompts=50]
#
# GPU assignment:
#   GPU 0: LLaVA-1.5-7B     (rdo env)
#   GPU 1: Qwen2.5-VL-7B    (qwen3-vl env, transformers >= 4.52)
#   GPU 2: InternVL2-8B     (rdo env, trust_remote_code)
#   GPU 3: InstructBLIP-7B  (rdo env)

N_PROMPTS=${1:-50}
SCRIPT="experiments/phase3/exp_3d_layerwise_ablation.py"
LOG_DIR="experiments/phase3/logs"
mkdir -p "$LOG_DIR"

echo "Starting Exp 3D with n_prompts=${N_PROMPTS}"

CUDA_VISIBLE_DEVICES=0 conda run -n rdo \
    python "$SCRIPT" --model llava_7b --device cuda:0 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_llava7b.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 conda run -n qwen3-vl \
    python "$SCRIPT" --model qwen2vl_7b --device cuda:1 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_qwen2vl.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 conda run -n rdo \
    python "$SCRIPT" --model internvl2_8b --device cuda:2 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_internvl2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 conda run -n rdo \
    python "$SCRIPT" --model instructblip_7b --device cuda:3 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_instructblip.log" 2>&1 &

wait
echo "Exp 3D complete. Results in results/phase3/{model}/exp_3d_layerwise_results.json"
