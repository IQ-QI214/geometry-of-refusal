#!/bin/bash
# P0-B: Full-layer 3A scan (stride=2), 4 models on 4 GPUs in parallel
# Expected: ~30min (inference only, no generation)
# Output: results/phase3/{model}/exp_3a_results_full.json + exp_3a_directions_full.pt

set -e

cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p experiments/phase3/logs

echo "[$(date)] Starting P0-B: Full-layer 3A scan..."

CUDA_VISIBLE_DEVICES=0 conda run -n rdo python experiments/phase3/exp_3a_amplitude_reversal.py --model llava_7b --device cuda:0 --full_layers > experiments/phase3/logs/3a_full_llava7b.log 2>&1 &
PID_LLAVA=$!

CUDA_VISIBLE_DEVICES=1 conda run -n qwen3-vl python experiments/phase3/exp_3a_amplitude_reversal.py --model qwen2vl_7b --device cuda:0 --full_layers > experiments/phase3/logs/3a_full_qwen2vl.log 2>&1 &
PID_QWEN=$!

CUDA_VISIBLE_DEVICES=2 conda run -n rdo python experiments/phase3/exp_3a_amplitude_reversal.py --model internvl2_8b --device cuda:0 --full_layers > experiments/phase3/logs/3a_full_internvl2.log 2>&1 &
PID_INTERN=$!

CUDA_VISIBLE_DEVICES=3 conda run -n rdo python experiments/phase3/exp_3a_amplitude_reversal.py --model instructblip_7b --device cuda:0 --full_layers > experiments/phase3/logs/3a_full_instructblip.log 2>&1 &
PID_IBLIP=$!

echo "  LLaVA PID=$PID_LLAVA, Qwen PID=$PID_QWEN, InternVL PID=$PID_INTERN, InstructBLIP PID=$PID_IBLIP"
echo "  Logs: experiments/phase3/logs/3a_full_*.log"

wait
echo "[$(date)] P0-B: Full-layer 3A scan completed!"

# Quick verification
python -c "
import json
for model in ['llava_7b', 'qwen2vl_7b', 'internvl2_8b', 'instructblip_7b']:
    try:
        with open(f'results/phase3/{model}/exp_3a_results_full.json') as f:
            r = json.load(f)
        n = len(r['probe_layers'])
        print(f'{model}: {n} probe layers, narrow_waist=layer {r[\"narrow_waist_layer\"]} (cos={r[\"narrow_waist_cos\"]:.4f})')
    except Exception as e:
        print(f'{model}: FAILED - {e}')
"
