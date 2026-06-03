#!/bin/bash
set -e

ROOT="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal"
MMSAFETY="/inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench"

cd "$ROOT"
mkdir -p logs

echo "Starting Qwen3-VL on GPU 0..."
conda run -n qwen3-vl python -m experiments.mibd.run_phase1 \
  --model qwen3vl --gpu 0 \
  --config experiments/mibd/configs/phase1_probe.yaml \
  --data-dir data/saladbench_splits \
  --mmsafety-dir "$MMSAFETY" \
  > logs/mibd_phase1_qwen3vl.log 2>&1 &
PID_QWEN=$!

echo "Starting InternVL3 on GPU 1..."
conda run -n rdo python -m experiments.mibd.run_phase1 \
  --model internvl3 --gpu 1 \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
  --data-dir data/saladbench_splits \
  --mmsafety-dir "$MMSAFETY" \
  > logs/mibd_phase1_internvl3.log 2>&1 &
PID_INTERN=$!

echo "Qwen3-VL PID=$PID_QWEN  InternVL3 PID=$PID_INTERN"
echo "Logs: logs/mibd_phase1_qwen3vl.log  logs/mibd_phase1_internvl3.log"
wait $PID_QWEN && echo "Qwen3-VL DONE" || echo "Qwen3-VL FAILED (exit $?)"
wait $PID_INTERN && echo "InternVL3 DONE" || echo "InternVL3 FAILED (exit $?)"
