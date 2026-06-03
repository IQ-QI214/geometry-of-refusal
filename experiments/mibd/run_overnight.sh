#!/usr/bin/env bash
# run_overnight.sh — Phase 1.5 overnight run, 2 GPUs in parallel
#
# GPU 0 (rdo env):
#   Step 1: InternVL3 generate refusal labels
#   Step 2: InternVL3 refusal Phase 1.5 audit
#
# GPU 1 (qwen3-vl env):
#   Step 1: Qwen3-VL harmfulness Phase 1.5 audit
#   Step 2: Qwen3-VL generate refusal labels
#   Step 3: Qwen3-VL refusal Phase 1.5 audit
#
# Usage:
#   cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
#   bash experiments/mibd/run_overnight.sh
#
# All output is tee'd to per-step log files under results/mibd/phase1_probe/.
# A master log is written to results/mibd/phase1_probe/overnight_master.log.
# On completion a summary is appended to the master log.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MASTER_LOG="results/mibd/phase1_probe/overnight_master.log"
mkdir -p results/mibd/phase1_probe/internvl3_8b
mkdir -p results/mibd/phase1_probe/qwen3_vl_8b

MMSAFETY="/inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench"
DATA_DIR="data/saladbench_splits"

_ts() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(_ts)] $*" | tee -a "$MASTER_LOG"; }

log "====== overnight run start ======"
log "ROOT=$ROOT"
log "GPU 0 → InternVL3 refusal labels + refusal audit (rdo env)"
log "GPU 1 → Qwen3-VL harmfulness audit + refusal labels + refusal audit (qwen3-vl env)"

# ─────────────────────────────────────────────────────────────
# GPU 0 pipeline (background)
# ─────────────────────────────────────────────────────────────
gpu0_pipeline() {
    local ts
    ts="$(_ts)"
    log "[GPU0] === Step 1/2: InternVL3 generate refusal labels ==="

    conda run -n rdo python -m experiments.mibd.generate_refusal_labels \
        --model internvl3 \
        --gpu 0 \
        --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
        --data-dir "$DATA_DIR" \
        --mmsafety-dir "$MMSAFETY" \
        --output results/mibd/phase1_probe/internvl3_8b/refusal_labels.json \
        --log-file results/mibd/phase1_probe/internvl3_8b/refusal_labels_gen.log

    log "[GPU0] Step 1 done. refusal_labels.json written."

    log "[GPU0] === Step 2/2: InternVL3 refusal Phase 1.5 audit ==="

    conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
        --model internvl3 \
        --gpu 0 \
        --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
        --signal-type refusal \
        --refusal-labels results/mibd/phase1_probe/internvl3_8b/refusal_labels.json \
        --n-permutations 100 \
        --data-dir "$DATA_DIR" \
        --mmsafety-dir "$MMSAFETY" \
        --log-file results/mibd/phase1_probe/internvl3_8b/phase1p5_refusal.log

    log "[GPU0] Step 2 done. InternVL3 refusal audit complete."
}

# ─────────────────────────────────────────────────────────────
# GPU 1 pipeline (background)
# ─────────────────────────────────────────────────────────────
gpu1_pipeline() {
    log "[GPU1] === Step 1/3: Qwen3-VL harmfulness Phase 1.5 audit ==="

    conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
        --model qwen3vl \
        --gpu 1 \
        --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml \
        --signal-type harmfulness \
        --n-permutations 100 \
        --data-dir "$DATA_DIR" \
        --mmsafety-dir "$MMSAFETY" \
        --log-file results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_harmfulness.log

    log "[GPU1] Step 1 done. Qwen3-VL harmfulness audit complete."

    log "[GPU1] === Step 2/3: Qwen3-VL generate refusal labels ==="

    conda run -n qwen3-vl python -m experiments.mibd.generate_refusal_labels \
        --model qwen3vl \
        --gpu 1 \
        --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml \
        --data-dir "$DATA_DIR" \
        --mmsafety-dir "$MMSAFETY" \
        --output results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels.json \
        --log-file results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels_gen.log

    log "[GPU1] Step 2 done. refusal_labels.json written."

    log "[GPU1] === Step 3/3: Qwen3-VL refusal Phase 1.5 audit ==="

    conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
        --model qwen3vl \
        --gpu 1 \
        --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml \
        --signal-type refusal \
        --refusal-labels results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels.json \
        --n-permutations 100 \
        --data-dir "$DATA_DIR" \
        --mmsafety-dir "$MMSAFETY" \
        --log-file results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_refusal.log

    log "[GPU1] Step 3 done. Qwen3-VL refusal audit complete."
}

# ─────────────────────────────────────────────────────────────
# Launch both pipelines in parallel
# ─────────────────────────────────────────────────────────────
gpu0_pipeline >> "$MASTER_LOG" 2>&1 &
PID0=$!
log "GPU 0 pipeline started (pid=$PID0)"

gpu1_pipeline >> "$MASTER_LOG" 2>&1 &
PID1=$!
log "GPU 1 pipeline started (pid=$PID1)"

# ─────────────────────────────────────────────────────────────
# Wait and collect exit codes
# ─────────────────────────────────────────────────────────────
EXIT0=0; EXIT1=0
wait $PID0 || EXIT0=$?
log "GPU 0 pipeline finished (exit=$EXIT0)"

wait $PID1 || EXIT1=$?
log "GPU 1 pipeline finished (exit=$EXIT1)"

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
log ""
log "====== overnight run complete ======"
log ""
log "Results summary:"
for f in \
    results/mibd/phase1_probe/internvl3_8b/refusal_labels.json \
    results/mibd/phase1_probe/internvl3_8b/phase1p5_refusal.log \
    results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_harmfulness.log \
    results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels.json \
    results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_refusal.log; do
    if [[ -f "$f" ]]; then
        SIZE=$(wc -c < "$f")
        log "  ✓  $f  (${SIZE} bytes)"
    else
        log "  ✗  MISSING: $f"
    fi
done

log ""
if [[ $EXIT0 -eq 0 && $EXIT1 -eq 0 ]]; then
    log "All pipelines succeeded."
else
    log "WARNING: GPU0 exit=$EXIT0  GPU1 exit=$EXIT1 — check logs above."
fi
