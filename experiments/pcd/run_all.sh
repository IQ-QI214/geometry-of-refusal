#!/usr/bin/env bash
# run_all.sh — PCD pipeline orchestrator
# Run from: refusal_direction/  with PYTHONPATH=.
# Usage: bash ../experiments/pcd/run_all.sh [bootstrap|sweep|ablate|rdo|judge|all]

set -euo pipefail

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
GEMMA_PATH="/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it"
QWEN_LLM_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct"
QWEN_VLM_PATH="/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# Conda env notes:
#   rdo       — Qwen2.5-7B, general pipeline  (transformers 4.47)
#   qwen3-vl  — Qwen2.5-VL AND Gemma-3        (transformers 4.57.3)
#   Gemma-3 MUST use qwen3-vl; rdo lacks Gemma-3 support.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="$SCRIPT_DIR/exp_pcd_layer_sweep.py"
ABLATE_SCRIPT="$SCRIPT_DIR/exp_pcd_ablate.py"
JUDGE_SCRIPT="$SCRIPT_DIR/exp_pcd_evaluate.py"

# ---------------------------------------------------------------------------
run_bootstrap() {
    echo "=== [bootstrap] Running bootstrap on Qwen2.5-7B (L condition) ==="
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n rdo bash -c \
        "PYTHONPATH=. python $SWEEP_SCRIPT \
            --model_path $QWEN_LLM_PATH \
            --model_type qwen_llm \
            --condition L \
            --bootstrap \
            2>&1 | tee $SCRIPT_DIR/logs/bootstrap_qwen_llm.log"
    echo "=== [bootstrap] Done ==="
}

# ---------------------------------------------------------------------------
run_sweep() {
    echo "=== [sweep] Starting layer sweep — 6 conditions in 2 batches ==="

    # Batch 1: Qwen V-text, V-noise, V-blank (GPU 0,1,2) + Gemma V-text (GPU 3)
    # All use qwen3-vl env (Qwen VLM needs it; Gemma-3 needs it)
    echo "--- [sweep] Batch 1: Qwen V-text/V-noise/V-blank + Gemma V-text ---"
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl bash -c \
        "PYTHONPATH=. python $SWEEP_SCRIPT \
            --model_path $QWEN_VLM_PATH \
            --model_type qwen_vlm \
            --condition V-text \
            2>&1 | tee $SCRIPT_DIR/logs/sweep_qwen_vtext.log" &

    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n qwen3-vl bash -c \
        "PYTHONPATH=. python $SWEEP_SCRIPT \
            --model_path $QWEN_VLM_PATH \
            --model_type qwen_vlm \
            --condition V-noise \
            2>&1 | tee $SCRIPT_DIR/logs/sweep_qwen_vnoise.log" &

    CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n qwen3-vl bash -c \
        "PYTHONPATH=. python $SWEEP_SCRIPT \
            --model_path $QWEN_VLM_PATH \
            --model_type qwen_vlm \
            --condition V-blank \
            2>&1 | tee $SCRIPT_DIR/logs/sweep_qwen_vblank.log" &

    # Gemma: qwen3-vl env (NOT rdo — rdo has transformers 4.47 which lacks Gemma-3 support)
    CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n qwen3-vl bash -c \
        "PYTHONPATH=. python $SWEEP_SCRIPT \
            --model_path $GEMMA_PATH \
            --model_type gemma_vlm \
            --condition V-text \
            2>&1 | tee $SCRIPT_DIR/logs/sweep_gemma_vtext.log" &

    wait
    echo "--- [sweep] Batch 1 complete ---"

    # Batch 2: Gemma V-blank (GPU 0) + Gemma V-noise (GPU 1)
    # Both use qwen3-vl env
    echo "--- [sweep] Batch 2: Gemma V-blank + Gemma V-noise ---"
    CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl bash -c \
        "PYTHONPATH=. python $SWEEP_SCRIPT \
            --model_path $GEMMA_PATH \
            --model_type gemma_vlm \
            --condition V-blank \
            2>&1 | tee $SCRIPT_DIR/logs/sweep_gemma_vblank.log" &

    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n qwen3-vl bash -c \
        "PYTHONPATH=. python $SWEEP_SCRIPT \
            --model_path $GEMMA_PATH \
            --model_type gemma_vlm \
            --condition V-noise \
            2>&1 | tee $SCRIPT_DIR/logs/sweep_gemma_vnoise.log" &

    wait
    echo "--- [sweep] Batch 2 complete ---"
    echo "=== [sweep] All 6 conditions done ==="
}

# ---------------------------------------------------------------------------
run_ablate() {
    echo "=== [ablate] TODO: fill in ablation commands (Task 11) ==="
    # TODO: call exp_pcd_ablate.py for each model/condition once sweep artifacts exist
}

# ---------------------------------------------------------------------------
run_rdo() {
    echo "=== [rdo] TODO: fill in RDO commands (Task 12) ==="
    # TODO: run refusal direction orthogonalization for selected layers
}

# ---------------------------------------------------------------------------
run_judge() {
    echo "=== [judge] TODO: fill in judge commands (Task 13) ==="
    # TODO: call exp_pcd_evaluate.py on ablation outputs
}

# ---------------------------------------------------------------------------
# Ensure log directory exists
mkdir -p "$SCRIPT_DIR/logs"

# Dispatch
STAGE="${1:-all}"
case "$STAGE" in
    bootstrap)
        run_bootstrap
        ;;
    sweep)
        run_sweep
        ;;
    ablate)
        run_ablate
        ;;
    rdo)
        run_rdo
        ;;
    judge)
        run_judge
        ;;
    all)
        run_bootstrap
        run_sweep
        run_ablate
        run_rdo
        run_judge
        ;;
    *)
        echo "Usage: $0 [bootstrap|sweep|ablate|rdo|judge|all]"
        exit 1
        ;;
esac
