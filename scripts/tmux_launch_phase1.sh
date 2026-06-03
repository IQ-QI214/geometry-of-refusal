#!/bin/bash
# Create two tmux windows and launch MIBD Phase 1 experiments.
# Usage: bash scripts/tmux_launch_phase1.sh [--gpu0 0] [--gpu1 1] [--session mibd_phase1]
#
# After running this script:
#   tmux attach -t mibd_phase1          # attach to session
#   Ctrl-b 0  →  Qwen3-VL window
#   Ctrl-b 1  →  InternVL3 window
#   Ctrl-b d  →  detach (experiments keep running)

set -euo pipefail

ROOT="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal"
SESSION="mibd_phase1"
GPU0=0
GPU1=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu0)    GPU0="$2";    shift 2 ;;
    --gpu1)    GPU1="$2";    shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# ── kill existing session if present ──────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux] Session '$SESSION' already exists. Killing it first..."
  tmux kill-session -t "$SESSION"
fi

# ── create session with first window: Qwen3-VL ────────────────
tmux new-session -d -s "$SESSION" -n "qwen3vl" -x 220 -y 50

tmux send-keys -t "${SESSION}:qwen3vl" \
  "cd '$ROOT' && bash scripts/launch_qwen3vl.sh --gpu $GPU0" Enter

# ── create second window: InternVL3 ───────────────────────────
tmux new-window -t "$SESSION" -n "internvl3"

tmux send-keys -t "${SESSION}:internvl3" \
  "cd '$ROOT' && bash scripts/launch_internvl3.sh --gpu $GPU1" Enter

# ── status ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  MIBD Phase 1 experiments launched in tmux session:"
echo "    Session : $SESSION"
echo "    Window 0: qwen3vl   (GPU $GPU0)"
echo "    Window 1: internvl3 (GPU $GPU1)"
echo ""
echo "  Attach:  tmux attach -t $SESSION"
echo "  Switch:  Ctrl-b 0 / Ctrl-b 1"
echo "  Detach:  Ctrl-b d"
echo ""
echo "  Logs:"
echo "    $ROOT/logs/mibd_phase1/qwen3vl/"
echo "    $ROOT/logs/mibd_phase1/internvl3/"
echo ""
echo "  Results (after completion):"
echo "    $ROOT/results/mibd/phase1_probe/qwen3_vl_8b/phase1_report.md"
echo "    $ROOT/results/mibd/phase1_probe/internvl3_8b/phase1_report.md"
echo "============================================================"
