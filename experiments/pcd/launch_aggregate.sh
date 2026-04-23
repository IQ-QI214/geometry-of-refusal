#!/usr/bin/env bash
# launch_aggregate.sh — 汇总 8×6 矩阵（CPU，无需 GPU）
# 用法：bash experiments/pcd/launch_aggregate.sh

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== 汇总 PCD 8×6 矩阵 ==="
cd "$ROOT"
python3 experiments/pcd/aggregate.py \
    --root    results/pcd \
    --out_json results/pcd/pcd_8x6_matrix.json \
    --out_md   results/pcd/pcd_summary.md
echo ""
cat results/pcd/pcd_summary.md
