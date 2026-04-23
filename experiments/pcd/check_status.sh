#!/usr/bin/env bash
# check_status.sh — 一键查看所有实验的当前状态
# 用法：bash experiments/pcd/check_status.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PCD 实验状态检查                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo ""
echo "── 运行中的进程 ──"
procs=$(ps aux | grep "exp_pcd" | grep -v grep)
if [ -z "$procs" ]; then
    echo "  （无 exp_pcd 进程运行）"
else
    echo "$procs" | awk '{printf "  PID:%-7s GPU:%-3s %s\n", $2, $0, $11}'
fi

echo ""
echo "── Sweep（best_layer.json）──"
for cond in qwen_family/V-text qwen_family/V-blank-resweep qwen_family/V-noise \
            gemma_family/V-text gemma_family/V-blank gemma_family/V-noise; do
    bl="$ROOT/results/pcd/$cond/best_layer.json"
    md="$ROOT/results/pcd/$cond/mean_diffs.pt"
    if [ -f "$bl" ]; then
        info=$(python3 -c "import json; d=json.load(open('$bl')); \
            print(f\"L{d['layer']} pos={d['pos']} filter={d['filter_passed']}\")" 2>/dev/null)
        printf "  ✅ %-35s %s\n" "$cond" "$info"
    elif [ -f "$md" ]; then
        printf "  🔶 %-35s mean_diffs✅ best_layer❌\n" "$cond"
    else
        printf "  ❌ %-35s 未开始\n" "$cond"
    fi
done

echo ""
echo "── Ablate（dim_responses.json）──"
for cond in qwen_family/V-text qwen_family/V-blank-resweep qwen_family/V-noise \
            gemma_family/V-text gemma_family/V-blank gemma_family/V-noise; do
    f="$ROOT/results/pcd/$cond/dim_responses.json"
    if [ -f "$f" ]; then
        n=$(python3 -c "import json; d=json.load(open('$f')); \
            print(len(d.get('responses', d) if isinstance(d,dict) else d))" 2>/dev/null || echo "?")
        printf "  ✅ %-35s %s 条响应\n" "$cond" "$n"
    else
        printf "  ❌ %-35s\n" "$cond"
    fi
done

echo ""
echo "── Evaluate（dim_responses_eval.json）──"
for cond in qwen_family/V-text qwen_family/V-blank-resweep qwen_family/V-noise \
            gemma_family/V-text gemma_family/V-blank gemma_family/V-noise; do
    f="$ROOT/results/pcd/$cond/dim_responses_eval.json"
    if [ -f "$f" ]; then
        info=$(python3 -c "
import json, sys
d = json.load(open('$f'))
kw  = d.get('asr_keyword', float('nan'))
lg3 = d.get('asr_lg3', float('nan'))
ar  = d.get('arditi_joint_asr', float('nan'))
print(f'asr_kw={kw:.3f} lg3={lg3:.3f} arditi={ar:.3f}')
" 2>/dev/null || echo "解析失败")
        printf "  ✅ %-35s %s\n" "$cond" "$info"
    else
        printf "  ❌ %-35s\n" "$cond"
    fi
done

echo ""
echo "── 下一步 ──"
all_sweep=1; all_ablate=1; all_eval=1
for cond in qwen_family/V-text qwen_family/V-blank-resweep qwen_family/V-noise \
            gemma_family/V-text gemma_family/V-blank gemma_family/V-noise; do
    [ -f "$ROOT/results/pcd/$cond/best_layer.json" ]       || all_sweep=0
    [ -f "$ROOT/results/pcd/$cond/dim_responses.json" ]    || all_ablate=0
    [ -f "$ROOT/results/pcd/$cond/dim_responses_eval.json" ] || all_eval=0
done

if   [ $all_sweep -eq 0 ];  then echo "  → bash experiments/pcd/launch_sweep.sh"
elif [ $all_ablate -eq 0 ]; then echo "  → bash experiments/pcd/launch_ablate.sh"
elif [ $all_eval -eq 0 ];   then echo "  → bash experiments/pcd/launch_evaluate.sh"
else                              echo "  → bash experiments/pcd/launch_aggregate.sh"
fi
