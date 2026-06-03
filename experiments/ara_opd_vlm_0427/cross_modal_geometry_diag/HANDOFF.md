# ARA-OPD-VLM 诊断阶段 — 会话交接文档

**更新时间**：2026-04-30  
**当前状态**：GPU 0 运行 heretic probe（Task 2），GPU 1/2/3 空闲。Task 4 冒烟测试待确认，Task 5 sweep 可立即并行启动。

**GPU 分配**：GPU 0 = heretic probe，GPU 1 = V-text sweep，GPU 2 = V-blank sweep，GPU 3 = V-noise sweep。sweep 完成后 ablate/evaluate/projector 用 GPU 1。

---

## 一、已完成的工作

| 任务 | 内容 | 状态 |
|---|---|---|
| Task 4 | Qwen3-VL model adapter（`qwen3_vlm_model.py` + `model_factory.py` 注册） | ✅ 完成，commit `bc6025d` |
| Task 8 | `compute_alignment.py`（余弦矩阵计算，已用 PCD 数据验证） | ✅ 完成 |
| Task 9 | `projector_causal_test.py`（projector 因果测试，forward pass bug 已修复） | ✅ 完成 |
| Task 10 | `aggregate_diag.py`（汇总脚本，生成 target_modules.json） | ✅ 完成 |
| Task 1 | smoke test 全部通过（A/B/C/D/E） | ✅ 完成 |

**compute_alignment.py 验证结果**（Qwen2.5-VL，与 PCD 一致）：
- c1（LLM vs V-text）= 0.671
- c2（V-text vs V-blank）= 0.804
- c3（LLM vs V-blank）= 0.492
- 级联预测 c1×c2 = 0.539，误差 -0.047 → **两效应接近级联**

---

## 二、待执行任务（qi 手动操作）

### Task 1 & 2：venv smoke test + heretic probe ✅ 运行中

**Task 1**：smoke test 全部通过（A/B/C/D/E ✅）

**Task 2**：heretic probe 已在 GPU 0 后台运行（PID 见 `heretic_probe/probe.pid`）

监控进度：

```bash
tail -f /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/probe.log
```

完成确认：

```bash
python3 -c "
import json
d = json.load(open('/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/heretic_probe_n50.json'))
print('total:', len(d))
"
```

**venv 启动方式**（如需重启）：

```bash
VENV_SITE=/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/.venv_gemma_probe/lib/python3.12/site-packages

nohup env PYTHONPATH="$VENV_SITE" CUDA_VISIBLE_DEVICES=0 \
    /opt/conda/envs/qwen3-vl/bin/python3 \
    /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/experiments/ara_sapp/exp_gemma4_heretic_probe.py all --n 50 \
    --output /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/heretic_probe_n50.json \
    > /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/probe.log 2>&1 &
echo $! > /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/heretic_probe/probe.pid
```

### Task 3：下载 Qwen3-VL-8B-Instruct（✅ 已完成）

模型已下载到 `/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B`（17.5GB，16 个文件）。

验证结果：
- `architectures`: `Qwen3VLForConditionalGeneration`（**非** Qwen2.5-VL 类名）
- `num_hidden_layers`: 36
- `model_type`: `qwen3_vl`

**已修复**：`qwen3_vlm_model.py` 已将 `Qwen2_5_VLForConditionalGeneration` 改为 `AutoModelForImageTextToText`，兼容新类名。

### Task 4 附加确认：adapter 冒烟测试（GPU 1）

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n qwen3-vl python3 -c "
import sys
sys.path.insert(0, '/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/refusal_direction')
from pipeline.model_utils.model_factory import construct_model_base
m = construct_model_base(
    '/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B',
    model_name='qwen3vl_8b'
)
print('model_type:', type(m).__name__)
print('num_layers:', len(m._get_model_block_modules()))
print('PASS')
"
```

### Task 5：Qwen3-VL Sweep（V-text 已完成，V-blank/V-noise 用 --reselect 重跑）

V-text 已完成（layer 12）。V-blank/V-noise 在 select_direction 阶段 OOM，mean_diffs 已缓存，用 `--reselect --select_batch_size 2` 从缓存重跑：

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

for COND_GPUS in "V-blank:0,2" "V-noise:1,3"; do
    COND="${COND_GPUS%%:*}"
    GPUS="${COND_GPUS##*:}"
    OUTDIR="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/sweep"
    LOGFILE="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/sweep.log"
    mkdir -p "$OUTDIR"
    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=refusal_direction \
        /opt/conda/envs/qwen3-vl/bin/python3 -u \
        experiments/pcd/exp_pcd_layer_sweep.py \
        --model_name qwen3vl_8b \
        --model_path /inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B \
        --condition $COND \
        --output_dir $OUTDIR \
        --select_n_val 128 \
        --select_batch_size 2 \
        --reselect" \
        > "$LOGFILE" 2>&1 &
    echo $! > "results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/sweep.pid"
    echo "[sweep] $COND on GPU $GPUS, PID=$!"
done
```

监控：

```bash
tail -f /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/V-blank/sweep.log
```

完成确认：

```bash
for COND in V-text V-blank V-noise; do
    F="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/sweep/best_layer.json"
    [ -f "$F" ] && echo "$COND: $(cat $F)" || echo "$COND: 未完成"
done
```

### Task 6：Ablate（sweep 全部完成后，GPU 0/1 并行，V-noise 顺序）

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

for COND_GPU in "V-text:0" "V-blank:1" "V-noise:0"; do
    COND="${COND_GPU%%:*}"
    GPU="${COND_GPU##*:}"
    OUTDIR="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND"
    LOGFILE="$OUTDIR/ablate.log"
    mkdir -p "$OUTDIR"
    nohup CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=refusal_direction \
        /opt/conda/envs/qwen3-vl/bin/python3 -u \
        experiments/pcd/exp_pcd_ablate.py \
        --model_name qwen3vl_8b \
        --model_path /inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B \
        --condition "$COND" \
        --sweep_dir "$OUTDIR/sweep" \
        --output_dir "$OUTDIR" \
        > "$LOGFILE" 2>&1 &
    echo $! > "$OUTDIR/ablate.pid"
    echo "[ablate] $COND on GPU $GPU, PID=$!"
done
```

完成确认：

```bash
for COND in V-text V-blank V-noise; do
    F="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/dim_responses.json"
    [ -f "$F" ] && echo "$COND OK" || echo "$COND: 未完成"
done
```

### Task 7：Evaluate（ablate 全部完成后，GPU 0/1 并行）

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

for COND_GPU in "V-text:0" "V-blank:1" "V-noise:0"; do
    COND="${COND_GPU%%:*}"
    GPU="${COND_GPU##*:}"
    OUTDIR="results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND"
    LOGFILE="$OUTDIR/evaluate.log"
    nohup CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=refusal_direction \
        /opt/conda/envs/qwen3-vl/bin/python3 -u \
        experiments/pcd/exp_pcd_evaluate.py \
        --model_name qwen3vl_8b \
        --model_path /inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B \
        --condition "$COND" \
        --responses_file "$OUTDIR/dim_responses.json" \
        --output_file "$OUTDIR/dim_eval.json" \
        > "$LOGFILE" 2>&1 &
    echo $! > "$OUTDIR/evaluate.pid"
    echo "[evaluate] $COND on GPU $GPU, PID=$!"
done
```

完成确认：

```bash
for COND in V-text V-blank V-noise; do
    F="/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/qwen3vl/$COND/dim_eval.json"
    [ -f "$F" ] && python3 -c "
import json; d=json.load(open('$F'))
print('$COND: asr_kw=', d.get('asr_kw'), 'asr_lg3=', d.get('asr_lg3'))
" || echo "$COND: 未完成"
done
```

### Task 8 重新运行（sweep 完成后，CPU 执行）

Qwen3-VL sweep 完成后重跑 compute_alignment.py，即可获得 Qwen3-VL 的 c2 值：

```bash
python3 /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/compute_alignment.py
```

### Task 9：Projector 因果测试（GPU 1，ablate 完成后）

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

nohup CUDA_VISIBLE_DEVICES=1 PYTHONPATH=refusal_direction \
    /opt/conda/envs/qwen3-vl/bin/python3 -u \
    experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.py \
    > results/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.log 2>&1 &
echo $! > results/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.pid

# 结果确认：
cat /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.json
```

### Task 10：最终汇总（所有 GPU 任务完成后，CPU 执行）

```bash
python3 /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/aggregate_diag.py
# 输出：target_modules.json + 本 HANDOFF.md 自动更新
```

---

## 三、关键文件位置

| 文件 | 路径 |
|---|---|
| 设计规格 | `docs/superpowers/specs/2026-04-27-ara-opd-vlm-design.md` |
| 实验计划 | `docs/superpowers/plans/2026-04-27-ara-opd-vlm-diag.md` |
| Qwen3-VL adapter | `refusal_direction/pipeline/model_utils/qwen3_vlm_model.py` |
| compute_alignment | `experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/compute_alignment.py` |
| projector_causal_test | `experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.py` |
| aggregate_diag | `experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/aggregate_diag.py` |
| 当前 alignment 结果 | `results/ara_opd_vlm_0427/cross_modal_geometry_diag/cross_modal_alignment.json` |

---

## 四、已知注意事项

1. **Qwen3-VL HF 类名**：已确认为 `Qwen3VLForConditionalGeneration`，`qwen3_vlm_model.py` 已改用 `AutoModelForImageTextToText` 兼容。模型路径为 `Qwen3-VL-8B`（非 Instruct 后缀）。

2. **venv 状态**：`.venv_gemma_probe/` 用 `/opt/conda/envs/qwen3-vl/bin/python3` 创建，transformers 5.5.4 已强制装入 venv。smoke test 全部通过。

3. **Gemma-3 c2 异常**：PCD 数据中 c2(V-text vs V-blank) = -0.006，原因是 Gemma 在 V-blank 条件下 best layer 跳至第 1 层（PCD findings §8.1），导致比较的是不同层的方向向量。分析时需特别注意。

4. **sweep 3 条件并行**：GPU 1/2/3 各跑一个条件，互不干扰。ablate/evaluate/projector 在 sweep 完成后用 GPU 1。

---

## 五、诊断完成后的下一步

所有 GPU 任务完成 → 运行 `aggregate_diag.py` → 生成 `target_modules.json` 后：
- **编写 ARA 实验计划**（ara_vlm/ 子目录）
- 根据 `target_modules.json` 中 `ara_target_modules_decision` 的值决定 ARA 攻击模块范围
