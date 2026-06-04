# MIBD Phase 1.5 交接文档 v2
**时间**: 2026-06-04  
**接续自**: `2026-06-04-mibd-phase1p5-handoff.md`

---

## 当前进度快照

### 已完成
| 模型 | 信号 | Phase 1.5 状态 |
|------|------|----------------|
| InternVL3-8B | harmfulness | ✅ PASS (perm=0.475±0.186, held-out=1.000) |
| Qwen3-VL-8B | harmfulness | ✅ PASS (perm=0.517±0.171, held-out=1.000) |
| InternVL3-8B | refusal | ❌ 未跑（本 session 修复了 bug，待启动） |
| Qwen3-VL-8B | refusal | ❌ 未跑（标签可能不足，见下方警告） |
| Gemma3-4B | harmfulness + refusal | ❌ 未开始 |

---

## 本 Session 关键修复

### Bug：probe 函数硬编码 `"harmful"/"harmless"` 标签

**根因**：以下函数内部直接 `lp.get("harmful")` / `lp.get("harmless")`，
导致 refusal signal（标签为 `"refusal"/"compliance"`）时返回空 dict，
触发 `"no V-text probes trained"` 错误。

**修复文件**（已 commit `2137d90`）：
- `experiments/mibd/probes/train.py` — `train_probes_for_condition()`, `compute_static_transfer_aucs()`：新增 `pos_label`/`neg_label` 参数，默认仍为 `"harmful"/"harmless"`（向后兼容）
- `experiments/mibd/audit/margins.py` — `compute_score_margins()`, `condition_margin_table()`：同上
- `experiments/mibd/audit/permutation.py` — `permutation_auc()`, `_nested_permutation()`, `_single_locus_permutation()`：同上
- `experiments/mibd/audit/held_out.py` — `array_held_out_auc_train_selected()` 内部 dict key：从 `"harmful"/"harmless"` 改为用 `pos_label`/`neg_label` 变量
- `experiments/mibd/run_phase1p5_audit.py` — 所有调用处透传 `pos_label`, `neg_label`

---

## 当前 refusal_labels.json 状态

| 模型 | 文件 | 总数 | refusal | compliance | 备注 |
|------|------|------|---------|------------|------|
| InternVL3 | `results/mibd/phase1_probe/internvl3_8b/refusal_labels.json` | 510 | 237 (46%) | 273 | 用新代码生成 ✅ |
| Qwen3-VL | `results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels.json` | 510 | 65 (13%) | 445 | **需确认是否用新代码生成** |

### ⚠️ Qwen refusal 样本极少
Qwen3-VL 的 refusal rate 只有 13%（65/510）。Phase 1.5 audit 加载 678 样本 × 5 conditions，
remap 后约 248 样本，每 condition 约 **6 个 refusal**。探针训练需要 ≥2 个 train pos，
加上 train/test split 后几乎不够，**可能导致所有 condition 的 refusal probe 都无法训练**。

**应对方案**（下个 session 决策）：
1. 只用 V-text condition 跑 Qwen refusal audit（不扩展到其他 condition）
2. 或放宽 `test_frac` 到 0.1
3. 或接受 Qwen refusal 数据不足，跳过此项，直接去 Gemma

---

## 接下来要做的命令

### Step 1：InternVL3 refusal audit（GPU 0）
```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

/opt/conda/bin/conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
  --model internvl3 --gpu 0 \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
  --signal-type refusal \
  --refusal-labels results/mibd/phase1_probe/internvl3_8b/refusal_labels.json \
  --n-permutations 100 \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/internvl3_8b/phase1p5_refusal.log
```

### Step 2：Qwen refusal labels 确认（如果之前进程被中断重跑了）
```bash
# 验证 key 格式
python3 -c "
import json
d = json.load(open('results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels.json'))
print('n=', len(d), 'sample_ids:', list(d.keys())[:3])
from collections import Counter; print(Counter(d.values()))
"
```
若 key 是 16 位 hex（如 `3ec4c0fb605b4e3f`），则是新代码生成，可直接用。

### Step 3：Qwen refusal audit（GPU 1，注意样本少的问题）
```bash
/opt/conda/bin/conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
  --model qwen3vl --gpu 1 \
  --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml \
  --signal-type refusal \
  --refusal-labels results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels.json \
  --n-permutations 100 \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_refusal.log
```
若报 `no V-text probes`（样本不足），说明 Qwen refusal signal 数据确实太少，记录原因后跳过。

### Step 4：Gemma3（最低优先级）
```bash
# Phase 1
/opt/conda/bin/conda run -n rdo python -m experiments.mibd.run_phase1 \
  --model gemma3 --gpu 0 \
  --config experiments/mibd/configs/phase1_probe_gemma.yaml \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/gemma3_4b/phase1.log

# Phase 1.5 harmfulness
/opt/conda/bin/conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
  --model gemma3 --gpu 0 \
  --config experiments/mibd/configs/phase1_probe_gemma.yaml \
  --signal-type harmfulness \
  --n-permutations 100 \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/gemma3_4b/phase1p5_harmfulness.log
```

---

## 关键代码路径

| 文件 | 作用 |
|------|------|
| `experiments/mibd/data/loaders.py` | `_stable_id()` — sha256 hash，**不含** `visual_condition` |
| `experiments/mibd/generate_refusal_labels.py` | 生成 refusal_labels.json，V-text only |
| `experiments/mibd/run_phase1p5_audit.py` | Phase 1.5 主流程 |
| `experiments/mibd/probes/train.py` | probe 训练，`pos_label`/`neg_label` 参数化 |
| `experiments/mibd/audit/permutation.py` | 嵌套 permutation test |
| `experiments/mibd/audit/held_out.py` | held-out AUC + train-only 选 locus |
| `experiments/mibd/audit/margins.py` | score margin 统计 |

## 环境说明

| 模型 | conda env | 激活方式 |
|------|-----------|---------|
| InternVL3-8B | `rdo` at `/opt/conda/envs/rdo` | `/opt/conda/bin/conda run -n rdo python ...` |
| Qwen3-VL-8B | `qwen3-vl` | `/opt/conda/bin/conda run -n qwen3-vl python ...` |
| Gemma3-4B | `rdo` | 同 InternVL3 |

> `conda activate rdo` 在 user shell（miniconda3）中不可用，**必须用 `conda run`**。

## Phase 1.5 判定标准

| 指标 | PASS 条件 |
|------|-----------|
| 嵌套 permutation mean | < 0.55 |
| held-out AUC | > 0.75 |
| train-only held-out AUC | > 0.70 |

---

## 分析文档位置

- `analysis/mibd/2026-06-03-phase1p5-gemma-mechanism-typing.md` — 机制分型表，当前 InternVL3 harmfulness 已填，其余 TBD
- `results/mibd/phase1_probe/internvl3_8b/phase1p5_report.md` — InternVL3 harmfulness 详细报告
- `results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_report.md` — Qwen harmfulness 详细报告
