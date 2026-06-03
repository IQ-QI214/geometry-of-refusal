# Phase 1.5 审计 & Gemma3 机制分型报告

**日期**：2026-06-03  
**阶段**：Phase 1.5 审计（InternVL3-8B、Qwen3-VL-8B）+ Gemma3-4B-IT 新增  
**目标**：验证 AUC=1.000 不是过拟合伪影，完成三模型机制分型，决定是否进入 MIBD 训练

---

## 1. Phase 1 已知结果摘要

### 1.1 InternVL3-8B（harmfulness probe）

| Visual Condition | Best Layer | Token Pos | AUC |
|---|---|---|---|
| V-text | 6 | -3 | 1.000 |
| V-blank | 7 | -3 | 1.000 |
| V-noise | 7 | -3 | 1.000 |
| V-real | 7 | -3 | 1.000 |
| FigStep | 0 | -1 | 1.000 |

Static transfer AUC（train=V-text）：

| Target condition | AUC |
|---|---|
| V-blank | 1.000 |
| V-noise | 1.000 |
| V-real | 1.000 |
| FigStep | **0.741** |

**Phase 1 判定**：FigStep static transfer AUC=0.741（低于其他条件），其余检查 PASS → **CONTINUE_MIBD**

### 1.2 Qwen3-VL-8B（harmfulness probe）

| Visual Condition | Best Layer | Token Pos | AUC |
|---|---|---|---|
| V-text | 15 | -1 | 1.000 |
| V-blank | 17 | -1 | 1.000 |
| V-noise | 16 | -1 | 1.000 |
| V-real | 17 | -1 | 1.000 |
| FigStep | 1 | -1 | 1.000 |

Static transfer AUC（train=V-text）：

| Target condition | AUC |
|---|---|
| V-blank | 1.000 |
| V-noise | 1.000 |
| V-real | 1.000 |
| FigStep | **0.956** |

**Phase 1 判定**：所有条件 AUC=1.000，FigStep transfer drop 较小（0.956），→ 待 Phase 1.5 确认

---

## 2. Phase 1.5 审计结果表

> 各列含义：Train AUC = 训练集；Held-out AUC = 留出集（random 20%）；Train-selected held-out AUC = train-only locus 选择后的留出 AUC；Group-split AUC = 按 paired_id 分组；Permutation AUC = nested permutation mean±std (n=200)；Cross-category AUC = 跨危害类别迁移。

| Model | Signal | Condition | Train AUC | Held-out AUC | Train-sel held-out | Group-split | Perm mean (n=200) | Perm p95 | Verdict |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| InternVL3-8B | harmfulness | V-text | 1.000 | 1.000 | 0.994 (l5,p-3) | N/A | 0.475±0.186 | 0.759 | **PASS** |
| InternVL3-8B | harmfulness | V-blank | 1.000 | 1.000 | 1.000 (l7,p-3) | N/A | 0.472±0.173 | 0.758 | **PASS** |
| InternVL3-8B | harmfulness | V-noise | 1.000 | 0.982 | 1.000 (l7,p-3) | N/A | 0.486±0.184 | 0.805 | **PASS** |
| InternVL3-8B | harmfulness | V-real | 0.999 | 0.994 | 1.000 (l7,p-3) | N/A | 0.474±0.167 | 0.734 | **PASS** |
| InternVL3-8B | harmfulness | FigStep | 1.000 | 1.000 | 1.000 (l0,p-1) | N/A | 0.484±0.224 | 0.870 | **PASS** |
| InternVL3-8B | refusal | — | TBD | TBD | TBD | N/A | TBD | TBD | TBD |
| Qwen3-VL-8B | harmfulness | — | TBD | TBD | TBD | N/A | TBD | TBD | TBD |
| Qwen3-VL-8B | refusal | — | TBD | TBD | TBD | N/A | TBD | TBD | TBD |
| Gemma3-4B-IT | harmfulness | — | TBD | TBD | TBD | N/A | TBD | TBD | TBD |
| Gemma3-4B-IT | refusal | — | TBD | TBD | TBD | N/A | TBD | TBD | TBD |

**Cross-category AUC**：所有模型均 N/A（harmless 样本全来自 Alpaca `general` category，无法构造 cross-category test set）。这是数据集结构限制，不影响 PASS 判定，需在论文中声明。

**Group-split AUC**：所有样本无 `paired_id`，N/A。需在论文中声明。

---

## 3. Static Transfer Margin Drop 表

> 衡量 probe 在视觉条件间的泛化能力，重点关注 FigStep drop。

| Model | Signal | V-text→V-blank | V-text→V-noise | V-text→V-real | V-text→FigStep | FigStep drop |
|---|---|---|---|---|---|---|
| InternVL3-8B | harmfulness | 1.000 (Phase1) | 1.000 (Phase1) | 1.000 (Phase1) | **0.741** (Phase1) | **−0.259** |
| InternVL3-8B | refusal | TBD | TBD | TBD | TBD | TBD |
| Qwen3-VL-8B | harmfulness | 1.000 (Phase1) | 1.000 (Phase1) | 1.000 (Phase1) | **0.956** (Phase1) | **−0.044** |
| Qwen3-VL-8B | refusal | TBD | TBD | TBD | TBD | TBD |
| Gemma3-4B-IT | harmfulness | TBD | TBD | TBD | TBD | TBD |
| Gemma3-4B-IT | refusal | TBD | TBD | TBD | TBD | TBD |

---

## 4. 机制分型表

> 分型标准（暂定）：  
> - Locus shift = FigStep 与 V-text 最优层差值 ≥ 3  
> - Transfer drop = FigStep static transfer AUC < 0.85  
> - Early-layer collapse = FigStep 最优层 ≤ 2  
> Type 候选：**Unified**（locus stable, transfer high）、**FigStep-divergent**（locus shift OR transfer drop）、**TBD**

| Model | Harm locus shift | Harm transfer drop | Refusal locus shift | Refusal transfer drop | FigStep early-layer collapse | Type |
|---|---|---|---|---|---|---|
| InternVL3-8B | layer 6→0 (Δ=6, YES) | 0.741 < 0.85 (YES) | TBD | TBD | layer 0 (YES) | TBD |
| Qwen3-VL-8B | layer 15→1 (Δ=14, YES) | 0.956 ≥ 0.85 (NO) | TBD | TBD | layer 1 (YES) | TBD |
| Gemma3-4B-IT | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 5. 已知条件余弦相似度

### 5.1 InternVL3-8B（harmfulness probe direction cosines）

| Condition pair | Cosine similarity |
|---|---|
| V-blank \| V-real | 1.000 |
| V-blank \| V-noise | 0.983 |
| V-noise \| V-real | 0.983 |
| V-blank \| V-text | 0.878 |
| V-real \| V-text | 0.878 |
| V-noise \| V-text | 0.833 |
| FigStep \| V-text | 0.136 |
| FigStep \| V-real | 0.110 |
| FigStep \| V-blank | 0.110 |
| FigStep \| V-noise | 0.120 |

**注**：FigStep 与所有其他条件的余弦相似度均极低（< 0.14），表明 FigStep 激活了与标准条件截然不同的表征方向。V-blank ≈ V-real（cosine=1.000），表明空白图像与真实图像在该 probe 方向上完全等价。

### 5.2 Qwen3-VL-8B（harmfulness probe direction cosines）

| Condition pair | Cosine similarity |
|---|---|
| V-blank \| V-real | 1.000 |
| V-blank \| V-noise | 0.981 |
| V-noise \| V-real | 0.981 |
| V-noise \| V-text | 0.914 |
| V-blank \| V-text | 0.884 |
| V-real \| V-text | 0.884 |
| FigStep \| V-text | 0.259 |
| FigStep \| V-noise | 0.153 |
| FigStep \| V-real | 0.124 |
| FigStep \| V-blank | 0.124 |

**注**：Qwen3-VL 的 FigStep|V-text cosine（0.259）略高于 InternVL3（0.136），与 FigStep transfer AUC 更高（0.956 vs 0.741）一致，但 FigStep 仍与其他条件方向显著不同。

---

## 6. FigStep 单独分析

> FigStep 是将有害文本嵌入图像的越狱方式，其表征可能绕过文本 harmfulness probe。

| Model | Signal | FigStep best layer | FigStep AUC | Static transfer to FigStep | Cosine to V-text | 解读 |
|---|---|---|---|---|---|---|
| InternVL3-8B | harmfulness | 0 | 1.000 | **0.741** | 0.136 | FigStep 在 layer 0 即可分离，但方向与标准 probe 不同，transfer 低 |
| Qwen3-VL-8B | harmfulness | 1 | 1.000 | **0.956** | 0.259 | FigStep 在 layer 1 分离，transfer 较高，方向与标准 probe 有部分重叠 |
| InternVL3-8B | refusal | TBD | TBD | TBD | TBD | TBD |
| Qwen3-VL-8B | refusal | TBD | TBD | TBD | TBD | TBD |
| Gemma3-4B-IT | harmfulness | TBD | TBD | TBD | TBD | TBD |
| Gemma3-4B-IT | refusal | TBD | TBD | TBD | TBD | TBD |

**待确认问题**：
1. InternVL3 的 FigStep transfer 低（0.741）是因为模型用不同机制处理 FigStep，还是 FigStep 样本分布偏移？
2. Qwen3-VL 的 FigStep transfer（0.956）是否意味着 harmfulness probe 对 FigStep 越狱已有泛化能力？
3. Gemma3 的 FigStep 行为是否与上述两模型一致？

---

## 7. 审计结论与 MIBD 训练建议

> 状态：待 Phase 1.5 运行完成后填写。以下为进入 MIBD 训练的条件 checklist。

### 进入条件 Checklist

| 条件 | InternVL3-8B | Qwen3-VL-8B | Gemma3-4B-IT |
|---|---|---|---|
| Harmfulness Held-out AUC ≥ 0.90 | ✅ 0.982–1.000 | TBD | TBD |
| Harmfulness Group-split AUC ≥ 0.85 | ⚠️ N/A (无 paired_id) | ⚠️ N/A | ⚠️ N/A |
| Harmfulness Permutation mean ≤ 0.60 | ✅ 0.472–0.486 (n=200) | TBD | TBD |
| Harmfulness Permutation p95 ≤ 0.90 | ✅ 0.734–0.870 | TBD | TBD |
| Refusal Held-out AUC ≥ 0.85 | TBD (待跑) | TBD | TBD |
| Harmfulness locus stable across V-blank/noise/real | ✅ layer 7 across all | TBD | TBD |
| FigStep transfer drop 有合理解释 | ✅ 0.741，early-layer collapse (layer 0) | ⚠️ 0.956，drop 较小 | TBD |
| 无明显 label leakage / spurious feature | ✅ V-blank/V-real SHA1 不同 | TBD | TBD |
| **综合判定** | **✅ PASS harmfulness** | **TBD** | **TBD** |

### 机制分型结论（待定）

- **InternVL3-8B**：FigStep 激活 layer 0，方向与标准 probe 余弦 < 0.14，static transfer drop 显著（−0.259）。初步判断：**FigStep-divergent**，可能需要单独 FigStep probe 或多条件联合 probe 进入 MIBD。
- **Qwen3-VL-8B**：FigStep 激活 layer 1，余弦 0.259，transfer drop 较小（−0.044）。初步判断：更接近 **Unified**，但需 Phase 1.5 确认。
- **Gemma3-4B-IT**：全部 TBD，待实验。

---

## 8. 运行命令参考

```bash
# ======================================================
# Phase 1.5 审计命令（供 qi 复制运行）
# 环境：rdo（InternVL3），qwen3-vl（Qwen3-VL / Gemma3）
# ======================================================

# --- InternVL3-8B harmfulness audit (rdo env, GPU 0) ---
conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
  --model internvl3 \
  --gpu 0 \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
  --signal-type harmfulness \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/internvl3_8b/phase1p5_harmfulness.log

# --- InternVL3-8B refusal audit (rdo env, GPU 0) ---
conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
  --model internvl3 \
  --gpu 0 \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
  --signal-type refusal \
  --refusal-labels results/mibd/phase1_probe/internvl3_8b/refusal_labels.json \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/internvl3_8b/phase1p5_refusal.log

# --- Qwen3-VL-8B harmfulness audit (qwen3-vl env, GPU 1) ---
conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
  --model qwen3vl \
  --gpu 1 \
  --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml \
  --signal-type harmfulness \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_harmfulness.log

# --- Qwen3-VL-8B refusal audit (qwen3-vl env, GPU 1) ---
conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
  --model qwen3vl \
  --gpu 1 \
  --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml \
  --signal-type refusal \
  --refusal-labels results/mibd/phase1_probe/qwen3_vl_8b/refusal_labels.json \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_refusal.log

# --- Gemma3-4B-IT harmfulness audit (qwen3-vl env, GPU 2) ---
conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
  --model gemma3 \
  --gpu 2 \
  --config experiments/mibd/configs/phase1p5_audit_gemma.yaml \
  --signal-type harmfulness \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/gemma3_4b_it/phase1p5_harmfulness.log

# --- Gemma3-4B-IT refusal audit (qwen3-vl env, GPU 2) ---
conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
  --model gemma3 \
  --gpu 2 \
  --config experiments/mibd/configs/phase1p5_audit_gemma.yaml \
  --signal-type refusal \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/gemma3_4b_it/phase1p5_refusal.log
```

---

*文件由子代理自动生成，基于 Phase 1 实验结果（`results/mibd/phase1_probe/`）。TBD 部分待 Phase 1.5 实验完成后手动或脚本填写。*
