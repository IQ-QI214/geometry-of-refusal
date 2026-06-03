# MIBD Phase 1.5 Audit 交接文档

> **时间**：2026-06-04
> **承接**：2026-06-03-mibd-phase1p5-gemma-mechanism-typing.md 执行会话
> **状态**：InternVL3 harmfulness Phase 1.5 审计通过，待跑 refusal / Qwen / Gemma

---

## 1. 本会话完成的工作

### 1.1 新增文件

| 文件 | 说明 |
|---|---|
| `experiments/mibd/audit/__init__.py` | 审计模块入口 |
| `experiments/mibd/audit/splits.py` | held-out / group / cross-category split 函数 |
| `experiments/mibd/audit/permutation.py` | Nested permutation test（每次重新选 locus） |
| `experiments/mibd/audit/held_out.py` | `array_held_out_auc` + `array_held_out_auc_train_selected`（train-only locus selection） |
| `experiments/mibd/audit/margins.py` | score margin statistics |
| `experiments/mibd/eval/phase1p5_report.py` | `AuditResult` dataclass + Markdown 报告生成 |
| `experiments/mibd/run_phase1p5_audit.py` | Phase 1.5 GPU 入口脚本 |
| `experiments/mibd/test_audit.py` | 16 个 CPU-only 测试，全部通过 |
| `experiments/mibd/configs/phase1_probe_qwen3vl.yaml` | Qwen3-VL Phase 1 配置（修复命名） |
| `experiments/mibd/configs/phase1_probe_gemma.yaml` | Gemma3-4B Phase 1 配置 |
| `experiments/mibd/configs/phase1p5_audit_gemma.yaml` | Gemma3-4B Phase 1.5 审计配置 |
| `analysis/mibd/2026-06-03-phase1p5-gemma-mechanism-typing.md` | 机制分型报告（含真实 Phase 1 数据，TBD 待填） |

### 1.2 修改文件

| 文件 | 修改内容 |
|---|---|
| `models/loader.py` | 新增 `load_gemma3()`（`AutoModelForImageTextToText`） |
| `models/adapters.py` | 新增 `Gemma3Adapter`（34层，forward hook，qwen3-vl env） |
| `data/loaders.py` | `load_harmbench_phase1` 新增 `mmsafety_dir` 参数，V-real 条件分配真实图像 |
| `run_phase1.py` | 新增 `--signal-type`、`--refusal-labels`、`--log-file`、`gemma3` 支持 |
| `run_phase1p5_audit.py` | 新增 `--n-permutations`（默认 100）、`--log-file`；修复全部 split 逻辑 |

---

## 2. 修复的 Bug 记录

### Bug 1：V-blank = V-real（数据流问题）
- **原因**：`load_harmbench_phase1` 所有样本 `image_path=None`，`build_image_for_condition` 对 V-real 也返回 `blank_image()`
- **修复**：`load_harmbench_phase1` 新增 `mmsafety_dir` 参数，V-real 从 `images_figstep` 随机分配真实图像
- **验证**：`[identity]` 诊断输出 V-blank/V-real SHA1 不同

### Bug 2：Permutation AUC 虚高（~0.77）
- **原因 1**（严重）：方向用 shuffled labels 训练，AUC 也对同一 shuffled labels 计算，等同于训练集 AUC
- **原因 2**（次要）：复用 full-data oracle locus，存在 selection leakage
- **修复**：
  - 升级为 **nested permutation**：每次置换在 shuffled train 上重新扫所有 locus，用真实 test labels 评估
  - `permutation_auc()` 现返回 `{mean, std, min, max, p95, n_valid}` dict
- **修复后**：V-blank/V-noise permutation mean 从 0.61 降到 ~0.49

### Bug 3：non-V-text held-out AUC 全为 N/A
- **原因**：主循环对非 V-text 条件直接写 -1.0 哨兵
- **修复**：新增 `_array_held_out_auc`，从已提取的 hidden state 数组直接计算所有条件的 held-out AUC，不需要额外 GPU

### Bug 4：Selection leakage（held-out AUC 偏高）
- **原因**：`best_layer/best_pos` 在全量数据上选择，再在同一 locus 报 held-out AUC
- **修复**：新增 `array_held_out_auc_train_selected`，只用 train hidden states 选 best locus，在 test 上评估；报告同时显示 full-data locus 和 train-only locus 对比

### Bug 5：-1.0 哨兵被误报为"低 AUC"
- **修复**：`_condition_verdict` 跳过哨兵值，报告全部改为 N/A 显示

### Bug 6：`held_out.py` key 体系混用（refusal signal 下出错）
- **原因**：`train_probe_map` 用硬编码 `"harmful"/"harmless"`，`test_hidden_map` 用动态 `pos_label/neg_label`
- **修复**：内部统一用 `"harmful"/"harmless"` key，外部参数只用于从原始 hidden map 取数据

---

## 3. 当前 InternVL3 harmfulness Phase 1.5 结果

| Condition | Train AUC | Held-out AUC | Nested Perm mean | Verdict |
|---|:---:|:---:|:---:|:---:|
| V-text | ~1.0 | ~1.0 | ~0.49 | PASS |
| V-blank | ~1.0 | ~1.0 | ~0.49 | PASS |
| V-noise | ~1.0 | ~1.0 | ~0.52 | PASS |
| V-real | ~1.0 | ~1.0 | ~0.49 | PASS |
| FigStep | ~1.0 | ~1.0 | ~0.50 | PASS |

**结论**：InternVL harmfulness 在当前协议下通过 Phase 1.5 审计。

**剩余限制**（不影响通过判定，但需在论文中声明）：
- group split N/A（无 `paired_id`）
- cross-category AUC 全 N/A（harmless 全来自 Alpaca `general` category）
- nested permutation p95 较宽（FigStep 0.91，V-noise 0.82）——建议跑 n=200/500 多 seed 稳定性验证

---

## 4. 下一步运行命令

所有命令在项目根目录 `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/` 执行。

### 4.1 稳定性验证（可选，建议先跑）

```bash
# InternVL3 harmfulness，n=200 稳定性验证
conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
  --model internvl3 --gpu 0 \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
  --signal-type harmfulness \
  --n-permutations 200 \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/internvl3_8b/phase1p5_harmfulness_n200.log
```

### 4.2 InternVL3 refusal audit

```bash
# 先需要生成 refusal_labels.json（模型推理输出），然后：
conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
  --model internvl3 --gpu 0 \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
  --signal-type refusal \
  --refusal-labels results/mibd/phase1_probe/internvl3_8b/refusal_labels.json \
  --n-permutations 100 \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/internvl3_8b/phase1p5_refusal.log
```

### 4.3 Qwen3-VL harmfulness audit

```bash
conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
  --model qwen3vl --gpu 1 \
  --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml \
  --signal-type harmfulness \
  --n-permutations 100 \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/qwen3_vl_8b/phase1p5_harmfulness.log
```

### 4.4 Gemma3-4B harmfulness audit

```bash
conda run -n qwen3-vl python -m experiments.mibd.run_phase1p5_audit \
  --model gemma3 --gpu 2 \
  --config experiments/mibd/configs/phase1p5_audit_gemma.yaml \
  --signal-type harmfulness \
  --n-permutations 100 \
  --data-dir data/saladbench_splits \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --log-file results/mibd/phase1_probe/gemma3_4b_it/phase1p5_harmfulness.log
```

---

## 5. 运行结果判读标准

```
PASS：
  - held-out AUC >= 0.90
  - nested permutation mean <= 0.60
  - V-blank/V-real identity 诊断显示 SHA1 不同
  - 报告无 -1.0000 字符串

WARN：
  - nested permutation mean 在 (0.60, 0.65]
  - no paired_id group split（可接受，需声明）
  - cross-category 全 N/A（已知数据集结构限制）

INVALID：
  - nested permutation mean > 0.65
  - train-only held-out AUC 明显低于 held-out AUC（selection leakage 未修）
  - V-blank/V-real hidden hash 仍相同
```

---

## 6. 代码模块总览

```
experiments/mibd/
├── configs/
│   ├── phase1_probe_internvl3.yaml    # InternVL3 Phase 1
│   ├── phase1_probe_qwen3vl.yaml      # Qwen3-VL Phase 1
│   ├── phase1_probe_gemma.yaml        # Gemma3-4B Phase 1
│   └── phase1p5_audit_gemma.yaml      # Gemma3-4B Phase 1.5
├── audit/
│   ├── splits.py                      # held-out / group / cross-category split
│   ├── permutation.py                 # nested permutation test
│   ├── held_out.py                    # train-only locus selection held-out AUC
│   └── margins.py                     # score margin statistics
├── data/
│   ├── loaders.py                     # 数据加载（V-real 已支持真实图像）
│   └── schema.py                      # MIBDSample dataclass
├── eval/
│   ├── phase1_report.py               # Go/No-Go 报告
│   └── phase1p5_report.py             # Phase 1.5 审计报告（AuditResult）
├── models/
│   ├── adapters.py                    # Qwen3VLAdapter / InternVL3Adapter / Gemma3Adapter
│   └── loader.py                      # load_qwen3vl / load_internvl3 / load_gemma3
├── run_phase1.py                      # Phase 1 入口（支持 --signal-type / --log-file）
├── run_phase1p5_audit.py              # Phase 1.5 入口（支持 --n-permutations / --log-file）
└── test_audit.py                      # 16 个 CPU-only 测试（全通过）
```

---

## 7. 待完成事项（下一个会话）

- [ ] 用 n=200/500 多 seed 重跑 InternVL3 harmfulness，确认 p95 稳定
- [ ] 生成 InternVL3 refusal labels（需要模型推理）并跑 refusal audit
- [ ] 跑 Qwen3-VL harmfulness + refusal audit
- [ ] 跑 Gemma3-4B harmfulness + refusal audit
- [ ] 填写 `analysis/mibd/2026-06-03-phase1p5-gemma-mechanism-typing.md` 的 TBD 占位符
- [ ] 更新机制分型表（Type A/B/C/D 判定）
- [ ] 决定是否进入 Phase 3 MIBD 训练
