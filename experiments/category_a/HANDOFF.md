# Geometry-of-Refusal — 交接文档 v2

> 更新: 2026-04-10 | 阶段: Category A A1 完成 + Judge 完成（4/5 模型）

---

## 1. 项目概览

**研究题目**: VLM Safety Geometry — visual encoder 架构如何决定拒绝机制的空间分布与可攻击性

**核心假说**: VLM 隐空间存在 refusal direction，消融该方向可攻破安全对齐；架构类型决定消融有效性

**当前研究阶段**: Category A 实验（论文 motivation 实验，证明 DSA 现象）

**目标投稿**: AAAI 2026 / ICLR 2027

**项目根目录**: `[PROJECT_ROOT]/geometry-of-refusal/`

**给新窗口 Claude 的快速启动**:
> "请读取 `experiments/category_a/HANDOFF.md` 和 `analysis/category_a/a1_full_analysis_2026-04-10.md`。前者是整体交接，后者是今天最新的分析结论。请你了解研究背景后，继续协助推进 Category A 实验。"

---

## 2. 环境信息

| Conda Env | transformers | 用于 |
|---|---|---|
| `rdo` | 4.47.0 | LLaVA-7B, LLaVA-13B, InternVL2-8B 的 A1/A2/A3 生成 |
| `qwen3-vl` | 4.57.3 | Qwen-7B, Qwen-32B 的 A1/A2/A3 生成 + 所有 judge 评估 |

**硬件**: 4×H100 80GB（GPU 节点，无网络，必须设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`）

**模型路径**:
- LLaVA-7B: HF cache (`llava-hf/llava-1.5-7b-hf`)
- LLaVA-13B: `/inspire/hdd/global_user/wenming-253108090054/models/llava-1.5-13b-hf`
- Qwen-7B: `/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct`
- Qwen-32B: `/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-VL-32B-Instruct`
- InternVL2-8B: `/inspire/hdd/global_user/wenming-253108090054/models/InternVL2-8B`
- Qwen3Guard-8B: `/inspire/hdd/global_user/wenming-253108090054/models/Qwen3Guard-Gen-8B`

---

## 3. 三分类框架（Phase 3 已验证，Category A 确认）

| 模型 | Type | NW 层 | 相对深度 | Amplitude Reversal | 最优攻击 | FHCR_judge (最优) |
|------|------|-------|---------|-------------------|---------|-----------------|
| LLaVA-7B | I (Bottleneck) | 16 | 50% | ✅ | nw_vmm | **74.5%** |
| LLaVA-13B | I (Bottleneck) | 20 | 50% | ✅ | nw_vmm | **75.5%** |
| Qwen-7B | II (Late Gate) | 24 | 86% | ❌ | all_vmm | 13.3% |
| Qwen-32B | II (Late Gate) | 55 | 86% | ❌ | all_vmm | 8.2% |
| InternVL2-8B | III (Diffuse) | 28 | 88% | ❌ | 无有效策略 | ~6% |

---

## 4. Category A 实验状态（截至 2026-04-10）

### A1 实验状态 ✅ 全部完成

| 模型 | 生成 | Judge |
|------|------|-------|
| LLaVA-7B | ✅ | ✅ |
| LLaVA-13B | ✅ | ✅ |
| Qwen-7B | ✅ | ✅ |
| Qwen-32B | ✅ | ✅ |
| InternVL2-8B | ✅ | ⏳ 未完成 |

**结果路径**: `results/category_a/{model}/`
- 生成结果: `a1_{config}_saladbench.json`
- Judge 结果: `a1_judged_qwen3guard_saladbench.json`（4 模型已有）

### A2 实验 ⏳ 未启动

因果验证（forced generation probe），需要 A1 完成后从中提取 harmful prefixes。

### A3 实验 ⏳ 仅有 LLaVA-7B smoke test (n=10)

需要全量运行（572 prompts × 5 models）。设计需根据今日发现更新（见下节）。

---

## 5. 今日重要发现（2026-04-10）

### 5.1 指标修正

**Bug**: `eval_utils.py` 中 `SCR_given_bypass` 计算错误，分子包含所有检测到 SC 关键词的响应（含初始拒绝），导致数值严重虚高（最高 33×）。

**修正**: 现在只统计 `initial_bypass=True AND self_correction_found=True`，值恒 ≤ 1.0。

修正文件: `experiments/phase2/common/eval_utils.py`

### 5.2 隐性拒绝（Stealth Refusal）现象

Qwen 模型在全层消融后，**不再使用标准拒绝语言，但仍生成伦理反驳内容**：
- "Certainly not! Discrimination is morally wrong..."
- Keyword 判为 `bypass=True, harmful=True`（假阳性）
- Judge 判为 `Safe, Refusal=Yes`（正确）

**Qwen-7B all_vmm 响应分布**: 3.5% 显式拒绝 | 83.2% 隐性拒绝 | 13.3% 真正有害

### 5.3 两层安全架构假说

- **Layer 1**（几何层）: 显式拒绝语言，由 refusal direction 编码，消融可破坏
- **Layer 2**（推理层）: 独立的伦理推理能力，仅 Type II (Qwen) 具有，不被消融

### 5.4 真实攻击效果对比

| Model | FHCR_judge (最优攻击) | 安全评估 |
|-------|----------------------|---------|
| LLaVA-7B | **74.5%** | 危险 ⚠️ |
| LLaVA-13B | **75.5%** | 危险 ⚠️ |
| Qwen-7B | 13.3% | 相对安全 |
| Qwen-32B | 8.2% | 安全 ✅ |

---

## 6. 代码变更记录（2026-04-10）

| 文件 | 变更 |
|------|------|
| `experiments/phase2/common/eval_utils.py` | 修正 SCR_given_bypass；新增 DSA_gap_abs/rel；改进 REFUSAL_STARTERS |
| `experiments/category_a/common/judge_utils.py` | 修复 `device_map` → `.to(device)`（去掉 accelerate 依赖）；`torch_dtype` → `dtype` |
| `experiments/category_a/exp_a1_judge.py` | 新增 `scr_bypass_judge`、`dsa_gap_rel` 输出字段 |
| `experiments/category_a/run_a1_judge.sh` | 改为 4 GPU 并行执行 |

---

## 7. 立即可执行任务队列

### Priority 1: InternVL2 Judge（GPU 节点，qwen3-vl env）

```bash
cd [PROJECT_ROOT]/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/category_a/exp_a1_judge.py \
    --model internvl2_8b --judge qwen3guard --dataset saladbench --device cuda:0 \
    > results/category_a/internvl2_8b/a1_judge.log 2>&1 &
tail -f results/category_a/internvl2_8b/a1_judge.log
```

### Priority 2: A3 全量运行（5 模型）

```bash
# LLaVA-7B (rdo env)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n rdo \
    python experiments/category_a/exp_a3_norm_prediction.py \
    --model llava_7b --device cuda:0 \
    > results/category_a/llava_7b/a3.log 2>&1 &

# LLaVA-13B (rdo env)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n rdo \
    python experiments/category_a/exp_a3_norm_prediction.py \
    --model llava_13b --device cuda:1 \
    > results/category_a/llava_13b/a3.log 2>&1 &

# Qwen-7B (qwen3-vl env)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/category_a/exp_a3_norm_prediction.py \
    --model qwen2vl_7b --device cuda:2 \
    > results/category_a/qwen2vl_7b/a3.log 2>&1 &

# Qwen-32B (qwen3-vl env)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/category_a/exp_a3_norm_prediction.py \
    --model qwen2vl_32b --device cuda:3 \
    > results/category_a/qwen2vl_32b/a3.log 2>&1 &

# InternVL2-8B (rdo env, 需在 judge 完成后或另一 GPU)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n rdo \
    python experiments/category_a/exp_a3_norm_prediction.py \
    --model internvl2_8b --device cuda:0 \
    > results/category_a/internvl2_8b/a3.log 2>&1 &
```

### Priority 3: A3 设计更新讨论（需与 qi 讨论后实施）

基于 Stealth Refusal 发现，A3 的新方向：
- A3 当前：预测 bypass vs refuse（实际上是预测 SC 关键词，混入了拒绝响应）
- A3 修订：在 baseline_mm 下区分三种响应类型的 norm 轨迹
  1. Explicit refusal (bypass=False)
  2. Genuine harmful completion (bypass=True, judge=Unsafe)
  3. Stealth refusal (bypass=True, judge=Safe)

**预期新 finding**: Stealth refusal 的 norm 轨迹是否与 genuine harmful 可区分？

### Priority 4: A2 因果验证（A3 后）

```bash
bash experiments/category_a/run_a2.sh
```

---

## 8. 关键文档路径

| 文档 | 路径 |
|------|------|
| 本交接文档 | `experiments/category_a/HANDOFF.md` |
| A1 完整分析报告 (今日) | `analysis/category_a/a1_full_analysis_2026-04-10.md` |
| 指标修正报告 | `analysis/category_a/a1_corrected_metrics.md` |
| Category A 设计 spec | `docs/specs/2026-04-09-category-a-dsa-validation-design.md` |
| 运行指南 | `experiments/category_a/RUN_GUIDE.md` |
| 研究周报 | `plan/weekly_research_report_0401.md` |
| 攻击算法全规划 | `plan/research_plan_attack_algorithm.md` |

---

## 9. 文件目录规范

```
geometry-of-refusal/
├── experiments/              # 仅放实验代码（.py, .sh）
│   ├── category_a/          # Category A 实验
│   │   ├── common/          # judge_utils, data_utils
│   │   ├── exp_a*.py        # 实验脚本
│   │   └── run_*.sh         # 启动脚本
│   ├── phase2/              # Phase 2 实验代码
│   ├── phase3/              # Phase 3 实验代码（含 common/model_adapters.py）
│   └── pilot/               # Pilot 实验
│
├── analysis/                 # 仅放分析报告（.md）
│   ├── category_a/          # Category A 分析（含今日报告）
│   │   ├── a1_full_analysis_2026-04-10.md  ← 今日新增
│   │   └── a1_corrected_metrics.md
│   └── phase3/              # Phase 3 分析报告
│       ├── phase3_exp3a_report.md
│       └── ...
│
├── scripts/                  # 一次性分析脚本（不属于实验流程）
│   └── recompute_a1_metrics.py   ← 从 analysis/ 迁移至此
│
├── docs/                     # 设计文档
│   ├── specs/               # 实验设计 spec（从 docs/superpowers/specs/ 迁移）
│   └── plans/               # 实现计划（从 docs/superpowers/plans/ 迁移）
│
├── plan/                     # 研究规划文档（保持现状）
│   ├── base_instructions-phase1.md
│   ├── weekly_research_report_0401.md
│   └── research_plan_attack_algorithm.md
│
├── data/                     # 数据集
│   └── saladbench_splits/
│
└── results/                  # 实验结果（.json, .pt）不在 git 跟踪
    ├── category_a/{model}/
    └── phase3/{model}/
```

**迁移建议**（当前可先不动，后续整理时遵照）:
- `analysis/recompute_a1_metrics.py` → `scripts/recompute_a1_metrics.py`
- `analysis/phase3_*.md` → `analysis/phase3/`
- `docs/superpowers/specs/` → `docs/specs/`
- `docs/superpowers/plans/` → `docs/plans/`
- 项目根目录的 `*.py`（plots.py, rdo.py 等）→ `scripts/`
