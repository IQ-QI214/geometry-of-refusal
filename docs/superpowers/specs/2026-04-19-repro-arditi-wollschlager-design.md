# Design: Repro Arditi-Wollschläger on Pure LLMs

**阶段名**: `repro-arditi-wollschlager-2026-04-19`
**创建日期**: 2026-04-19
**前置文档**: `plan/ARA+GRPO-0417.md`
**状态**: Spec — 待 qi 审核

---

## 1. 目标与范围

### 1.1 目标

**主目标 — Pipeline 正确性验证**

证明本 repo 中 `refusal_direction/pipeline/` (Arditi 2024 DIM) 和 `rdo.py` (Wollschläger 2025 RDO/Cone) 的实现在纯 LLM 上能复现原论文的**定性结论**。这是后续 VLM 研究（V1/T0/M1/A1/A3 等 PI 文档中的实验）的方法论基础。

**附带观测 — LLM 上的 SRR 数据**

在主实验产出的数据上顺带计算 `SRR = ASR_kw - ASR_LG3`，把 Qwen2.5-7B / Llama-3.1-8B 两个模型上 SRR 数字列在 FINDINGS 中。**不作为 narrative 分叉点**——因为 VLM 同样经过对齐训练，内生对齐不是 LLM 独有属性，SRR 在 LLM 上存在与否都不直接证明"Stealth Refusal 是 LLM 特有问题"。这部分只是免费的副产品观察，留待未来对照参考。

### 1.2 范围边界

| ✅ 包含 | ❌ 不包含 |
|---|---|
| Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct | VLM 实验 |
| DIM 全流程 + RDO k=1 + Cone k=3,5 | ARA, GRPO |
| Keyword + LlamaGuard3（+ StrongREJECT for RDO/Cone） | activation patching, SAE |
| SaladBench splits（已本地） | AdvBench, HarmBench |
| 定性结论对齐（趋势） | 数值精确复现 |

### 1.3 成功标准

| 方法 | Pipeline 验证通过标准 | SRR 附带观测 |
|------|---|---|
| DIM | ablation 后 `ASR_kw` 和 `ASR_LG3` 都显著 > baseline；两 judge concordance ≥ 85% | 记录 `SRR = ASR_kw - ASR_LG3` |
| RDO k=1 | 训练 loss 收敛；ASR 不低于 DIM | 同上 |
| Cone k=3,5 | 训练收敛；k↑ 不崩 | 观察 k↑ 对 SRR 影响 |

### 1.4 Kill 条件

- **K1**: T6 smoke test 在 Qwen 上连续 3 次修复仍崩溃 → 当前 model_utils 的 Qwen2.5 适配存在深层问题
- **K2**: T7 完成后两模型 ablation 都不降 `ASR_kw`（< baseline+5%）→ pipeline 不工作，需对比 Arditi 原 repo 确认 fork 差异

遇 K1/K2 立即停止并告知 qi，不自行 pivot。

---

## 2. 架构与模块

### 2.1 原则

1. **最小化侵入现有代码** — 只改 3 个已识别 bug，不动其他 pipeline 实现
2. **最大化利用现有代码** — 复用 `run_pipeline.py`, `rdo.py`, `scoring.py`；`p0_cone/common/` 和 `category_a/common/` 的 judge 调用代码可引用
3. **不确定先问** — spec 中所有"待确认"点必须在实现前解决，不静默选默认值

### 2.2 实验目录结构

```
experiments/repro_arditi_wollschlager/
├── README.md                       # 阶段总览 + 如何运行
├── PROGRESS.md                     # 逐任务进度（做了什么 / 得到什么 / 存在哪）
├── HANDOFF.md                      # 接手交接
├── FINDINGS.md                     # T12 最终产出
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── model_paths.py              # 模型 & judge 路径常量
│   ├── eval_judges.py              # Keyword + LG3 + SR 统一接口
│   └── stealth_analysis.py         # SRR 附带计算
├── run_dim.sh                      # 双模型并行（GPU0/GPU1）
├── run_rdo.sh                      # 双模型并行（GPU0/GPU1）
├── run_cone.sh                     # 双模型并行，每模型内部 k=2→5 串行
├── run_evaluate.py                 # 所有 completions → evaluation.json
├── smoke_test.py                   # T6 单模型 32-prompt 烟雾测试
└── logs/                           # 所有 GPU 运行日志
    ├── smoke_test.log
    ├── dim_qwen.log / dim_llama.log
    ├── rdo_qwen.log / rdo_llama.log
    ├── cone_qwen.log / cone_llama.log
    └── eval.log
```

### 2.3 结果目录结构

```
results/repro_arditi_wollschlager/
├── qwen2.5_7b/
│   ├── dim/
│   │   ├── direction.pt
│   │   ├── direction_metadata.json
│   │   ├── generate_directions/mean_diffs.pt
│   │   ├── select_direction/
│   │   ├── completions/
│   │   │   ├── baseline_jailbreakbench.json
│   │   │   ├── ablation_jailbreakbench.json
│   │   │   └── actadd_jailbreakbench.json  (if enabled)
│   │   └── evaluations/
│   ├── rdo_k1/
│   │   ├── rdo_direction.pt
│   │   ├── completions.json
│   │   └── evaluations/
│   └── cone_k{2,3,4,5}/
│       ├── basis.pt
│       ├── completions.json
│       └── evaluations/
├── llama3.1_8b/                    # 同上结构
├── evaluation.json                 # 所有 configs × judges 扁平数字
└── summary.md                      # 人类可读 ASR 表 + SRR 列
```

### 2.4 模块职责

**`common/model_paths.py`** — 单一真相源

```python
MODEL_PATHS = {
    "qwen2.5_7b":  "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct",
    "llama3.1_8b": "/inspire/hdd/global_user/wenming-253108090054/models/Llama-3.1-8B-Instruct",
}
JUDGE_PATHS = {
    "llamaguard3":          "/inspire/hdd/global_user/wenming-253108090054/models/llama-guard-3-8b",
    "strongreject_base":    "/inspire/hdd/global_user/wenming-253108090054/models/gemma-2b",
    "strongreject_adapter": "/inspire/hdd/global_user/wenming-253108090054/models/strongreject-15k-v1",
}
```

**`common/eval_judges.py`** — 评估三件套
- `judge_keyword(response) -> bool`: 全文 substring 匹配（与 Arditi 原文一致，**非** P0 的"首 60 字符"）；使用修订后的 prefix 列表（原 12 条 + 4 条 smart-quote 变体）
- `judge_llamaguard3(responses, prompts) -> List[bool]`: 本地 LG3-8B，返回 unsafe=True / safe=False
- `judge_strongreject(responses, prompts) -> List[float]`: gemma-2b + adapter，返回 [0,1] 分数

**复用已有代码**：实现前先读 `experiments/p0_cone/common/eval_pipeline.py` 和 `experiments/category_a/common/judge_utils.py`，可引用就直接引用，不重写。

**`common/stealth_analysis.py`** — SRR 附带计算
- 输入: keyword + LG3 judge 结果
- 输出: `SRR = ASR_kw - ASR_LG3`, `concordance(kw, LG3)`
- 数字写入 `summary.md` 的"附带观测"章节

**`run_evaluate.py`** — 单一评估入口
读取 `results/repro_arditi_wollschlager/{model}/{method}/completions/*.json` → 跑所有 judge → 写 `evaluation.json` + `summary.md` + 打印 ASCII 表到 stdout。

### 2.5 既有代码修改清单（最小侵入）

| 文件 | 改动 | 行数 | 理由 |
|------|------|---|------|
| `refusal_direction/pipeline/submodules/evaluate_jailbreak.py:11-24` | 追加 4 条 smart-quote 变体（`I'm sorry`, `I can't`, `I'm unable to`, `I'm not able to`，使用 U+2019） | +4 | Arditi 列表补齐 |
| `refusal_direction/pipeline/model_utils/qwen_model.py:79-92` | 重写 `orthogonalize_qwen_weights` 和 `act_add_qwen_weights` 对齐 Qwen2.5 架构（`model.model.embed_tokens`, `layers[i].self_attn.o_proj`, `layers[i].mlp.down_proj`） | ~15 | 现有实现是 Qwen-1 (`model.transformer.h`)，对 Qwen2.5 会 AttributeError |
| `refusal_direction/pipeline/model_utils/llama3_model.py:16, 22` | 删多余开头 `"` | -2 | 模板字符串 bug |

**不修改任何其他 pipeline 实现**。

**Sanity check**（实现前验证，无需改码）:
- `model_factory.py:30` 用 `'llama-3' in path_lower` 匹配 → 路径 `Llama-3.1-8B-Instruct` 小写后命中，**OK**
- `model_factory.py` 用 `'qwen' in path_lower` 匹配纯文本 Qwen → Qwen2.5-7B-Instruct 命中 `QwenModel`，**OK**（但需配合 T1 的架构修复）

---

## 3. 数据流与 4×H100 执行

### 3.1 数据流

```
data/saladbench_splits/{harmful,harmless}_{train,val,test}.json
       │ (harmful_test 取 n_test=128 条)
       ▼
[Phase 1: DIM]  refusal_direction/pipeline/run_pipeline.py
       │ 产出: direction.pt + mean_diffs.pt + metadata.json + baseline/ablation completions
       ▼
[Phase 2: RDO k=1] rdo.py --train_direction
       │ 读: DIM 产出物 (rdo.py:175-189)
       │ 产出: rdo_direction.pt + completions
       ▼
[Phase 3: Cone k=2→5 串行] rdo.py --train_cone --min_cone_dim 2 --max_cone_dim 5
       │ 每个 k 用 k-1 的 lowest_loss_vector 初始化（rdo.py:1092-1102）
       │ 产出: basis_k{2,3,4,5}.pt + completions
       ▼
[Phase 4: Evaluate] run_evaluate.py
       │ Keyword + LG3 (+ SR for RDO/Cone)
       ▼
results/repro_arditi_wollschlager/{evaluation.json,summary.md}
```

### 3.2 GPU 分配（4×H100，每实验单卡）

| Phase | GPU0 | GPU1 | GPU2 | GPU3 | 预估 |
|---|---|---|---|---|---|
| DIM smoke test (T6) | Qwen 32-prompt | — | — | — | ~30 min |
| DIM 全量 (T7) | Qwen | Llama | — | — | 3-5h |
| RDO k=1 (T8) | Qwen | Llama | — | — | 1-2h |
| Cone k=2→5 (T9) | Qwen (串行 k=2→5) | Llama (串行 k=2→5) | — | — | 4-6h |
| Evaluate (T10) | Qwen LG3 | Llama LG3 | Qwen SR | Llama SR | 1-2h |

### 3.3 环境变量

```bash
export HF_HUB_OFFLINE=1              # GPU 环境离线
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline            # 或 WANDB_DISABLED=true
export SAVE_DIR=results/repro_arditi_wollschlager
export DIM_DIR=.                     # 相对 SAVE_DIR
```

### 3.4 关键依赖

- **T8 依赖 T7**: `rdo.py:175-189` 读 DIM 产出物（`direction.pt` + `metadata.json` + `mean_diffs.pt`）
- **Cone 内部 k 串行**: `rdo.py:1092-1102` 每个 k 用 k-1 的 `lowest_loss_vector` 初始化，不能拆卡并行
- **W&B**: `wandb.init(mode="offline")` 或 env `WANDB_MODE=offline`；如调用失败用 `mode="disabled"`

---

## 4. 任务拆解（T0–T12）

### PROGRESS.md 登记规则

**每个任务完成时必须在 PROGRESS.md 登记三件事**：
1. **做了什么**（简述动作）
2. **得到什么结果**（关键数字或产物）
3. **保存在哪**（文件路径 + 日志路径）

示例：
```
### T7 双模型 DIM 全量 — 完成 2026-04-20 14:32
- 做了什么: 并行跑 run_pipeline.py on Qwen/Llama, n_test=128
- 得到什么: Qwen direction.pt (shape=[3584]), Llama direction.pt (shape=[4096])
           baseline ASR_kw: Qwen=__, Llama=__; ablation ASR_kw: Qwen=__, Llama=__
- 保存在: results/repro_arditi_wollschlager/{qwen2.5_7b,llama3.1_8b}/dim/
         日志: experiments/repro_arditi_wollschlager/logs/dim_{qwen,llama}.log
```

### 任务清单

| # | 任务 | 依赖 | 产出 | 预估 | Gate |
|---|------|------|------|------|------|
| T0 | 建目录 + PROGRESS.md + HANDOFF.md + README.md | — | 目录 | 10min CPU | — |
| T1 | 修 `qwen_model.py` Qwen2.5 架构适配 | — | diff | 1-2h CPU | — |
| T2 | 修 `llama3_model.py` 模板引号 | — | diff | 5min CPU | — |
| T3 | 追加 smart-quote prefixes 到 `evaluate_jailbreak.py` | — | diff | 5min CPU | — |
| T4 | 写 `common/{model_paths,eval_judges,stealth_analysis}.py` | — | 代码 | 2-3h CPU | — |
| T5 | 写 `run_*.sh` + `smoke_test.py` + `run_evaluate.py` | T4 | 脚本 | 1-2h CPU | — |
| T6 | **Qwen DIM smoke test**（32 prompts） | T1-T5 | direction.pt + ASR | ~30min GPU | **Gate 1** |
| T7 | **双模型 DIM 全量并行** | T6 通过 | direction × 2 + completions | 3-5h GPU | **Gate 2** |
| T8 | **双模型 RDO k=1 并行** | T7 | rdo_direction × 2 | 1-2h GPU | — |
| T9 | **双模型 Cone k=2→5 并行（内部串行）** | T8 | cone basis × 2 | 4-6h GPU | — |
| T10 | **Evaluate 全量** (Keyword + LG3 + SR) | T7, T8, T9 | evaluation.json | 1-2h GPU | **Gate 3** |
| T11 | 计算 SRR 附带指标 | T10 | 数字列入 summary.md | 10min CPU | — |
| T12 | FINDINGS 报告 | T11 | FINDINGS.md | 30min CPU | — |

### Gate 规则

- **Gate 1 @ T6**: smoke test 通过（`direction.pt` 生成 + ablation ASR > baseline）→ qi 确认 → 启动 T7
- **Gate 2 @ T7**: DIM 全量完成且数字合理 → qi 确认 → 启动 T8/T9
- **Gate 3 @ T10**: Evaluation 完成 → qi 审核 ASR 表和 concordance → 决定 T12 结论走向

### Gate 触发机制

- 所有 GPU 脚本用 `| tee logs/<task>.log` 自动保存日志
- 评估脚本在日志末尾打印 **结构化 summary block**（ASCII 表 + JSON）
- qi 完成任务后无需手动复制数字 — claude 直接读日志文件

### T6 Smoke Test 通过标准

1. `run_pipeline.py` 不崩溃跑完
2. `direction.pt` 文件生成（非空）
3. `ASR_keyword(ablation) > ASR_keyword(baseline)`（任何 delta 都算通过，定性）

---

## 5. 评估输出格式

### 5.1 终端/日志输出

> **以下数字为展示格式用途的示例占位，非预期值。**

```
==================== ASR Summary ====================
Model: qwen2.5_7b
┌──────────┬──────────┬──────────┬──────────┬──────┬─────────┐
│ Config   │ ASR_kw   │ ASR_LG3  │ ASR_SR   │ SRR  │ Concord │
├──────────┼──────────┼──────────┼──────────┼──────┼─────────┤
│ baseline │   5.5%   │   6.3%   │   4.8%   │ -0.8 │  91.4%  │
│ dim      │  85.2%   │  72.6%   │  78.1%   │ 12.6 │  86.8%  │
│ rdo_k1   │  82.0%   │  70.3%   │  75.5%   │ 11.7 │  87.5%  │
│ cone_k3  │  78.4%   │  62.1%   │  67.8%   │ 16.3 │  83.9%  │
│ cone_k5  │  76.8%   │  58.5%   │  63.2%   │ 18.3 │  81.7%  │
└──────────┴──────────┴──────────┴──────────┴──────┴─────────┘
(同样表格给 llama3.1_8b)

==================== SRR (附带观测) ====================
Qwen2.5-7B:  SRR(best)=18.3% @ cone_k5
Llama3.1-8B: SRR(best)=__%  @ ____
(注: VLM 也有内生对齐，此数据不作判定结论)
```

### 5.2 `evaluation.json` 结构

```json
{
  "qwen2.5_7b": {
    "baseline": {"asr_kw": 0.055, "asr_lg3": 0.063, "asr_sr": null, "srr": null, "concord": 0.914, "n": 128},
    "dim":      {"asr_kw": 0.852, "asr_lg3": 0.726, "asr_sr": null, "srr": 0.126, "concord": 0.868, "n": 128},
    "rdo_k1":   {...},
    "cone_k3":  {...},
    "cone_k5":  {...}
  },
  "llama3.1_8b": {...}
}
```

### 5.3 FINDINGS.md 编写规范

FINDINGS 写给 qi 后续阅读和论文讨论用。**不粘贴大段完整结果**（如全量 completions、长 JSON）——用文件路径指向即可，让 qi 自行查阅。但正常的结果总结、分析、洞察都要写全。

**必含章节**：

```markdown
# Findings — Repro Arditi-Wollschläger
日期: 2026-04-XX
完整结果目录: results/repro_arditi_wollschlager/
完整评估 JSON: results/repro_arditi_wollschlager/evaluation.json

## 1. Pipeline Correctness 验证结论
- DIM 在 Qwen2.5-7B 上: baseline/ablation 的 ASR_kw 和 ASR_LG3 对比（列数字 + concordance）
- DIM 在 Llama-3.1-8B 上: 同上
- RDO k=1 / Cone k=3 / Cone k=5: 每个 config 的关键数字
- 判定: Pipeline 是否通过验证 / 哪些部分有问题

## 2. 观察到的现象
- 定性规律（e.g., k↑ 时 ASR 变化方向）
- 两模型对比差异
- 与 Arditi / Wollschläger 原论文的趋势对比

## 3. LLM 上的 SRR 附带观测
- Qwen2.5-7B 最高 SRR 是多少，在哪个 config
- Llama-3.1-8B 同上
- 解读说明：VLM 也有内生对齐，本节只作为数据记录，不引申结论

## 4. 新发现与洞察
- 复现过程中发现的意外现象
- 对 VLM 研究 narrative 的影响（如有）
- 对下一步实验的启示

## 5. Pipeline 局限与已知问题
- 本次复现的边界条件
- 修复过的 bug 清单（指向 T1/T2/T3 的 diff）
- 未覆盖的情况

## 6. 推荐的下一步实验
- 基于本次发现，具体下一个 experiment 的建议

## 附录：完整结果文件导航
- ASR 表: results/repro_arditi_wollschlager/summary.md
- Per-config 评估: results/repro_arditi_wollschlager/{model}/{method}/evaluations/
- 生成样本: results/repro_arditi_wollschlager/{model}/{method}/completions/
- GPU 日志: experiments/repro_arditi_wollschlager/logs/
```

FINDINGS 目标篇幅: 2-4 页（不含附录）。核心数字以表格呈现，完整 raw data 用路径引用。

---

## 6. 风险矩阵

| # | 风险 | 触发 | Fallback |
|---|------|------|---------|
| R1 | Qwen2.5 orthogonalize 修复有 bug | T6 AttributeError / 形状不匹配 | 对照 `llama3_model.py` 重校；不行就回退到 inference-hook 版本（不做 weight ortho），仍能完成验证，注明局限 |
| R2 | Llama-3.1 模板修复引入新问题 | T6 baseline ASR 异常 | 改用 `tokenizer.apply_chat_template` |
| R3 | W&B offline 报错 | T8/T9 wandb.init 失败 | 优先试 `WANDB_MODE=offline`（需本地 wandb 目录可写）；失败则切 `WANDB_DISABLED=true`；再不行 monkey-patch wandb 模块 |
| R4 | Qwen baseline ASR 已>80% | T7 ablation delta 不显著 | 看 ablation delta，SaladBench filter 会筛掉 baseline 不拒绝样本 |
| R5 | Llama-3.1 不是 Arditi 原版本 | T7 数字偏离 | 不算失败——定性趋势对即可 |
| R6 | Cone k=2→5 串行超时 (>6h) | T9 运行时 | 先问 qi，再决定是否跳过 k=4 |
| R7 | 两 LLM 都不存在 SRR | T11 附带观测 | 信息性结果，正常记入 FINDINGS 附带观测章节 |
| R8 | LG3/SR judge 不稳定 | T10 concordance<80% | 人工抽查 20-30 条，识别 judge 失效类型 |
| R9 | 磁盘空间不够 | T9 I/O 失败 | 优先删 Cone k=2/k=4 中间产出 |
| R10 | LLM→VLM 迁移假设 | T12 总结 | 在 FINDINGS 明写局限 |

---

## 7. 总决策清单

| 维度 | 决定 |
|------|------|
| 阶段命名 | `repro-arditi-wollschlager-2026-04-19` |
| 目标 | Pipeline 验证（主）+ SRR 附带观测 |
| 模型 | Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct |
| 数据 | SaladBench splits, n_test=128 (config 默认) |
| 方法 | DIM + RDO k=1 + Cone k=3,5 (实训 k=2→5) |
| 评估 | DIM: Keyword + LG3；RDO/Cone: +StrongREJECT |
| GPU | 每实验单卡，双模型并行，评估阶段 4 卡 |
| 日志 | 所有 GPU 命令 `tee` 到固定路径 |
| PROGRESS 规则 | 每任务三段式: 做了什么 / 得到什么 / 保存在哪 |
| FINDINGS 规则 | 写总结+分析+洞察；不粘贴大段结果，用路径引用 |
| Gate | T6 / T7 / T10 三道 qi 确认关卡 |
| Kill | T6 连续 3 次崩 / T7 两模型都不降 ASR |

---

## 8. 待实现期确认的点（暂无未决问题）

所有设计选项均已与 qi 确认。实现阶段若出现任何新的不确定点，遵循"先问再做"原则。
