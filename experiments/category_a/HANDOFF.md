# Geometry-of-Refusal — 交接文档

> 更新: 2026-04-09 | 阶段: Category A 实验运行中

---

## 1. 项目概览

**研究题目**: VLM Safety Geometry — visual encoder 架构如何决定拒绝机制的空间分布与可攻击性

**当前研究阶段**: Category A 实验（论文 motivation 实验，证明 DSA 现象）

**目标投稿**: AAAI 2026 / ICLR 2027

**项目根目录**: `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/`

---

## 2. 环境信息

| Conda Env | transformers | 用于 |
|---|---|---|
| `rdo` | 4.47.0 | LLaVA-7B, LLaVA-13B, InternVL2-8B 的 A1/A2/A3 生成 |
| `qwen3-vl` | 4.57.3 | Qwen-7B, Qwen-32B 的 A1/A2/A3 生成 + 所有 judge 评估 |

**硬件**: 4×H100 80GB（GPU 节点，无网络）

**模型路径**:
- LLaVA-7B: HF cache (`llava-hf/llava-1.5-7b-hf`)
- LLaVA-13B: HF cache (`llava-hf/llava-1.5-13b-hf`) — 已下载
- Qwen-7B: `/inspire/hdd/.../models/Qwen2.5-VL-7B-Instruct`
- Qwen-32B: `/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-VL-32B-Instruct`
- InternVL2-8B: `/inspire/hdd/.../models/InternVL2-8B`
- Qwen3Guard-8B: `/inspire/hdd/.../models/Qwen3Guard-Gen-8B`

---

## 3. 已实现的代码（全部已提交到 git）

```
experiments/category_a/
├── common/
│   ├── __init__.py
│   ├── data_utils.py          # SaladBench(572) + HarmBench 加载
│   └── judge_utils.py         # Qwen3Guard + LlamaGuard3 评估封装
├── exp_a1_dsa_validation.py   # A1 生成脚本（5配置×N prompts，支持 --resume）
├── exp_a1_judge.py            # A1 Qwen3Guard/LlamaGuard3 评估
├── exp_a2_dsa_causality.py    # A2 Forced generation probe
├── exp_a3_norm_prediction.py  # A3 per-step norm 记录 + AUROC 分析
├── run_a1_gen.sh              # A1 并行启动（自动切换 conda env）
├── run_a1_judge.sh            # A1 judge 评估
├── run_a2.sh                  # A2 并行启动
├── run_a3.sh                  # A3 启动（支持传 GPU id）
└── RUN_GUIDE.md               # 完整运行指南
```

**关键设计**:
- `--n_prompts N`: smoke test 时限制 prompt 数量（0=全量）
- `--resume`: A1 断点续跑，按 config×prompt_idx 粒度恢复
- 所有 `run_*.sh` 用 `conda run --no-capture-output -n <env>` 自动切换环境

---

## 4. 实验状态（截至 2026-04-09）

### Phase 0: Direction 提取 ✅ 完成

| 模型 | NW 层 | 相对深度 | Amplitude Reversal | 类型 |
|---|---|---|---|---|
| LLaVA-7B | 16 | 50% | ✅ | Type I (Bottleneck) |
| **LLaVA-13B** | **20** | **50%** | **✅** | **Type I (新验证)** |
| Qwen-7B | 24 | 86% | ❌ | Type II (Late Gate) |
| **Qwen-32B** | **55** | **86%** | **❌** | **Type II (新验证)** |
| InternVL2-8B | 28 | 88% | ❌ | Type III (Diffuse) |

**重要发现**: LLaVA-13B NW 层=20（50%），与 7B 完全一致 → **LLaMA-2 backbone 决定 crossover 深度，规模无影响**。Qwen-32B 同 Qwen-7B，Type II。

### Phase 1: A1 实验（5模型×5配置×572 prompts）— 部分运行

| 模型 | 状态 | 进度 |
|---|---|---|
| LLaVA-7B | 🔄 运行中 | 4/5 config 完成，ablation_nw_vmm 150/572 |
| LLaVA-13B | ⏳ 未启动 | — |
| Qwen-7B | ⏳ 未启动 | — |
| Qwen-32B | ⏳ 未启动 | — |
| InternVL2-8B | 🔄 运行中 | 3/5 config 完成，baseline_mm 200/572 |

### Smoke Tests ✅ 全部通过

- A1 (LLaVA-7B, n=10): FHCR_kw ablation_nw=90%，与 Phase 3 一致
- A3 (LLaVA-7B, n=10): AUROC max_norm=0.833，mean_norm=1.0，Spike precedes SC=1.0 🎯

---

## 5. 下一步任务队列

### 立即可做（任意顺序）

```bash
# 启动剩余 3 个模型的 A1 (脚本会 --resume 不重跑已有进度)
bash experiments/category_a/run_a1_gen.sh

# A3 在 LLaVA-7B A1 完成后启动（或现在在空闲 GPU 上并行）
bash experiments/category_a/run_a3.sh cuda:1  # 指定空闲 GPU
```

### A1 全部完成后

```bash
# A2 因果验证（需要 A1 的 harmful prefixes）
bash experiments/category_a/run_a2.sh

# Qwen3Guard 评估（qwen3-vl env）
bash experiments/category_a/run_a1_judge.sh
```

---

## 6. 检查实验进度的快速命令

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

# 查看各模型 A1 进度
for m in llava_7b llava_13b qwen2vl_7b qwen2vl_32b internvl2_8b; do
    prog=$(cat results/category_a/$m/a1_progress_saladbench.json 2>/dev/null || echo "{}")
    files=$(ls results/category_a/$m/a1_*_saladbench.json 2>/dev/null | wc -l)
    echo "$m: $files/5 configs done | $prog"
done

# 查看 A1 日志
tail -f results/category_a/llava_7b/a1_gen.log
tail -f results/category_a/gpu2_a1_gen.log   # InternVL2 + Qwen-7B

# A1 完成后查看汇总指标
python -c "
import json, glob
for path in sorted(glob.glob('results/category_a/*/a1_baseline_mm_saladbench.json')):
    with open(path) as f: d = json.load(f)
    m = d['metrics_kw']
    print(f\"{d['model']}: IBR={m['initial_bypass_rate']:.3f} SCR={m['self_correction_rate_overall']:.3f} FHCR={m['full_harmful_completion_rate']:.3f}\")
"
```

---

## 7. 关键文档路径

| 文档 | 路径 |
|---|---|
| 研究周报（完整研究背景） | `plan/weekly_research_report_0401.md` |
| 攻击算法设计（含 Category A-D 完整计划） | `plan/research_plan_attack_algorithm.md` |
| Category A 设计 spec | `docs/superpowers/specs/2026-04-09-category-a-dsa-validation-design.md` |
| Category A 实现计划 | `docs/superpowers/plans/2026-04-09-category-a-dsa-validation.md` |
| 运行指南（完整步骤） | `experiments/category_a/RUN_GUIDE.md` |

---

## 8. Phase 3 已验证结论（背景）

已完成：LLaVA-7B、Qwen-7B、InternVL2-8B、InstructBLIP-7B 的跨模态 direction 分析 + ablation 攻击：
- **三分类框架**: Type I（LLaVA）/ Type II（Qwen）/ Type III（InternVL2）
- **Amplitude Reversal 预测最优攻击策略**
- **LLM Backbone Crossover Hypothesis**: crossover 深度由 LLM backbone 决定，非 ViT
- Phase 3 结果路径: `results/phase3/{llava_7b,qwen2vl_7b,internvl2_8b,instructblip_7b}/`

---

## 9. 给 Claude 的快速上手指令

新会话开始时，告诉 Claude：

> "请读取 `plan/base_instructions-phase1.md`、`plan/weekly_research_report_0401.md` 和 `experiments/category_a/HANDOFF.md`，这是我的研究项目。"

然后可以直接问：
- "检查 A1 实验进度"
- "A1 完成了，帮我启动 A2"
- "分析 A1 结果，写分析报告"
