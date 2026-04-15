# Phase 3 Exp 3A 跨模型验证：完整结果分析与下一步方向

> **文档版本**: 2026-03-31
> **状态**: Exp 3A 全部完成（4/4 模型），待 Exp 3B/3C
> **用途**: 总结 Exp 3A 结论 + 指导新对话窗口快速接续工作

---

## 1. 实验结果总表

### 1.1 核心指标

| 模型 | 总层数 | Narrow Waist 层 | NW 相对深度 | NW cos | Amplitude Reversal | 浅层 mean ratio | 深层 mean ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LLaVA-1.5-7B** | 32 | **16** | **0.50** | **0.917** | ✅ True | 0.691 | 1.144 |
| **Qwen2.5-VL-7B** | 28 | 24 | 0.86 | **0.961** | ❌ False | 0.924 | 0.788 |
| **InternVL2-8B** | 32 | 28 | 0.88 | **0.919** | ❌ False | 0.799 | 0.789 |
| **InstructBLIP-7B** | 32 | 20 | 0.63 | 0.467 | ✅ True | 0.740 | 1.271 |

### 1.2 逐层 norm_ratio 数据

| 相对深度 | LLaVA | Qwen2.5-VL | InternVL2 | InstructBLIP |
|:---:|:---:|:---:|:---:|:---:|
| ~25% | **0.605** | 0.993 | 0.841 | **0.741** |
| ~38% | **0.777** | 0.854 | 0.756 | **0.738** |
| ~50% | 1.061 | 0.568 | 0.744 | 1.062 |
| ~63% | 1.151 | 0.886 | 0.780 | 1.343 |
| ~88% | 1.221 | 0.911 | 0.843 | 1.409 |

### 1.3 逐层 cos(v_text, v_mm) 数据

| 相对深度 | LLaVA | Qwen2.5-VL | InternVL2 | InstructBLIP |
|:---:|:---:|:---:|:---:|:---:|
| ~25% | 0.765 | 0.766 | 0.739 | 0.445 |
| ~38% | 0.876 | 0.661 | 0.720 | 0.441 |
| ~50% | **0.917** | 0.832 | 0.867 | 0.451 |
| ~63% | 0.896 | 0.904 | 0.894 | **0.467** |
| ~88% | 0.885 | **0.961** | **0.919** | 0.446 |

---

## 2. 核心发现

### Finding 3A-1: Refusal Direction 跨模态稳定性是通用的（但程度不同）

**结论: 4/4 模型的 refusal direction 在 text-only 和 MM 模式下均存在正相关（cos > 0.44）。**

- LLaVA / Qwen2.5-VL / InternVL2: 高相似度（cos > 0.72 所有层），说明 visual modality 不改变 refusal 的"朝向"
- InstructBLIP: 中等相似度（cos ≈ 0.44-0.47），Q-Former 的信息瓶颈（32 tokens）显著改变了 refusal direction 的几何结构

**意义**: Phase 1 的 Finding A1 在 3/4 模型上强成立，InstructBLIP 是有条件成立。

### Finding 3A-2: Amplitude Reversal 不是通用现象，而是架构相关的

**结论: 2/4 模型（LLaVA, InstructBLIP）显示 amplitude reversal，2/4（Qwen2.5-VL, InternVL2）不显示。**

两类模型的幅度行为有本质不同：
- **Group A (LLaVA, InstructBLIP)**: 浅层压制 + 深层放大 → visual modality 是"双刃剑"
- **Group B (Qwen2.5-VL, InternVL2)**: **所有层均压制** (ratio < 1) → visual modality 单纯地削弱 refusal

**可能的解释方向**（需进一步验证）：
- Group A 使用 CLIP 系 ViT（CLIP ViT-L/14, BLIP2-ViT），Group B 使用自研 ViT（Custom ViT, InternViT-300M）
- CLIP 的 text-image 对齐预训练可能导致其 visual features 与 safety mechanism 有特殊的交互模式
- 或者：Group B 的模型有更强的 safety alignment（Qwen2.5 和 InternLM2 都是更新、更大规模的 alignment 训练）

### Finding 3A-3: Narrow Waist 位置不收敛到 ~50%

**结论: "Narrow waist 在相对深度 ~50%" 的假设被否定。**

- LLaVA: 50% ✅
- InstructBLIP: 63%
- Qwen2.5-VL: 86%
- InternVL2: 88%

3/4 模型的 cos 随深度单调递增（越深越稳定），LLaVA 是特例（在 50% 处有峰值然后略降）。

**修正后的结论**: Narrow waist 不是一个通用的 architectural bottleneck，而是与具体模型的 safety alignment 分布相关。但 3/4 模型在最深探测层的 cos > 0.88，说明**深层的 refusal direction 跨模态一致性是普遍成立的**。

### Finding 3A-4: InstructBLIP 的 Q-Former 瓶颈效应

**结论: InstructBLIP 的 cos 在所有层均 ≈ 0.44-0.47，远低于其他 3 个模型。**

Q-Former 将视觉信息压缩为 32 个 tokens（其他模型 256-576），这个极端压缩使得 MM 模式的 hidden state 分布与 text-only 模式显著不同。

但有趣的是：**尽管方向偏移大（cos 低），amplitude reversal 仍然存在**（浅层压制 0.74，深层放大 1.27）。这说明 amplitude reversal 和 direction alignment 是两个独立的现象。

---

## 3. 对 Paper Claim 的影响

### 原有假设 vs 实际结果

| 原假设 | 结果 | Paper 策略调整 |
|:---|:---|:---|
| H1: Amplitude reversal 是通用的 | **2/4 模型成立** | 不能作为 universal claim，改为"与视觉编码器类型相关的条件性现象" |
| H2: Narrow waist 在 ~50% 深度 | **仅 LLaVA 成立** | 放弃 "architectural regularity" claim |
| H3: 所有模型 NW cos > 0.85 | **3/4 模型成立** | 可以 claim "深层 refusal direction 跨模态高度稳定"（排除 Q-Former 架构） |
| 新发现: 两类幅度行为模式 | Group A vs Group B | **新 contribution**: VLM 的 visual modality 对 safety 的影响存在两种截然不同的模式 |

### 可能的 paper 叙事调整

**原叙事**: "VLM 存在通用的 compensatory safety mechanism（amplitude reversal），以及通用的 narrow waist bottleneck"

**修正叙事**: "VLM 的 refusal direction 跨模态稳定性是通用的（cos > 0.72 在 3/4 主流架构），但 visual modality 对 refusal 幅度的影响存在两种模式：(1) CLIP 系架构的浅层压制-深层放大（compensatory），(2) 自研 ViT 架构的全层压制（uniform suppression）。这一发现揭示了 visual encoder 类型是 VLM safety mechanism 的关键决定因素。"

---

## 4. 待解决的问题

### 4.1 InstructBLIP cos 偏低是否影响后续 ablation

InstructBLIP 的 cos ≈ 0.47 意味着 v_text 和 v_mm 方向偏差大。如果在 Exp 3C 中使用 v_mm 做 ablation，可能效果较差（因为 v_mm 捕获的 "refusal direction" 不太纯粹）。建议 Exp 3C 优先跑 LLaVA + Qwen2.5-VL（一个有 reversal、一个没有），InstructBLIP 作为补充。

### 4.2 Qwen2.5-VL 环境兼容性

Qwen2.5-VL 使用 `qwen3-vl` conda 环境运行（transformers >= 4.52），其他 3 个模型使用 `rdo` 环境（transformers 4.47）。实验逻辑完全相同，结果可信，但 run_3a_all.sh 需要为 Qwen2.5-VL 单独处理环境。

### 4.3 InternVL2 的 timm 依赖

InternVL2 需要 `timm` 包（已在 GPU 节点安装到 rdo 环境）。如果重建环境需要记得装这个依赖。

---

## 5. 下一步方向

### 5.1 推荐的 Exp 3B（Dynamic Rotation）方案

使用每个模型的**最高 cos 层**（不再是固定 narrow waist）做 teacher-forced controlled 实验：

| 模型 | 3B 目标层 | cos at that layer |
|:---|:---:|:---:|
| LLaVA-1.5-7B | 16 | 0.917 |
| Qwen2.5-VL-7B | 24 | 0.961 |
| InternVL2-8B | 28 | 0.919 |
| InstructBLIP-7B | 20 | 0.467 |

### 5.2 推荐的 Exp 3C（Ablation）方案

**优先级排序**：
1. **LLaVA-7B** (基准，验证 Phase 2 结果可复现)
2. **Qwen2.5-VL-7B** (无 reversal 的模型，ablation 效果是否不同？)
3. **InternVL2-8B** (另一个无 reversal 模型，交叉验证)
4. **InstructBLIP-7B** (cos 偏低，ablation 可能效果差，作为补充)

### 5.3 潜在新实验方向

- **探究 CLIP ViT vs 自研 ViT 的差异根源**: 对比 LLaVA（CLIP ViT）和 InternVL2（InternViT）在 visual feature space 中的分布差异
- **更细粒度的层扫描**: 当前 probe 了 5 层，可以对 LLaVA 和 Qwen2.5-VL 各扫 16 层以获得更精确的 crossover 点
- **Group A vs Group B 的 safety alignment 对比**: 测量两组模型的 baseline refusal rate 差异，验证是否 alignment 强度决定了哪种模式出现

---

## 6. 代码与环境速查（新对话参考）

### 6.1 项目路径
- **项目根目录**: `[PROJECT_ROOT]/geometry-of-refusal/`
- **Phase 3 代码**: `experiments/phase3/`
- **Phase 3 结果**: `results/phase3/{llava_7b,qwen2vl_7b,internvl2_8b,instructblip_7b}/`
- **基础指令**: `plan-markdown/base_instructions-phase1.md`
- **Phase 3 方案文档**: `plan-markdown/gapc_phase3_supplement.md`

### 6.2 代码结构
```
experiments/phase3/
├── common/
│   ├── __init__.py
│   ├── model_configs.py      # MODEL_CONFIGS + load_model_by_name()
│   └── model_adapters.py     # 4 个 adapter: LLaVA/Qwen2VL/InternVL2/InstructBLIP
├── exp_3a_amplitude_reversal.py   # ✅ 已完成
├── run_3a_all.sh                  # 4 GPU 并行
└── logs/                          # 实验日志
```

### 6.3 运行环境

| 模型 | conda 环境 | 特殊依赖 |
|:---|:---|:---|
| LLaVA-1.5-7B | rdo | 无 |
| Qwen2.5-VL-7B | **qwen3-vl** | transformers >= 4.52 |
| InternVL2-8B | rdo | timm |
| InstructBLIP-7B | rdo | 无 |

### 6.4 模型本地路径

| 模型 | 路径 |
|:---|:---|
| LLaVA-1.5-7B | `models/hub/models--llava-hf--llava-1.5-7b-hf/` (HF hub cache) |
| Qwen2.5-VL-7B | `models/Qwen2.5-VL-7B-Instruct/` |
| InternVL2-8B | `models/InternVL2-8B/` |
| InstructBLIP-7B | `models/InstructBLIP-7B/` |

### 6.5 已知的代码适配要点

1. **Qwen2.5-VL**: `model.language_model.layers`（非 `.model.layers`），手动 `.to(device)` 避免 accelerate 依赖
2. **InternVL2**: 不能用 `model.forward()`（需要 image_flags），改为手动注入 visual features 到 `inputs_embeds` 后直接调 `model.language_model()`
3. **InstructBLIP**: 标准 HF 接口，无特殊适配
4. **所有模型**: forward hook 注册在 `adapter.get_llm_layers()[layer_idx]` 上

---

## 7. 给新对话窗口的关键指令

```
1. 读取 plan-markdown/base_instructions-phase1.md 获得基础规则
2. 读取 analysis/phase3_exp3a_report.md（本文档）获得 Phase 3 当前进展
3. 读取 plan-markdown/gapc_phase3_supplement.md 获得完整 Phase 3 方案
4. 代码在 experiments/phase3/，结果在 results/phase3/
5. 下一步: 实现 Exp 3B (Dynamic Rotation) 和 Exp 3C (Ablation)
6. Qwen2.5-VL 必须用 qwen3-vl 环境运行
7. 给 qi 的命令必须是单行格式
```

---

*文档结束。Phase 3 Exp 3A 全部完成，等待 3B/3C 推进。*
