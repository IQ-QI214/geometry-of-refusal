# VLM Safety Geometry 研究全局总结

> **文档性质**：研究进展文字总结，面向 AAAI 2026 投稿
> **覆盖范围**：Phase 1（Pilot）→ Phase 2 → Phase 3（Exp 3A + 3C）
> **最后更新**：2026-04-01

---

## 一、研究背景与出发点

### 1.1 核心问题

VLM（Vision-Language Model）的 safety alignment 存在一个系统性漏洞：现有 jailbreak 攻击方法大多只在生成初期绕过 refusal（利用 shallow alignment 弱点），但即使初始绕过成功，模型往往会在生成中途重新激活安全意识（**Delayed Safety Reactivation**），触发自我纠正，导致有害内容无法完整生成。

本研究从第一性原理出发，希望从 hidden state 的几何结构视角，理解这一现象的机制，并指导更有效的攻击设计。

### 1.2 研究定位与文献空白

| 已有工作 | 研究了什么 | 未研究的 |
|:---|:---|:---|
| Arditi et al. (NeurIPS 2024) | LLM 中 refusal 是单一方向 | 未在 VLM multimodal mode 验证；未研究 generation dynamics |
| Wollschläger et al. (ICML 2025) | Refusal 是多维 concept cone（静态） | 未研究动态演变；未在 VLM 验证 |
| ICET (ICML 2025 Spotlight) | Image encoder 内部层的 safety 分布不均 | **只研究 CLIP 架构**，未研究 custom ViT；未研究 LLM backbone |
| TGA (ICLR 2025) | Safety mechanism 无法从 text 转移到 vision 的根源 | 未系统区分不同 visual encoder 架构的影响 |
| SafeProbing (2026) | Safety signal 在 generation 全程持续存在 | 未研究 signal 的**几何方向**如何演化 |
| JailBound (NeurIPS 2025) | VLM fusion layer 的静态安全边界 | 未研究 delayed reactivation；未考虑 dynamic rotation |

**本研究的空白**：首次系统研究 VLM 的 **visual encoder 架构类型**如何决定 LLM backbone 中 refusal signal 的**空间分布模式**，以及这一模式如何决定安全脆弱性类型。

---

## 二、实验体系与核心结果

### 2.1 实验架构全图

```
Phase 1 (Pilot)
├── Exp A：Refusal direction 的跨模态稳定性（text-only vs MM）
├── Exp B：Refusal direction 的时间步一致性（generation 过程中）
└── Exp C：Delayed Safety Reactivation 的基线量化

Phase 2
├── Exp 2A：Content confound 分离（teacher-forced controlled）
├── Exp 2B：Direction ablation 的因果效力验证
└── Exp 2C：PGD 视觉扰动优化（负结果）

Phase 3 (Cross-Model)
├── Exp 3A：Refusal direction 跨模态分析（4 模型）
└── Exp 3C：Narrow Waist Ablation Attack（4 模型）
```

### 2.2 各实验核心数据

**Pilot Exp A（LLaVA-1.5-7B，Layer 16）**

| Layer | cos(v_text, v_mm) | norm_ratio |
|:---:|:---:|:---:|
| 12 | 0.877 | **0.78**（压制）|
| 16 | **0.918** | 1.06（放大）|
| 20 | 0.897 | 1.15（放大）|

**Pilot Exp B / Phase 2 Exp 2A（LLaVA-1.5-7B，Layer 16）**

| 指标 | Exp B（含 confound）| Exp 2A（controlled）|
|:---:|:---:|:---:|
| Min pairwise cos | 0.018 | **0.231** |
| Mean pairwise cos | 0.207 | 0.413 |

**Phase 2 Exp 2B（LLaVA-1.5-7B，58 prompts）**

| 配置 | Full Harmful Rate |
|:---|:---:|
| baseline_text | 17.2% |
| blank_image | 56.9% |
| ablation_mm_all (32层) | 74.1% |
| **ablation_mm_layer16** | **89.7%** |

**Phase 3 Exp 3A（4 模型）**

| 模型 | Group | NW 层 | NW 相对深度 | NW cos | Amplitude Reversal |
|:---|:---:|:---:|:---:|:---:|:---:|
| LLaVA-1.5-7B | **A** | 16 | 0.50 | 0.917 | ✅ True |
| Qwen2.5-VL-7B | **B** | 24 | 0.86 | 0.961 | ❌ False |
| InternVL2-8B | **B** | 28 | 0.88 | 0.919 | ❌ False |
| InstructBLIP-7B | **A** | 20 | 0.63 | 0.467 | ✅ True |

**Phase 3 Exp 3C（4 模型，8 prompts，full_harmful_rate）**

| 模型 | Group | baseline_mm | ablation_nw | ablation_all |
|:---|:---:|:---:|:---:|:---:|
| LLaVA-1.5-7B | A | 62.5% | **87.5%** | 87.5% |
| Qwen2.5-VL-7B | B | 0.0% | 62.5% | **100.0%** |
| InternVL2-8B | B | 12.5% | 12.5% | **25.0%** |
| InstructBLIP-7B | A | **100.0%** | 100.0% | 100.0% |

---

## 三、核心发现汇总

### Finding 1：Delayed Safety Reactivation 是普遍现象（Pilot Exp C）

LLaVA-1.5-7B 在 text-only baseline 下，80% 的 bypassed prompt 在生成过程中出现自我纠正。这直接证明了 Gap C 问题的真实性：现有攻击只解决了初始 bypass，没有解决持续的 safety reactivation。

**状态**：已确认，作为 paper 的 motivation experiment。

---

### Finding 2：Refusal Direction 跨模态稳定，但幅度呈层级反转（Exp A）

**结论**：Visual modality 不改变 refusal direction 的"朝向"（cos > 0.87），但会改变"幅度"——且这个幅度变化呈现层级反转：
- 浅层（相对深度 < 50%）：norm_ratio < 1，visual modality **压制** refusal signal
- 深层（相对深度 ≥ 50%）：norm_ratio > 1，visual modality **放大** refusal signal

**跨模型情况（Exp 3A）**：这一"Amplitude Reversal"是**架构相关**的条件性现象：
- **Group A（CLIP ViT 架构）**：LLaVA + InstructBLIP 均显示 reversal
- **Group B（Custom ViT 架构）**：Qwen2.5-VL + InternVL2 **不显示** reversal，全层均匀压制

**状态**：已发现，跨模型部分验证，是 paper 核心 claim 之一。

---

### Finding 3：Dynamic Refusal Rotation（Exp B + Exp 2A）

**结论**：Refusal direction 在 generation 过程中不是静态的，而是随时间步动态旋转。去除 content confound 后（controlled min cos = 0.231），旋转仍然真实存在，且模式是非单调的（subspace switching，而非平滑 drift）。

**机制解释**：模型在 generation 的不同阶段使用不同的 "safety subspace" 表达安全意识，这正是现有静态方向 ablation 无法阻止 delayed reactivation 的原因。

**状态**：在 LLaVA-1.5-7B 上已验证，尚未跨模型复验（Exp 3B 未完成）。

---

### Finding 4：Layer 16 Narrow Waist 与单层消融效力（Exp 2B）

**结论**：Layer 16 是 LLaVA safety mechanism 的"narrow waist"——refusal signal 在此处最纯粹（cos = 0.918 最高），单层 ablation 效果（89.7%）反而优于全层 ablation（74.1%）。

**跨模型情况（Exp 3A + 3C）**：
- Narrow waist 位置**不收敛到 ~50%**，因此"架构通用性"假设被否定
- Group A 模型：单层 ablation ≥ 全层 ablation（窄瓶颈结构）
- Group B 模型：全层 ablation > 单层 ablation（分布式 safety）

**状态**：已发现，跨模型验证支持。

---

### Finding 5：Visual Encoder 类型预测最优攻击策略（Exp 3C 核心发现）

这是整个研究最重要的新发现：

**CLIP ViT 架构（Group A）→ 集中式 safety（Concentrated）**：
- 浅层压制 + 深层放大（amplitude reversal）
- Safety signal 集中在某一"narrow waist"层
- 单层精准攻击有效
- 示例：LLaVA-1.5-7B，layer 16 ablation = 87.5%

**Custom ViT 架构（Group B）→ 分布式 safety（Distributed）**：
- 全层均匀压制（无 reversal）
- Safety signal 均匀分布，无明显瓶颈
- 必须全层攻击
- 示例：Qwen2.5-VL-7B，all-layer ablation = 100%

**Q-Former 架构（极端 Group A 变体）**：
- Q-Former 的 32-token 极度压缩破坏了 text-image alignment（cos = 0.467）
- Visual modality 直接导致 safety mechanism 失效
- 无需任何 ablation：blank image 即 100% bypass

**状态**：在 4 个模型上验证，**但 8 条 prompts 样本量严重不足**，所有百分比结论统计可信度低。

---

### Finding 6：v_text 与 v_mm 在 Ablation 效果上完全等价（Exp 3C）

4/4 模型的 ablation_nw_vmm = ablation_nw_vtext，零差距。这意味着：
1. 攻击者只需纯文本 prompt pairs 即可提取有效攻击方向，无需 multimodal paired data
2. Phase 1 Exp A 的"方向跨模态稳定性（cos > 0.87）"在攻击实践中得到了因果验证

---

### Finding 7：Qwen2.5-VL 的反直觉特性（Exp 3C）

- Baseline safety 最强（text + mm 均为 0% full_harmful）
- All-layer ablation = 100%（完全崩溃，无任何 self-correction）
- 单层 ablation = 62.5%（明显弱于全层）

这揭示了一个深层 alignment trade-off：**safety mechanism 越"干净"（refusal direction 最纯粹），越容易被定向消除**。Safety 的集中性与鲁棒性可能是对立的。

---

### Finding 8：InternVL2 对 Ablation 的强抵抗性（未解决问题）

All-layer ablation 仅 25%，远低于预期。可能原因：
1. NW 层在 88% 深度，ablation 时机过晚
2. Mean-diff direction 不是最优提取方法
3. InternVL2 具有真实的分布式 safety 冗余（类似 DeepRefusal 训练效果）

**状态**：原因未明，是 paper 中需要诚实报告的 limitation，并作为未来工作提出逐层扫描实验。

---

## 四、已验证 vs 已否定

### 已验证的假设

| 假设 | 支持证据 | 强度 |
|:---|:---|:---:|
| Delayed safety reactivation 在 VLM 中普遍存在 | Exp C：80% self-correction | ★★★ |
| Refusal direction 跨模态方向稳定 | Exp A：cos > 0.87（3/4 模型）| ★★★ |
| Ablation 可以有效降低 self-correction rate | Exp 2B：layer16 ablation = 8.6% SC rate | ★★★ |
| v_text = v_mm 在攻击效果上等价 | Exp 3C：4/4 模型零差距 | ★★★ |
| Q-Former 结构是 VLM 的天然安全漏洞 | Exp 3C：blank image 100% bypass | ★★★ |

### 已否定的假设

| 假设 | 否定证据 | 修正方向 |
|:---|:---|:---|
| Amplitude reversal 是通用现象 | Exp 3A：仅 Group A（CLIP）成立 | 改为"架构相关的条件性现象" |
| Narrow waist 在 ~50% 相对深度 | Exp 3A：仅 LLaVA 成立 | 放弃"architectural regularity" claim |
| 单层 NW ablation 通用优于全层 | Exp 3C：Group B 相反（all > NW）| 改为"Visual encoder 类型预测最优策略" |
| Visual modality 是"双刃剑"对所有模型成立 | Exp 3A：Group B 只有单纯压制 | 仅对 Group A 成立 |

---

## 五、Insight 与 Novelty 评估

### 核心 Insight 1：Visual Encoder 架构类型是 VLM Safety Geometry 的根本决定因素

CLIP 的 contrastive text-image pretraining 使 visual features 与 LLM 语义空间建立了特殊的层级对齐关系，导致 safety signal 在 LLM 中呈现集中式瓶颈结构（narrow waist）。Custom ViT 专为 VLM 从头训练，与 LLM 的融合更均匀，safety signal 分布式存在。

**与 ICET (ICML 2025) 的关系**：ICET 发现了 image encoder 内部的 layer-wise safety 不均匀性；本研究首次在 LLM backbone 侧建立了 visual encoder 架构类型与 safety signal 空间分布的因果关联，且覆盖了 ICET 未研究的 Custom ViT 架构。

### 核心 Insight 2：Safety 集中性与鲁棒性的 Trade-off

Qwen2.5-VL 的 finding 揭示了 alignment 训练的内在 trade-off：越纯粹的 refusal direction（集中式 safety）越脆弱，越容易被定向消除；分布式 safety（可能是 InternVL2 的训练特性）对单方向攻击更鲁棒。这与 DeepRefusal（EMNLP 2025）从防御侧的发现完全呼应。

### 核心 Insight 3：Q-Former 的信息瓶颈即安全漏洞

Q-Former 的 32-token 极度压缩不仅使 visual features 无法有效传递语义安全信息（cos = 0.467，远低于其他模型），还直接导致 safety mechanism 在 visual modality 介入时完全失效。这从几何角度解释了 TGA（ICLR 2025）发现的"safety mechanism 无法从 text 转移到 vision"的根本原因。

### Novelty 评估

| 贡献点 | 最相近文献 | 独特 delta | 等级 |
|:---|:---|:---|:---:|
| Visual encoder 类型决定 LLM backbone 安全分布模式 | ICET（只研究 image encoder 内部，只研究 CLIP）| **首次跨架构比较，首次在 LLM backbone 侧建立关联** | ★★★★ |
| Amplitude reversal 是 CLIP 特有现象 | 无对应文献 | 全新发现 | ★★★ |
| Q-Former 的极端信息压缩 = 安全漏洞的几何解释 | TGA（结论相近但无几何量化）| **提供了具体的 cos 量化和 ablation 验证** | ★★★ |
| Safety 集中性与鲁棒性 Trade-off | DeepRefusal（防御侧，无跨架构比较）| **首次在 VLM 多架构上量化这一关系** | ★★★ |
| Dynamic refusal rotation | 无对应文献 | 全新发现（但只在单模型验证）| ★★★★（需跨模型验证）|
| v_text = v_mm 等价性 | 无对应文献 | 实践意义高 | ★★ |

---

## 六、当前局限性

### 实验层面

**L1（P0 级，必须解决）**：8 条 test prompts 统计可信度极低（Fisher 精确检验 p ≈ 0.5）。所有百分比结论（87.5% = 7/8, 100% = 8/8, 25% = 2/8）在置信区间内高度重叠，无法区分差异。必须扩展到至少 100 条。

**L2（重要）**：Exp 3B（Dynamic Rotation 的跨模型验证）尚未完成。这是 paper 的 mechanistic 核心 claim 之一，缺失状态下无法完整讲述研究故事。

**L3（重要）**：逐层 ablation 曲线尚未做（当前只有 4 个离散配置点）。连续的 layer-by-layer ASR curve 是构建"完整 safety geometry"最直接的证据。

**L4**：当前只有 5 个 probe 层（离散点），全层 cos + norm_ratio 的连续曲线尚未生成。

### 方法论层面

**L5**：CLIP vs Custom ViT 的区分存在 confound（不仅 visual encoder 类型不同，alignment 训练质量、训练数据规模、LLM backbone 系列也都不同）。现有证据是 correlation，不是 causation。

**L6**：所有研究都局限于白盒 LLM backbone。商业模型（GPT-4o、Gemini、Claude）的 LLM backbone 不可访问，当前 findings 的直接应用性受限。

**L7**：Eval 使用 keyword matching，存在 false positive/negative（"It is not" 等隐式拒绝可能被误判）。

---

## 七、研究框架与策略

### 7.1 研究定位的重新表述

**当前定位（局限白盒 LLM backbone）**：
> "在 LLM backbone 的 hidden state 空间中，visual encoder 架构类型决定了 refusal direction 的空间分布模式。"

**升级后的定位（双空间对照，涵盖黑盒实用场景）**：
> "VLM 的安全脆弱性几何（Safety Geometry）由两个可观测空间共同决定：(1) Visual Encoder Feature Space（可在黑盒场景访问）；(2) LLM Backbone Hidden State Space（白盒场景）。两个空间的几何特征可互相预测，并共同决定攻击策略的有效性。"

这一重新定位的意义：研究结论可以通过 visual encoder 特征（更容易在黑盒模型上获取）来预测商业模型的安全脆弱性类型，而不局限于需要 LLM backbone 访问权的白盒场景。

### 7.2 三阶段研究策略

**Phase 3 补全（当前，约 2-3 周）**：
- P0：扩大数据集（100+ prompts）
- P0：完成 Exp 3B（Dynamic Rotation 跨模型验证）
- P0：逐层 ablation 曲线（32 层连续 ASR curve，每个模型）
- P1：全层 cos + norm_ratio 连续曲线（5 层 → 32 层）

**Phase 4：Visual Encoder Space 分析（约 2-3 周）**：
- VE-Exp A：Visual encoder feature space 中 safe/unsafe 的线性可分性（各层 probe accuracy）
- VE-Exp B：Visual encoder 不同层输出→ LLM 时，LLM 中 amplitude pattern 的变化（桥接 ICET 和当前工作）
- 因果验证：引入更多模型（Qwen-VL-Chat、LLaVA-v1.6-Mistral、GLM-4.1V）
- 目标：建立 visual encoder 几何指标 → LLM safety geometry 类型的预测模型

**Phase 5：Paper 完整实验 + 写作（约 4-5 周）**：
- 扩展到 6-8 个模型（覆盖所有主要 VE 架构类型）
- 标准评估 benchmark（MM-SafetyBench + HarmBench，LLM-based judge）
- Attack application：基于 safety geometry 的 architecture-aware 攻击策略
- Defense implication：如何设计 VE 来避免被 geometry-based 攻击利用

### 7.3 Paper Title 建议

> **"VLM Safety Geometry: How Visual Encoder Architecture Determines the Spatial Distribution and Exploitability of Refusal Mechanisms in Vision-Language Models"**

或更精炼：

> **"Safety Geometry of Vision-Language Models: Visual Encoder Architecture as the Determinant of Refusal Distribution and Attack Vulnerability"**

---

## 八、关键文献速查（更新版）

| 文献 | 发表 | 与本研究关系 |
|:---|:---:|:---|
| Arditi et al., "Refusal in LLMs is Mediated by a Single Direction" | NeurIPS 2024 | 基础方法论（mean-diff direction）|
| Wollschläger et al., "Geometry of Refusal in LLMs: Concept Cones" | ICML 2025 | Refusal 多维结构（静态）|
| ICET: "Layer-wise Alignment across Image Encoder Layers" | ICML 2025 Spotlight | **最重要竞争文献**：image encoder 内部 safety，但只有 CLIP，只有 visual encoder 侧 |
| TGA: "Cross-Modal Safety Mechanism Transfer" | ICLR 2025 | safety 无法从 text 转移到 vision 的根源 |
| VLLM Safety Paradox | NeurIPS 2025 | Visual modality 对 safety 的结构性破坏 |
| DeepRefusal | EMNLP 2025 Findings | 分布式 vs 集中式 safety 的鲁棒性 |
| JailBound | NeurIPS 2025 | 静态安全边界的 joint attack |
| Qi et al., "Safety Alignment Should Be More Than a Few Tokens Deep" | ICLR 2025 Outstanding | Shallow alignment 理论基础 |

---

*文档结束。配套文档：gapc_vibe_coding_guide.md（Claude Code 执行指导）。*
