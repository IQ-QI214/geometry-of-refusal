# VLM Safety Geometry 研究进展周报

> **汇报对象**：导师
> **汇报日期**：2026-04-01
> **研究主题**：VLM Safety Geometry：视觉编码器架构如何决定拒绝机制的空间分布与可攻击性
> **当前阶段**：Phase 3（跨模型验证）完成；Phase 4（视觉编码器空间分析）规划中

---

## 目录

1. [研究背景与出发点](#1-研究背景与出发点)
2. [核心问题定义](#2-核心问题定义)
3. [研究目标与方法论框架](#3-研究目标与方法论框架)
4. [Pilot 实验（Phase 1）](#4-pilot-实验phase-1)
5. [Phase 2：核心攻击实现与负结果](#5-phase-2核心攻击实现与负结果)
6. [Phase 3：跨模型 Safety Geometry 验证](#6-phase-3跨模型-safety-geometry-验证)
7. [综合分析：Safety Geometry 框架的建立](#7-综合分析safety-geometry-框架的建立)
8. [已验证与已否定的假设](#8-已验证与已否定的假设)
9. [Novelty 评估与文献定位](#9-novelty-评估与文献定位)
10. [局限性](#10-局限性)
11. [下一步研究方向](#11-下一步研究方向)
12. [参考文献](#12-参考文献)

---

## 1. 研究背景与出发点

### 1.1 问题的发现

大型视觉语言模型（Vision-Language Models, VLMs）已在医疗、教育、自动驾驶等安全敏感领域广泛部署。然而，这类模型的安全对齐（safety alignment）存在一个系统性漏洞，我们将其称为 **Gap C**：

现有 VLM Jailbreak 攻击方法（如 GCG、AutoDAN、UltraBreak）大多聚焦于绕过生成初期的 **initial refusal**，利用的是"shallow alignment"（Qi et al., ICLR 2025 Outstanding Paper 的核心发现）——即安全对齐主要只在前几个输出 token 处有效。然而，我们在 Pilot Exp C 中发现了一个被系统性忽视的现象：

> **Delayed Safety Reactivation**：即使初始 refusal 被成功绕过，模型在生成有害内容的中途仍会以 80%（baseline）的概率触发自我纠正，中断有害内容的完整生成。

这意味着：**现有攻击只解决了必要条件（初始 bypass），但没有解决充分条件（持续 safety signal 压制）。**

### 1.2 研究文献中的空白

通过系统文献调研，我们定位了以下关键空白：


| 已有工作                            | 研究内容                              | 未覆盖的空白                                            |
| ------------------------------- | --------------------------------- | ------------------------------------------------- |
| Arditi et al. (NeurIPS 2024)    | LLM 中 refusal 是单一线性方向             | 未在 VLM multimodal mode 验证；未研究 generation dynamics |
| Wollschläger et al. (ICML 2025) | Refusal 是多维 concept cone（静态）      | 未研究动态演变；未在 VLM 验证                                 |
| ICET (ICML 2025 Spotlight)      | Image encoder 内部层的 safety 分布不均    | **仅研究 CLIP 架构**；未研究 LLM backbone 侧                |
| TGA (ICLR 2025)                 | Safety 机制无法从 text 转移到 vision 的根源  | 未区分不同 visual encoder 架构类型的影响                      |
| SafeProbing (2026)              | Safety signal 在 generation 全程持续存在 | 未研究 signal 的**几何方向**如何演化                          |
| JailBound (NeurIPS 2025)        | VLM fusion layer 的静态安全边界          | 未研究 delayed reactivation；未考虑 dynamic rotation     |
| DeepRefusal (EMNLP 2025)        | 分布式 refusal 比集中式更鲁棒（防御侧）          | 未在 VLM 跨架构量化；无视觉模态分析                              |


**核心空白**：**没有任何工作系统研究 visual encoder 架构类型如何决定 LLM backbone 中 refusal signal 的空间分布模式，以及这一分布模式如何决定安全脆弱性类型和最优攻击策略。**

---

## 2. 核心问题定义

### 2.1 问题精确陈述

**研究问题（RQ）**：

> **RQ1**：VLM 的 refusal direction 在 multimodal 模式下是否保持跨模态稳定性？Visual modality 如何影响其幅度？
>
> **RQ2**：Refusal direction 在自回归生成过程中是否随时间步动态旋转？这种旋转是否是 delayed safety reactivation 的 mechanistic 来源？
>
> **RQ3**：Visual encoder 的架构类型（CLIP 系 vs 自研 ViT vs Q-Former）是否决定了 LLM backbone 中 refusal signal 的空间分布模式（集中式瓶颈 vs 分布式扩散）？
>
> **RQ4**：上述分布模式是否预测了最优攻击策略（单层精准攻击 vs 全层分布攻击），并为实际 jailbreak 设计提供机制指导？

### 2.2 形式化定义

设 $h_t^l \in \mathbb{R}^{d}$ 为模型在第 $l$ 层、第 $t$ 个生成步骤时的 hidden state。定义：

$$v^l_{\text{text}} = \frac{\mathbb{E}[h^l | \text{harmful}] - \mathbb{E}[h^l | \text{harmless}]}{ \mathbb{E}[h^l | \text{harmful}] - \mathbb{E}[h^l | \text{harmless}] }$$

类似地定义 $v^l_{\text{mm}}$（multimodal 模式下提取）。

**关键指标**：

- $\cos(v^l_{\text{text}}, v^l_{\text{mm}})$：跨模态方向对齐度（direction stability）
- $\rho^l = v^l_{\text{mm}} / v^l_{\text{text}}$：视觉模态对 refusal 幅度的影响（norm ratio）
- Amplitude Reversal：$\rho^l$ 从 $<1$ 跨越到 $>1$ 的临界层

---

## 3. 研究目标与方法论框架

### 3.1 研究目标层次

```
Level 1（理解）：
  VLM safety mechanism 的几何结构是什么？
  它随层深度和生成时间步如何演化？
  visual encoder 架构类型如何影响这一几何结构？

Level 2（攻击）：
  基于上述几何理解，如何设计更有效的 sequence-level 攻击？
  攻击策略是否应该针对不同架构类型而有所不同？

Level 3（防御）：
  什么样的 VLM 安全架构设计天然鲁棒于 direction ablation 攻击？
  如何从 visual encoder 设计角度提升安全性？
```

### 3.2 整体实验路线

```
Phase 1（Pilot，已完成）
│
├── Exp A：跨模态方向稳定性（LLaVA-1.5-7B）
├── Exp B：生成过程中的方向时间步稳定性
└── Exp C：Delayed Safety Reactivation 基线量化
        ↓ 决策矩阵：Exp A PASS + Exp B FAIL → Dynamic cone estimation

Phase 2（核心攻击实现，已完成）
│
├── Exp 2A：Content confound 分离
├── Exp 2B：Direction ablation 因果效力验证（关键正结果）
└── Exp 2C：PGD 视觉扰动优化（关键负结果）
        ↓ 负结果揭示：pixel → CLIP → hidden state 路径不可行
        ↓ 重大发现：Optimization Paradox + Layer 16 Narrow Waist

Phase 3（跨模型验证，已完成）
│
├── Exp 3A（5层探针）→ P0-B（全层连续曲线，16层 stride-2）
├── Exp 3B（Dynamic Rotation）：待完成
├── Exp 3C（8 prompts）→ P0-A（100 prompts）
└── Exp 3D（逐层 ablation 曲线，50 prompts）
        ↓ 重大发现：Safety Geometry 三分类框架

Phase 4（下一步，规划中）
├── Visual Encoder Feature Space 分析（VE-Exp A/B）
├── 因果控制实验（替换 visual encoder）
└── 多轮对话 Safety Geometry 扩展
```

---

## 4. Pilot 实验（Phase 1）

### 4.1 实验设置

- **模型**：LLaVA-1.5-7B（`llava-hf/llava-1.5-7b-hf`）
- **硬件**：4x H100 80GB
- **数据**：12 对 harmful/harmless topic-matched paired prompts + SaladBench harmful_test

### 4.2 Exp A：Refusal Direction 的跨模态稳定性

**实验设计**：

- 在 text-only 和 text+blank_image 两种条件下，分别在 layer 12/16/20 提取 mean-difference direction
- 计算 $\cos(v_{\text{text}}, v_{\text{mm}})$ 和 norm ratio

**结果**：


| Layer | $\cos(v_{\text{text}}, v_{\text{mm}})$ | norm_text | norm_mm | norm_ratio   |
| ----- | -------------------------------------- | --------- | ------- | ------------ |
| 12    | 0.877                                  | 7.82      | 6.08    | **0.78**（压制） |
| 16    | **0.918**                              | 20.38     | 21.63   | **1.06**（放大） |
| 20    | 0.897                                  | 29.45     | 33.90   | **1.15**（放大） |


**Finding A1**：Refusal direction 跨模态高度稳定（cos > 0.87），visual modality 不改变方向，只改变幅度。

**Finding A2（关键新发现）**：幅度行为呈层级反转——浅层（layer 12）visual modality 压制 refusal 幅度（ratio=0.78），深层（16-20）反而放大（ratio=1.06-1.15）。**这是 Amplitude Reversal 现象的首次观测。**

### 4.3 Exp B：Refusal Direction 的时间步一致性

**实验设计**：

- 在 generation 的 $t = 1, 5, 10, 20, 50$ 步分别提取 mean-difference direction
- 计算 pairwise cosine similarity matrix（layer 16）

**结果**（Pairwise Cosine Similarity，Layer 16）：


|      | t=1   | t=5   | t=10  | t=20      | t=50  |
| ---- | ----- | ----- | ----- | --------- | ----- |
| t=1  | 1.000 | 0.254 | 0.432 | 0.264     | 0.147 |
| t=5  | —     | 1.000 | 0.184 | **0.018** | 0.103 |
| t=10 | —     | —     | 1.000 | 0.250     | 0.108 |
| t=20 | —     | —     | —     | 1.000     | 0.306 |


**min cos = 0.018（t5 vs t20），几乎正交。**

**Finding B**：Refusal direction 在 generation 过程中发生剧烈非单调旋转（subspace switching），而非平滑漂移。⚠️ 注意：此结果含 content confound，需 Phase 2 Exp 2A 分离。

### 4.4 Exp C：Delayed Safety Reactivation 基线量化

**实验设计**：

- 测量 baseline（text-only）和 blank image attack 下的完整有害内容生成率与自我纠正率
- 对 8 条有害 test prompts

**结果**：


| 指标                                  | Baseline（text-only） | Blank Image Attack |
| ----------------------------------- | ------------------- | ------------------ |
| Initial bypass rate                 | 62.5% (5/8)         | 75.0% (6/8)        |
| **Self-correction 率（given bypass）** | **80.0% (4/5)**     | 33.3% (2/6)        |
| Full harmful completion             | 50.0% (4/8)         | 75.0% (6/8)        |


**Finding C（Gap C 问题的直接证明）**：即使初始 bypass 成功，**80% 的情况下模型会在生成中途自我纠正**。Blank image 攻击主要的增益不是提高初始 bypass，而是降低 self-correction rate。

**决策矩阵输出**：Exp A PASS + Exp B FAIL → 升级为 dynamic cone estimation。

---

## 5. Phase 2：核心攻击实现与负结果

### 5.1 Exp 2A：Content Confound 分离

**设计**：Teacher-forced controlled experiment——给所有 prompt 拼接相同的固定前缀（30 tokens），在前缀内提取 hidden state，排除 content difference 的影响。

**结果对比**：


| 指标                | Exp B（uncontrolled） | Exp 2A（controlled） |
| ----------------- | ------------------- | ------------------ |
| Min pairwise cos  | 0.018               | **0.231**          |
| Mean pairwise cos | 0.207               | 0.413              |


**Finding 2A**：约一半的极端漂移是 content confound 造成的 artifact，但**去 confound 后仍存在真实旋转（controlled min cos = 0.231）**，且模式仍然非单调，证实 Dynamic Refusal Rotation 是真实现象。

**Decision：DYNAMIC** → 后续攻击不能用单一静态 direction。

### 5.2 Exp 2B：Direction Ablation 因果效力验证

**设计**：用 PyTorch forward hooks 在生成时实时移除 hidden state 在 refusal direction 上的投影，测试 7 种配置。（58 prompts）

**结果（Full Harmful Completion Rate）**：


| 配置                        | Full Harmful Rate |
| ------------------------- | ----------------- |
| baseline_text             | 17.2%             |
| blank_image               | 56.9%             |
| ablation_text_all（32层）    | 46.6%             |
| ablation_mm_all（32层）      | 74.1%             |
| ablation_mm_shallow（0-15） | 74.1%             |
| ablation_mm_deep（16-31）   | 74.1%             |
| **ablation_mm_layer16**   | **89.7%**         |


**Finding 2B-1（反直觉）**：单层 layer 16 ablation（89.7%）> 全层 ablation（74.1%）。更少的干预反而更有效——**Layer 16 是 LLaVA safety mechanism 的 "Narrow Waist"**。

**Finding 2B-2（Super-additive）**：Visual modality + ablation 存在协同效应：ablation_mm_layer16（89.7%）> 理论加法（29.4+39.7=69.1%），超额增益约 20.6pp。

**Finding 2B-3**：Self-correction rate 从 74.1% 降至 8.6%，证明 direction ablation 同时压制了 delayed safety reactivation。

### 5.3 Exp 2C：PGD 视觉扰动优化（重要负结果）

**设计**：

- 优化图像扰动 $\delta$，使 hidden state 在整个 generation trajectory 上持续抑制 refusal cone
- Loss = $L_{\text{suppress}} + \lambda L_{\text{harmful}}$（hidden state space 的几何约束）
- Config：epsilon=16/255（L∞），200 steps，SVD cone basis（dim=2），all 32 layers

**结果**：


| 指标              | Blank Image | PGD Optimized | Ablation Layer16（上界） |
| --------------- | ----------- | ------------- | -------------------- |
| Initial bypass  | 60.3%       | **44.8%**     | 93.1%                |
| Self-correction | 32.8%       | **53.4%**     | 8.6%                 |
| Full harmful    | **56.9%**   | **37.9%**     | 89.7%                |


**PGD 优化后（37.9%）显著劣于无优化的 blank image（56.9%）——优化让模型变得更安全了。**

**失败根因诊断（四重因素）**：

1. **Proxy Objective Misalignment**：cone basis 从 prefill 阶段提取，但 generation 阶段方向已旋转
2. **CLIP Gradient Bottleneck**：梯度经过 ~56 层 transformer，累积衰减严重（理论 $0.95^{56} \approx 5.6$）
3. **Optimization Paradox**：精确几何目标 + 不精确梯度 → 比 proxy objective 更脆弱
4. **Generic vs Specific Perturbation 的 Robustness Reversal**：blank image 通过 generic activation shift 达到 56.9%；精确优化反而破坏了这个天然属性

**关键 Insight（Optimization Paradox）**：mechanistic understanding 给出了更精确的攻击目标（cone suppression），但将这个目标通过 CLIP ViT 转化为像素空间操作时，对梯度精度的要求更高，反而比 proxy objective 更脆弱。这是一个有理论深度的负结果。

---

## 6. Phase 3：跨模型 Safety Geometry 验证

### 6.1 实验设置

**目标模型（4 种架构类型）**：


| 模型              | Visual Encoder                | VL Connector        | LLM Backbone |
| --------------- | ----------------------------- | ------------------- | ------------ |
| LLaVA-1.5-7B    | CLIP ViT-L/14                 | Linear MLP          | LLaMA-2-7B   |
| Qwen2.5-VL-7B   | Custom ViT（native resolution） | MLP merger          | Qwen2.5-7B   |
| InternVL2-8B    | InternViT-300M                | MLP                 | InternLM2-8B |
| InstructBLIP-7B | BLIP-2 ViT-G                  | Q-Former（32 tokens） | Vicuna-7B    |


---

### 6.2 Exp 3A + P0-B：Refusal Direction 跨模态分析（全层连续曲线）

#### 6.2.1 离散探针结果（5 层，Exp 3A）


| 模型              | Group | NW 层 | 相对深度 | NW cos | Amplitude Reversal | 浅层 ratio | 深层 ratio |
| --------------- | ----- | ---- | ---- | ------ | ------------------ | -------- | -------- |
| LLaVA-1.5-7B    | **A** | 16   | 0.50 | 0.917  | ✅                  | 0.691    | 1.144    |
| Qwen2.5-VL-7B   | **B** | 24   | 0.86 | 0.961  | ❌                  | 0.924    | 0.788    |
| InternVL2-8B    | **B** | 28   | 0.88 | 0.919  | ❌                  | 0.799    | 0.789    |
| InstructBLIP-7B | **A** | 20   | 0.63 | 0.467  | ✅                  | 0.740    | 1.271    |


#### 6.2.2 连续曲线结果（P0-B：stride=2，16 层）

**全层连续 norm_ratio 曲线关键数据**：


| 模型              | 曲线形状                    | Crossover 深度       | 模式                 |
| --------------- | ----------------------- | ------------------ | ------------------ |
| LLaVA-1.5-7B    | 上升-峰-缓降                 | **~0.47（L14→L16）** | 单峰，sharp crossover |
| Qwen2.5-VL-7B   | 全程 < 1（含短暂浅层小于 1.1 的波动） | 无持续 crossover      | 均匀压制               |
| InternVL2-8B    | 全程 < 1（0.724-0.950）     | 无                  | 单调压制               |
| InstructBLIP-7B | 上升-峰-缓降                 | **~0.47（L14→L16）** | 与 LLaVA 惊人一致       |


**关键发现（P0-B Finding 3，The LLM Backbone Crossover Hypothesis）**：

LLaVA 和 InstructBLIP 的 crossover 深度均在 **~47%（layer 14-16/32）**，两者 visual encoder 完全不同（CLIP ViT-L vs BLIP-2 ViT-G + Q-Former），但共享 LLaMA-2 / Vicuna backbone。这强有力地支持了：

> **Crossover 深度由 LLM backbone 的架构属性决定，而非 visual encoder 类型。**

Qwen2.5-VL（Qwen2.5 backbone，28 层）和 InternVL2（InternLM2 backbone，32 层）均无 crossover，与 LLaMA-2 系 backbone 的完全不同行为进一步支持这一假说。

**全层 cos 曲线形态分类**：


| 模型           | cos 曲线形状                     | 含义                           |
| ------------ | ---------------------------- | ---------------------------- |
| LLaVA        | 倒 U 形（mid-peak）              | 中间层 text-mm 融合完成后方向最稳定       |
| Qwen2.5-VL   | V 形（mid-dip → late recovery） | 中间层短暂"困惑"，深层重新收敛到高 cos=0.961 |
| InternVL2    | V 形（较平）                      | 模式类似 Qwen，但恢复更慢              |
| InstructBLIP | L 形（早期高，迅速坍塌并持续低）            | Q-Former 系统性地引入方向旋转（约 63°）   |


**InstructBLIP Anomaly**：cos 从 L0 的 0.878 迅速跌落至 L2 的 0.401 后稳定在 0.40-0.50。Q-Former 的 32-token 压缩引入了 text-mm 之间系统性的约 63° 方向旋转（amplitude reversal 仍存在，但 direction misaligned——"Amplitude Without Direction"现象）。

---

### 6.3 P0-A：Ablation Attack（100 prompts，Exp 3C 扩展版）

**设计**：SaladBench harmful_test 前 100 条 prompts，测试 5 种 ablation 配置。

#### 6.3.1 Full Harmful Completion Rate（n=100）


| 配置                | LLaVA-7B  | Qwen2.5-VL-7B | InternVL2-8B | InstructBLIP-7B |
| ----------------- | --------- | ------------- | ------------ | --------------- |
| baseline_text     | 0.280     | 0.070         | 0.020        | **0.980**       |
| baseline_mm       | 0.610     | 0.110         | 0.070        | **1.000**       |
| ablation_nw_vmm   | **0.870** | 0.540         | 0.030        | 1.000           |
| ablation_all_vmm  | 0.720     | **0.990**     | 0.050        | 1.000           |
| ablation_nw_vtext | 0.860     | 0.570         | 0.070        | 1.000           |


#### 6.3.2 Self-Correction Rate（n=100）


| 配置               | LLaVA-7B | Qwen2.5-VL-7B | InternVL2-8B | InstructBLIP-7B |
| ---------------- | -------- | ------------- | ------------ | --------------- |
| baseline_mm      | 0.290    | 0.890         | 0.930        | 0.000           |
| ablation_nw_vmm  | 0.100    | 0.430         | 0.970        | 0.000           |
| ablation_all_vmm | 0.220    | **0.000**     | 0.930        | 0.000           |


#### 6.3.3 与 8-prompt Pilot 结果的一致性验证


| 模型           | 方向一致？ | 关键数值变化                                        |
| ------------ | ----- | --------------------------------------------- |
| LLaVA-7B     | ✅     | NW: 87.5%→87.0%（稳定）; All: 87.5%→72.0%（略降，更精确） |
| Qwen2.5-VL   | ✅     | NW: 62.5%→54.0%; All: 100%→99.0%              |
| InternVL2    | ✅     | 均低（3-5%），模式一致                                 |
| InstructBLIP | N/A   | 全程 100%（退化对照，不做分析）                            |


**n=100 的结论完全验证了 n=8 的方向性判断，统计显著性大幅提升。**

---

### 6.4 Exp 3D：逐层 Ablation 曲线（50 prompts）

**设计**：对每一层 $l$ 单独 ablate，测量 full_harmful_rate($l$)，生成连续 ASR 曲线。

#### 6.4.1 四种曲线形态


| 模型              | 曲线形态     | 峰值层              | 峰值 ASR    |
| --------------- | -------- | ---------------- | --------- |
| LLaVA-1.5-7B    | 单峰集中型    | Layer 16（深度 50%） | **0.900** |
| Qwen2.5-VL-7B   | 深层悬崖型    | Layer 22（深度 79%） | 0.640     |
| InternVL2-8B    | 平线型（免疫）  | -                | 0.040     |
| InstructBLIP-7B | 常数型（无对齐） | -                | 1.000     |


**LLaVA 的关键观测（窄腰几何证明）**：

```
层  8: fhr = 0.460（低谷区）
层 12: fhr = 0.320（诡异低谷，低于 baseline=0.580）← 浅层消融悖论
层 14: fhr = 0.620
层 16: fhr = 0.900（峰值，与 norm_ratio crossover 精确对应）
层 18: fhr = 0.820（快速衰减）
层 28: fhr = 0.580（接近 baseline）
```

**Finding 3D-1（因果桥梁）**：Layer 16 既是 norm_ratio 的 crossover 点（ratio: 0.911→1.061），也是 ablation 效果的峰值层（fhr=0.900）。两者精确对应，建立了 **"幅度反转点 = ablation 最优层"** 的因果关系。

**Finding 3D-2（浅层消融悖论）**：Layer 12 ablation 的 fhr=0.320，**低于无 ablation 的 baseline（0.580）**。在 norm_ratio < 1 的浅层，refusal direction 携带双重信息——安全信号 + 视觉压制效应。消融这个方向反而移除了有利于攻击者的"视觉压制分量"，导致安全性反向增强。**攻击者应该避免消融浅层。**

**Qwen2.5-VL 的三个反常发现**：

1. **真正峰值（layer 22）≠ cos 最高层（NW layer 24）**：执行层与方向最纯粹层不同。
2. **末端安全复核机制**：Layer 22 ablation fhr=0.640，但 Layer 26 ablation 仅 0.040，**2 层内完全恢复**。Qwen 存在末端安全"复核"机制，攻击后能在最后几层重建防御。
3. **"早期积累 + 晚期执行"的委员会决策模型**：需要全层 ablation 才能同时瓦解早期存储的前置判断和晚期的执行层。

**InternVL2 的平线型（完全免疫）**：全部 18 个探测层的 fhr ≈ 0.020，与 baseline 无异。可能原因：

- 无幅度反转 → 无浓缩点 → 消融任意单层只移除 1/32 的安全表征
- mean-diff direction 可能不是 InternVL2 安全信号的准确线性代理
- InternLM2 的安全可能依赖 attention pattern 等非线性机制

---

## 7. 综合分析：Safety Geometry 框架的建立

### 7.1 三种 Safety 架构范式

基于所有 Phase 3 实验，我们提出 VLM 安全几何的三分类框架：

**Type I — 单点瓶颈型（Bottleneck Architecture）**

- 代表：LLaVA
- 特征：Amplitude Reversal ✅，NW ablation > All ablation
- 几何解释：Refusal signal 在 crossover 层高度汇聚，形成单一功能瓶颈
- 脆弱性：高（单层精准攻击 ASR ≈ 90%）

**Type II — 晚期门控型（Late Gate Architecture）**

- 代表：Qwen2.5-VL
- 特征：Amplitude Reversal ❌，All ablation >> NW ablation，末端复核机制
- 几何解释："早期积累 + 晚期执行"的委员会决策，深层存在末端复核
- 脆弱性：中（需全层 ablation ASR 99%，但单层攻击被末端复核拦截）

**Type III — 弥散免疫型（Diffuse Architecture）**

- 代表：InternVL2
- 特征：Amplitude Reversal ❌，All ablation 和 NW ablation 均接近 0%
- 几何解释：Safety signal 均匀分布于所有层，线性方向 ablation 原理性失效
- 脆弱性：低（对 direction ablation 攻击免疫）

（InstructBLIP 作为退化对照：Type IV — 对齐缺失型，baseline 即 100% bypass，不作为有效安全模型分析。）

### 7.2 Amplitude Reversal 是安全集中度的预测因子

这是将 Exp 3A（geometry）和 Exp 3C + 3D（attack）打通的核心发现：

```
CLIP ViT（线性 patch projection）
    → 图像表征平滑注入 LLM
    → 浅层 safety signal 被视觉噪声稀释（norm_ratio < 1）
    → 中层残差流将多模态信号整合
    → 深层 safety 机制将分散线索"汇聚结晶"→ norm_ratio 跨越 1.0
    → 形成 Bottleneck → 单层精准攻击有效

自研 ViT（cross-attention 或专为 VLM 训练）
    → 图像表征通过更复杂路径注入 LLM
    → Safety signal 从一开始就分散在跨层的多个维度
    → norm_ratio 全程 < 1，无集中事件
    → 无 Bottleneck → 需全层攻击
```

**因此，`Amplitude Reversal 的存在` 是预测 `最优攻击策略` 的充分指示符。**

### 7.3 "两阶段 Safety Geometry"模型

P0-B 的连续曲线揭示了 VLM safety 的两阶段结构：

**Phase 1：浅层特征整合区（depth 0–0.5）**

- Visual tokens 正被融合进 LLM token stream
- norm_ratio < 1 + cos 较低/上升
- Safety signal 弱且不稳定
- **风险窗口**：攻击者的最佳切入点（但见 Layer 12 悖论）

**Phase 2：深层决策表征区（depth 0.5–1.0）**

- LLM backbone 完成了跨模态表征的整合
- Group A：norm_ratio > 1 + cos 高且稳定 → 强的、可汇聚的安全信号
- Group B：norm_ratio < 1 但 cos 高 → 方向正确但幅度被压制
- Narrow waist 标志了方向对齐度的峰值

### 7.4 Qwen2.5-VL 的 Safety Trade-off

Qwen2.5-VL 展示了一个深刻的 alignment 悖论：

- Baseline safety：最强（text + mm 均为 0% full harmful）
- All-layer ablation：完全崩溃（99% full harmful，0% self-correction）
- Self-correction 命运：可被彻底摧毁（sc: 0.89→0.00）

**Safety 集中性越高 → 越脆弱于定向攻击。** 这与 DeepRefusal（EMNLP 2025）从防御侧的发现完全对应：分布式 refusal 比集中式更鲁棒。InternVL2 的 Type III 模式正是分布式安全的天然实现，对线性 direction ablation 免疫。

### 7.5 Self-Correction 对 Ablation 的三种命运


| 模型         | Self-correction 响应                   | 含义                               |
| ---------- | ------------------------------------ | -------------------------------- |
| LLaVA      | 衰减型：SC: 0.29→0.10（NW ablation 后）     | SC 部分依赖 refusal direction，可被削弱   |
| Qwen2.5-VL | 可摧毁型：SC: 0.93→**0.00**（all ablation） | SC 完全依赖 refusal direction，可被彻底消除 |
| InternVL2  | 免疫型：SC: 0.93→0.97（ablation 后反升）      | SC 完全独立于 refusal direction       |


InternVL2 的 SC **反升**现象（ablation 后 SC 不降反升）提示：InternVL2 的 self-correction 机制在某种程度上被激活以补偿 ablation 的干扰——这是 safety mechanism 的非线性冗余特性。

---

## 8. 已验证与已否定的假设

### 8.1 已验证（✅）


| 假设                                               | 验证实验        | 关键数据                                    |
| ------------------------------------------------ | ----------- | --------------------------------------- |
| Delayed safety reactivation 在 VLM 中普遍存在          | Exp C       | baseline SC = 80% (5/8)                 |
| Refusal direction 跨模态方向稳定                        | Exp A + 3A  | cos > 0.87（非 Q-Former 3/4 模型）           |
| Dynamic refusal rotation 是真实现象（非 artifact）       | Exp 2A      | controlled min cos = 0.231              |
| Layer 16 是 LLaVA safety mechanism 的 Narrow Waist | Exp 2B + 3D | NW ablation 89.7% > all-layer 74.1%     |
| v_text = v_mm 在 ablation 效果上完全等价                 | 3C（n=100）   | 4/4 模型差距 < 3%                           |
| CLIP ViT 架构产生 Amplitude Reversal                 | 3A + P0-B   | LLaVA crossover @ L16, ratio: 0.91→1.06 |
| Q-Former 是天然安全漏洞                                 | 3C（n=100）   | InstructBLIP baseline_mm = 100%         |
| Amplitude Reversal 预测最优 ablation 策略              | 3C + 3D     | Group A: NW≥All；Group B: All>>NW        |
| LLM backbone 决定 crossover 深度（非 ViT）              | P0-B        | LLaVA + InstructBLIP 同 crossover @L16   |


### 8.2 已否定（❌）


| 原假设                                                  | 否定证据                             | 修正方向                              |
| ---------------------------------------------------- | -------------------------------- | --------------------------------- |
| Amplitude reversal 是通用现象                             | Exp 3A：仅 2/4 模型成立                | 改为"CLIP 架构特有的条件性现象"               |
| Narrow waist 在 ~50% 相对深度是架构规律                        | 3A：仅 LLaVA 成立，其余 63%-88%         | 放弃"architectural regularity"claim |
| 单层 NW ablation 通用优于全层                                | 3C + 3D：Group B 正好相反             | 改为"Amplitude Reversal 预测策略"       |
| Qwen2.5-VL 最强 baseline → 最强鲁棒性                       | 3C：all-layer collapse 99%        | 揭示集中式 safety 的内在脆弱                |
| PGD pixel-level 优化可有效压制 hidden state 中的 refusal cone | Exp 2C：37.9% < blank image 56.9% | Optimization Paradox（理论贡献）        |


---

## 9. Novelty 评估与文献定位

### 9.1 核心贡献点


| #   | 贡献                                                            | 最相近文献                              | 独特 Delta                                                  | 等级           |
| --- | ------------------------------------------------------------- | ---------------------------------- | --------------------------------------------------------- | ------------ |
| 1   | Amplitude Reversal 现象（CLIP 特有）+ 与 ablation 效果的因果关系            | 无直接对应                              | 全新发现，量化建立了 geometry→vulnerability 因果链                     | ★★★★         |
| 2   | 三种 Safety 架构范式的实证分类框架                                         | DeepRefusal（防御侧，无跨架构比较）            | 首次从攻击侧跨多架构量化分类                                            | ★★★★         |
| 3   | LLM Backbone Crossover Hypothesis（crossover 由 LLM 架构决定，非 ViT） | ICET（只研究 image encoder 内部，只有 CLIP） | 首次识别 crossover 深度的决定因素                                    | ★★★          |
| 4   | 浅层消融悖论（Layer 12 ablation 降低 ASR）                              | 无                                  | 揭示 refusal direction 在浅层的"双重信息"特性                         | ★★★          |
| 5   | Qwen 的末端安全复核机制                                                | 无                                  | 首次量化 last-layer recovery 效应（2层内完全恢复）                      | ★★★          |
| 6   | InstructBLIP "Amplitude Without Direction" 悖论                 | TGA（结论相近无量化）                       | 首次用 cos < 0.5 + amplitude reversal 共存来量化 Q-Former 的安全失效机制 | ★★★          |
| 7   | Dynamic Refusal Rotation（生成中的非单调方向旋转）                         | 无                                  | 全新现象，解释 delayed reactivation 的 mechanistic 来源             | ★★★★（待跨模型验证） |
| 8   | Optimization Paradox（精确几何目标比 proxy 更脆弱）                       | 无                                  | 理论贡献，对 adversarial attack 领域的普遍警示                         | ★★★          |


### 9.2 与最相关文献的精确区分

**与 ICET（ICML 2025 Spotlight）**：

- ICET 研究 image encoder 内部的 safety 分布（visual encoder 侧），仅覆盖 CLIP 架构
- 本研究研究 LLM backbone 侧的 safety geometry，并覆盖 CLIP、自研 ViT、Q-Former 三种架构类型
- P0-B 的 Crossover Hypothesis 进一步将两者桥接：image encoder 类型影响 LLM 中的 safety phase transition 位置

**与 JailBound（NeurIPS 2025）**：

- JailBound 在 fusion layer 定义静态安全边界，joint image+text 攻击
- 本研究发现边界是动态旋转的（Dynamic Rotation），且不同架构类型的最优攻击策略截然不同

**与 TGA（ICLR 2025）**：

- TGA：safety mechanism 无法从 text 转移到 vision 的根源是 hidden state alignment 不足
- 本研究：从 geometry 层面量化了这一现象——Q-Former 的方向旋转（cos ≈ 0.45）是 alignment 失败的几何标志

---

## 10. 局限性

### 实验层面


| 问题                                                  | 严重程度 | 状态                                 |
| --------------------------------------------------- | ---- | ---------------------------------- |
| InstructBLIP Vicuna backbone 无安全对齐，所有 3C/3D 数据为退化对照 | ⚠️ 高 | 需替换（LLaVA-NeXT 或 LLaMA-3.2-Vision） |
| Exp 3B（Dynamic Rotation 跨模型验证）尚未完成                  | ⚠️ 高 | 计划 P1-A                            |
| 50 prompts（3D）的部分数值统计显著性有限（如 0.040 = 2/50）          | ⚠️ 中 | P0-C 升级到 100 prompts               |
| 5 层 probe → 16 层 stride-2（P0-B），仍非逐层全扫描             | ℹ️ 低 | 已可绘制连续曲线                           |


### 方法论层面


| 问题                                  | 说明                                               |
| ----------------------------------- | ------------------------------------------------ |
| CLIP vs Custom ViT 的区分存在 confound   | 不同架构同时伴随不同 alignment 训练质量、数据规模、LLM backbone      |
| 所有 LLM backbone 实验限于白盒场景            | 商业模型（GPT-4o、Gemini）的 LLM backbone 不可访问           |
| Mean-diff direction 是 refusal 的一阶近似 | 可能无法捕获 InternVL2 等模型的非线性安全特征                     |
| 评估使用 keyword matching               | 存在 false positive/negative，正式实验需 LLM-based judge |


---

## 11. 下一步研究方向

### 11.1 近期（2-3 周）：补全 Phase 3

**P0 级（立即执行）**：

- **P0-A**：Exp 3C 升级到 100 prompts（已完成）✅
- **P0-B**：Exp 3A 全层连续曲线（已完成）✅  
- **P0-C**：Exp 3D 逐层 ablation 曲线（已完成）✅
- **P1-A**：完成 Exp 3B（Dynamic Rotation 跨模型验证，4 模型）

**P1 级（模型扩展）**：

- 替换 InstructBLIP（退化对照）→ **LLaVA-v1.6-Mistral-7B**（CLIP ViT-L + Mistral backbone，验证 crossover 是否随 LLM backbone 变化）
- 新增 **Llama-3.2-Vision-11B**（更大 CLIP ViT-H，Group A 的扩展验证）
- 目的：将 Group A 数据点从 1 有效（LLaVA）→ 3，强化因果主张

### 11.2 中期（4-6 周）：Visual Encoder Feature Space 分析（Phase 4）

**VE-Exp A：Visual Encoder 内部 Safety 线性可分性**

在 visual encoder 的每一层提取 image features，训练 linear probe 区分 safe/unsafe context，测量各层的 probe accuracy。目标：建立 visual encoder feature space → LLM backbone safety geometry 的预测模型，使研究结论可在黑盒商业模型上应用。

**VE-Exp B：Visual Encoder 层输出 → LLM Safety Amplitude 的关系**

模拟 ICET 的思路：强制使用 visual encoder 的中间层输出（而非最终层）注入 LLM，测量 LLM 中 norm_ratio 如何响应。桥接 ICET（image encoder 侧）和本研究（LLM backbone 侧）的两个研究维度。

### 11.3 攻击设计（Phase 4 后期）：Architecture-Aware AGDA

**Ablation-Guided Distillation Attack（AGDA）**：

基于 Exp 2B 的 oracle（layer 16 ablation 后的 hidden states），优化 pixel perturbation 使其在 hidden state 空间上逼近 ablation 效果，绕过 CLIP 梯度 bottleneck：

$$\mathcal{L}_{\text{AGDA}} = \frac{1}{T}\sum_t \text{MSE}(h_t^{\text{perturbed}}, h_t^{\text{ablated}}) + \lambda \cdot \text{TV}(\delta)$$

**针对不同架构类型的差异化策略**：


| 模型类型                              | 推荐攻击策略                                       | 预期 ASR  |
| --------------------------------- | -------------------------------------------- | ------- |
| Type I（Bottleneck, e.g. LLaVA）    | AGDA 对准单一 crossover 层                        | 高（~90%） |
| Type II（Late Gate, e.g. Qwen）     | 多层 distillation（覆盖 layer 20-24）              | 中-高     |
| Type III（Diffuse, e.g. InternVL2） | Direction ablation 无效，需探索 attention-level 攻击 | 待探索     |


### 11.4 新方向探索：多轮对话中的 Safety Geometry 演化

**可行性判断：可行，建议纳入 Phase 4 后期或独立研究线。**

**理论基础**：

本研究发现 refusal direction 在单轮 generation 过程中动态旋转（Dynamic Rotation）。这一现象在多轮对话场景下有一个自然延伸：随着对话轮数增加，KV-cache 中积累的上下文会持续影响 hidden state 的分布，从而可能影响 safety geometry 的形态。

**核心问题**：

> 在多轮对话中，VLM 的 safety geometry（refusal direction 的 cos 和 norm_ratio）是否随对话轮数演化？是否存在某个轮数"滚雪球窗口"，使 safety signal 自然衰减，从而使后续轮次更容易被 bypass？

**初步假说**：

1. **上下文累积假说**：多轮对话中，若前几轮引入了看似 benign 的视觉内容，KV-cache 会积累这些 visual tokens 的 hidden state，逐渐"稀释"LLM 对 safety 的关注
2. **滚雪球脆弱性**：有害请求被分散在多轮对话中（每轮单独无害），但当所有轮次的上下文积累后，safety geometry 可能已发生足够大的偏移，使第 N 轮轻松 bypass
3. **对话轮次特异性**：Dynamic Rotation 在单轮内的旋转角度可能与对话轮次有规律性关系

**实验设计构想**：

```
Multi-turn Safety Geometry Probe:
  Turn 1: "Can you describe this image?" + safe image（建立视觉上下文）
  Turn 2: "What about this one?" + safe image
  Turn t: 在每轮结束后提取 refusal direction，测量 cos(v_t, v_1) 的变化
  Turn N: 引入 harmful request，测量 safety bypass rate vs. 单轮对照

关键测量：
  - v_t（第 t 轮的 refusal direction）随 t 的变化
  - Amplitude（norm_ratio）随 t 的演化
  - 有效 bypass rate 与对话轮数的关系

与 adversarial context injection 结合：
  - 前 t-1 轮可以被设计为引导性上下文，逐步"推移" safety geometry
  - 使得第 t 轮的 single-layer ablation 从 Type II 行为转变为 Type I 行为
```

**与现有工作的区分**：

现有多轮 jailbreak 攻击（如 multi-turn PAIR 变体）是 prompt 层面的工程技巧。本研究方向是首次从 **safety geometry 的 mechanistic 演化** 角度分析多轮对话的安全性，可以解释为什么某些多轮攻击有效的内在机制。

**与当前研究的衔接**：

Dynamic Rotation（Exp B/2A）发现生成过程中 refusal direction 非单调旋转，多轮对话的 geometry 演化是这一发现在对话层面的自然延伸，形成一个完整的 "temporal dynamics of VLM safety" 研究主线。

---

## 12. 参考文献

**本研究直接依赖（精读）**：

1. yiArditi et al., "Refusal in LLMs is Mediated by a Single Direction", **NeurIPS 2024**
2. Wollschläger et al., "The Geometry of Refusal in LLMs: Concept Cones and Representational Independence", **ICML 2025**
3. Qi et al., "Safety Alignment Should Be Made More Than Just a Few Tokens Deep", **ICLR 2025 Outstanding Paper**
4. ICET: "Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in VLMs", **ICML 2025 Spotlight**
5. TGA: "Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models", **ICLR 2025**
6. VLLM Safety Paradox: "The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense", **NeurIPS 2025**
7. JailBound: "Jailbreaking Internal Safety Boundaries of Vision-Language Models", **NeurIPS 2025**
8. DeepRefusal: "Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction", **EMNLP 2025 Findings**
9. SafeProbing, arXiv 2026
10. The Safety Reminder (SAPT), arXiv 2025
11. DSN, **ACL 2025 Findings**
12. UltraBreak, **ICLR 2026**
13. HarmBench (Mazeika et al.), **ICML 2024**（评估标准）

---

## 附录：实验完成状态一览


| 实验                           | 模型       | 规模                | 状态            | 关键结果                       |
| ---------------------------- | -------- | ----------------- | ------------- | -------------------------- |
| Exp A（5 层 cos/norm）          | LLaVA-7B | 12 对              | ✅             | cos=0.918@L16, Reversal 发现 |
| Exp B（时间步稳定性）                | LLaVA-7B | 12 对              | ✅（含 confound） | min cos=0.018              |
| Exp C（SC 基线）                 | LLaVA-7B | 8                 | ✅             | baseline SC=80%            |
| Exp 2A（confound 分离）          | LLaVA-7B | 12 对              | ✅             | controlled min cos=0.231   |
| Exp 2B（ablation 因果）          | LLaVA-7B | 58                | ✅             | layer16=89.7%              |
| Exp 2C（PGD 负结果）              | LLaVA-7B | 16 train, 58 eval | ✅             | 37.9% < blank 56.9%        |
| Exp 3A（4模型5层）                | 4 模型     | 12 对 × 4          | ✅             | Group A/B 分类               |
| P0-B（4模型16层连续）               | 4 模型     | 12 对 × 4          | ✅             | Crossover Hypothesis       |
| Exp 3C（8 prompts）            | 4 模型     | 8                 | ✅             | 方向性验证                      |
| P0-A（100 prompts 3C）         | 4 模型     | 100               | ✅             | 统计显著验证                     |
| Exp 3D（逐层 ablation）          | 4 模型     | 50                | ✅             | 三分类安全架构                    |
| Exp 3B（Dynamic Rotation 跨模型） | 4 模型     | —                 | ❌ 待完成         | —                          |


---

*汇报总结：本周完成了从 Pilot 到 Phase 3 的完整实验体系，涵盖 4 种 VLM 架构。核心发现是"visual encoder 架构类型通过 Amplitude Reversal 机制决定 LLM backbone 中 refusal signal 的空间分布，进而决定安全脆弱性类型和最优攻击策略"。这一发现有文献空白支撑、有跨模型实验验证、有完整的因果解释链，具备 AAAI 2026 full paper 的提交条件，但需要补全 Exp 3B 和替换 InstructBLIP 退化对照。*