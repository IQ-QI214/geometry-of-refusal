# Gap C 研究全局总结与 Phase 3 跨模型验证方案

> **文档版本**：2026-03-31
> **状态**：Phase 1 + Phase 2 完成，Phase 3 规划中
> **目标会议**：AAAI 2026
> **研究方向**：VLM Safety Mechanism 的动态几何结构 —— 从 Pilot 到完整攻击论文

---

## 目录

1. [研究问题与核心假说](#1-研究问题与核心假说)
2. [已完成实验的完整发现总结](#2-已完成实验的完整发现总结)
3. [关键理论 Insight 整理](#3-关键理论-insight-整理)
4. [与已有文献的精确定位](#4-与已有文献的精确定位)
5. [Novelty 评估（更新版）](#5-novelty-评估更新版)
6. [Phase 3：跨模型验证 Vibe Coding 方案](#6-phase-3跨模型验证-vibe-coding-方案)
7. [后续攻击设计路线图](#7-后续攻击设计路线图)
8. [AAAI 2026 Paper 框架草案](#8-aaai-2026-paper-框架草案)
9. [关键文献速查表](#9-关键文献速查表)

---

## 1. 研究问题与核心假说

### 1.1 Problem Statement（精确版本）

VLM 的 safety alignment 存在一个系统性的漏洞：**Delayed Safety Reactivation**。
即使 jailbreak 攻击成功绕过了初始 refusal token，模型会在生成有害内容的中途重新激活 safety awareness，触发 self-correction（SAPT/Safety Reminder, arXiv 2025）。我们的 Pilot Exp C 在 LLaVA-1.5-7B 上定量验证：**baseline 条件下 80% 的 bypassed prompt 在生成过程中出现 self-correction**。

这个问题的 hidden state 层面机制此前从未被系统研究：safety signal 在整个 generation 序列中持续存在（SafeProbing, arXiv 2026），但其**几何方向**如何演化，以及 visual modality 如何影响这个演化过程，在现有顶会文献中是完全空白的。

### 1.2 Research Questions（第一性原理推导）

```
RQ1（已被 Pilot 验证）:
  VLM 的 refusal direction 是否跨模态（text vs multimodal）稳定？
  → 答案：YES，cos=0.918（layer 16），但幅度行为呈现层级反转

RQ2（已被 Phase 2 部分验证）:
  Refusal direction 在 generation 过程中是否稳定？
  → 答案：NO，真实旋转存在（controlled min sim = 0.231），非平滑的 subspace switching

RQ3（已被 Phase 2 验证）:
  LLaVA 的 safety mechanism 是否存在单层 bottleneck？
  → 答案：YES，layer 16 是 "narrow waist"（89.7% vs 74.1% 全层）

RQ4（待完整验证，Phase 3 核心）:
  以上 RQ1-RQ3 的发现是否在多个 VLM 架构上普遍成立？
  → 目标：3+ 模型验证

RQ5（待攻击设计，Phase 4）:
  基于以上 mechanistic understanding，能否设计比 blank image 更强的 sequence-level attack？
  → 目标：全有害 completion rate > 80%，self-correction rate < 10%
```

---

## 2. 已完成实验的完整发现总结

### 2.1 Pilot Experiment A：Refusal Direction 的模态稳定性

**实验设置**：

- 模型：LLaVA-1.5-7B，4x H100
- 数据：12 对 harmful/harmless paired prompts（topic-matched）
- 方法：text-only forward → 提取 mean-difference direction $v_{\text{text}}$；text+blank_image → 提取 $v_{\text{mm}}$；计算 cos similarity 和 norm ratio
- 层：layer 12, 16, 20

**核心数据**：


| Layer | cos(v_text, v_mm) | norm_text | norm_mm | norm_ratio |
| ----- | ----------------- | --------- | ------- | ---------- |
| 12    | 0.877             | 7.82      | 6.08    | **0.78**   |
| 16    | **0.918**         | 20.38     | 21.63   | **1.06**   |
| 20    | 0.897             | 29.45     | 33.90   | **1.15**   |


**关键 Findings**：

**Finding A1（已确认）：Refusal direction 跨模态高度稳定（cos > 0.87）**
Visual modality 不改变 refusal 的"朝向"（方向），这意味着在纯 LLM backbone 上提取的 refusal direction 在 VLM 的 multimodal mode 下仍然有效——无需为 VLM 重新设计 direction extraction pipeline。这与 Du et al. (2025) 的"refusal direction drift 发生在 fine-tuning 阶段"的结论一致，推理时 visual modality 的加入不引起额外 drift。

**Finding A2（核心新发现）：Layer-wise Amplitude Reversal——视觉模态对 refusal 幅度的影响呈层级反转**

- 浅层（layer 12）：norm_ratio = 0.78 < 1 → visual modality **压制** refusal 信号幅度
- 深层（layer 16, 20）：norm_ratio = 1.06, 1.15 > 1 → visual modality 反而**放大** refusal 信号

**理论解释（待验证假说）**：这可能反映了一种 "Compensatory Safety Mechanism"——浅层的 safety signal 被 visual modality 压制后，深层的 safety mechanism 通过放大来补偿。如果这个假说成立，它提供了一个全新的 mechanistic 解释：**blank image 之所以既能帮助初始 bypass（利用浅层压制），又无法完全阻止 self-correction（因为深层会补偿放大），正是这个 layer-wise amplitude reversal 的直接推论**。这是一个尚未被任何顶会论文研究过的 VLM-specific 现象。

**注意**：Exp A 验证的是 **refusal direction（单向量，mean-difference 提取）**，不是 refusal cone（多维，需 SVD 提取）。这两个概念在 paper 写作时必须精确区分。

---

### 2.2 Pilot Experiment B：Refusal Direction 的时间步一致性

**实验设置**：

- 固定 harmful prompt，在 t=1,5,10,20,50 个 decoding step 分别提取 mean-difference direction
- 计算所有时间步对之间的 pairwise cosine similarity（layer 16）

**核心数据**（Pairwise Cosine Similarity, Layer 16）：


|      | t=1   | t=5   | t=10  | t=20  | t=50  |
| ---- | ----- | ----- | ----- | ----- | ----- |
| t=1  | 1.000 | 0.254 | 0.432 | 0.264 | 0.147 |
| t=5  | —     | 1.000 | 0.184 | 0.018 | 0.103 |
| t=10 | —     | —     | 1.000 | 0.250 | 0.108 |
| t=20 | —     | —     | —     | 1.000 | 0.306 |
| t=50 | —     | —     | —     | —     | 1.000 |


**min = 0.018（t5 vs t20），mean = 0.207**

**关键 Findings（含 confound 警告）**：

**Finding B1：Refusal direction 在 generation 中接近正交化**
t5 和 t20 之间 cos=0.018 几乎正交，说明模型在不同 decode 阶段使用几乎完全不同方向表达 safety awareness。

**Finding B2：非平滑的 Subspace Switching（注意：此结论存在 confound）**
漂移模式是非单调的（t1 vs t10 的 cos=0.432 反而高于 t1 vs t5 的 cos=0.254），提示不是平滑 drift 而是跳跃式 subspace switching。
⚠️ **Confound 警告**：Exp B 的 mean-difference 同时捕获了 refusal signal 和 content difference，未去 confound。

---

### 2.3 Phase 2 Exp 2A：Confound Resolution

**目标**：分离 Exp B 中 content difference confound 和真实 refusal direction drift。

**方法**：Teacher-forced controlled experiment——给所有 prompts 拼接相同的固定前缀（30 tokens），在前缀内提取 hidden state，确保 mean-difference 只捕获安全属性差异。

**核心数据对比**：


| 指标                | Exp B（uncontrolled） | Exp 2A（controlled） | 差异     |
| ----------------- | ------------------- | ------------------ | ------ |
| Min pairwise cos  | 0.018               | **0.231**          | +0.213 |
| Mean pairwise cos | 0.207               | 0.413              | +0.206 |


**关键 Findings**：

**Finding 2A-1：约一半的漂移来自 content confound**
controlled min (0.231) 远高于 uncontrolled min (0.018)，说明 Exp B 的极端漂移中约 50% 是 content difference 的 artifact。

**Finding 2A-2（核心）：剩余漂移是真实的 Dynamic Refusal Rotation**
控制 content confound 后，min sim=0.231 仍远低于 0.40 阈值（Phase 2 plan 中的阈值），且模式仍是非单调的。**这证明 VLM 的 safety mechanism 在 generation 过程中确实发生了几何方向的显著旋转——这不是 artifact，是 LLaVA 安全机制的真实特征。**

**Decision：DYNAMIC**——后续攻击需要 timestep-specific directions 或 SVD cone basis，不能用单一静态 direction。

---

### 2.4 Phase 2 Exp 2B：Direction Ablation Attack 因果效力验证

**目标**：验证 Exp A 提取的 mean-diff direction 在 LLaVA generation 中有因果效力（相关性 ≠ 因果性）。

**方法**：PyTorch forward hooks 在生成时实时移除 hidden state 在 refusal direction 上的投影。测试 7 种配置。

**核心数据**：


| Config                    | Initial Bypass | Self-Correction | Full Harmful | n   |
| ------------------------- | -------------- | --------------- | ------------ | --- |
| baseline_text             | 22.4%          | 74.1%           | 17.2%        | 58  |
| blank_image               | 60.3%          | 32.8%           | **56.9%**    | 58  |
| ablation_text_all（32层）    | 48.3%          | 50.0%           | 46.6%        | 58  |
| ablation_mm_all（32层）      | 77.6%          | 22.4%           | 74.1%        | 58  |
| ablation_mm_shallow（0-15） | 77.6%          | 18.9%           | 74.1%        | 58  |
| ablation_mm_deep（16-31）   | 77.6%          | 18.9%           | 74.1%        | 58  |
| **ablation_mm_layer16**   | **93.1%**      | **8.6%**        | **89.7%**    | 58  |


**关键 Findings**：

**Finding 2B-1（Counter-intuitive）：Layer 16 单层 ablation 最强（89.7%）> 全层 ablation（74.1%）**
更少的干预反而更有效。原因：全层 ablation 在浅层（0-15）干扰了与 content 混合的 signal，引入生成质量下降；layer 16 是 refusal signal 最纯粹的位置（cos=0.918），精确 ablation 的副作用最小。**这确立了 Layer 16 作为 LLaVA safety mechanism 的 "Narrow Waist"。**

**Finding 2B-2：Visual Modality + Ablation 存在 Super-additive Interaction**

- ablation_mm_layer16 (89.7%) > 理论加法 (29.4 + 39.7 = 69.1%)
- 超额增益 ≈ 20.6 pp
- Visual modality 改变了 hidden state distribution，使 ablation 更精准命中 refusal signal 的核心分量。

**Finding 2B-3：Delayed Safety Reactivation 被有效抑制**
ablation_mm_layer16 的 self-correction rate 从 baseline 74.1% 降至 8.6%——**直接证明 refusal direction ablation 不仅影响初始 refusal，还影响整个序列的 delayed safety reactivation**。这是 sequence-level suppression 的 causal evidence。

**重要区分**：Exp 2B 验证的仍是 **refusal direction（单向量）的 ablation**，不是 cone ablation。

---

### 2.5 Phase 2 Exp 2C：PGD Visual Perturbation（负结果）

**目标**：优化像素扰动 delta，通过梯度反传使 hidden state 在整个 generation trajectory 上持续抑制 refusal cone。

**Config**：PGD epsilon=16/255，200 steps，SVD cone basis（dim=2），all 32 layers，16 training prompts。

**核心数据**：


| Metric                  | Blank Image | PGD Optimized | Ablation Layer16 |
| ----------------------- | ----------- | ------------- | ---------------- |
| Initial bypass rate     | 60.3%       | **44.8%**     | 93.1%            |
| Self-correction rate    | 32.8%       | **53.4%**     | 8.6%             |
| Full harmful completion | **56.9%**   | **37.9%**     | 89.7%            |


**结论：PGD 优化后的扰动（37.9%）显著劣于无优化的 blank image（56.9%）。这是一个重要的负结果。**

**失败根因诊断（四重因素）**：

1. **Proxy Objective Misalignment**：cone basis 从 prefill 阶段提取（5个 teacher-forced positions），但在 generation 阶段方向已旋转（Exp 2A），用旧方向的 loss 无法准确反映 generation 中的 refusal signal
2. **CLIP Gradient Bottleneck**：梯度路径经过 ~56 层 transformer（32 LLaMA + 24 CLIP ViT），累积衰减严重；delta 在 step 20 就达到 epsilon 饱和，但梯度方向失真
3. **Generic vs Specific Perturbation 的 Robustness Reversal**：Blank image 通过 generic activation shift 破坏 safety boundary（model-agnostic，robust to direction rotation）；PGD 试图用 specific perturbation 替代，但丢失了 generic 的 robustness
4. **Teacher-forcing vs Auto-regressive Gap**：loss 在 teacher-forcing 下计算，ASR 在 auto-regressive generation 下评估，两种模式下 hidden state 分布不同

**理论 Insight（Optimization Paradox）**：**mechanistic understanding 给出了更精确的攻击目标（cone suppression），但将这个目标通过 CLIP ViT 转化为像素空间操作时，对梯度精度的要求反而比 proxy objective（最大化 affirmative token 概率）更高。精确几何目标 + 不精确梯度 → 比 proxy objective 更脆弱。**

---

## 3. 关键理论 Insight 整理

### Insight 1：Layer-wise Compensatory Amplitude Mechanism

**假说**（基于 Exp A，尚未因果验证）：

```
浅层（layer 0-15）：visual modality → 压制 refusal 幅度（norm_ratio < 1）
                    → 有利于 initial bypass
                    ↓
深层（layer 16-31）：visual modality → 放大 refusal 幅度（norm_ratio > 1）
                    → 补偿浅层的压制
                    → 随着 generation 深入，deep layer 的补偿放大效应累积
                    → 最终触发 Delayed Safety Reactivation (self-correction)
```

**如果这个假说成立的意义**：这提供了 blank image 效果的 mechanistic 完整解释——blank image 是把双刃剑，它在浅层帮助 bypass，在深层反过来强化 safety。要实现完整的 sequence-level suppression，需要同时抑制深层的补偿放大效应。

**验证方案**（Phase 3b 增加）：测量 blank image 条件下 generation 不同时间步的 layer-wise hidden state norm，与 self-correction 发生位置做 Pearson correlation。

---

### Insight 2：Dynamic Refusal Rotation 的 Subspace Switching 模型

**观测**（Exp B + 2A）：

- Uncontrolled min cos = 0.018（包含 content confound）
- Controlled min cos = 0.231（去 confound 后的真实旋转）
- 漂移模式非单调（subspace switching，非平滑 drift）

**理论模型**：VLM 的 safety mechanism 不是单一的 monolithic 机制，而是在 generation 的不同阶段激活不同的 "safety subspace"（类似 Gao et al. (2025) 发现的 dominant + non-dominant directions 系统）。不同 subspace 之间的转换是跳跃式的，转换时间点可能与生成内容的语义上下文相关。

**对攻击设计的 Implication**：

1. 静态 refusal direction 只能抑制当前激活的 subspace，无法覆盖整个 trajectory
2. 需要 **timestep-aware direction tracking**（动态方向追踪）才能实现序列级压制
3. 或者绕过方向追踪，改用 ablation-guided distillation（直接拟合 ablation 效果，不依赖方向估计精度）

---

### Insight 3：Layer 16 Narrow Waist

**观测**（Exp A + 2B）：

- Layer 16 的 cos(v_text, v_mm) = 0.918（最高，refusal signal 最纯粹）
- Layer 16 ablation（89.7%）> 全层 ablation（74.1%）

**理论解释**：Layer 16（深度 = 32/2 = 50% 相对位置）是 refusal signal 从浅层"混合区"过渡到深层"执行区"的汇聚点，此处 refusal direction 与 content direction 的 overlap 最小，是 safety mechanism 的功能性瓶颈。

**VLM-specific 贡献点**：LARF（EMNLP 2025）和 Gao et al.（2025）在纯 LLM 中发现类似现象，但 (a) 未在 VLM multimodal mode 下验证，(b) 未发现 norm_ratio 的层级反转现象，(c) 未量化 visual modality 与 ablation 的 super-additive interaction。

---

### Insight 4：Optimization Paradox

**核心 Claim**：在 VLM 的 pixel-to-hidden-state 攻击路径中，越精确的几何目标（hidden state space 的 cone suppression）比越粗糙的 proxy 目标（output token 的 affirmative probability）更脆弱，因为精确目标对梯度信噪比的要求更高，而 CLIP ViT 的梯度传递对下游 safety signal 是 bottleneck。

**对应文献**：Schlarmann & Hein（ICCV 2023, ICLR 2025）已发现 CLIP vision encoder 的梯度质量对 downstream task 不可靠，但未在 safety 语境和 hidden state geometric objective 下研究。

---

## 4. 与已有文献的精确定位

### 4.1 直接竞争文献（需要明确区分）


| 文献                   | 发表               | 与你的关系                            | 你的 Delta                                                                        |
| -------------------- | ---------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| Arditi et al.        | NeurIPS 2024     | 提供 mean-difference direction 方法  | 你在 VLM multimodal mode 下验证；发现 direction 稳定但幅度呈层级反转                              |
| Wollschläger et al.  | ICML 2025        | 提供 concept cone（多维）框架            | 你的 Exp B/2A 发现 cone 在 generation 中动态旋转——这是 Wollschläger 的静态 cone 框架无法解释的        |
| JailBound            | NeurIPS 2025     | Joint image+text attack，静态安全边界   | 你的发现说明静态边界是不完整的；你研究 dynamic geometry 和 delayed reactivation                     |
| LARF / Gao et al.    | EMNLP/arXiv 2025 | Narrow waist 在 LLM 中的发现          | 你在 VLM multimodal mode 下验证；额外发现 visual modality 的 layer-wise amplitude reversal |
| SafeProbing          | arXiv 2026       | Safety signal 在 generation 中持续存在 | 你研究的是 signal 的**几何方向**如何演化，而非 signal 是否存在——两者 orthogonal                        |
| SAPT/Safety Reminder | arXiv 2025       | 防御侧解决 delayed safety awareness   | 你是攻击侧的对应研究（如何永久阻止 safety reactivation）                                          |


### 4.2 Text+Image Joint Attack 文献（Phase 4 相关）


| 文献                                        | 方法                                            | 你的改进空间                                                            |
| ----------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------- |
| Jailbreak in Pieces (ICLR 2024 Spotlight) | CLIP embedding space，image+text compositional | 你针对 hidden state space，目标是 dynamic refusal suppression            |
| JailBound (NeurIPS 2025)                  | Fusion layer 静态边界，joint image+text            | 你增加 dynamic direction tracking，解决 delayed reactivation            |
| BAP (2024)                                | Image 扰动 + CoT text 优化                        | 你有 mechanistic grounding（layer 16 narrow waist），更 principled      |
| JPS (ACM MM 2025)                         | Visual perturbation + Textual Steering 协同     | 你的 text 侧专注于浅层 refusal 抑制，image 侧专注于深层补偿 suppression              |
| UltraBreak (ICLR 2026)                    | Semantic loss + transferability               | 你用 ablation-guided distillation 替代 cone suppression loss，更 robust |


---

## 5. Novelty 评估（更新版）


| #   | 发现/贡献                                                                              | 已有文献的空白                                                         | Novelty 等级                      | Paper 定位                     |
| --- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------- | ---------------------------- |
| 1   | Delayed Safety Reactivation 的精确量化（baseline 80% self-correction in VLM）             | SAPT 仅定性描述，未在 VLM 上量化                                           | **Moderate**（motivation figure） | Paper Section 2              |
| 2   | Refusal direction 在 VLM multimodal mode 下的模态稳定性（cos=0.918）                         | 未在 VLM 的 MM mode 下系统验证                                          | **Moderate**                    | Paper Section 3              |
| 3   | **Layer-wise Amplitude Reversal**（浅层压制→深层放大）                                       | **完全空白，无对应文献**                                                  | **Significant**                 | Paper Section 3 (核心 finding) |
| 4   | **Dynamic Refusal Rotation with Subspace Switching**（controlled min sim=0.231，非单调） | **完全空白：SafeProbing 不研究方向，Wollschläger 不研究动态**                   | **Significant**                 | Paper Section 4 (核心 finding) |
| 5   | Layer 16 as Narrow Waist in VLM（89.7% single-layer ablation）                       | LARF/Gao 在 LLM-only，未在 VLM multimodal mode + 未发现 super-additive | **Moderate-Significant**        | Paper Section 3              |
| 6   | Visual modality + Ablation 的 Super-additive Interaction（20.6pp 超额增益）               | 完全空白                                                            | **Moderate**                    | Paper Section 3              |
| 7   | **Optimization Paradox**（精确几何目标比 proxy 更脆弱）                                        | 无在 safety geometric attack 语境下的研究                               | **Significant**（需 Exp 2D 支撑）    | Paper Section 5              |
| 8   | Gradient Information Bottleneck 的量化（CLIP ViT → hidden state 路径）                    | Schlarmann 研究了 robustness，未量化 safety signal gradient            | **Moderate**                    | Paper Section 5              |


**顶会可行性判断**：Finding 3 + Finding 4 + Finding 7 构成三个 Significant 级别发现，配合 Finding 5 的 counter-intuitive 结果，整体构成一篇 **NeurIPS/ICLR mechanistic analysis paper 或 AAAI full paper 的核心内容**。但 **前提是跨模型验证（Phase 3）确认 generality**，否则所有 claim 只是单模型 anecdote。

---

## 6. Phase 3：跨模型验证 Vibe Coding 方案

### 6.0 总体设计原则

**目标**：验证 Finding A2（Layer-wise Amplitude Reversal）、Finding 2A-2（Dynamic Refusal Rotation）、Finding 2B-1（Layer 16 Narrow Waist）在 3 个不同 VLM 上的 generality。

**原则**：

1. 代码全部基于 Phase 1/2 的已有脚本直接 adapt，不重写
2. 每个实验独立可复现，结果存 JSON
3. 所有实验用相同的 12 对 harmful/harmless prompts（来自 Exp A）
4. 优先验证最核心的 3 个 Findings，不追求完整 ablation（Phase 4 补全）

**目标模型**：


| 模型                       | HuggingFace ID                      | 理由                                  | 预计显存  |
| ------------------------ | ----------------------------------- | ----------------------------------- | ----- |
| LLaVA-1.5-13B            | `llava-hf/llava-1.5-13b-hf`         | 同架构不同规模，验证 scale 效应                 | ~30GB |
| LLaVA-NeXT-7B (Mistral)  | `llava-hf/llava-v1.6-mistral-7b-hf` | 不同 LLM backbone + 更新 visual encoder | ~18GB |
| InstructBLIP-7B (Vicuna) | `Salesforce/instructblip-vicuna-7b` | 不同 VLM 架构族（Q-Former based）          | ~16GB |


---

### 6.1 Exp 3A：跨模型 Amplitude Reversal 验证

**目标**：验证 "浅层压制、深层放大" 的 layer-wise amplitude reversal 是否在所有目标模型上存在，以及 "narrow waist" 层（cos 最高）的相对位置是否收敛。

**实验配置**：

```python
# ============================================================
# Exp 3A: Cross-Model Amplitude Reversal Verification
# 直接修改自 exp_a_modality_stability.py
# ============================================================

MODELS = {
    "llava_13b": "llava-hf/llava-1.5-13b-hf",
    "llava_next_7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "instructblip_7b": "Salesforce/instructblip-vicuna-7b",
}

# 每个模型需要适配的参数
MODEL_CONFIG = {
    "llava_13b": {
        "total_layers": 40,              # LLaMA-13B 层数
        "probe_layers": [10, 16, 20, 28, 35],  # 相对比例约 25%,40%,50%,70%,87.5%
        "image_token_count": 576,        # 同 7B
        "get_hidden_states_fn": "llava_1x_hidden_states",  # 复用同一 fn
    },
    "llava_next_7b": {
        "total_layers": 32,
        "probe_layers": [8, 12, 16, 20, 28],
        "image_token_count": 2880,       # NeXT 使用 anyres，token 数量更多
        "get_hidden_states_fn": "llava_next_hidden_states",  # 需要适配
    },
    "instructblip_7b": {
        "total_layers": 32,
        "probe_layers": [8, 12, 16, 20, 28],
        "image_token_count": 32,         # Q-Former 输出 32 个 query tokens
        "get_hidden_states_fn": "instructblip_hidden_states",  # Q-Former 架构不同
    },
}

# 数据：复用 Exp A 的 12 对 prompts
PAIRED_PROMPTS = load_from("results/exp_a_directions.pt")["paired_prompts"]

# 图像：blank gray（与 Phase 1 完全相同）
BLANK_IMAGE = create_blank_gray(336, 336)  # LLaVA; InstructBLIP 可能需要 224

# 关键指标（每层）：
# - cos(v_text, v_mm)：方向稳定性
# - norm_text, norm_mm, norm_ratio：幅度行为
# - 标记 "narrow waist" 为 cos 最高的层

# 判断标准（关键）：
# H1（验证）：所有模型的 norm_ratio 是否在浅层 < 1，深层 > 1？
# H2（新发现）：narrow waist 层的相对位置（layer/total_layers）是否收敛到某个常数（如 ~50%）？
```

**架构适配注意事项**：

```python
# LLaVA-NeXT 适配：
# - anyres 模式：图像被分割为多个 tiles，生成 2880+ 个 visual tokens
# - hidden_states 偏移量需根据实际 tile 数动态计算
# - 推荐先用 processor.apply_chat_template() 获取 token 长度，再定位 text tokens

# InstructBLIP 适配：
# - Q-Former 架构：visual 信息通过 32 个 learnable query tokens 压缩后注入 LLM
# - hidden_states 中没有 576 个 visual patch tokens，而是 32 个 Q-Former output tokens
# - 从 model.language_model.model.layers[i] 提取 hidden states（Vicuna backbone）
# - 注意：InstructBLIP 的 text tokens 位置需要通过 tokenizer offset 定位

# 通用提示：
# - 对每个模型先跑 1 对 prompt 的 debug run，打印 hidden_states 的 shape
# - 确认 seq_len 与预期 token 数匹配后再跑全部 12 对
```

**期望结果与结论**：


| 结果                                   | 结论                                                      |
| ------------------------------------ | ------------------------------------------------------- |
| 3/3 模型都有 amplitude reversal（浅压制、深放大） | 这是 VLM 的 architectural regularity，与 visual encoder 类型无关 |
| 2/3 模型有 amplitude reversal           | 部分架构特异，需要分析哪个因素决定是否出现 reversal                          |
| 0/3 模型有（或 narrow waist 位置完全不同）       | LLaVA-1.5-7B 的 finding 是模型特异的，大幅降低 paper claim 强度       |
| Narrow waist 相对位置收敛到 ~50%            | **强力支持 architectural regularity claim**                 |


**输出文件**：`results/exp_3a_{model_name}_results.json`，格式：

```json
{
  "model": "llava_13b",
  "total_layers": 40,
  "layer_results": [
    {"layer": 10, "cos": 0.88, "norm_text": 5.2, "norm_mm": 4.1, "norm_ratio": 0.79},
    ...
  ],
  "narrow_waist_layer": 20,
  "narrow_waist_relative_depth": 0.50,
  "amplitude_reversal_exists": true,
  "reversal_crossover_layer": 12
}
```

**预计 GPU 时间**：~1h per model（inference only，no gradient）

---

### 6.2 Exp 3B：跨模型 Dynamic Refusal Rotation 验证

**目标**：验证 "refusal direction 在 generation 中存在真实旋转（controlled min sim < 0.40）" 是否在所有目标模型上成立，以及旋转模式（non-monotone subspace switching）是否一致。

**实验配置（直接 adapt 自 exp_2a_confound_resolution.py）**：

```python
# ============================================================
# Exp 3B: Cross-Model Dynamic Refusal Rotation Verification
# 修改自 exp_2a_confound_resolution.py（使用 controlled 版本）
# ============================================================

# 固定前缀（与 Exp 2A 完全相同，确保 apple-to-apple 比较）
FIXED_PREFIX = "Sure, here is a detailed explanation. First, let me provide some background information. Then, I will describe the step-by-step process. Let me begin by"  # ~30 tokens

# 提取位置（前缀内的 positions）
PREFIX_POSITIONS = [5, 10, 15, 20, 25]  # 模拟 generation 深度

# 关键参数（每个模型需要适配）：
# - hidden_seq_len 的计算方式（LLaVA NeXT 的 tile 数不同）
# - prefix_start_in_hidden = hidden_seq_len - prefix_token_count
# - 从序列末端定位前缀 token（与 Exp 2A 完全相同的逻辑）

# 计算 pairwise cosine similarity matrix（5x5）
# 每个 position 提取 mean-difference direction，然后计算所有对的 cos

# 判断标准：
# > 0.80：静态（no real drift）
# 0.40-0.80：部分漂移
# < 0.40：真实动态旋转（与 Exp 2A 的 controlled min sim=0.231 一致为 DYNAMIC）

# 额外分析：non-monotone test
# 检验 cos(t3, t4) < cos(t1, t5) 是否成立（non-monotone 的 signature）
```

**关键控制变量**：

- 使用相同的 12 对 harmful/harmless prompts（跨模型 apple-to-apple）
- 使用相同的 fixed prefix（控制 content difference）
- Layer 选择：每个模型使用其 narrow waist 层（由 Exp 3A 确定）

**期望结果与结论**：


| 结果                               | 结论                                         |
| -------------------------------- | ------------------------------------------ |
| 3/3 模型 controlled min sim < 0.40 | Dynamic rotation 是 VLM 的通用特征，非 LLaVA-7B 特异 |
| 3/3 模型都有 non-monotone pattern    | Subspace switching 是通用机制                   |
| min sim 在不同模型间差异大                | Rotation 幅度可能与 safety alignment 强度相关（额外发现） |


**输出文件**：`results/exp_3b_{model_name}_results.json`

**预计 GPU 时间**：~30min per model（teacher-forced，inference only）

---

### 6.3 Exp 3C：跨模型 Layer Narrow Waist + Ablation 验证

**目标**：验证 (a) narrow waist 层在不同模型上的 ablation 效果，(b) layer 16 的特殊地位是否是 LLaVA 特异的，还是所有 VLM 都在 "相对深度 ~50%" 有类似现象。

**实验配置（adapt 自 exp_2b_ablation_attack.py）**：

```python
# ============================================================
# Exp 3C: Cross-Model Narrow Waist Ablation Verification
# 修改自 exp_2b_ablation_attack.py
# ============================================================

# 每个模型需要测试的 ablation 配置
ABLATION_CONFIGS = {
    "baseline_mm": [],                   # no ablation, with blank image
    "ablation_narrow_waist": [exp_3a_narrow_waist_layer],  # Exp 3A 确定的最优层
    "ablation_relative_50pct": [int(total_layers * 0.50)],  # 固定相对 50% 深度
    "ablation_all": list(range(total_layers)),
    "ablation_shallow": list(range(total_layers // 2)),
    "ablation_deep": list(range(total_layers // 2, total_layers)),
}

# 评估数据：Exp C 的 8 个 test prompts（直接复用）
# + SaladBench harmful_test 前 20 条（轻量验证）

# 关键指标：
# - Full harmful completion rate
# - Self-correction rate
# - Initial bypass rate

# 判断标准：
# narrow_waist ablation > all_layers ablation：确认 narrow waist 现象
# relative_50pct ablation ≈ narrow_waist ablation：narrow waist 是 ~50% depth 的 universal property

# LLM-judge：用 gpt-4o-mini 或 llama-3-8b-instruct 替代 keyword matching
# 原因：8-20 条 prompts，eval 成本低；消除 keyword matching 的 false positive 问题
```

**架构适配注意事项**：

```python
# LLaVA-NeXT 的 ablation hook 适配：
# model.language_model.model.layers[i]  # 与 LLaVA-1.5 相同

# InstructBLIP 的 ablation hook 适配：
# model.language_model.model.layers[i]  # Vicuna backbone，结构相同
# 注意：Q-Former 的 query tokens 不在 LLM 的 hidden states 中，不需要 ablation

# 通用注意：
# ablation direction 需要用各自模型的 Exp 3A 提取的 v_mm（不能跨模型复用）
# 确保 ablation 在 generation loop 中实时应用（pre_hook，不是 post_hook）
```

**输出文件**：`results/exp_3c_{model_name}_results.json`

**预计 GPU 时间**：~2h per model（包含 generation）

---

### 6.4 实验依赖关系与执行顺序

```
Exp 3A（Amplitude Reversal, ~1h/model）
  ↓ 输出各模型的 narrow_waist_layer
  ├──→ Exp 3B（Dynamic Rotation, ~30min/model, 可并行）
  └──→ Exp 3C（Ablation, ~2h/model, 依赖 3A 的 narrow waist layer）

总时间：(1h + 0.5h + 2h) × 3 models = ~10.5h GPU
GPU 分配：GPU0: llava_13b, GPU1: llava_next_7b, GPU2: instructblip_7b
并行 3A + 3B：总时间压缩到 ~(1h + 2h) × 1 = ~3h
```

**并行执行脚本**：

```bash
#!/bin/bash
# run_phase3_parallel.sh

# Exp 3A 三个模型并行
CUDA_VISIBLE_DEVICES=0 python experiments/phase3/exp_3a/exp_3a_amplitude_reversal.py \
    --model llava_13b &

CUDA_VISIBLE_DEVICES=1 python experiments/phase3/exp_3a/exp_3a_amplitude_reversal.py \
    --model llava_next_7b &

CUDA_VISIBLE_DEVICES=2 python experiments/phase3/exp_3a/exp_3a_amplitude_reversal.py \
    --model instructblip_7b &

wait
echo "Exp 3A complete. Starting 3B + 3C..."

# 3A 完成后，3B 和 3C 可以并行（不同 GPU）
# 3B 在 GPU0 (最快)
CUDA_VISIBLE_DEVICES=0 python experiments/phase3/exp_3b/exp_3b_dynamic_rotation.py \
    --models all &

# 3C 需要 generation，用 GPU1+2+3
CUDA_VISIBLE_DEVICES=1 python experiments/phase3/exp_3c/exp_3c_ablation.py \
    --model llava_13b &
CUDA_VISIBLE_DEVICES=2 python experiments/phase3/exp_3c/exp_3c_ablation.py \
    --model llava_next_7b &
CUDA_VISIBLE_DEVICES=3 python experiments/phase3/exp_3c/exp_3c_ablation.py \
    --model instructblip_7b &

wait
echo "Phase 3 complete."
```

---

### 6.5 Exp 3D（补充）：Amplitude Reversal 与 Self-Correction 的因果验证

**目标**：验证 Insight 1 的假说——深层 amplitude amplification 是 delayed self-correction 的 mechanistic cause。

**设计**：

```python
# 关键对比实验：
# 条件 A：blank_image + 只 ablate 浅层（0 to narrow_waist-1）
#   → 浅层的 visual modality 压制 refusal 被进一步增强
#   → 深层的 amplitude amplification 保持原样
#   → 预期：高 initial bypass，高 self-correction rate（因为深层补偿保留）

# 条件 B：blank_image + 只 ablate narrow_waist 层
#   → 这是 Exp 2B 的最优配置（89.7%）
#   → 深层的 narrow waist 被精确消除
#   → 预期：高 initial bypass，低 self-correction rate

# 关键比较：
# 如果条件 A 的 self-correction rate 显著高于条件 B，
# 且两者的 initial bypass rate 相近，
# 则深层的 amplitude amplification（narrow waist 处）是 self-correction 的 causal driver

# 额外分析：
# 测量 blank_image 下，在 self-correction 发生前后的时间步 t，
# layer 12（浅层）vs layer 16（narrow waist）的 refusal signal 幅度变化
# 如果 self-correction 发生时 layer 16 的幅度突然升高 → 直接因果证据
```

**预计时间**：~1h GPU（inference + generation）

---

### 6.6 Phase 3 决策矩阵


| Exp 3A-3C 结果                                                | 决策                                                                                     |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 3/3 模型 amplitude reversal + dynamic rotation + narrow waist | **强 generality claim，直接进入 Phase 4 攻击设计**                                               |
| 2/3 模型成立                                                    | 分析差异因素（architecture? training data? alignment strength?），作为 paper 的 nuance             |
| 1/3 模型成立                                                    | LLaVA-1.5 特异性太强，需要更多模型或 pivot to case study                                            |
| 0/3 模型成立                                                    | Kill the generality claim，paper 降格为 "LLaVA-specific mechanistic study"（仍有价值但 venue 降低） |


---

## 7. 后续攻击设计路线图（Phase 4）

### 7.1 核心方法：Ablation-Guided Distillation Attack（AGDA）

**动机**：Exp 2C 失败的根本原因是 proxy objective misalignment + gradient bottleneck。Ablation attack（Exp 2B）提供了完美的 "oracle"——89.7% full harmful rate，8.6% self-correction。**如果我们能让 pixel perturbation 的 hidden state 效果接近 ablation 的效果，就绕过了 cone direction estimation 的不确定性，也降低了对梯度精度的要求。**

**方法**：

```
Step 1: 对每个 training prompt，运行 ablation_mm_layer16
         → 保存所有 time steps 的 layer 16 hidden states 为 target: h*_{t,16}

Step 2: 优化 pixel perturbation delta:
         Loss = (1/T) Σ_t MSE(h_{t,16}^{perturbed}, h_{t,16}^{ablated})
              + λ * TV(delta)          ← 平滑正则化
              + μ * ||delta||_{2}      ← 幅度惩罚

Step 3: PGD 在像素空间优化 delta（初始化为 blank gray image）
         epsilon = 16/255, steps = 200
         但目标是拟合 ablation 的效果，而非最小化 cone projection

Step 4: 多代理模型迁移性增强（Phase 4b）
         在 LLaVA-7B + LLaVA-13B + LLaVA-NeXT 上同时优化
         加入 input diversity transforms（resize, crop, color jitter）
```

**理论优势**：

1. Loss 是 MSE in hidden state space，梯度方向清晰（拟合目标，非最小化投影）
2. 不依赖 cone direction 的精确估计（ablation 效果是 oracle，不需要知道 direction）
3. Teacher-forcing gap 仍然存在，但 distillation target 比 cone projection 更 robust

### 7.2 Text Modality 的角色设计

**设计原则**：职责分工，modality 协同，各司其职。

```
Image modality（已验证）：
  → 利用 generic activation shift（blank image baseline）
  → 针对 narrow waist 层的 refusal signal（AGDA delta）
  → 负责 sequence-level 的持续 suppression

Text modality（Phase 4b 新增）：
  → 针对浅层（layer 0 to narrow_waist-1）的 refusal direction
  → 方法：adversarial suffix（GCG-style，但目标是浅层 hidden state）
  → 负责 initial bypass 的强化
  → 不携带 explicit harmful content（避免 content filter）

联合优化：
  Loss_joint = Loss_AGDA(delta) + α * Loss_suffix(text_suffix)
  其中 Loss_suffix = Σ_{l ∈ shallow_layers} (h_{t,l} · r̂_l)^2  ← 浅层 direction 抑制
```

**与 JailBound 的精确区分**：JailBound 在 fusion layer 优化 static boundary；你在 **dynamic generation trajectory** 上优化，针对 **layer-specific** 的目标（浅层 text suppression + 深层 image suppression），且使用 **ablation-guided distillation** 而非 geometric boundary crossing。

### 7.3 AAAI 2026 完整实验清单

```
必须实验：
□ Phase 3A-3C：3 模型跨模型验证（基础）
□ Phase 4a：AGDA 白盒攻击，LLaVA 家族
   - 指标：full harmful rate, self-correction rate, ASR on HarmBench/SaladBench
   - 对比 baseline：blank image / GCG / JailBound / Image Hijacks
□ Phase 4b：Multi-surrogate 迁移攻击
   - 目标：LLaVA-NeXT-7B（未见过的模型）
   - 黑盒迁移 ASR 目标：> 50%
□ Ablation study：
   - AGDA vs PGD（cone suppression） ← 解释 Exp 2C 失败
   - Image-only vs Image+Text joint
   - Single-layer (narrow waist) vs All-layers
   - Static direction vs Dynamic direction
□ LLM-based evaluation（GPT-4o-mini 或 HarmBench classifier）

建议实验（提升 paper strength）：
□ Phase 3D：Amplitude reversal 的因果验证
□ Exp 2D：PGD hyperparameter sweep（排除 engineering failure）
□ Amplitude reversal 的 layer-wise norm trajectory 可视化（Figure 1 候选）
```

---

## 8. AAAI 2026 Paper 框架草案

**Tentative Title**：
*"Dynamic Safety Geometry in Vision-Language Models: From Layer-wise Amplitude Reversal to Sequence-level Jailbreak"*

**Abstract 关键词**：
dynamic refusal rotation, layer-wise amplitude reversal, narrow waist, sequence-level jailbreak, delayed safety reactivation, ablation-guided distillation, vision-language model safety

**Paper 结构**：

```
Section 1: Introduction
  - VLM jailbreak 现状：现有攻击只解决 initial refusal，忽略 delayed reactivation
  - Motivation experiment：baseline 80% self-correction（Figure 1）
  - Our contributions：3 mechanistic findings + 1 attack method

Section 2: Background & Related Work
  - Refusal direction/cone：Arditi et al. (2024), Wollschläger et al. (2025)
  - Shallow alignment：Qi et al. (2025)
  - Delayed safety awareness：SAPT (2025), SafeProbing (2026)
  - VLM safety：Safety Paradox (2025), JailBound (2025)

Section 3: Mechanistic Analysis
  3.1 Layer-wise Amplitude Reversal（Finding A2 + cross-model validation）
  3.2 Narrow Waist in VLMs（Finding 2B-1 + cross-model validation）
  3.3 Super-additive Visual-Ablation Interaction（Finding 2B-2）

Section 4: Dynamic Refusal Rotation
  4.1 Content-controlled experiment design（Exp 2A methodology）
  4.2 Evidence for true direction rotation（controlled min sim < 0.40）
  4.3 Subspace switching pattern（non-monotone analysis）
  4.4 Implications for sequence-level attack design

Section 5: Ablation-Guided Distillation Attack (AGDA)
  5.1 Optimization Paradox（Exp 2C negative result）
  5.2 AGDA design（from oracle ablation to pixel perturbation）
  5.3 Text+Image joint attack（modality role assignment）
  5.4 Results：ASR, self-correction rate, transferability

Section 6: Ablation Study
  - AGDA vs PGD; image-only vs joint; narrow waist vs all layers

Section 7: Conclusion & Discussion
  - Implications for defense design（protect the narrow waist）
  - Limitations（single VLM family, white-box setting）
```

---

## 9. 关键文献速查表

### 必读（实验直接相关）


| 文献                                                                            | 发表                    | arXiv      | 核心贡献                                           |
| ----------------------------------------------------------------------------- | --------------------- | ---------- | ---------------------------------------------- |
| Arditi et al., "Refusal in LLMs is Mediated by a Single Direction"            | NeurIPS 2024          | 2406.11717 | Mean-difference direction 方法                   |
| Wollschläger et al., "Geometry of Refusal in LLMs: Concept Cones"             | ICML 2025             | —          | Refusal cone（多维），RDO 方法                        |
| Qi et al., "Safety Alignment Should Be Made More Than Just a Few Tokens Deep" | ICLR 2025 Outstanding | —          | Shallow alignment；Gap C 的入口                    |
| "VLLM Safety Paradox"                                                         | NeurIPS 2025          | —          | Visual modality 对 safety 的结构性破坏                |
| SafeProbing                                                                   | arXiv 2026            | —          | Safety signal 在 generation 中持续存在               |
| The Safety Reminder (SAPT)                                                    | arXiv 2025            | —          | Delayed safety awareness；防御侧的 Gap C            |
| DSN                                                                           | ACL 2025 Findings     | —          | 直接抑制 refusal direction > 最大化 affirmative token |


### 竞争文献（需精确区分）


| 文献                            | 发表                  | 与你的区别                                                            |
| ----------------------------- | ------------------- | ---------------------------------------------------------------- |
| JailBound                     | NeurIPS 2025        | 静态边界 + joint 优化；你是动态几何 + sequence-level                          |
| UltraBreak                    | ICLR 2026           | Output token proxy + transferability；你是hidden state distillation |
| Jailbreak in Pieces           | ICLR 2024 Spotlight | CLIP embedding space；你是 LLM hidden state space                   |
| LARF                          | EMNLP 2025          | LLM-only narrow waist；你在 VLM + visual modality 效应                |
| Gao et al., Hidden Dimensions | arXiv 2025          | LLM-only dominant direction；你在 VLM + layer-wise reversal         |
| Image Hijacks (Bailey et al.) | NeurIPS 2023        | VLM UAP 先驱；baseline + 参考代码                                       |
| BAP                           | 2024                | Image扰动+CoT text；你有 mechanistic grounding（narrow waist）          |


### 评估 Benchmark


| 名称                                    | 用途                                        |
| ------------------------------------- | ----------------------------------------- |
| HarmBench (Mazeika et al., ICML 2024) | 标准攻击评估，**必须使用**                           |
| SaladBench                            | 当前 pilot 使用中，harmful_train + harmful_test |
| MM-SafetyBench                        | 多模态 safety 评估                             |


---

*文档结束。Phase 3 实验执行优先级：Exp 3A（最优先）→ 并行 Exp 3B + 3C → Exp 3D（if time permits）。*
*Phase 4 开始前提：Phase 3 至少有 2/3 模型的跨模型验证支持 generality claim。*