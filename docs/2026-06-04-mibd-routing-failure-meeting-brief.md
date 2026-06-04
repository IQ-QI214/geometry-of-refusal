# MIBD 研究方向汇报：从 Modality Drift 到 Evidence-to-Policy Routing Failure

日期：2026-06-04  
目标：确定后续主线是否固定为 **VLM 安全中的动态风险证据到行为策略路由失败**，并明确它与 ReGap、VLM-Guard、TGA 等工作的本质区别。

---

## 1. 一句话结论

我们不应继续把问题表述为普通的 **modality gap** 或 **refusal direction transfer failure**。这些已经被 ReGap / VLM-Guard / TGA 等工作覆盖。更清晰、也更有顶会潜力的问题是：

> **VLM 中的风险证据可能已经在 hidden states 中可读，但视觉载体会改变其编码位置和方向；现有固定方向、固定层或全局 drift correction 方法无法保证这些动态风险证据被因果地路由到下游安全行为策略。**

因此，本文应研究：

> **Causal Routing Failure between Dynamic Safety Evidence and Stable Behavioral Gates in Vision-Language Models.**

---

## 2. 与 ReGap 的本质区别

ReGap 的核心对象是 **modality-induced drift**。它研究文本安全能力迁移到多模态输入时，沿 text-aligned refusal direction 的可分性被压缩，从而出现 **Safety Geometry Collapse**。其解决方式是估计并抵消 drift，使 refusal separability 恢复。

我们的核心对象不是 drift，而是 **risk evidence 到 safe behavior 的因果路由**。这一区别不是表述差异，而是 failure type、causal object 和 intervention target 都不同。

| 维度 | ReGap | 我们的方向 |
|---|---|---|
| 研究对象 | text-aligned refusal direction 与 modality-induced drift | safety sensor geometry 与 behavioral gate geometry 的因果关系 |
| 失败类型 | multimodal input 压缩 refusal separability | risk evidence 可读，但没有驱动正确安全策略 |
| 默认目标 | 恢复文本 refusal direction 上的安全可分性 | 将动态、多位置的风险证据接入稳定行为 gate |
| 表征假设 | 存在一个 text-aligned refusal axis，可通过 drift correction 恢复 | 不要求视觉风险编码与文本方向一致，允许 risk evidence 迁移到不同 layer / position / direction |
| 干预对象 | counteract estimated modality drift | bridge sensor evidence into behavioral gate residual |
| 关键证据 | drift 越强，refusal separability 越弱，ASR 越高 | risk probe AUC 高，但行为错误；patch/bridge 到 gate 后行为被修复 |
| 如果风险不可读 | 属于 perception failure，不是我们的问题 | 同左 |
| 如果风险可读且 drift correction 即可修复 | ReGap 充分 | 我们需证明存在 drift correction 不能覆盖的 sensor-gate routing gap |

**给导师的关键表达：**

> ReGap asks whether multimodal inputs move representations away from a text refusal geometry. We ask whether the model can causally use visually encoded risk evidence to activate the downstream safe-policy gate. A drift can exist without routing failure; routing failure can exist even when risk evidence is linearly decodable. Therefore, the causal unit of analysis is different.

更形式化地说，ReGap 关心：

$$
\text{separability}_{d_\text{text-refusal}}(x) \downarrow
\quad \Rightarrow \quad
\text{safety failure}
$$

我们关心：

$$
\exists s_{\ell,p}(x) \text{ such that risk is decodable, but }
B(x) \neq \text{safe-policy}
$$

并进一步要求：

$$
\operatorname{Bridge}(s_{\ell,p}(x) \rightarrow h_g)
\Rightarrow
B(x) = \text{safe-policy}
$$

如果最后我们只做「动态 drift correction」，会与 ReGap 撞车。必须做 **sensor-gate dissociation + causal bridge**。

---

## 3. 研究问题

### 3.1 问题定义

给定一个 VLM $M$，输入为多模态请求：

$$
x = (q, v)
$$

其中 $q$ 是文本请求，$v$ 是视觉输入。模型在每个 layer-position locus $(\ell,p)$ 产生 hidden state：

$$
h_{\ell,p}(x)
$$

我们区分两个对象：

1. **Safety Sensor Geometry**：哪些 hidden-state locus 可以读出输入是否含有风险证据。
2. **Behavioral Gate Geometry**：干预哪些 hidden-state locus 会因果改变模型的安全策略行为。

风险传感器定义为：

$$
s_{\ell,p}(x) = w_{\ell,p}^{\top} h_{\ell,p}(x)
$$

行为 gate 定义为对安全策略 margin 具有最大因果影响的 locus：

$$
G_{\ell,p}
=
\Delta\left[
\log p(y^{safe-policy}) -
\log p(y^{unsafe})
\right]
$$

### 3.2 核心假设

本文要验证的不是「VLM 不安全」，而是更强的机制假设：

> **存在一类多模态安全失败样本，其中风险证据在 hidden states 中可读，但该证据没有被稳定路由到下游安全行为 gate。**

形式化为：

$$
\exists x:
\quad s(x) > \tau,
\quad B(x) \in \{\text{unsafe-compliance}, \text{wrong-helpful}\},
\quad \operatorname{Bridge}(s(x), h_g) \rightarrow \text{safe-policy}
$$

同时要求 benign counterpart 不被破坏：

$$
\Delta \text{OverRefusal}_{benign} \leq \epsilon,
\quad
\Delta \text{Degeneration} \leq 0
$$

---

## 4. Motivation

### 4.1 已有方法隐含的两个操作性假设

不能说所有已有工作「显式假设」风险识别必然驱动安全行为。更准确的说法是：

> 现有 VLM safety 方法大多把安全失败建模为 output-level alignment failure、modality gap 或 fixed safety geometry failure；它们通常不显式区分「风险证据是否被内部编码」和「该证据是否被因果用于生成安全行为」。

典型文献：

| 工作 | 核心做法 | 与我们问题的关系 |
|---|---|---|
| Arditi et al., 2024 | 在 LLM 中发现 single refusal direction，擦除该方向可阻止 refusal，加入该方向可诱导 refusal | 提供 fixed direction intervention 基线，但对象是 LLM refusal，不处理视觉条件下 sensor relocation |
| VLM-Guard, 2025 | 从 safety-aligned LLM 提取 safety steering direction，并将 VLM 表示投影到该方向的正交子空间 | 典型 fixed safety direction / orthogonal projection 方法 |
| TGA, ICLR 2025 | 用 text-guided hidden-state alignment 促进视觉安全机制迁移 | 仍以 text hidden space 作为对齐目标，不区分 sensor 与 gate |
| ReGap, 2026 | 估计并修正 modality-induced drift，恢复 refusal separability | 最接近工作，但目标是 drift correction，不是 sensor-gate routing |
| SPA-VL, 2025 | 构造大规模 VLM safety preference data，通过偏好对齐提高 harmlessness/helpfulness | output-level alignment，不解释风险证据是否被内部使用 |
| FigStep, 2025 | 将有害文本转为 typography image，展示 VLM 对视觉有害输入脆弱 | 支撑视觉载体会改变 safety behavior，但没有机制性定位 sensor-gate failure |
| MM-SafetyBench, 2024 | 系统评估 MLLM 在多模态有害输入下的安全性 | 提供 benchmark 支撑，但不是机制论文 |

参考文献链接：

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*: https://arxiv.org/abs/2406.11717
- VLM-Guard: https://arxiv.org/abs/2502.10486
- ReGap: https://arxiv.org/abs/2605.18104
- FigStep: https://arxiv.org/abs/2311.05608
- MM-SafetyBench: https://arxiv.org/abs/2311.17600
- SPA-VL: https://arxiv.org/abs/2406.12030

### 4.2 第一性原理路径

VLM safety failure 至少可以分成三类：

1. **Perception failure**：模型没有编码风险证据。
2. **Representation relocation**：模型编码了风险证据，但编码位置 / 方向随视觉载体变化。
3. **Routing failure**：风险证据可读，但没有驱动正确的安全行为策略。

已有 benchmark 主要证明第 1 或第 2 类可能存在；已有 alignment 方法主要直接优化最终输出。我们的切入点是第 3 类：

> 如果风险证据已经存在，那么继续做普通 SFT / DPO 不是最机制化的方案。更直接的问题是：为什么已有证据没有进入安全行为 gate？

这也是 OPD / 正交子空间消融可以自然接上的原因：它们证明 fixed geometry intervention 不稳定，但不应继续被作为最终方法，而应作为 baseline，说明「单一方向消融」不足以解决动态 risk evidence routing。

---

## 5. Challenge

### 5.1 为什么传统方法难解决

**Output-level alignment** 的问题：

- 它优化最终 response preference。
- 但不能保证模型内部使用了视觉风险证据。
- 如果训练集里 attack pattern 固定，模型可能学到 surface shortcut，而不是 evidence-to-policy mapping。

**Fixed direction / fixed subspace intervention** 的问题：

- 它假设存在稳定的 safety/refusal direction。
- 但我们的结果显示 FigStep 会激活极早层、几乎正交的 harmfulness direction。
- 如果 direction/locus 随视觉载体迁移，固定干预要么无效，要么造成 over-refusal / degeneration。

**Global modality alignment / drift correction** 的问题：

- 它处理平均 drift 或 text-refusal separability。
- 但 routing failure 是样本级、输入依赖、多 locus 的问题。
- 把视觉表示强行拉回文本方向，可能掩盖视觉证据本身的动态编码结构。

### 5.2 当前 pilot 是否触达核心 challenge

当前 pilot 已触达一半：

- 已证明 risk evidence 在多个视觉条件下高度可解码。
- 已证明 FigStep 的 locus / direction 明显迁移。
- 已证明 fixed direction intervention 在 VLM 上不稳定。

但尚未完成最关键的因果闭环：

- 还没有 condition-specific behavior labels。
- 还没有识别 behavioral gate。
- 还没有证明 bridge sensor evidence 到 gate 后可以修复 behavior。

因此，当前阶段只能声称：

> We have evidence for dynamic safety evidence relocation and fixed-geometry failure.

还不能声称：

> We have proven routing failure.

Routing failure 必须通过 Phase 2 的 causal patching / oracle bridge 证明。

---

## 6. 已有实验结果与其支持的结论

### 6.1 LLM refusal geometry 复现：fixed geometry 在 LLM 上有效

| 模型 | 方法 | ASR_kw | ASR_LG3 | SRR | n | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B | DIM ablation | 100.0% | 94.5% | +5.5pp | 128 | LLM 上 single direction 消融强有效 |
| Qwen2.5-7B | Cone k=3 | 98.4% | 93.8% | +4.7pp | 128 | 多维 cone 保持高 ASR，SRR 接近评估器底线 |
| Qwen2.5-7B | RDO k=1 | 90.6% | 76.6% | +14.1pp | 128 | 单维 RDO 对 Qwen 不充分 |
| Llama3.1-8B | DIM ablation | 98.4% | 89.8% | +8.6pp | 128 | LLM 上 fixed direction 仍然有效 |
| Llama3.1-8B | Cone k=3 | 100.0% | 90.6% | +9.4pp | 128 | LLM 语义 ASR 高，SRR 较低 |

支持结论：

> 在纯 LLM 条件下，refusal geometry 的固定方向 / 低维子空间干预是有效且可复现的。因此，VLM 中的失败不是因为我们的 direction pipeline 本身无效。

### 6.2 PCD：LLM-to-VLM refusal direction transfer 失败

| 条件 | Best layer | ASR_kw | ASR_LG3 | Arditi | cos vs Qwen L | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Qwen L baseline | 17 | 1.000 | 0.945 | — | 1.000 | 纯文本 LLM 消融强有效 |
| Qwen V-text | 17 | 0.977 | 0.570 | 0.570 | 0.671 | VL 对齐后方向旋转，语义 ASR 下降 37.5pp |
| Qwen V-blank | 15 | 0.914 | 0.422 | 0.414 | 0.492 | 视觉 token 存在进一步降低效果 |
| Qwen V-noise | 16 | 1.000 | 0.508 | 0.508 | 0.545 | noise 与 blank 类似，内容不是主要因素 |
| Gemma V-text | 29 | 0.969 | 0.102 | 0.102 | — | keyword 绕过但语义安全仍强 |
| Gemma V-blank | 1 | 0.898 | 0.016 | 0.016 | — | 最优层从深层跳到极早层 |
| Gemma V-noise | 1 | 0.898 | 0.008 | 0.008 | — | blank/noise 几乎等价 |

支持结论：

1. **VL alignment shift**：Qwen V-text 与 LLM direction cosine 仅 0.671，且 ASR_LG3 从 94.5% 降至 57.0%。
2. **Visual token presence effect**：Qwen V-text 到 V-blank 额外下降约 15pp。
3. **Content is not the main driver**：Qwen V-blank vs V-noise cosine = 0.893；Gemma V-blank vs V-noise cosine = 0.996。
4. **Fixed intervention 不稳定**：Gemma 在 V-blank/V-noise 下最优层跳到 layer 1，说明固定层/固定方向假设失效。

### 6.3 MIBD Phase 1：risk evidence 可读，但 FigStep 编码路径迁移

| 模型 | 条件 | Best layer | Token pos | AUC | 结论 |
|---|---:|---:|---:|---:|---|
| InternVL3-8B | V-text | 6 | -3 | 1.000 | 标准视觉条件下 harmfulness 可读 |
| InternVL3-8B | V-blank | 7 | -3 | 1.000 | blank 条件可读 |
| InternVL3-8B | V-noise | 7 | -3 | 1.000 | noise 条件可读 |
| InternVL3-8B | V-real | 7 | -3 | 1.000 | real image 条件可读 |
| InternVL3-8B | FigStep | 0 | -1 | 1.000 | FigStep 在极早层可读 |
| Qwen3-VL-8B | V-text | 15 | -1 | 1.000 | 标准条件下 harmfulness 可读 |
| Qwen3-VL-8B | V-blank | 17 | -1 | 1.000 | blank 条件可读 |
| Qwen3-VL-8B | V-noise | 16 | -1 | 1.000 | noise 条件可读 |
| Qwen3-VL-8B | V-real | 17 | -1 | 1.000 | real image 条件可读 |
| Qwen3-VL-8B | FigStep | 1 | -1 | 1.000 | FigStep 在极早层可读 |

支持结论：

> FigStep 不是简单地让模型无法编码 harmfulness。相反，harmfulness 仍然高度可读，但其最佳 locus 从中层/深层迁移到 layer 0/1。

### 6.4 Static transfer 与方向余弦：FigStep 与标准条件使用不同安全证据方向

| 模型 | V-text → FigStep static transfer AUC | FigStep vs V-text cosine | FigStep best layer | 判断 |
|---|---:|---:|---:|---|
| InternVL3-8B | 0.741 | 0.136 | 0 | 明显 FigStep-divergent |
| Qwen3-VL-8B | 0.956 | 0.259 | 1 | transfer 仍高，但方向明显不同 |

补充方向余弦：

| 模型 | 对比 | Cosine |
|---|---|---:|
| InternVL3-8B | V-blank vs V-real | 1.000 |
| InternVL3-8B | V-blank vs V-noise | 0.983 |
| InternVL3-8B | FigStep vs V-text | 0.136 |
| InternVL3-8B | FigStep vs V-noise | 0.120 |
| Qwen3-VL-8B | V-blank vs V-real | 1.000 |
| Qwen3-VL-8B | V-blank vs V-noise | 0.981 |
| Qwen3-VL-8B | FigStep vs V-text | 0.259 |
| Qwen3-VL-8B | FigStep vs V-real | 0.124 |

支持结论：

> 普通视觉条件之间方向高度一致；FigStep 则激活明显不同的 harmfulness direction。这说明「风险证据」不是消失，而是视觉载体依赖地迁移。

### 6.5 Phase 1.5 审计：probe 有效性初步通过

| 模型 | 条件 | Held-out AUC | Train-selected held-out AUC | Permutation mean | Verdict |
|---|---|---:|---:|---:|---|
| InternVL3-8B | V-text | 1.000 | 0.994 | 0.475 ± 0.186 | PASS |
| InternVL3-8B | V-blank | 1.000 | 1.000 | 0.472 ± 0.173 | PASS |
| InternVL3-8B | V-noise | 0.982 | 1.000 | 0.486 ± 0.184 | PASS |
| InternVL3-8B | V-real | 0.994 | 1.000 | 0.474 ± 0.167 | PASS |
| InternVL3-8B | FigStep | 1.000 | 1.000 | 0.484 ± 0.224 | PASS |
| Qwen3-VL-8B | V-text | 1.000 | 1.000 | 0.517 ± 0.171 | PASS |
| Qwen3-VL-8B | V-blank | 0.994 | 1.000 | 0.514 ± 0.167 | PASS |
| Qwen3-VL-8B | V-noise | 1.000 | 1.000 | 0.505 ± 0.176 | PASS |
| Qwen3-VL-8B | V-real | 1.000 | 1.000 | 0.491 ± 0.164 | PASS |
| Qwen3-VL-8B | FigStep | 1.000 | 1.000 | 0.515 ± 0.204 | PASS |

局限：

- 当前没有 paired_id，因此 group-split AUC 缺失。
- harmless 样本类别结构不足，因此 cross-category AUC 缺失。
- 这意味着 probe 有效性已初步成立，但还不能达到论文最终标准。

---

## 7. Solution Sketch

### 7.1 不是普通 SFT / DPO

我们不应把方法定位成「再做一个 safety fine-tuning」。原因：

1. SFT/DPO 优化的是输出，不保证模型使用风险证据。
2. 如果数据分布固定，模型可能学到 attack-format shortcut。
3. 我们已有结果说明问题可能发生在 hidden evidence 到 behavior 的中间路由，而不是简单标签不足。

### 7.2 MIBD Bridge

方法目标：

> 从多个候选 sensor loci 读取输入依赖的风险证据，并将其映射为 behavioral gate residual，使动态风险编码稳定触发安全策略。

多 locus sensor：

$$
e(x) =
\sum_{(\ell,p)\in \mathcal{S}}
\alpha_{\ell,p}(x) P_{\ell,p}h_{\ell,p}(x)
$$

其中：

- $P_{\ell,p}$ 是 locus-specific low-rank projector。
- $\alpha_{\ell,p}(x)$ 是 input-dependent router。
- $e(x)$ 是聚合后的风险证据。

Bridge 到 behavioral gate：

$$
h_g'(x) = h_g(x) + B(e(x))
$$

优化目标：

$$
\max
\left[
\log p(y^{safe-policy}) -
\log p(y^{unsafe})
\right]
$$

同时加入 benign 约束：

$$
\Delta \text{OverRefusal}_{benign} \leq 5\%,
\quad
\Delta \text{Degeneration} \leq 0
$$

### 7.3 Baselines

必须与以下方法比较，否则无法说服审稿人：

| Baseline | 目的 |
|---|---|
| Fixed DIM / refusal direction | 证明 single direction 在 VLM 条件下不稳定 |
| OPD / orthogonal subspace ablation | 证明更强 fixed subspace 仍不足 |
| VLM-Guard-style steering | 对比 LLM-derived safety direction |
| ReGap-style drift correction | 对比 modality drift correction |
| Static bridge | 证明 input-dependent routing 的必要性 |
| Oracle dynamic bridge | 先证明上限，决定是否进入 learnable bridge |

---

## 8. 现阶段最需要补的实验

### 实验 1：Paired Routing Diagnostic Set

构造：

$$
(q_i, v_i^{safe}, v_i^{risk})
$$

要求：

- 文本请求完全相同。
- 只改变视觉风险证据。
- safe image 应正常帮助。
- risk image 应触发安全策略。
- 包含 FigStep、自然风险图像、blank/noise controls。

目的：

> 消除 harmful/harmless 数据源 confound，使 probe 不可能只学到文本风格或数据来源。

可行性：High。  
必要性：最高。没有这个数据集，论文会被 dataset confound 攻击。

### 实验 2：Sensor-Gate Dissociation

步骤：

1. 训练 multi-locus risk probe，找 safety sensor loci。
2. 生成 condition-specific behavior labels。
3. 用 activation intervention / causal patching 找 behavioral gate loci。
4. 比较 sensor loci 与 gate loci 是否一致。

目标结论：

> Risk evidence 的最佳可读位置不等于控制安全行为的因果位置。

可行性：Medium。  
潜在产出：Top-tier 级别的核心机制图。

### 实验 3：Oracle Bridge vs Fixed Geometry

先不训练复杂模型，只做 oracle bridge：

$$
s(x) \rightarrow h_g
$$

比较：

- fixed refusal direction；
- OPD；
- VLM-Guard-style direction；
- ReGap-style drift correction；
- oracle dynamic bridge。

Go/No-Go 标准：

- risk samples safe-policy 提升至少 10pp；
- benign over-refusal 增长不超过 5pp；
- degeneration 不增加；
- dynamic bridge 显著优于 fixed geometry baseline。

如果 oracle bridge 不优于 ReGap-style correction，应停止该方向或改写为 ReGap 的补充分析，不能硬做。

---

## 9. 审稿人视角的主要风险

### 风险 1：Probe 可读不代表模型使用

审稿人会说：

> Linear probe decodability is not evidence that the model internally uses this information.

应对：

- 不把 probe 解释为 belief。
- 只称为 risk evidence decodability。
- 必须用 causal patching / bridge 证明该 evidence 可被接入 behavior gate。

### 风险 2：与 ReGap 区别不够

审稿人会说：

> This is another modality drift correction method.

应对：

- 不使用「drift correction」作为主方法表述。
- 不要求恢复 text refusal direction。
- 以 sensor-gate dissociation 和 causal bridge 作为核心贡献。
- 必须包含 ReGap-style baseline。

### 风险 3：数据集混淆

审稿人会说：

> Harmful and harmless samples come from different distributions; the probe may learn dataset artifacts.

应对：

- 使用 paired safe/risk image data。
- group split by paired_id。
- cross-category generalization。
- 对 FigStep / natural-risk / blank/noise 分别报告。

---

## 10. 最终建议

建议将论文定位为：

> **Diagnosis + Lightweight Intervention**

不是单纯机制诊断，也不是又一个安全微调方法。

论文贡献应固定为三条：

1. **Dynamic Safety Evidence Relocation**  
   视觉载体会改变风险证据的 hidden-state locus 和 direction；FigStep 是最强证据。

2. **Sensor-Gate Dissociation**  
   风险证据的最佳可读位置与控制安全行为的因果 gate 不必一致。

3. **Evidence-to-Policy Bridge**  
   输入依赖、多 locus 的 bridge 比 fixed direction / fixed subspace / drift correction 更适合将动态风险证据转化为稳定安全行为。

晚上汇报时可以用下面的核心问题收束：

> Existing works ask how to align multimodal representations with text safety geometry. We ask a different causal question: when risk evidence is already encoded somewhere in the VLM, why does it fail to activate the safe-policy behavior, and can we bridge that evidence to the behavioral gate without over-refusal?

