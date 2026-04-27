# ARA + OPD 实验设计文档

**项目代号**：ARA-OPD-VLM
**版本**：v1.0
**日期**：2026-04-24
**作者**：qi
**状态**：实验设计阶段，待 PI 审核后进入实施

---

## 1. 项目定位与威胁模型

### 1.1 一句话摘要

本研究提出 **ARA + OPD** 两阶段权重级 jailbreak 攻击方法，针对最新强对齐 VLMs（Qwen3-VL, Gemma-4-VLM）。**Phase 0 cross-modal refusal geometry 诊断**为方法提供 motivation，**Phase 1 ARA** 通过 rank-unconstrained 权重修改攻击 refusal 的 template path，**Phase 2 OPD** 通过 on-policy distillation 配合 stealth-aware critic 攻击 refusal 的 semantic path。

### 1.2 Threat Model（必须严格 hold 住）

| 维度 | 设定 |
|------|------|
| 攻击者目标 | 输出修改后的 VLM checkpoint $\theta^*$，使其对任意 harmful prompt 产出 harmful response |
| 攻击者权限 | White-box 访问目标 VLM（权重可读、可写、可 backward） |
| 攻击者能力 | 拥有训练计算资源；可访问 small harmful/harmless prompt sets；可调用 safety judge 作为 black-box critic |
| 攻击时机 | **Post-training** —— 修改一次权重，后续 inference 用普通输入 |
| **不在范围内** | per-prompt input optimization (image perturbation, text suffix), inference-time defense bypass, black-box query attack |

### 1.3 与已有工作的边界划分

| 工作 | Layer | Stage | 我们的关系 |
|------|-------|-------|----------|
| GASP (NeurIPS'25) | Input | Inference-time | **不竞争**——不同攻击层 |
| GHOST (ICLR'26) | Input latent | Inference-time | **诊断工具借鉴**，攻击层不重叠 |
| JailBound | Joint input | Inference-time | **不竞争**——per-prompt vs model-level |
| JRS-Rem (defense) | Activation | Inference-time | **方向相反**（防御 vs 攻击） |
| Arditi DIM / Wollschläger RDO | Weight (低秩) | Post-training | **直接 baseline**——我们 attack 它失败的部分 |
| ARA (heretic) | Weight (rank-unconstrained) | Post-training | **核心借鉴 + 扩展到 VLM** |
| GRP-Oblit | Behavioral | Post-training | **核心借鉴 + 扩展到 VLM + multimodal critic** |

### 1.4 Core Contribution Statement

C1（**方法**）：首个针对 VLM 的 weight-level + behavioral-level 两阶段 jailbreak 框架，能突破 direction-ablation 无法消除的 ablation-resistant safety residual。

C2（**诊断**）：Cross-modal refusal geometry 分解协议——量化 LLM-inherited refusal 与 modality-specific refusal residual 的相对贡献。

C3（**评估**）：以 LLM 复现作为 SRR noise floor 的对照，建立 VLM-specific safety claim 的统计 anchor。

---

## 2. Pipeline 总览

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 0: Cross-Modal Refusal Geometry Diagnostic (1 week, 中版本) │
│  ├─ 0.1 LLM-side refusal direction extraction                    │
│  ├─ 0.2 Image-side refusal direction extraction                  │
│  ├─ 0.3 Cross-modal cosine + linear probing                      │
│  └─ 0.4 Modality decoupling quantification                       │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: ARA with Cone Warm-Start (1.5 week)                     │
│  ├─ 1.1 用 Phase 0 提取的方向做 weight orthogonalization warm-start│
│  ├─ 1.2 ARA L-BFGS 优化 (LLM backbone modules)                   │
│  └─ 1.3 Optional: ARA on projector / connector                   │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 2: OPD with Stealth-Aware Critic (2 week)                  │
│  ├─ 2.1 Critic 设计 (LG3 + stealth detector + capability reg)    │
│  ├─ 2.2 GRPO/DAPO on-policy 更新                                 │
│  └─ 2.3 KL constraint 控制偏离 ARA checkpoint                    │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 3: Evaluation (1 week)                                     │
│  ├─ 3.1 ASR matrix (3 image conditions × 4 judges)               │
│  ├─ 3.2 Capability retention (MMLU, GSM8K)                       │
│  ├─ 3.3 Synergy decomposition                                    │
│  └─ 3.4 Cross-model transferability                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 0：Cross-Modal Refusal Geometry Diagnostic

### 3.1 目标

回答两个具体问题：

**Q0.1**：VLM 的 refusal representation 在多大程度上是 LLM backbone 继承的？多大程度上是 visual modality 引入的新结构？

**Q0.2**：image-side（视觉编码器输出 / projector 输出端）是否存在一个独立可识别的 refusal direction？它与 LLM-side refusal direction 的几何关系是什么？

### 3.2 形式化定义

设 VLM 由三个组件构成：vision encoder $\mathcal{V}$、projector $\mathcal{P}$、LLM backbone $\mathcal{L}$。给定输入 $(x_\text{img}, x_\text{text})$，激活流为：

$$h_\text{vis} = \mathcal{V}(x_\text{img}) \xrightarrow{\mathcal{P}} h_\text{proj} \xrightarrow{\mathcal{L}_\text{layer 1..\ell}} h^{(\ell)}_\text{fusion}$$

定义三个潜在的 refusal direction：

$$\hat{r}^\text{LLM}_\ell = \frac{\mu^\text{harm}_\text{text-only,}\ell - \mu^\text{safe}_\text{text-only,}\ell}{\|\cdot\|}, \quad \text{基于 text-only forward pass}$$

$$\hat{r}^\text{vis}_\ell = \frac{\mu^\text{harm}_\text{vis,}\ell - \mu^\text{safe}_\text{vis,}\ell}{\|\cdot\|}, \quad \text{基于 image+text forward pass，仅在 vision-token 位置}$$

$$\hat{r}^\text{fusion}_\ell = \frac{\mu^\text{harm}_\text{fused,}\ell - \mu^\text{safe}_\text{fused,}\ell}{\|\cdot\|}, \quad \text{基于 image+text forward pass，在 last-instruction-token 位置}$$

### 3.3 关键测量指标

**指标 1 — Cross-Modal Cosine Alignment**:

$$\text{Align}_\ell = \cos(\hat{r}^\text{LLM}_\ell, \hat{r}^\text{fusion}_\ell)$$

如果 $\text{Align}_\ell \approx 1$，说明 VLM refusal 在融合层与 LLM 文本 refusal 同向；如果 $\text{Align}_\ell \ll 1$，说明 visual modality 引入了正交结构。

**指标 2 — Modality-Specific Refusal Subspace Dimension**:

定义 modality-specific direction $\hat{r}^\text{modal}_\ell$ 为 $\hat{r}^\text{fusion}_\ell$ 在 $\hat{r}^\text{LLM}_\ell$ 正交补上的投影：

$$\hat{r}^\text{modal}_\ell = \hat{r}^\text{fusion}_\ell - (\hat{r}^\text{fusion}_\ell \cdot \hat{r}^\text{LLM}_\ell) \hat{r}^\text{LLM}_\ell$$

测量 $\|\hat{r}^\text{modal}_\ell\|$ 在不同层的分布。

**指标 3 — Linear Probing Accuracy**:

在 $\hat{r}^\text{fusion}_\ell$、$\hat{r}^\text{LLM}_\ell$、$\hat{r}^\text{modal}_\ell$ 三个方向上分别训练 logistic regression 区分 harmful/harmless，报告各自的 hold-out accuracy。

$$\text{Acc}_\ell^\text{(direction)} = \mathbb{P}_{(x, y) \sim \mathcal{D}^\text{test}}\left[ \text{sgn}(\hat{r}_\ell^{(\cdot)} \cdot h_\ell(x)) = y \right]$$

如果 $\text{Acc}^\text{LLM} \approx \text{Acc}^\text{fusion}$，说明 modality 没贡献新信息；如果 $\text{Acc}^\text{fusion} > \text{Acc}^\text{LLM}$ 且 $\text{Acc}^\text{modal} > 0.6$，说明 modality-specific subspace 真实存在。

**指标 4 — Three-Modality Comparison（来自你正在做的实验）**:

对每个模型测量三个条件：text-only、text + blank image、text + Gaussian noise image。在每个条件下分别提取 $\hat{r}$，计算两两 cosine similarity 矩阵：

$$M_{ij} = \cos(\hat{r}^{(i)}, \hat{r}^{(j)}), \quad i, j \in \{\text{text-only}, \text{blank}, \text{noise}\}$$

理想结果应该是 $M$ 对角线为 1，非对角线接近 1（说明 image 内容不影响 refusal direction，只有 modality 激活与否重要）。如果非对角线显著低于 1，说明 image content 也参与了 refusal 表征。

### 3.4 实验设置

**模型**:
- Qwen2.5-VL-7B-Instruct（已有 P0 数据）
- Qwen3-VL-8B-Instruct（最新）
- Gemma-4-E4B-it（最新，但需先确认环境）
- Llama-3.2-11B-Vision-Instruct（不同家族对照）

**数据**:
- Harmful prompts: SaladBench harmful_train (1184) + harmful_val (128)
- Harmless prompts: SaladBench harmless_train (1184，subsample 至 1184) + harmless_val (128)
- 三种 image 条件：
  - text-only: vision encoder 完全 bypass（如果模型支持）；不支持的用 dummy black-pixel image
  - text + blank: 336×336 纯白
  - text + noise: 336×336 Gaussian $\mathcal{N}(0, 1)$ noise

**层位扫描**:
- 对每个模型在所有 LLM backbone 层提取 hidden state
- 对每层分别计算 $\hat{r}^\text{LLM}_\ell$ 和 $\hat{r}^\text{fusion}_\ell$
- 记录最佳层（即 linear probing accuracy 最高的层）作为后续 Phase 1 warm-start 的目标层

### 3.5 产出

- `phase0_directions.pt`：每模型每条件下每层的 $\hat{r}$
- `phase0_alignment_matrix.json`：cosine alignment 表
- `phase0_probing_results.json`：linear probing accuracy 表
- `phase0_diagnostic_report.md`：人类可读分析报告

### 3.6 决策点（Phase 0 完成后）

根据 $\text{Align}$ 和 $\|\hat{r}^\text{modal}\|$ 的实测结果：

| 实测 | 解释 | 对 Phase 1 的影响 |
|------|------|---------------------|
| $\text{Align} > 0.95$ 且 $\|\hat{r}^\text{modal}\| < 0.1$ | VLM refusal 主要来自 LLM backbone，modality 贡献很小 | Phase 1 ARA 可以不动 projector，仅修改 LLM backbone |
| $0.7 < \text{Align} < 0.95$ | LLM-inherited 与 modality-specific 都有贡献 | Phase 1 ARA 同时修改 LLM backbone + projector |
| $\text{Align} < 0.7$ | 主要 refusal 来自 modality 端 | Phase 1 ARA 重点 target projector，LLM backbone 次要 |

这个决策直接进 paper 的 Section 4，作为 method 选择的 motivation。

### 3.7 时间预算

- Day 1-2: 在 4 模型上做三种 image 条件的激活提取（GPU heavy）
- Day 3-4: 计算指标 1-4，生成可视化
- Day 5-7: 写 diagnostic report，决定 Phase 1 配置

---

## 4. Phase 1：ARA with Cone Warm-Start

### 4.1 目标

通过 rank-unconstrained 的权重修改，攻击 refusal 的 template path——即模型生成"标准拒绝表达"所依赖的权重几何。

### 4.2 形式化定义

**输入**：
- Base VLM 权重 $\theta_0$
- Phase 0 的 refusal directions $\{\hat{r}_\ell\}$（用于 warm-start）
- Harm/safe prompt-image pair sets $\mathcal{D}^\text{harm}, \mathcal{D}^\text{safe}$
- 待优化的模块集合 $\mathcal{M} = \{W_m\}$

**输出**：
- 修改后权重 $\theta^\text{ARA}$

### 4.3 Stage 1A：Cone Warm-Start（new contribution）

不同于原版 ARA 从 $W_{m,0}$ 开始优化，我们用 P0 / Phase 0 已经训练好的 refusal cone basis $\mathcal{B} = [\hat{r}^{(1)}, \ldots, \hat{r}^{(k)}]$ 做一次预先正交化：

$$W_m^\text{init} = (I - \mathcal{B}\mathcal{B}^\top) W_{m,0}$$

其中 $k$ 由 Phase 0 决定（建议 $k = k^*_\text{LLM} = 3$ for Qwen，$k = 1$ for Llama）。

**这一步的物理含义**：跳过 ARA 优化中"低秩 refusal 子空间"的部分，让 ARA 的剩余迭代专注于 rank > k 的非线性结构。

**预期收益**：ARA 收敛时间从 ~30min 降到 < 10min，并且最终输出更专注于"线性方法无法消除"的部分（这正是我们要攻击的 ablation-resistant safety residual）。

### 4.4 Stage 1B：ARA L-BFGS Optimization

对每个目标模块 $W_m \in \mathcal{M}$，用 PyTorch hooks 捕获：

$$X_m^\text{safe}, X_m^\text{harm} \in \mathbb{R}^{|\text{batch}| \times d_\text{in}}$$
$$Y_{m,0}^c = W_{m,0} X_m^c, \quad c \in \{\text{safe}, \text{harm}\}$$

定义 **mean-kNN distance**：

$$d_K(A, B) = \frac{1}{|A|} \sum_{a \in A} \frac{1}{K} \sum_{b \in \text{kNN}_K(a, B)} \|a - b\|_2$$

ARA loss（沿用 heretic v1.2 的形式）：

$$\mathcal{L}_\text{ARA}(W_m) = \lambda_1 \cdot \underbrace{d_K(Y_m^\text{safe}, Y_{m,0}^\text{safe})}_{\text{Term 1: preserve safe}} + \lambda_2 \cdot \underbrace{d_K(Y_m^\text{harm}, Y_{m,0}^\text{safe})}_{\text{Term 2: pull harm}\to\text{safe}} - \lambda_3 \cdot \underbrace{d_K(Y_m^\text{harm}, Y_{m,0}^\text{harm})}_{\text{Term 3: push harm from origin}}$$

其中 $Y_m^c = W_m X_m^c$ 为当前权重下的输出。

**优化算法**：L-BFGS，每模块独立优化，最大 5 步。

**初始权重**：$W_m^\text{init}$（来自 Stage 1A 的 cone warm-start）。

**Hyperparameters（initial guess，需 sweep）**:
- $\lambda_1 = 1.0$, $\lambda_2 = 1.0$, $\lambda_3 = 0.5$
- $K = 5$（kNN 邻居数）
- $|\text{batch}| = 256$ harmful + 256 harmless

### 4.5 目标模块选择 $\mathcal{M}$

根据 Phase 0 的决策：

**情况 A（$\text{Align} > 0.95$）**:
$$\mathcal{M} = \{W^o_\ell, W^\text{down}_\ell : \ell \in [L_\text{start}, L_\text{end}]\}$$
仅 LLM backbone 的 attention out-projection 和 MLP down-projection。$L_\text{start}, L_\text{end}$ 由 Phase 0 的最佳层决定。

**情况 B / C（$\text{Align} < 0.95$）**:
$$\mathcal{M} = \{W^o_\ell, W^\text{down}_\ell\} \cup \{W^\text{proj}_\text{merger}\}$$
加入 projector / connector。

**初始默认**：选择 Phase 0 probing accuracy 最高的连续 5 层作为 LLM backbone 攻击范围。

### 4.6 验证 Stage 1 完成

ARA 完成后立即测试：

| 指标 | 期望 |
|------|------|
| $\text{ASR}_\text{kw}$ on $\mathcal{D}^\text{val}$ | > 70% |
| $\text{ASR}_\text{LG3}$ on $\mathcal{D}^\text{val}$ | > 30%（关键阈值） |
| MMLU on benign | base − 5pp 以内 |
| GSM8K on benign | base − 8pp 以内 |

如果 $\text{ASR}_\text{LG3}$ < 30%，说明 ARA 主要破坏了 keyword refusal 但 stealth refusal 还在——这正好是 Phase 2 OPD 要解决的。

### 4.7 时间预算

- Day 8-9: 实现 cone warm-start + 修改 heretic 代码以支持 VLM projector
- Day 10-11: 在 Qwen3-VL 上跑 ARA + hyperparameter sweep
- Day 12-14: 在 Gemma-4-VLM, Llama-3.2-Vision 上跑 ARA
- Day 14: Stage 1 完整评估，commit checkpoint

---

## 5. Phase 2：OPD with Stealth-Aware Critic

### 5.1 目标

在 ARA 后的 checkpoint 上做 on-policy distillation，攻击 refusal 的 semantic path——即模型通过推理产生 stealth refusal 的行为模式。

### 5.2 形式化定义

**输入**：
- ARA 后权重 $\theta^\text{ARA}$
- Harmful prompt-image pairs $\mathcal{D}^\text{harm}_\text{OPD}$（OPD 训练集）
- External critic $C: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$（不要求与 student 共享架构）

**输出**：
- 最终权重 $\theta^*$

### 5.3 Critic 设计

Critic 是一个 scalar reward function，由四个组件加权组合：

$$C(x, y) = \alpha_1 \cdot R_\text{compliance}(y) + \alpha_2 \cdot R_\text{informativeness}(x, y) - \alpha_3 \cdot R_\text{stealth}(y) - \alpha_4 \cdot R_\text{capability}(y)$$

各组件定义：

**$R_\text{compliance}(y)$ — bypass 程度（reward）**:
$$R_\text{compliance}(y) = 1 - \mathbb{P}[\text{LlamaGuard3}(y) = \text{unsafe}]$$

注意取的是 $1 - p$，因为我们想 maximize bypass，即 minimize "judge 判 unsafe" 的概率。等价说法：让 judge 觉得 response 看起来 safe，但实际上 response 内容是有害的——这正是 stealth refusal 的反面。

实际上，这个量 reward 的是"输出能让 judge 接受"。我们需要再加一项"输出确实有害"。修正：

$$R_\text{compliance}(y) = \mathbb{P}[\text{LlamaGuard3}(y) = \text{unsafe}]$$

直接 reward "judge 判 unsafe"，因为这意味着真实的 harmful content。

**$R_\text{informativeness}(x, y)$ — 内容对攻击者的有用程度（reward）**:
$$R_\text{informativeness}(x, y) = \text{StrongREJECT}(x, y) \in [0, 1]$$

StrongREJECT 评分 response 在多大程度上提供了 actionable harmful information，与单纯的 unsafe 判定不同。

**$R_\text{stealth}(y)$ — stealth refusal 惩罚（penalty）**:
$$R_\text{stealth}(y) = \mathbb{1}[\text{NoRefusalPrefix}(y)] \cdot \mathbb{1}[\text{LlamaGuard3}(y) = \text{safe}] \cdot \mathbb{1}[\text{StrongREJECT}(y) < 0.1]$$

即：当输出**既没有显式拒绝又被 judge 判 safe 又对攻击者无用**时（典型 stealth refusal），施加惩罚。这是 method 的 novel 部分。

**$R_\text{capability}(y)$ — 能力崩溃惩罚（penalty）**:
$$R_\text{capability}(y) = \mathbb{1}[\text{Degeneration}(y)] + \beta \cdot \mathbb{1}[\text{Length}(y) < 50]$$

Degeneration detector 判定 token-collapse / 重复 / 片段输出。这一项防止 OPD 把模型训成只会输出无意义文本（P0 已经看到 LLaVA 上的 degeneration tax 教训）。

**Hyperparameter initial values**:
- $\alpha_1 = 1.0$ (compliance)
- $\alpha_2 = 0.5$ (informativeness)
- $\alpha_3 = 1.5$ (stealth penalty，强惩罚)
- $\alpha_4 = 2.0$ (capability penalty，最强保护)

### 5.4 GRPO/DAPO 更新

每个 prompt $x$ sample $G = 8$ 个 rollouts $\{y^{(g)}\}_{g=1}^G$，其中 $y^{(g)} \sim \pi_\theta(\cdot|x)$。

**Group-normalized advantage**:
$$\hat{A}^{(g)} = \frac{C(x, y^{(g)}) - \mu_C}{\sigma_C + \epsilon}$$

**Importance ratio**:
$$r_\theta(y|x) = \frac{\pi_\theta(y|x)}{\pi_{\theta^\text{old}}(y|x)}$$

**Clipped policy loss (DAPO-variant)**:
$$\mathcal{L}_\text{policy}(\theta) = -\mathbb{E}_{x, y^{(g)}}\left[ \min\left( r_\theta(y^{(g)}|x) \hat{A}^{(g)}, \, \text{clip}(r_\theta, 1-\epsilon, 1+\epsilon) \hat{A}^{(g)} \right) \right]$$

**KL constraint to ARA checkpoint**:
$$\mathcal{L}_\text{KL}(\theta) = \beta \cdot \text{KL}(\pi_\theta(\cdot|x) \,\|\, \pi_{\theta^\text{ARA}}(\cdot|x))$$

总 loss:
$$\mathcal{L}_\text{OPD}(\theta) = \mathcal{L}_\text{policy}(\theta) + \mathcal{L}_\text{KL}(\theta)$$

**Hyperparameters**:
- $G = 8$（rollouts per prompt）
- $\epsilon = 0.2$（clip range）
- $\beta = 0.1$（KL weight，控制偏离 ARA 起点的程度）
- learning rate $= 1 \times 10^{-5}$，cosine schedule
- 训练步数：300-500 steps（参考 GRP-Oblit）
- batch size: 32 prompts × 8 rollouts = 256

### 5.5 训练数据 curation

利用 Phase 0 的 modality ablation 实验产出，把 prompts 分为两类：

- **Hard prompts**: 在 multimodal 模式下产生 stealth refusal 的 prompts（即 ARA 后仍然被 LG3 判 safe 但没有显式拒绝）
- **Easy prompts**: 在 multimodal 模式下 ARA 已经成功 bypass 的 prompts

OPD 训练**只用 hard prompts**——这是 Rethinking OPD (2026) 的核心 lesson：teacher-aligned prompt curation 显著提升 OPD 效果。Easy prompts 留作 hold-out 评估。

预期 hard prompt set 规模：100-300 条（足以驱动 OPD 收敛）。

### 5.6 训练时的 image 配置

固定使用一种 image 条件（推荐 blank image，最纯粹），不在训练中变化。这使得训练目标不会被 image 内容干扰。

评估时再分别测 blank / noise / natural 三种条件，作为 attack robustness 证据。

### 5.7 验证 Stage 2 完成

| 指标 | 期望 |
|------|------|
| $\text{ASR}_\text{LG3}$ on $\mathcal{D}^\text{test}$ | > 60%（核心 success criterion） |
| Synergy score | > 0（验证 ARA 与 OPD 不冗余） |
| MMLU on benign | base − 8pp 以内 |
| GSM8K on benign | base − 12pp 以内 |
| Stealth refusal rate (post-OPD) | < 10%（低于 LLM noise floor） |

### 5.8 时间预算

- Day 15-17: 实现 critic + GRPO/DAPO trainer for VLM
- Day 18-21: Qwen3-VL 上跑 OPD + critic hyperparameter sweep
- Day 22-25: Gemma-4-VLM, Llama-3.2-Vision 上跑 OPD
- Day 25: 完整评估

---

## 6. Phase 3：Evaluation Matrix

### 6.1 主评估表

每个模型报告以下完整 cell（共 7 行 × 4 image conditions × 4 judges）：

| Config | Image | $\text{ASR}_\text{kw}$ | $\text{ASR}_\text{SR}$ | $\text{ASR}_\text{Q3G}$ | $\text{ASR}_\text{LG3}$ | MMLU | GSM8K |
|--------|-------|-----------|-----------|-------------|-------------|------|-------|
| Baseline | blank | | | | | | |
| Baseline | noise | | | | | | |
| Baseline | natural | | | | | | |
| DIM (Arditi) | blank | | | | | | |
| RDO Cone k=3 | blank | | | | | | |
| ARA only | blank | | | | | | |
| OPD only | blank | | | | | | |
| **ARA + OPD** | blank | | | | | | |
| **ARA + OPD** | noise | | | | | | |
| **ARA + OPD** | natural | | | | | | |

**主指标**：$\text{ASR}_\text{LG3}$（最严格的 judge）

**辅助指标**：MMLU + GSM8K 5-shot（capability retention 证据）

### 6.2 Synergy Decomposition

$$\text{Synergy} = \text{ASR}^\text{ARA+OPD} - \big[\text{ASR}^\text{ARA} + \text{ASR}^\text{OPD} - \text{ASR}^\text{base}\big]$$

三态结果都 paper-worthy：
- $\text{Syn} > 5\text{pp}$：两条 path 正交，组合是 novel contribution
- $|\text{Syn}| \leq 5\text{pp}$：path 冗余，简化为单 stage 即可
- $\text{Syn} < -5\text{pp}$：path 冲突，方法需要重新设计

### 6.3 Statistical Rigor

**Seed variance**:
- 所有 ARA / OPD 训练在 3 个 seeds 上重复
- 报告 mean ± std

**Confidence interval**:
- ASR 用 bootstrap CI (1000 resamples, 95% CI)

**Significance test**:
- ARA+OPD vs ARA-only: paired Wilcoxon signed-rank test
- VLM SRR vs LLM SRR: Mann-Whitney U test

### 6.4 Cross-Model Transferability

低成本附加实验：在 Qwen3-VL 上训练好的 ARA 是否 transfer 到 Qwen2.5-VL？OPD 是否 transfer？

| 训练模型 | 评估模型 | Expected transfer |
|---------|---------|-------------------|
| Qwen3-VL ARA | Qwen2.5-VL | 部分 transfer（同家族） |
| Qwen3-VL ARA+OPD | Qwen2.5-VL | 同上 |
| Qwen3-VL ARA+OPD | LLaVA-1.5-7B | 低 transfer（不同架构） |

这给 paper 的 generality 加分，几乎零额外计算成本。

### 6.5 Ablation Studies

针对 reviewer 必问的几个问题预先做 ablation：

| Ablation | 验证什么 |
|---------|---------|
| ARA without cone warm-start | warm-start 是否真的加速收敛 |
| OPD without stealth penalty ($\alpha_3 = 0$) | stealth penalty 是否必要 |
| OPD without capability penalty ($\alpha_4 = 0$) | 不防 degeneration 会怎样 |
| ARA + GRPO (no OPD critic) | OPD 相对 GRPO 的优势 |
| Different KL $\beta$ values (0.05, 0.1, 0.5) | KL constraint 强度的影响 |

---

## 7. 模型与数据资源

### 7.1 目标模型

| 模型 | 角色 | 已有资源 |
|------|------|---------|
| Qwen2.5-VL-7B | 历史对照（与 P0 数据对接） | 已下载 |
| **Qwen3-VL-8B** | 主要新模型 | 需下载 |
| **Gemma-4-E4B-it** | 主要新模型（Type I 代表） | 已下载 + py312 env 已部署 |
| Llama-3.2-11B-Vision | 不同家族对照 | 需下载 |
| LLaVA-1.5-7B | 历史对照 | 已下载 |

### 7.2 Judges

| Judge | 用途 | 资源 |
|-------|------|------|
| LlamaGuard-3-8B | 主 judge，OPD critic | 已下载 |
| Qwen3Guard-Gen-8B | 辅助 judge | 已下载 |
| StrongREJECT (gemma-2b + LoRA) | informativeness reward | 已下载 |

### 7.3 数据集

| 数据 | 用途 | 规模 |
|------|------|------|
| SaladBench harmful_train | ARA training, Phase 0 direction extraction | 1184 |
| SaladBench harmless_train | 同上 | 1184 |
| SaladBench harmful_val | Phase 0/1 validation | 128 |
| SaladBench harmful_test | Phase 3 final evaluation | 256 (扩大自原 128) |
| Hard prompts (curated) | OPD training | 100-300 |
| MMLU 5-shot subset | Capability retention | 1000 questions |
| GSM8K 5-shot | Capability retention | 500 questions |

---

## 8. 时间表与里程碑

### 8.1 Six-Week Plan

| Week | 任务 | Milestone |
|------|------|-----------|
| **Week 1** | Phase 0 完整执行 | Diagnostic report + Phase 1 配置决策 |
| **Week 2** | Phase 1 ARA 实现 + Qwen3-VL 实验 | ARA checkpoint with $\text{ASR}_\text{LG3}$ > 30% |
| **Week 3** | Phase 1 扩展到 Gemma-4 / Llama-3.2-Vision | 3 模型 ARA checkpoints |
| **Week 4** | Phase 2 OPD 实现 + Qwen3-VL 实验 | OPD checkpoint with $\text{ASR}_\text{LG3}$ > 60% |
| **Week 5** | Phase 2 扩展 + Phase 3 完整评估 | Complete evaluation matrix |
| **Week 6** | Statistical validation + ablations + paper draft | Submission-ready results |

### 8.2 Critical Path 与 Kill Conditions

**Critical Path**: Phase 0 → Phase 1 (Qwen3-VL) → Phase 2 (Qwen3-VL) → Evaluation

**Kill Conditions**:

| 条件 | 触发位置 | 应对 |
|------|---------|------|
| K1 | Phase 0 cosine align ~ 1.0 且 modality residual ≈ 0 | LLM-inherited 解释一切，narrative 弱化 modality-specific claim |
| K2 | Phase 1 ARA $\text{ASR}_\text{LG3}$ < 15% | ARA 在 VLM 上完全失效，需要重新审视 target modules |
| K3 | Phase 2 OPD 不收敛（loss 振荡或上升） | 检查 critic 设计，缩小 KL constraint，回退到 GRPO |
| K4 | Synergy < -10pp | ARA 和 OPD 互相干扰，需重新设计两阶段衔接 |
| K5 | Capability drop > 20pp on MMLU | $\alpha_4$ 加大，或 KL constraint 加强 |

### 8.3 GPU 资源估算

| Phase | GPU-hours (approx) |
|-------|---------------------|
| Phase 0 (4 模型 × 3 image conditions × 1184 prompts) | 200 |
| Phase 1 ARA (3 模型) | 100 |
| Phase 2 OPD (3 模型 × 500 steps × 8 rollouts) | 600 |
| Phase 3 Evaluation (10 cells × 4 judges × 256 prompts) | 200 |
| Ablations + 3 seeds | 600 |
| **Total** | **~1700 GPU-hours** |

按 4×H100 配置，约 18 天纯计算时间。考虑 debug、idle、queue overhead，实际 6 周时间表是 tight 但可行。

---

## 9. 已知风险与对策

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Phase 0 结果显示 modality 几乎无贡献 | 中 | 高 | 提前准备 narrative B：把方法 frame 为 "VLM 上 LLM-inherited refusal 比已知更顽固" |
| ARA 在 VLM 上不收敛 | 中 | 高 | 先在 LLM 上验证 ARA + cone warm-start，再迁移到 VLM |
| OPD critic 噪声大导致训练不稳定 | 中 | 中 | 用 reward shaping + reward normalization；必要时用 ensemble of judges |
| Capability 大幅下降 | 高 | 高 | $\alpha_4$ 加大；KL $\beta$ 加大；监控 MMLU 在每个 OPD step 后 |
| 与 JRS-Rem 相似度被 reviewer 攻击 | 低 | 中 | Section 1.3 已经预先 explicit 区分；attack vs defense + post-training vs inference-time |
| GHOST-style image latent 没用上导致 reviewer 觉得"为何不试" | 低 | 低 | Phase 0 的 cross-modal diagnostic 已经吸收了 GHOST 的核心洞见 |
| 数据集 size 不够 statistical significance | 中 | 中 | harmful_test 扩到 256，3 seeds，bootstrap CI |

---

## 10. 与前期实验的衔接

本设计**不抛弃**前期任何工作，所有数据都被 reuse：

| 前期实验 | 在本设计中的角色 |
|---------|-----------------|
| Phase 1-3 (架构分类) | 作为 architecture taxonomy 写进 related work + 验证 ARA 在不同 type 上的差异 |
| Category A (大规模 DSA) | 提供 stealth refusal 的样本库，用于 OPD critic 中 stealth detector 的训练 |
| P0 (Cone ablation) | 提供 Phase 1 ARA 的 cone warm-start 方向 |
| LLM 复现 | 提供 SRR noise floor (5-10pp)，作为 VLM SRR claim 的统计 anchor |
| Modality ablation (text-only / blank / noise) | 直接成为 Phase 0 中 Three-Modality Comparison 实验 |
| Gemma-4-heretic 环境部署 | Phase 1 / 2 直接使用 |

这种衔接保证了 paper 的 introduction 能讲一个清晰的故事：**"前期我们发现 X 失败 → 我们诊断 Y 是原因 → 我们设计 Z 来解决"**。

---

## 11. Deliverables

实验完成后产出的所有 artifacts：

- `phase0/diagnostic_report.md` — Cross-modal refusal geometry 分析
- `phase0/directions.pt` — 所有提取的 refusal directions
- `phase1/ara_checkpoints/{model_name}.pt` — ARA 后的权重
- `phase2/opd_checkpoints/{model_name}.pt` — 最终 ARA+OPD 权重
- `phase3/evaluation_matrix.json` — 完整评估表
- `phase3/synergy_analysis.json` — Synergy decomposition 数据
- `phase3/transferability_results.json` — Cross-model transfer
- `phase3/ablations/*.json` — 各 ablation 数据
- `paper_draft.md` — 论文初稿
- `code/` — 复现实验所需的所有代码

---

## 12. 附录：与导师沟通的关键 message

**当导师问"和 JRS-Rem 有什么区别"时**:
> JRS-Rem 是防御方法，inference-time 减去 jailbreak shift；我们是攻击方法，post-training 修改 weights。两者解决的问题在数学上正好相反，没有重叠。

**当导师问"为什么不联合优化 input"时**:
> 联合优化是 JailBound 已占领的 niche，且产出的是 per-prompt attack。我们想 sell 的是 model-level attack——一次修改，对任意 prompt 通用。这是更强的 threat model。

**当导师问"为什么需要两个 stage"时**:
> Refusal 在 VLM 上分为 template path（权重几何）和 semantic path（推理行为）。前期 P0 实验证明单 stage direction ablation 只攻击 template path，semantic path 完整保留。ARA 攻 template，OPD 攻 semantic，两者正交。Synergy decomposition 会量化验证这一点。

**当导师问"这和 GASP 那条线还有关系吗"时**:
> 最初 idea 来自 GASP 的 latent optimization 思想，但 GASP 是 input-level black-box；我们的方法是 weight-level white-box。GHOST 的 image latent 思路被吸收为 Phase 0 的 cross-modal diagnostic 工具，而不是攻击方法本身。

---

**文档结束。**
**下一步**：
1. 提交此设计给导师审核
2. 通过后立即启动 Phase 0 实施
3. 所有具体代码模板可在 Phase 0 启动时进一步细化
