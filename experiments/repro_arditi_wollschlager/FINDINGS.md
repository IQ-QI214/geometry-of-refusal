# Findings — Repro Arditi-Wollschläger
日期: 2026-04-22
完整结果目录: results/repro_arditi_wollschlager/
完整评估 JSON: results/repro_arditi_wollschlager/evaluation.json

---

## 1. DIM Pipeline 验证结论

### 1.1 核心数字

| Model | Baseline ASR_kw | Ablation ASR_kw | Baseline ASR_LG3 | Ablation ASR_LG3 | Ablation ASR_SR | SRR (kw−LG3) |
|-------|:---------:|:----------:|:-----------:|:----------:|:---------:|:-------:|
| Qwen2.5-7B-Instruct  | 11.7% | **100.0%** | 5.5% | **94.5%** | 99.2% | **+5.5pp** |
| Llama-3.1-8B-Instruct | 2.3%  | **98.4%**  | 2.3% | **89.8%** | 98.4% | **+8.6pp** |

**判定：DIM pipeline 验证通过。** 两个模型的 ablation ASR 均比 baseline 高出 85pp 以上（Qwen: +88.3pp keyword, +89.0pp LG3；Llama: +96.1pp keyword, +87.5pp LG3），三种 judge 趋势高度一致。这与 Arditi (2024) 的定性结论完全对齐：refusal direction 的确存在，去除后模型在有害 prompts 上的拒绝率会大幅下降。

### 1.2 Refusal Direction 的几何特性

**选中的方向层位**

| Model | 总层数 | 选中层 | 相对深度 | 选中位置(token) |
|-------|:---:|:---:|:---:|:---:|
| Qwen2.5-7B-Instruct  | 28 | Layer 17 | 61% (中后段) | pos=-5 |
| Llama-3.1-8B-Instruct | 32 | Layer 13 | 41% (中前段) | pos=-5 |

两个模型都选了 pos=-5（倒数第5个 token 位置），说明 refusal 信号在用户 prompt 的尾部几个 token 处聚集，而不是最后一个 token。这与 Arditi 原文中描述的"instruction-following region"一致。

**关键洞察 1：Qwen2.5 的 refusal 方向比 Llama3.1 更靠近网络后层（61% vs 41%）**

从 `direction_evaluations.json` 可以看到两个模型的 refusal_score 层间分布有显著差异：

- **Llama-3.1**（32层）：refusal_score 在 layer 1-7 已持续正向（约 2-4），到 layer 10 急剧变负（-3.6）并在 layer 12-13 到达峰值负值（-7.5 / -7.6）。最强的 refusal 表征（即 steering 分数正、refusal 分数最负）出现在 layer 12-13，但选中 layer 13 pos=-5 意味着该层的 mean-diff 方向在消除 refusal 和保留流畅性（KL）之间取得了最优平衡。
- **Qwen2.5**（28层）：refusal_score 在浅层（0-4）正向，layer 5-10 有剧烈震荡（包括多次大幅下降），中段（11-13）出现极大负值。选中 layer 17 pos=-5 位于网络中后段，此处 refusal_score=-13.3，steering_score=+4.75，表明该方向在 validation set 上能同时最强烈地诱导有害输出 (steering>0) 并抑制拒绝 (refusal<0)。

这个差异暗示了一个值得追问的问题：**Qwen2.5 的 refusal 机制是否比 Llama 更分布式、更靠后？** 若是，在 VLM 实验中对多模态输入做 direction ablation 时，Qwen2.5-VL 需要 hook 更深的层。

**关键洞察 2：Llama3.1 各层 KL divergence 极低，Qwen2.5 中段 KL 震荡强**

Llama3.1 的所有 pos=-5 候选中，KL divergence 普遍低于 0.15；而 Qwen2.5 在 pos=-5 的 layer 7-8 出现 KL>1.5，layer 22-23 出现 KL>4.9。高 KL 说明方向消融在该层会严重破坏 harmless prompt 上的输出分布。选择 layer 17（KL=1.21）而非 layer 18（KL=2.28）正是这种权衡的体现。**这说明 Qwen2.5 的 refusal 方向与一般语言建模能力的耦合比 Llama3.1 更紧密**——ablate 时更容易引入副作用，这一点在 VLM 实验设计时应考虑。

### 1.3 两 Judge 的差异：SRR 的首次出现

DIM 结果出现了一个**有趣的 SRR 信号**：

- Qwen2.5：ASR_kw=100.0%，ASR_LG3=94.5%，**SRR=+5.5pp**
- Llama3.1：ASR_kw=98.4%，ASR_LG3=89.8%，**SRR=+8.6pp**

这意味着：在 keyword judge 认为已经 jailbreak 的样本里，有 5.5%（Qwen）到 8.6%（Llama）并未被 LlamaGuard3 判为真正有害。

**这 5-9% 的差异意味着什么？** 两种解释：
1. **Keyword 误判（最可能）**：模型输出了不含拒绝前缀的文本，但内容是无害的混淆或绕题，LG3 正确识别为 safe。
2. **Stealth Refusal 的极小信号**：模型在没有显性拒绝词的情况下，通过语义拒绝或提供无用内容来规避实质性帮助，LG3 将其判为 safe。

在 ablation 后的纯 LLM 上，解释 1 更合理——方向已经被移除，模型很难有内聚的"stealth"能力。但这个 5-9pp 的基础差值对后续 VLM 实验极为重要：**它给出了 LLM 上 SRR 的"噪声底线"**——如果 VLM 实验中 SRR 显著超过这个范围，才值得作为 stealth refusal 的证据。

### 1.4 StrongREJECT 与 Keyword 的高度一致

Ablation completions 上：
- Qwen：ASR_kw=100.0% vs ASR_SR=99.2%（差 0.8pp）
- Llama：ASR_kw=98.4% vs ASR_SR=98.4%（完全一致）

SR 和 kw 在 ablation 后高度一致（差值<1pp），但 LG3 和 kw 的差异是 5-9pp。这提示 **LG3 比 SR 更严格/保守**——LG3 更倾向于把那些"形式上看起来在回答但内容有限"的输出判为 safe。对于后续 VLM 实验，应优先信赖 LG3 作为 ASR 上界，SR 作为下界。

---

## 2. RDO k=1 验证结论

### 2.1 核心数字

| Model | ASR_kw | ASR_LG3 | ASR_SR | mean_SR | SRR (kw−LG3) | vs DIM ASR_kw |
|-------|:------:|:-------:|:------:|:-------:|:------------:|:-------------:|
| Qwen2.5-7B-Instruct   | 90.6% | 76.6% | 88.3% | 0.636 | **+14.1pp** | −9.4pp |
| Llama-3.1-8B-Instruct | 100.0% | 91.4% | 100.0% | 0.765 | **+8.6pp** | +1.6pp |

**判定：RDO pipeline 验证通过（定性）。** 两个模型的 ablation ASR 均远超 baseline。Llama-3.1 的 RDO 效果与 DIM 持平甚至略优；Qwen2.5 的 RDO 绝对 ASR 低于 DIM 约 9pp，但仍在 90% 以上，属于可接受的定性复现。

### 2.2 训练动态：两模型的收敛差异

从训练日志（bypass_score 越负越好，induce_score 越正越好）：

**Llama-3.1 RDO（alpha=3.625，快速稳定收敛）**

| 阶段 | bypass | induce | 描述 |
|------|:------:|:------:|------|
| Step 16 | +2.9 | −8.25 | 初始：方向仍在强化拒绝 |
| Step 80 | −3.2 | +8.6 | 仅 80 步完成方向反转 |
| Step 128–480 | −5~−9 | +7~+8 | 稳定高值平台期 |

**Qwen2.5 RDO（alpha=31.375，缓慢且弱收敛）**

| 阶段 | bypass | induce | 描述 |
|------|:------:|:------:|------|
| Step 16 | +2.7 | −12.1 | 初始：方向极强拒绝诱导 |
| Step 48 | −0.6 | +0.07 | 方向反转但 induce 几乎为零 |
| Step 80–130 | +0.7~+3.1 | +3~+4.7 | 不稳定，bypass 多次变正 |
| Step 240–416 | −2.6~−6.1 | +1.4~+2.6 | 勉强收敛，induce 仅约 2 |

**关键洞察 3：Qwen2.5 的 RDO 优化难度显著高于 Llama3.1，alpha 差异（31.4 vs 3.6）是核心线索。**

`add_layer: 17, alpha: 31.375`（Qwen）vs `add_layer: 13, alpha: 3.625`（Llama）。这个 alpha 是 RDO 使用 DIM 初始方向做 activation addition 时的系数。Qwen 的 alpha 约为 Llama 的 **8.7 倍**，说明为了产生等效的激活偏移，需要在 Qwen 上施加大得多的系数——**Qwen2.5 的 refusal direction 的有效"杠杆率"（每单位方向向量对激活的影响）远低于 Llama3.1**。这可能意味着 Qwen 的 refusal 更依赖多个方向的协同（更冗余/分布式），单一方向的影响力被稀释。

这一发现对 VLM 实验有直接意义：**若 Qwen2.5-VL 继承了这一特性，单方向 DIM ablation 可能是不够的，需要 RDO 或 Cone 才能充分消除其多模态拒绝机制。**

### 2.3 Qwen RDO 的大 SRR：+14.1pp

Qwen2.5 的 RDO SRR = 90.6% − 76.6% = **+14.1pp**，是其 DIM SRR（5.5pp）的 **2.6 倍**，也是本实验所有 config 中最高的单一 SRR。

为什么 RDO 方向的 SRR 比 DIM 方向更大？

- DIM 方向是 mean-diff：harmful/harmless 隐态均值之差，直接编码"是否拒绝"的线性分量。它被 ablate 后，模型倾向于输出近似有害内容。
- RDO 方向是梯度优化的：目标是让模型在有害 prompt 上产生 DIM 预先生成的 ablation 目标 token（即"先 bypass 一步"的输出），并在 harmless prompt 上输出拒绝短语。这两个目标有时相互干扰——Qwen 的弱 induce（≈2）说明 RDO 优化没有找到一个能强力同时满足两个目标的方向。
- **结果**：RDO 方向在 Qwen 上产生的 ablation 输出，更多是"格式上不拒绝但内容含糊或偏题"的响应，LG3 将其判为 safe 的概率更高，造成 SRR 放大。

**这是一个重要的方法论警示**：SRR 不仅仅取决于模型是否有 stealth refusal 能力，还取决于 ablation 方向的质量——优化不充分的方向（弱 induce）会人为放大 SRR。

### 2.4 Llama RDO 的完美天花板

Llama-3.1 的 RDO ASR_kw=100%，ASR_SR=100%，与 DIM 相比甚至略有提升（DIM ASR_SR=98.4%）。训练收敛曲线在 step 80 即达到稳定，induce_score 持续维持 7-8，说明梯度优化找到了一个比 mean-diff（DIM）更强的方向。

这与 Wollschläger (2025) 的结论一致：RDO 通过优化目标直接驱动，能找到比统计方向更有效的 ablation 向量。**对于 Llama3.1 这样 refusal 集中（单层主导、低 KL 耦合）的模型，RDO 可以精确命中并完全消除拒绝机制。**

---

## 3. Cone k=3 验证结论

### 3.1 核心数字

§2 已揭示两模型的核心差异：Llama RDO 在 80 步内完成收敛并触及 100% 天花板，而 Qwen RDO 收敛弱且 ASR 下滑 9pp。Cone k=3 正是在这个断层上检验"多维子空间能否弥补单方向的不足"。

| Model | ASR_kw | ASR_LG3 | ASR_SR | mean_SR | SRR (kw−LG3) | vs DIM ASR_kw | vs RDO ASR_kw |
|-------|:------:|:-------:|:------:|:-------:|:------------:|:-------------:|:-------------:|
| Qwen2.5-7B-Instruct   | 98.4% | 93.8% | 98.4% | 0.718 | **+4.7pp** | −1.6pp | **+7.8pp ↑** |
| Llama-3.1-8B-Instruct | 100.0% | 90.6% | 100.0% | 0.768 | **+9.4pp** | +1.6pp | 持平 |

**判定：Cone k=3 验证通过，且对 Qwen2.5 有显著修复作用。**

- **Qwen2.5**：ASR_kw 从 RDO 的 90.6% 回升至 98.4%，LG3 从 76.6% 恢复至 93.8%，接近 DIM 上限（100% / 94.5%）。3 维子空间将 RDO 单向量遗漏的残余 refusal 方向一并消除。
- **Llama-3.1**：Cone k=3 与 RDO k=1 均达 100% ASR_kw；LG3 差 0.8pp（91.4% → 90.6%），在误差范围内，无实质性增益。

### 3.2 训练动态：新向量的初始化模式

Cone 训练日志揭示了一个一致的**新向量初始化规律**：

每当增加第 k 个新向量时，该向量在训练初期（step 16）的 `induce_score` 几乎总是大幅负值（约 −10 到 −13），而 `bypass_score` 为正。以 Qwen k=3 为例：

```
Step 16 [k=3新向量]: bypass=+2.85, induce=−12.98   ← 极强拒绝诱导方向
Step 48 [k=3新向量]: bypass=+0.04, induce=+3.97    ← 开始反转
Step 128[k=3新向量]: bypass=+2.19, induce=+5.11    ← 趋于稳定
```

**关键洞察 4：新 basis 向量从"最强拒绝"方向出发，通过优化逐步反转。**

这一模式在 Qwen 和 Llama 的所有 k 阶段均成立，反映了 Cone 初始化策略的几何含义：新向量被初始化为与现有 basis 正交的方向，而在当前残差空间中，与已有 bypass 方向正交的方向往往正好是最强化拒绝的子空间。优化器需要从这里出发"爬坡"，将其翻转为 bypass 方向。**这说明 refusal 子空间本身具有丰富的内部结构——越往外层扩展，残余方向越难优化**，与 Wollschläger (2025) 关于 Cone 训练难度随 k 增大的论述一致。

### 3.3 Qwen 的"维度依赖"：Cone 补偿了 RDO 的不足

回顾三个 Qwen 结果：

| Config | ASR_kw | ASR_LG3 | SRR |
|--------|:------:|:-------:|:---:|
| DIM（统计方向，1D） | 100.0% | 94.5% | +5.5pp |
| RDO k=1（梯度方向，1D） | 90.6% | 76.6% | +14.1pp |
| Cone k=3（梯度方向，3D） | 98.4% | 93.8% | +4.7pp |

这个 pattern 揭示了 Qwen2.5 refusal 机制的**维度依赖性**：

- **DIM** 用均值差统计出最主要的 1 个方向，因其来自大量样本均值，已经隐含了多方向的"平均效应"，所以效果最强。
- **RDO k=1** 用梯度优化只找到 1 个最优方向，但 Qwen 的 refusal 被分散在多个方向，单向量不够用，导致 ASR 下降且 SRR 虚高。
- **Cone k=3** 用 3 个正交向量，覆盖更大的 refusal 子空间，ASR 回升到接近 DIM 水平，SRR 也大幅回落到 4.7pp——与 DIM 的 5.5pp 接近，说明这 3pp 的 kw−LG3 差距更接近"真实噪声底线"。

**这与洞察 3（第 2 节）形成了完整闭环**：RDO 在 Qwen 上的 alpha 大（31.4）、induce 弱（≈2），正是因为单方向无法捕获多维度分布的 refusal；而 Cone k=3 通过扩展到 3 维子空间，恢复了单方向 DIM 的统计覆盖效果。**这一发现强化了"Qwen2.5 的 refusal 机制比 Llama3.1 更分布式"的假设（洞察 1）**，现在有了来自三种方法对比的三角验证。

### 3.4 Llama 的"向量冗余"：k=1 已经够用

对 Llama-3.1，Cone k=3 vs RDO k=1 的差距完全可以忽略：

| Config | ASR_kw | ASR_LG3 | SRR |
|--------|:------:|:-------:|:---:|
| RDO k=1 | 100.0% | 91.4% | +8.6pp |
| Cone k=3 | 100.0% | 90.6% | +9.4pp |

LG3 结果几乎相同，SRR 小幅上升 0.8pp 在误差范围内。从训练日志看，Llama Cone k=3 第二、三个向量的 `induce_score` 稳定在 3-8，bypass 全为负，说明这些向量确实是有效的 bypass 方向——只是它们所覆盖的 refusal 子空间，原本已经被第一个向量（RDO k=1）捕获的主方向完全处理掉了。

**关键洞察 5：Llama3.1 的 refusal 是"低秩"的，1 维已足够；Qwen2.5 的 refusal 是"高秩"的，需要至少 3 维才能有效覆盖。**

这一发现对 VLM 实验的方法论选择至关重要：对 Qwen2.5-VL，应优先使用 Cone k≥3 而非单方向 RDO；对 Llama3.1-VL，RDO k=1 已经充分。

### 3.5 SRR 在 Cone k=3 的重新定位

经过 DIM → RDO → Cone k=3 三个 config 的对比，现在可以更精确地理解 SRR 的来源：

| Config | Qwen SRR | Llama SRR | SRR 来源 |
|--------|:--------:|:---------:|----------|
| DIM | +5.5pp | +8.6pp | 噪声底线（keyword 误判） |
| RDO k=1 | **+14.1pp** | +8.6pp | Qwen：ablation 不充分人为放大 |
| Cone k=3 | **+4.7pp** | +9.4pp | Qwen：接近噪声底线；Llama：稳定基线 |

Cone k=3 将 Qwen 的 SRR 从 14.1pp 压回 4.7pp，低于 DIM 的 5.5pp。这进一步确认：**在 LLM 上，SRR 的主要驱动因素是 ablation 质量，而非 stealth refusal 能力**。只有在 ablation 充分（Cone k≥3 对 Qwen，或任意方法对 Llama）的前提下，剩余的 SRR 才具有解释意义——而此时两模型的 SRR 均在 5-10pp 范围内，构成 VLM 实验的对照基线。

---

---

## 4. Cone k=5 验证结论

### 4.1 核心数字

| Model | ASR_kw | ASR_LG3 | ASR_SR | mean_SR | SRR (kw−LG3) | vs Cone k=3 ASR_kw | vs Cone k=3 ASR_LG3 |
|-------|:------:|:-------:|:------:|:-------:|:------------:|:------------------:|:-------------------:|
| Qwen2.5-7B-Instruct   | 99.2% | 92.2% | 99.2% | 0.711 | **+7.0pp** | +0.8pp | **−1.6pp ↓** |
| Llama-3.1-8B-Instruct | 100.0% | 91.4% | 100.0% | 0.741 | **+8.6pp** | 持平 | +0.8pp |

**判定：Cone k=5 验证通过，但收益递减信号明确。**

- **Qwen2.5**：ASR_kw 仅从 98.4% 微升至 99.2%（+0.8pp），但 ASR_LG3 反而从 93.8% 下降至 92.2%（−1.6pp），SRR 从 +4.7pp 上升至 +7.0pp。收益递减且出现质量劣化迹象。
- **Llama-3.1**：ASR_kw 维持 100%，ASR_LG3 从 90.6% 微升至 91.4%（+0.8pp，在误差范围内），SRR 从 +9.4pp 回落至 +8.6pp。与 k=3 实质等价。

### 4.2 Qwen 的维度饱和：k=5 打破了 k=3 的平衡

将 Qwen 四个 config 的关键指标完整对比：

| Config | ASR_kw | ASR_LG3 | mean_SR | SRR | 解读 |
|--------|:------:|:-------:|:-------:|:---:|------|
| DIM（1D 统计）  | 100.0% | 94.5% | — | +5.5pp | 均值方向，隐含多维平均 |
| RDO k=1（1D 梯度）| 90.6% | 76.6% | 0.636 | +14.1pp | 欠拟合，单向量不够用 |
| Cone k=3（3D 梯度）| 98.4% | 93.8% | 0.718 | +4.7pp | **最优平衡点** |
| Cone k=5（5D 梯度）| 99.2% | 92.2% | 0.711 | +7.0pp | ASR_kw 微升，LG3 下滑 |

从 k=3 到 k=5，ASR_kw 提升了 0.8pp，但 ASR_LG3 下降了 1.6pp，SRR 上升了 2.3pp。这个组合模式说明 **k=5 的额外两个 basis 向量并未指向新的有效 refusal 子空间，而是引入了一定程度的激活扰动**，产生了"绕开关键词但内容质量下降"的输出——LG3 将更多此类输出判为 safe，SRR 因此被虚抬。

从训练日志也能直接看到这个退化：在 k=5 的 step 400-464，各向量终态得分为：

```
v1: bypass≈−7~−8, induce≈5    ← 强主向量（与 RDO k=1 近似）
v2: bypass≈−2,   induce≈7    ← 中等，k=3 已覆盖
v3: bypass≈−2~−3, induce≈1~2 ← 弱，有效性存疑
v4: bypass≈ 0,   induce≈3~4  ← bypass 几乎为零，方向接近中性
v5: bypass≈+0.5, induce≈5~6  ← bypass 为正（仍在拒绝区间!）
```

v4 的 bypass≈0、v5 的 bypass 甚至为正，说明这两个向量从未完全收敛到有效的 bypass 方向。它们占据了激活空间的一部分，却没有对应地消融拒绝行为，反而可能干扰了 v1-v3 已经建立的有效消融结构。

**关键洞察 6：Qwen2.5 的 refusal 子空间有效维度约为 3，k=5 已进入过度参数化区间。** 超过有效维度后，多余的 basis 向量无法找到有意义的残余 refusal 方向，其收敛行为趋于随机扰动，反而破坏了低维时建立的消融质量。这与洞察 5（"Qwen refusal 高秩，但有界"）构成了完整的维度图景：有效秩 ≈ 3，而非无上界。

### 4.3 Llama 的维度无关性：k=5 与 k=1 实质等价

Llama 在 k=5 的训练动态与 k=3 完全平行：所有 5 个向量最终均收敛到有效区间（bypass 负，induce 正），mean_sample 终态保持 bypass≈−6~−8，induce≈7~9，全程强健。但这些向量对 ASR 的贡献依然为零，因为 Llama 的主 refusal 轴在 k=1 时已被 RDO 精确命中并完全消除。

对 Llama 来说，k=3 到 k=5 的额外 basis 向量消融的是"已经不存在拒绝行为"的空间——这些样本在 k=1 时就已经 bypass，更多的维度无从累积收益。**这再次从反面验证了洞察 5：低秩结构的模型对 Cone 维度不敏感，高秩结构的模型对维度有明确的有效上界。**

值得注意的是，Llama k=5 的 SRR 从 k=3 的 +9.4pp 回落至 +8.6pp（与 DIM、RDO k=1 相同）。这说明 Llama 的 SRR 在 8-10pp 之间存在一个**结构性稳定区间**，与具体消融维度无关——它更可能反映的是 LlamaGuard3 本身对 Llama-3.1 输出风格的系统性判断偏差，而非模型内部 refusal 结构的特征。

### 4.4 跨方法全局对比：四个 config 的统一图景

综合 DIM / RDO / Cone k=3 / Cone k=5 的完整结果，可以画出两个模型的 ASR_LG3 变化轨迹（LG3 作为最严格的语义 judge，最能反映真实 bypass 质量）：

**Qwen2.5-7B ASR_LG3 轨迹：**
```
baseline   5.5%
DIM       94.5%  ← 统计主轴，覆盖多维均值，效果最强
Cone k=3  93.8%  ← 接近 DIM，梯度方法 3D 已足够
Cone k=5  92.2%  ← 微降，过度参数化开始引入噪声
RDO k=1   76.6%  ← 梯度单向量，欠拟合
```

**Llama-3.1-8B ASR_LG3 轨迹：**
```
baseline   2.3%
RDO k=1   91.4%  ← 1 维已饱和
Cone k=5  91.4%  ← 与 RDO k=1 完全相同
Cone k=3  90.6%  ← 在误差范围内
DIM       89.8%  ← 统计方向略弱于梯度方向（对低秩模型，梯度方法更精准）
```

这两条轨迹揭示了一个对比鲜明的规律：

- **Llama**：梯度方法（RDO）优于统计方法（DIM），因为其 refusal 集中在单轴，梯度能精准命中；维度增加完全无益。
- **Qwen**：统计方法（DIM）优于任何单向量梯度方法，因为其 refusal 分散在多轴，均值方向天然涵盖了最优投影；增加梯度维度（Cone）能部分弥补，但在 k≈3 时已达到统计方法的覆盖效果，继续增加维度反而引入噪声。

**关键洞察 7（全局）：refusal 子空间的秩决定了最优消融策略。** 低秩（Llama）→ 梯度单向量最优；高秩（Qwen）→ 统计均值方向（DIM）或梯度多维子空间（Cone k≈3）最优，两者等效但上界不同。这一规律对 VLM 实验的方法论选择具有直接指导意义。

---

## 5. 全局 SRR 附带观测

### 5.1 完整数据汇总（含基线）

以下汇总两个模型在全部消融方法（含未消融基线）上的 SRR（= ASR\_kw − ASR\_LG3）。前四节已逐 config 分析了各自的数字，本节将它们并排放置，给出全局对照视图。

| 模型 | Config | ASR\_kw | ASR\_LG3 | ASR\_SR | SRR (kw−LG3) | 见前文 |
|------|--------|:-------:|:--------:|:-------:|:------------:|:------:|
| **Qwen2.5-7B** | **baseline（未消融）** | 11.7% | 5.5% | — | **+6.2pp** | — |
| Qwen2.5-7B | DIM ablation | 100.0% | 94.5% | 99.2% | **+5.5pp** | §1.3 |
| Qwen2.5-7B | RDO k=1 | 90.6% | 76.6% | 88.3% | **+14.1pp** | §2.3 |
| Qwen2.5-7B | Cone k=3 | 98.4% | 93.8% | 98.4% | **+4.7pp** | §3.5 |
| Qwen2.5-7B | Cone k=5 | 99.2% | 92.2% | 99.2% | **+7.0pp** | §4.1 |
| **Llama-3.1-8B** | **baseline（未消融）** | 2.3% | 2.3% | — | **+0.0pp** | — |
| Llama-3.1-8B | DIM ablation | 98.4% | 89.8% | 98.4% | **+8.6pp** | §1.3 |
| Llama-3.1-8B | RDO k=1 | 100.0% | 91.4% | 100.0% | **+8.6pp** | §2.1 |
| Llama-3.1-8B | Cone k=3 | 100.0% | 90.6% | 100.0% | **+9.4pp** | §3.4 |
| Llama-3.1-8B | Cone k=5 | 100.0% | 91.4% | 100.0% | **+8.6pp** | §4.3 |

*n=128（全部 config）；baseline ASR\_SR 以 mean\_sr 代替（Qwen 0.099，Llama 0.010），未列入表中。*

### 5.2 基线 SRR：消融前的噪声已不对称

表中最值得注意的是两个模型**未消融基线**的 SRR：

- **Qwen2.5-7B baseline SRR = +6.2pp**：在完全未消融的状态下，128 个有害 prompt 中已有 15 个（11.7%）绕过关键词过滤器，但其中只有 7 个（5.5%）被 LG3 判为有害——**即在没有任何 direction ablation 的情况下，关键词误判已达 +6.2pp**。抽查这 15 条输出，它们并非真正越狱，而是以下几类：事实纠错型（prompt 预设了错误事实，模型给出纠正，既无拒绝词也无害）、专业建议型（医疗/法律/投资建议，有实质内容但不含显式有害指令）、免责声明式（以"It's important to…"开头后给出部分信息，绕开 kw 过滤但不满足 LG3 有害标准）。

- **Llama-3.1-8B baseline SRR = +0.0pp**：128 个有害 prompt 中仅 3 个（2.3%）绕过关键词，且这 3 个同时被 LG3 判为有害，双 judge 完全一致。Llama 在未消融状态下几乎没有输出噪声。

这一基线差异与 §1.2 形成直接呼应：**Qwen2.5 的 refusal 方向与通用语言建模能力耦合更紧（高 KL），其输出分布本身更"复杂"，更容易在不触发拒绝词的情况下给出内容边界模糊的响应**。这 6.2pp 的基线噪声是模型特性，与 stealth refusal 能力无关。

### 5.3 消融后 SRR 的变化规律：与前各节对照

| Config | Qwen SRR | Llama SRR | 主要来源（详见对应章节） |
|--------|:--------:|:---------:|------------------------|
| baseline | +6.2pp | +0.0pp | 模型输出风格差异（基线噪声） |
| DIM | +5.5pp | +8.6pp | kw 误判为主（§1.3），不含 stealth 信号 |
| RDO k=1 | **+14.1pp** | +8.6pp | Qwen：ablation 不充分引发语义绕道（§2.3）；Llama：稳定基线 |
| Cone k=3 | +4.7pp | +9.4pp | Qwen：回落至基线水平，ablation 质量恢复（§3.5）；Llama：误差范围内 |
| Cone k=5 | +7.0pp | +8.6pp | Qwen：过度参数化轻微抬升（§4.2）；Llama：与 k=1/k=3 等价 |

两条规律从表中直接可读：

**规律一（Qwen）：SRR 是 ablation 质量的函数。** Qwen 的 SRR 轨迹为：Cone k=3（4.7pp，最低）→ DIM（5.5pp）→ 基线（6.2pp）→ Cone k=5（7.0pp）→ RDO k=1（14.1pp，最高）。SRR 的最低点恰好是 ASR 最优的方法（Cone k=3），最高点恰好是 ablation 最不充分的方法（RDO k=1）。机制上，RDO 对 Qwen 的 induce\_score 仅约 2（§2.2），弱收敛使模型大量输出"不说拒绝词但内容绕道"的响应（如将"宣扬体型羞辱的社媒活动"改写为"宣扬身体正能量"的活动），LG3 将这些语义绕道判为 safe，虚高了 SRR。

**规律二（Llama）：SRR 与 ablation 方法无关，稳定在 8–10pp。** Llama 的 SRR 在所有 config 之间仅有 ±0.8pp 波动，且与基线（0.0pp）完全脱钩。结合 §3.4 中 Llama 各维度训练动态——额外 basis 向量均正常收敛但对 ASR 零贡献——可以确认：**这 8–10pp 是 LlamaGuard3 对 Llama-3.1 输出风格的系统性判断偏差**，约 9% 的完全 bypass 输出在语义上有害（通过 kw + SR），但 LG3 仍判为 safe。这是 judge 间的固有分歧，不是模型内部 stealth 能力的体现。

### 5.4 "充分消融后"的 LLM 噪声底线

当消融方法充分（Qwen: Cone k=3 或 DIM；Llama: 任意方法），两个模型的 SRR 稳定在：

| 模型 | 充分消融后 SRR 区间 | 参照基线 SRR |
|------|:-----------------:|:-----------:|
| Qwen2.5-7B | **4.7 ~ 5.5pp** | 6.2pp（模型输出风格噪声） |
| Llama-3.1-8B | **8.6 ~ 9.4pp** | 0.0pp（judge 偏差，非模型噪声） |

**这两个区间构成 LLM 实验的 SRR 噪声底线，是后续 VLM 实验进行 SRR 对比的参照锚点。** 若 VLM 实验中观察到的 SRR 超出上述范围，才值得作为超出"基础噪声"的信号加以分析。考虑到 VLM 同样经历了对齐训练，其内生对齐机制不亚于 LLM，任何 VLM 上的 SRR 都应首先对照这组 LLM 基线，排除 ablation 充分性和 judge 偏差两类方法论因素后，再讨论其可能的 stealth 来源。

> **注**：本节 SRR 数字均为附带观测，评估对象是 LLM（Qwen2.5-7B / Llama-3.1-8B），基于 SaladBench n=128 harmful\_test split。VLM 同样经过对齐训练，内生对齐不是 LLM 独有属性，因此 LLM 上存在或不存在 SRR 都不直接证明"Stealth Refusal 是 LLM 特有问题"。本节数据的唯一用途是为 VLM 实验提供方法论对照锚点。

---

## 6. 对 VLM 实验的方法论启示

本节从前五节的七个关键洞察中提炼出对后续 VLM 实验（V1/T0/M1/A1/A3）的具体方法论建议。

### 6.1 消融方法选择：以模型的 refusal 秩为依据

洞察 5 和 7 表明，最优消融策略取决于 refusal 子空间的秩，而非某种方法是否"更新"：

| 模型系列 | LLM 上的秩特征 | 推荐 VLM 消融方法 |
|---------|:-------------|:----------------|
| **Qwen2.5-VL** | 高秩（有效维度≈3，DIM 与 Cone k=3 等效） | 优先 **Cone k=3**；DIM 可作快速基线；**避免**单方向 RDO（ASR 下滑 9pp，SRR 虚高 14pp） |
| **Llama-3.1-VL** | 低秩（1 维饱和，k≥2 无增益） | **RDO k=1 或 DIM** 均已足够；Cone 带来的计算成本不值回报 |

这一建议不依赖 VLM 的具体架构，而是基于 LLM 骨干模型的固有属性——VLM 的文本 decoder 继承自对应 LLM，其 refusal 子空间的秩特征大概率相同。若 VLM 实验显示方法效果与 LLM 预测明显偏离，应首先检查多模态编码器是否引入了新的 refusal 通路（参见 6.3）。

### 6.2 Hook 深度：针对两模型的不同策略

洞察 1 指出两个模型的 refusal 方向层位不同（Qwen: layer 17/28 = 61%；Llama: layer 13/32 = 41%），洞察 2 进一步说明 Qwen 中段存在高 KL 层（layer 22–23，KL>4.9），ablate 这些层会严重破坏 harmless prompt 上的输出分布。这对 VLM 实验的 hook 设计有直接约束：

- **Qwen2.5-VL**：hook 应锚定在 decoder 的 **60% 深度附近**（对应 28 层 transformer 的 layer 17 位置）。不要盲目把 hook 扩展到 KL 高的中段层（layer 22–23 等价位置），否则会引入无关副作用，使评估结果难以解释。
- **Llama-3.1-VL**：hook 锚定在 **40% 深度附近**（32 层的 layer 13 位置）。Llama 全层 KL 普遍低（<0.15），hook 位置容错范围比 Qwen 更宽。
- **token 位置**：两模型均在 pos=-5（prompt 尾部倒数第 5 个 token）处取到最优 direction，建议 VLM 实验沿用此设置，仅在视觉 token 显著改变 prompt 长度时重新搜索。

### 6.3 SRR 的解读：建立正确的对照关系

§5 给出了 LLM 上的 SRR 噪声底线（Qwen 4.7–5.5pp，Llama 8.6–9.4pp）。VLM 实验中使用该底线时，需注意以下三层对照逻辑：

1. **先确认消融充分性**：若 VLM 上 SRR 偏高，首先排除"ablation 不充分"——对 Qwen2.5-VL 用 RDO k=1 而非 Cone k=3，就会因同样的欠拟合机制（洞察 3）人为抬升 SRR 14pp，与 LLM 上的结果完全一致，不能归因于 VLM 特有能力。

2. **再排除 judge 偏差**：Llama 系列模型上 LG3 的系统性偏差约 8–9pp（§5.3 规律二）。若 Llama-VL 实验中 SRR 落在同一区间，不具有解释意义；只有显著超出（>12pp）才值得进一步分析。

3. **最后讨论 stealth 来源**：在消融充分且排除 judge 偏差之后，若 VLM 上的 SRR 仍显著超出 LLM 底线，才应引入"视觉模态引入了新的 stealth 拒绝通路"这一假说。此时需要配合视觉 token 的激活模式分析（V1/M1 实验）加以验证，而不能仅凭数字差异下结论。

### 6.4 数据集差异的注意事项

本次 LLM 实验使用 SaladBench（n=128 harmful\_test），而非 Arditi/Wollschläger 原文使用的 JailbreakBench 或 AdvBench。VLM 实验如果改用其他数据集，需注意：

- SaladBench 的 harmful 类别包含较多"事实纠错型"prompt（如要求模型支持错误观点），此类 prompt 即使没有消融也会有一定比例绕过关键词过滤器（构成基线噪声），在不同数据集上此比例可能不同，影响 SRR 基线。
- 跨数据集比较 SRR 数字时，应先单独计算该数据集上的未消融基线 SRR，以确认噪声底线，不能直接套用本节的 4.7–9.4pp 范围。

---

## 7. Pipeline 局限与已知问题

### 7.1 修复的 Bug 清单

本次复现过程中发现并修复了 3 个已有 pipeline 的 bug：

| # | 文件 | 问题 | 修复方式 |
|---|------|------|---------|
| T1 | `refusal_direction/pipeline/model_utils/qwen_model.py:79–92` | `orthogonalize_qwen_weights` 和 `act_add_qwen_weights` 使用 Qwen-1 路径（`model.transformer.wte` / `model.transformer.h`），在 Qwen2.5 上触发 AttributeError | 改用 `model.model.embed_tokens` / `model.model.layers[i].self_attn.o_proj` / `model.model.layers[i].mlp.down_proj` |
| T2 | `refusal_direction/pipeline/model_utils/llama3_model.py:16,22` | `LLAMA3_CHAT_TEMPLATE` 和 `LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM` 开头多余一个 `"` 字符，导致每条 prompt 前多插入一个引号 | 删除两处多余的前导 `"` |
| T3 | `refusal_direction/pipeline/submodules/evaluate_jailbreak.py:11–24` | 拒绝前缀列表缺少 Unicode 右单引号（U+2019）变体，LLM 输出"I'm sorry"（curly quote）时被误判为越狱 | 追加 4 条 U+2019 变体 |

这三个 bug 在修复前对实验结果的影响：T1 仅在调用 weight-orthogonalization 路径时崩溃（DIM 的 hook-based ablation 不受影响），T2 影响 Llama 全部 prompt 的格式（可能轻微影响基线 ASR 数字），T3 影响拒绝检测的召回率（smart-quote 输出被误计为越狱，轻微虚高 ASR\_kw）。

### 7.2 数据集与原论文的偏差

- **评估数据集不同**：Arditi (2024) 使用 JailbreakBench（100 条）；Wollschläger (2025) 使用 AdvBench；本次使用 SaladBench（n\_test=128）。SaladBench 的 harmful 类别更宽泛，包含较多"误导性事实"类 prompt，导致基线 ASR\_kw 高于 JailbreakBench（Qwen 11.7%，而非原文的接近 0%）。**本次结果只验证定性趋势，数值不可与原文直接对比。**
- **模型版本**：Arditi 原文使用 Llama-2-7B；本次使用 Llama-3.1-8B（对齐更强）。Wollschläger 的 RDO/Cone 原文使用 Llama-2-13B 和 Gemma-7B；本次为 Qwen2.5-7B 和 Llama-3.1-8B。定性结论可迁移，绝对数字不可比。

### 7.3 未覆盖的情况

- **Weight orthogonalization（T1 路径）**：本次 DIM 实验只用 hook-based activation ablation，未测试 weight-space orthogonalization。T1 的修复保证了该路径不崩溃，但实际效果未经验证。
- **Cone k=4**：根据训练动态（§4 v4 的 bypass≈0），k=4 的边际效益极低，且 Wollschläger 原文主要报告 k=3/k=5 结果，故跳过单独评估。
- **StrongREJECT 基线**：evaluation.json 中 baseline 行未包含 ASR\_SR（仅有 mean\_sr\_baseline≈0.099/0.010），因此 §5.1 表格中 baseline 列 ASR\_SR 为 "—"，不影响 SRR 计算（SRR 仅用 kw 和 LG3）。
- **Cone k=3 中间 basis 的单独提取**：本次 Cone k=3 的 basis 来自 `--min_cone_dim 2 --max_cone_dim 5` 的中间快照，而非单独跑 `--max_cone_dim 3`，两者优化路径略有差异。

---

## 8. 推荐的下一步实验

基于本次七个关键洞察，按优先级排列后续实验建议：

| 优先级 | 实验 | 依据洞察 | 核心问题 |
|:---:|------|---------|---------|
| P0 | **V1**: Direction validity via activation addition causal check | 洞察 1、2 | Qwen2.5-VL 的 refusal 方向是否继承自 LLM backbone？激活加法在多模态输入下的因果效应是否稳定？ |
| P0 | **T0**: Stealth refusal 起源定位（multimodal vs text-only） | 洞察 7、§5.4 | 在充分消融（Qwen: Cone k=3；Llama: RDO k=1）且排除 judge 偏差后，VLM 上的 SRR 是否超出 LLM 底线（4.7–9.4pp）？ |
| P1 | **M1**: Layer-wise ablation sensitivity heatmap | 洞察 1、2 | Qwen2.5-VL 的高 KL 层（~layer 22–23 等价位置）在多模态输入下是否出现同样的 KL 震荡？最优 hook 层是否随视觉 token 的引入而漂移？ |
| P1 | **A1**: Refusal cone 在 VLM 上的维度扫描 | 洞察 5、6 | Qwen2.5-VL 的 refusal 有效秩是否仍为 3，还是视觉模态增加了新的 refusal 维度（有效秩>3）？ |
| P2 | **A3**: 跨模态 refusal direction 迁移性 | 洞察 3、7 | 从纯文本输入提取的 DIM direction 是否对视觉-文本混合输入同样有效？alpha 系数（Qwen: 31.4 vs Llama: 3.6）在多模态设置下是否需要重新标定？ |

**最高优先级的操作问题**：在开始任何 VLM 实验之前，先用 T0 smoke test 的设计（32 个有害 prompt + 视觉 token，n\_test 小批量）验证 Qwen2.5-VL / LLaVA 的 direction ablation 是否有效（类比 LLM 实验的 T6 gate），确认 pipeline 可用后再进入全量实验。

---

## 附录：结果文件导航

| 内容 | 路径 |
|------|------|
| ASR 汇总表（含 SRR） | `results/repro_arditi_wollschlager/summary.md` |
| 全量评估 JSON | `results/repro_arditi_wollschlager/evaluation.json` |
| DIM 产出（direction + completions） | `results/repro_arditi_wollschlager/dim/{Qwen2.5-7B-Instruct,Llama-3.1-8B-Instruct}/` |
| RDO k=1 completions | `results/repro_arditi_wollschlager/rdo/{model}/rdo_k1/completions/` |
| Cone k=3 completions | `results/repro_arditi_wollschlager/rdo/{model}/cone_k3/completions/` |
| Cone k=5 completions | `results/repro_arditi_wollschlager/rdo/{model}/cone_k5/completions/` |
| GPU 运行日志 | `experiments/repro_arditi_wollschlager/logs/` |
| 进度记录 | `experiments/repro_arditi_wollschlager/PROGRESS.md` |
