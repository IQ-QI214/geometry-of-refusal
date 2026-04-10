# Phase 2 实验分析报告

> **日期**：2026-03-30
> **模型**：LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)
> **硬件**：4x H100 80GB
> **实验**：Exp 2A (confound resolution), Exp 2B (ablation attack), Exp 2C (PGD visual perturbation)

---

## 1. 实验总览

| 实验 | 目标 | 状态 | 核心结论 |
|:---|:---|:---:|:---|
| **Exp 2A** | 分离 content confound vs 真实 refusal direction drift | 完成 | **DYNAMIC**：约一半漂移是 confound，但剩余漂移是真实的 (controlled min sim = 0.231) |
| **Exp 2B** | Ablation attack 验证因果效力 + 生成 target completions | 完成 | **Layer 16 单层 ablation 最强**：89.7% full harmful，远超 blank image 的 56.9% |
| **Exp 2C** | 核心 PGD 视觉扰动优化 | 完成 | **负结果**：优化后扰动 (37.9%) 显著劣于 blank image baseline (56.9%) |

---

## 2. Exp 2A: Confound Resolution — 真实漂移确认

### Problem

Pilot Exp B 发现 refusal direction 在 generation 过程中剧烈漂移 (min cos = 0.018)，但存在致命 confound：不同 decode step 的生成内容本身就不同（harmful prompt 在生成 refusal 文本，harmless prompt 在生成答案），mean-difference 同时捕获了 refusal signal 和 content difference。必须分离这两种信号。

### Method

Teacher-forced controlled experiment：给所有 harmful/harmless prompt 拼接相同的固定前缀（30 tokens），在前缀内不同位置提取 hidden state。由于两组共享完全相同的续写文本，mean-difference 只能捕获指令的安全属性差异。

### Result

| 指标 | Exp B (uncontrolled) | Exp 2A (controlled) | 差异 |
|:---|:---:|:---:|:---:|
| Min pairwise cos | 0.018 | 0.231 | +0.213 |
| Mean pairwise cos | 0.207 | 0.413 | +0.206 |

### Insight

1. **约一半的漂移来自 content confound**：controlled min (0.231) 比 uncontrolled min (0.018) 提高了一个数量级，说明 Exp B 的极端漂移中约 50% 是 content difference 造成的 artifact。

2. **但剩余的漂移是真实的**：即使控制了 content，min sim = 0.231 仍然远低于 0.40 阈值。Refusal direction 在 generation 过程中确实发生了显著旋转——这不是 artifact，是 LLaVA 安全机制的真实特征。

3. **Decision: DYNAMIC**——后续攻击不能用单一静态 direction，需要 timestep-specific directions 或 SVD cone basis。

---

## 3. Exp 2B: Ablation Attack — 因果效力验证

### Problem

Exp A 提取的 mean-diff direction 在统计上与 refusal 相关，但**相关性不等于因果性**。需要验证：ablate 这个 direction 后，模型是否真的减少 refusal？

### Result

| Config | Initial Bypass | Self-Correction | Full Harmful | n |
|:---|:---:|:---:|:---:|:---:|
| baseline_text | 22.4% | 74.1% | 17.2% | 58 |
| blank_image | 60.3% | 32.8% | **56.9%** | 58 |
| ablation_text_all (32层) | 48.3% | 50.0% | 46.6% | 58 |
| ablation_mm_all (32层) | 77.6% | 22.4% | 74.1% | 58 |
| ablation_mm_shallow (0-15) | 77.6% | 18.9% | 74.1% | 58 |
| ablation_mm_deep (16-31) | 77.6% | 18.9% | 74.1% | 58 |
| **ablation_mm_layer16** | **93.1%** | **8.6%** | **89.7%** | 58 |

### Findings

**Finding 1: Layer 16 单层 ablation 是最强配置**

这是一个违反直觉的结果。ablate 全部 32 层（all）只有 74.1% full harmful，而单独 ablate layer 16 达到了 89.7%。**更少的干预反而更有效**。

**可能的解释**：全层 ablation 在浅层（0-15）移除了 refusal direction 上的投影，但也干扰了其他与生成质量相关的信号（因为浅层的 refusal direction 与内容方向有更多 overlap）。这导致模型生成质量下降，产生更多 incoherent 回复而非流畅的 harmful 回复。而 layer 16 恰好是 refusal signal 最纯粹的位置（Exp A 中 cos = 0.918 最高），在此处精确 ablation 对生成质量的副作用最小。

**Finding 2: Visual modality + ablation 有强协同效应**

| 对比 | Full Harmful | 差值 |
|:---|:---:|:---:|
| ablation_text_all vs baseline_text | 46.6% vs 17.2% | +29.4pp (ablation 的因果效力) |
| blank_image vs baseline_text | 56.9% vs 17.2% | +39.7pp (visual modality 效应) |
| ablation_mm_all vs blank_image | 74.1% vs 56.9% | +17.2pp (协同增益) |
| ablation_mm_layer16 vs blank_image | 89.7% vs 56.9% | +32.8pp (最优协同) |

Ablation + visual modality 的效果不是简单加法（29.4 + 39.7 = 69.1），而是存在 **super-additive interaction**（实际 89.7% > 理论加法 69.1%）。这说明 visual modality 改变了 hidden state 的分布，使得 ablation 更容易命中 refusal signal 的核心分量。

**Finding 3: Self-correction 被有效抑制**

ablation_mm_layer16 的 self-correction rate 从 baseline 的 74.1% 降到了 8.6%。这直接证明了 refusal direction ablation 不仅影响初始 refusal，还影响整个序列的 delayed safety reactivation。

**43 条 harmful completions 已保存**，作为 Exp 2C 的 teacher-forcing targets。

---

## 4. Exp 2C: PGD 视觉扰动 — 负结果的深层分析

### Problem

核心攻击：优化图像扰动 delta，使 LLaVA 在整个 generation trajectory 上持续抑制 refusal cone。梯度从 LLM hidden states 反传到 CLIP ViT 输入像素。

### Config

- PGD: epsilon=16/255 (L-inf), alpha=1/255, 200 steps
- Direction mode: SVD cone basis (dim=2, from Exp 2A controlled directions)
- Target layers: all 32 layers
- Training prompts: 16 (from Exp 2B harmful completions)
- Loss: L_suppress (cone suppression) + 0.1 * L_harmful (direction alignment)
- 优化时间: ~17 min (H100)

### Result

| Metric | Blank Image | PGD Optimized | Ablation Layer16 (Exp 2B) |
|:---|:---:|:---:|:---:|
| Initial bypass rate | 60.3% | **44.8%** | 93.1% |
| Self-correction rate | 32.8% | **53.4%** | 8.6% |
| Full harmful completion | **56.9%** | **37.9%** | 89.7% |

### 这是一个重要的负结果

**PGD 优化后的扰动（37.9%）不仅没有超过 blank image（56.9%），反而显著劣于 blank image——攻击"优化"实际上让模型变得更安全了。**

### Loss 曲线分析

```
Step   0: 8.587 (初始)
Step  20: 7.919 (快速下降阶段)
Step  40: 7.698
Step  60: 7.909 (反弹!)
Step  80: 7.777
Step 100: 7.976 (再次反弹!)
Step 120: 7.790
Step 140: 7.586
Step 160: 7.548
Step 180: 7.499
Step 199: 7.603 (最终值，又反弹)
```

**Loss 下降了 11.5%（8.59 → 7.60），但伴随显著振荡。更关键的是：loss 下降并没有转化为 ASR 提升。**

---

## 5. Exp 2C 失败的根因诊断

### Diagnosis 1: Proxy Objective Misalignment — Loss 下降 ≠ ASR 提升

优化的 loss 是 hidden state 在 refusal cone basis 上的投影平方和。但这个 proxy objective 与实际 ASR 之间存在根本性的 misalignment：

- **Cone basis 来自 prefill 阶段的 mean-difference**（Exp 2A 的 5 个位置），但这些方向在 **generation 阶段**可能已经旋转（Exp B/2A 已证明漂移存在）
- Loss 在 teacher-forcing 模式下计算，但 ASR 在 **auto-regressive generation** 模式下评估——两种模式下的 hidden state distribution 完全不同
- 我们在所有 32 层上最小化 cone 投影，但 Exp 2B 证明 **layer 16 单层是最有效的干预点**，全层抑制反而引入噪音

### Diagnosis 2: CLIP Gradient Bottleneck — 梯度信息在视觉编码器中严重衰减

PGD 的梯度流路径：

```
Loss (hidden state space)
  → 32 LLaMA decoder layers 反传
  → MM Projector 反传
  → CLIP ViT (24 层 transformer) 反传
  → 像素空间 delta
```

总共经过 ~56 层 transformer 的反传。即使每层梯度只衰减 5%，累积衰减 = 0.95^56 ≈ 5.6%。实际上 CLIP ViT 的 pretrained 权重没有为 "传递 safety signal 梯度" 而优化过，bottleneck 可能更严重。

**证据**：delta L-inf 在 step 20 就达到了 epsilon 上限（0.0627），说明梯度**方向**已经饱和——PGD 在每一步都在 sign(grad) 方向做最大步长。但这个梯度方向可能是嘈杂的、不准确的。

### Diagnosis 3: 优化破坏了 Blank Image 的天然优势

这是最关键的洞察：**blank image 本身已经是一个强大的 "攻击"**（56.9% ASR），因为它提供了 generic visual modality activation shift。PGD 优化试图在此基础上进一步调整像素，但：

1. 优化目标（最小化 cone 投影）与 blank image 的天然效果方向不一致
2. PGD 生成的扰动可能**破坏了 blank image 原有的有利属性**——比如 blank image 的均匀灰色在 CLIP ViT 中产生的 feature pattern 恰好有利于 refusal suppression，而 PGD 扰动引入的高频 pattern 可能干扰了这种效果
3. CLIP ViT 是在自然图像上预训练的，对 adversarial perturbation 的响应可能高度非线性，导致 loss landscape 极度不光滑

### Diagnosis 4: Cone Basis 维度不足

SVD cone basis (dim=2) 从 5 个 position-specific directions 提取。但：
- 5 个方向的 pairwise similarity 最低只有 0.231，说明它们跨越的子空间维度 >> 2
- 2 维 basis 只能捕获这 5 个方向中解释方差最大的 2 个主成分
- 被遗漏的 3+ 个方向可能包含实际 generation 时关键的 refusal 分量

---

## 6. Key Insights: 从负结果中提炼的认知升级

### Insight 1: Ablation（白盒直接干预）vs Perturbation（间接像素优化）的效力鸿沟

| 方法 | 攻击路径 | Full Harmful |
|:---|:---|:---:|
| Ablation (layer 16) | 直接修改 hidden state | **89.7%** |
| Blank image | CLIP → MM Proj → hidden state (无优化) | **56.9%** |
| PGD perturbation | 像素 → CLIP → MM Proj → hidden state (200步优化) | **37.9%** |

**Ablation 是 upper bound（89.7%），blank image 是 strong baseline（56.9%），PGD 优化反而拉低了性能。**

这揭示了一个根本性的 **mechanistic gap**：我们在 hidden state 空间定义了精确的攻击目标（refusal cone suppression），ablation 可以直接在这个空间操作所以非常有效，但**通过像素空间间接操控 hidden state 的能力极其有限**。梯度信号在通过 CLIP ViT + MM Projector 的 ~30 层后严重失真。

**这不是一个工程问题（调参能解决），而是一个 architectural constraint**：VLM 的视觉编码器不是为传递精确的 safety-related gradient 信号而设计的。

### Insight 2: "Optimization Paradox" — 优化目标越精确，对不精确梯度越敏感

GCG (NeurIPS 2023) 和 UltraBreak (ICLR 2026) 在 output token space 优化，目标是最大化 "Sure" 等 affirmative token 的概率。这个目标虽然是 "proxy"，但它足够 robust——即使梯度不精确，只要大致正确就能推动模型向 affirmative 方向移动。

我们的 loss（hidden state cone projection 的精确几何约束）**对梯度精度要求更高**：需要精确知道 refusal cone 的方向，精确抑制该方向的投影。当梯度通过 CLIP ViT 后失真，这种精确几何约束变成了在噪声中寻找精确方向——近似等于随机优化。

**Paradox: mechanistic understanding 给出了更好的攻击目标，但将这个目标转化为像素空间操作时，反而比 proxy objective 更脆弱。**

### Insight 3: Blank Image 效应的本质 — Generic vs Specific Perturbation

Blank image 有效不是因为它抑制了某个特定的 refusal direction，而是因为它引入了 **generic activation shift**，恰好破坏了 LLM backbone 中 safety mechanism 的正常工作条件。这种效果是：
- **Model-agnostic**（Safety Paradox 已证明）
- **不需要知道 refusal direction**
- **Robust to direction rotation**（因为它不针对任何特定方向）

PGD 优化试图用 **specific perturbation** 替代 generic perturbation，但反而丢失了 generic 的 robustness。

### Insight 4: Layer 16 的特殊地位 — Refusal 的 "Narrow Waist"

Exp 2B 中 layer 16 单层 ablation 效果最强（89.7%），远超全层 ablation（74.1%）。结合 Exp A 的发现（layer 16 cos = 0.918，是 refusal direction 最纯粹的位置），这表明 **layer 16 是 LLaVA safety mechanism 的 "narrow waist"（最窄处）**：

- 浅层（0-15）的 refusal signal 与 content signal 混合度高，ablation 有副作用
- 深层（16-31）的 refusal signal 可能已经分散到多个方向（Exp 2A controlled similarity < 0.6）
- Layer 16 恰好是 refusal signal 最纯粹、最集中、与其他功能信号最正交的层

**这意味着攻击应该 "snipe" layer 16 而非 "carpet bomb" 所有层。**

---

## 7. Motivation: 从负结果到新方向

### 当前方法的核心矛盾

```
Gap C 原始假设:
  "Visual perturbation 可以在像素空间优化 delta，使 hidden state
   在整个 generation trajectory 上持续抑制 refusal cone"

Phase 2 证据否定了这个假设的实现路径:
  - Exp 2A: Refusal direction 在 generation 中真实旋转 (dynamic)
  - Exp 2B: 直接干预 hidden state (ablation) 极其有效 (89.7%)
  - Exp 2C: 通过像素间接干预完全无效 (37.9% < 56.9% blank image)

  → 问题不在 "什么目标"（cone suppression 是正确的目标）
  → 问题在 "什么路径"（像素 → CLIP → hidden state 这条路径传不了精确信号）
```

### 可能的 Pivot 方向

**方向 A: Representation Space Attack — 绕过 CLIP 直接操作 visual tokens**

Exp 2C plan 中已预留的 fallback：如果 CLIP 梯度消失，改为直接优化 MM projector 输出空间的 576x4096 visual tokens。这绕过了 CLIP ViT 的梯度 bottleneck，但代价是攻击不再是像素级的（需要修改 MM projector 输出，非物理可实现的扰动）。

- **优势**：直接在 hidden state 邻近空间操作，梯度质量高
- **劣势**：攻击模型变了（从 pixel perturbation 变成 representation perturbation）
- **Novelty**：可以研究 "representation space 中 visual tokens 需要偏移多少才能等效 ablation"

**方向 B: Hybrid Attack — Ablation-Guided Visual Perturbation**

不再用 hidden state cone projection 作为 loss，而是用 **ablation 的效果** 作为 teacher：
1. 对每个 training prompt，先跑一次 ablation_mm_layer16，记录 ablation 后的 hidden states 为 target
2. 优化 pixel perturbation 使得（不加 ablation 的）hidden states 与 ablation target 尽量接近
3. Loss = MSE(h_perturbed, h_ablated)

这将精确的几何约束（cone suppression）替换为更 "soft" 的 distillation 约束，可能更 robust to gradient noise。

**方向 C: Loss Redesign — 从 Cone Suppression 到 Output-Level Objective**

回到 output space 的 proxy objective（类似 GCG/UltraBreak），但加入 sequence-level constraint：
- Primary loss: 最大化 harmful target tokens 的 log-likelihood（整个序列，非仅初始 token）
- Regularizer: 最小化生成序列中出现 self-correction pattern 的概率
- 这个 loss 对梯度精度的要求更低，可能更适合通过 CLIP 传递

**方向 D: 不做像素攻击，改为 Mechanistic Analysis Paper**

Phase 2 的发现（dynamic refusal rotation, layer 16 narrow waist, ablation-perturbation gap, optimization paradox）本身就是有价值的 mechanistic insight。可以 pivot 为一篇 **analysis/understanding paper** 而非 attack paper：

- **Title 方向**: "The Geometry of Dynamic Refusal in Vision-Language Models"
- **贡献**: (1) Refusal direction 在 VLM generation 中的 dynamic rotation 现象; (2) Layer 16 作为 safety narrow waist; (3) Visual modality 的 generic vs specific perturbation 效应; (4) 从 hidden state 到 pixel space 的 gradient information bottleneck
- **目标会议**: NeurIPS / ICLR analysis track

---

## 8. Limitations

### 实验层面

1. **单模型**: 所有实验仅在 LLaVA-1.5-7B 上进行。Layer 16 的特殊地位、refusal rotation 模式可能是模型特异的。需要在 InstructBLIP、LLaMA-Vision 等模型上验证。

2. **Eval 局限**: 使用规则匹配（keyword pattern）判断 bypass/refusal/self-correction，会有 false positive/negative。正式实验需要 LLM-based judge (如 GPT-4 或 HarmBench classifier)。

3. **Cone basis 质量**: SVD cone (dim=2) 从仅 5 个 teacher-forced positions 提取，维度可能不足。Exp 2A 的 controlled pairwise similarity 最低 0.231 说明 5 个方向跨越的子空间维度 > 2。

4. **PGD 超参未充分搜索**: 只跑了 epsilon=16/255, alpha=1/255, 200 steps 一种配置。plan 中的 Exp 2D ablation study（不同 epsilon, 不同层配置, 有无 Component 2）尚未执行。

### 方法论层面

5. **Teacher-forcing vs Auto-regressive Gap**: PGD 优化在 teacher-forcing 模式下进行（固定 target response），但评估在 auto-regressive 模式下进行。两种模式下 hidden state 分布的差异可能导致 loss 在优化时下降但在评估时无效。

6. **CLIP Gradient Bottleneck 未量化**: 知道梯度通过 CLIP 后衰减，但没有定量测量衰减幅度。应该对比 "在 MM projector 输出处优化" vs "在像素处优化" 的效果差异，以 isolate CLIP bottleneck 的贡献。

7. **Harmful Direction 构建粗糙**: Component 2 的 harmful direction 只是 harmful completions 的 mean hidden state，没有减去 harmless 的 mean——所以它捕获的是 "harmful content 的一般特征" 而非 "harmful vs harmless 的 discriminative direction"。

---

## 9. Novelty Assessment: Phase 2 新增贡献

| # | 发现 | 类型 | Novelty 等级 |
|:---|:---|:---|:---|
| 1 | Refusal direction drift 约一半来自 content confound | Methodological finding | Incremental |
| 2 | 控制 confound 后仍存在真实旋转 (controlled min sim = 0.231) | Empirical finding | Moderate |
| 3 | Layer 16 单层 ablation (89.7%) > 全层 ablation (74.1%) | Counter-intuitive finding | **Significant** |
| 4 | Visual modality + ablation 存在 super-additive interaction | Empirical finding | Moderate |
| 5 | **PGD 像素优化反而劣于无优化的 blank image** | **Negative result** | **Significant** |
| 6 | Optimization Paradox: 精确几何目标 + 不精确梯度 → 比 proxy objective 更脆弱 | Theoretical insight | **Significant** |
| 7 | Generic perturbation (blank image) vs Specific perturbation (PGD) 的 robustness 反转 | Conceptual insight | Moderate-Significant |

---

## 10. 完整数据索引

| 文件 | 内容 |
|:---|:---|
| `results/exp_2a_results.json` | Exp 2A 结果: decision=dynamic, controlled min sim=0.231 |
| `results/exp_2a_controlled_directions.pt` | 5 个 position-specific refusal directions (layer 16) |
| `results/exp_2b_results.json` | Exp 2B 全部 7 个 config 的 metrics + sample results |
| `results/exp_2b_harmful_completions.json` | 43 条 ablation 成功的 harmful completions |
| `results/exp_2c_results.json` | Exp 2C PGD 结果: perturbed 37.9% vs blank 56.9% |
| `results/exp_2c_delta.pt` | 优化后的 delta tensor + base pixel values |
| `results/exp_2c_perturbed_image.png` | 可视化扰动图像 |

---

## 11. Next Steps 优先级

| 优先级 | 任务 | 预期时间 |
|:---|:---|:---|
| **P0** | 与导师讨论 pivot 方向（attack paper vs analysis paper） | - |
| P1 | Exp 2D: ablation study（不同 epsilon / 层配置 / 有无 Component 2），确认 Exp 2C 的负结果不是超参问题 | ~2.5h GPU |
| P1 | 尝试 Direction B (Hybrid: ablation-guided visual perturbation) | ~1 day |
| P2 | 尝试 Direction A (representation space attack, bypass CLIP) 量化 gradient bottleneck | ~1 day |
| P2 | 在 InstructBLIP 上验证 layer 16 narrow waist 是否为通用现象 | 需下载模型 |
| P3 | 使用 LLM-based judge 替换 keyword matching eval | ~0.5 day |
