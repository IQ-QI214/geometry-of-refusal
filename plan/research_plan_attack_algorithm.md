# VLM Safety Geometry Attack：完整算法设计与实验清单

> 版本：v1.0 | 日期：2026-04-09
> 研究定位：Phenomenon Discovery → Mechanistic Explanation → Attack Algorithm
> 目标投稿：AAAI 2026 / ICLR 2027（备选）

---

## 第一部分：整体研究逻辑链（一环扣一环）

```
[现象发现] Delayed Safety Awareness (DSA)
    即使初始 bypass 成功，80% 概率中途自我纠正 → 现有攻击只解决充分条件的一半
         ↓ 机制问题
[机制解释] Refusal Direction 的动态几何结构
    1. 跨模态稳定性：direction 在 text/mm 之间高度一致（cos > 0.87）
    2. Amplitude Reversal：浅层压制、深层放大（CLIP 架构特有）
    3. Dynamic Rotation：generation 过程中 direction 非单调旋转（min cos = 0.231）
    4. Layer 16 Narrow Waist：safety signal 在 CLIP 系模型中高度集中
         ↓ 三分类框架
[架构分类] Type I（Bottleneck）/ Type II（Late Gate）/ Type III（Diffuse）
    Amplitude Reversal 的存在 → 预测最优攻击策略
         ↓ 算法设计
[攻击算法] AGDA-V：Architecture-aware Geometry-guided Distillation Attack for VLMs
    利用三分类框架，针对不同架构设计差异化优化策略
         ↓ 验证
[实验验证] 白盒 ASR + 跨架构迁移 + 黑盒 Transfer
```

---

## 第二部分：AGDA-V 算法完整设计

### 2.1 符号定义

| 符号 | 含义 |
|---|---|
| $x = (x_v, x_t)$ | 输入：image + text |
| $\delta$ | 可优化的 image perturbation（在 CLIP embedding space） |
| $f_\theta$ | original model (student) |
| $f_\theta^{\text{abl}}$ | ablated model (teacher)：带 forward hook 的实时消解模型 |
| $h_t^l$ | step $t$、layer $l$ 的 hidden state |
| $v^l$ | layer $l$ 的 refusal direction（从 prefill 阶段提取） |
| $M_\phi$ | mapper 网络：CLIP embedding → LLM hidden state 空间的近似 |
| $e_\delta$ | 经过扰动后的 CLIP embedding |

### 2.2 整体框架：两阶段优化

```
┌─────────────────────────────────────────────────────────┐
│                  AGDA-V 两阶段优化框架                      │
├─────────────────────────────────────────────────────────┤
│  Stage 1: TF Warm-up (Teacher-Forced，快速收敛)             │
│    • teacher model 预先采样 T* 序列（一次性，不需要梯度）        │
│    • 在固定 T* context 下优化 δ                             │
│    • 梯度路径：logp → LLM → projection → CLIP emb → δ       │
│    • 优化稳定，快速找到大致方向                                │
│                                                         │
│  Stage 2: AR Fine-tune (Auto-Regressive + Gumbel)       │
│    • student 自己生成 context（AR）                         │
│    • Gumbel-Softmax（temperature annealing）使采样可微       │
│    • 真实模拟 mid-sequence safety dynamics                  │
│    • 从 Stage 1 的 δ_0 出发，精确调优                        │
└─────────────────────────────────────────────────────────┘
```

### 2.3 三目标损失函数

$$\mathcal{L}(\delta) = \mathcal{L}_1 + \lambda_2 \mathcal{L}_2 + \lambda_3 \mathcal{L}_3$$

**Loss 1：行为蒸馏损失（Behavioral Distillation）**

$$\mathcal{L}_1 = \frac{1}{T} \sum_{t=1}^{T} \text{KL}\left( p_{\text{teacher}}(\cdot | \text{ctx}_t) \| p_{\text{student}+\delta}(\cdot | \text{ctx}_t) \right)$$

- TF 阶段：$\text{ctx}_t = [x_t, T_1^*, ..., T_{t-1}^*]$，完全可微
- AR 阶段：$\text{ctx}_t = [x_t, \tilde{T}_1, ..., \tilde{T}_{t-1}]$，$\tilde{T}$ 为 Gumbel-Softmax 采样

**Loss 2：几何约束损失（Geometric Constraint），架构自适应**

*Type I（LLaVA，Bottleneck）——单层精准约束：*

$$\mathcal{L}_2^{I} = \frac{1}{T} \sum_{t=1}^{T} \max\left(0, \langle h_t^{16}, v^{16} \rangle - \epsilon \right)$$

通过 mapper $M_\phi$ 实现无 CLIP 梯度瓶颈的近似：

$$\mathcal{L}_2^{I, \text{approx}} = \max\left(0, \langle M_\phi(e_\delta), v^{16} \rangle - \epsilon \right)$$

*Type II（Qwen，Late Gate）——多层覆盖 + 末端复核特别惩罚：*

$$\mathcal{L}_2^{II} = \sum_{l \in \{18,20,22,24\}} w_l \cdot \max\left(0, \langle h^l, v^l \rangle - \epsilon \right) + w_{26} \cdot \max\left(0, \langle h^{26}, v^{26} \rangle - \epsilon \right)$$

其中 $w_{26} > w_l$（末端复核层权重更高）

*Type III（InternVL2，Diffuse）——退化为行为蒸馏（Loss2 置零）：*

$$\mathcal{L}_2^{III} = 0 \quad \text{（单层/多层 direction ablation 无效，Loss1 主导）}$$

若 SVD cone（$k=3$）有效则：$\mathcal{L}_2^{III} = \sum_{k=1}^{3} \max(0, \langle h^l, v_k^l \rangle - \epsilon_k)$

**Loss 3：扰动正则化损失**

$$\mathcal{L}_3 = \|\delta\|_1 + \alpha \cdot \text{TV}(\delta)$$

$\ell_1$ 约束稀疏性，TV（Total Variation）约束视觉平滑性

### 2.4 Mapper 网络设计（解决 CLIP 梯度瓶颈）

受 GHOST 启发，训练一个轻量级 mapper $M_\phi$：

**结构**：2 层 MLP，输入维度 = CLIP embedding dim（768 for ViT-L），输出维度 = LLM hidden state dim（4096 for LLaVA-7B）

**训练目标**：

$$\mathcal{L}_{\text{mapper}} = \text{MSE}(M_\phi(e^{CLIP}), h^{16})$$

在 harmful/harmless paired data 上训练，冻结 CLIP 和 LLM，只训练 mapper。

**作用**：Loss2 的梯度路径从 "$h^{16} \to ...\to \text{CLIP encoder} \to \delta$"（56 层，梯度消失）变为 "$M_\phi(e_\delta) \to e_\delta \to \delta$"（2 层 MLP，梯度高效）

### 2.5 Universal Adversarial Image Prefix 的训练

**目标**：训练一个通用的 adversarial image embedding $e^*$，对任意 harmful prompt 都能有效压制 refusal signal

$$e^* = \arg\min_{e} \mathbb{E}_{x_t \sim \mathcal{D}_{\text{harmful}}} \left[ \mathcal{L}_1(x_t, e) + \lambda_2 \mathcal{L}_2(x_t, e) + \lambda_3 \|e - e_0\|_2 \right]$$

**训练流程**：

```
1. 初始化：e = CLIP_encode(blank_image)，这是你已知有效的 baseline 起点
2. 训练集：HarmBench 前 200 条 harmful prompts
3. 每个 batch（size=16）从训练集随机采样
4. Stage 1（TF warm-up，200 steps）：固定 teacher 序列，优化 e
5. Stage 2（AR fine-tune，100 steps）：Gumbel-softmax，temperature 从 0.5 → 0.1
6. 评估：在 held-out 50 prompts 上测 FHCR 和 SCR
7. 输出：e*（CLIP embedding），通过 diffusion unCLIP 生成对应图像（可视化用）
```

### 2.6 架构自适应策略总结

| 架构类型 | 代表模型 | Loss2 策略 | 扰动粒度 | 预期 ASR |
|---|---|---|---|---|
| Type I（Bottleneck） | LLaVA | 单层 layer 16 | Patch-level | ~90% |
| Type II（Late Gate） | Qwen2.5-VL | 多层 + 末端复核 | Semantic-level | ~70-80% |
| Type III（Diffuse） | InternVL2 | Loss2 = 0（纯蒸馏） | Semantic-level | ~30-40%（待探索） |

### 2.7 迁移性方案（白盒 → 灰盒 → 黑盒）

**白盒（已有）**：直接优化，完整梯度，4×H100 可承受

**灰盒（可尝试）**：只用 encoder + mapper，不用 LLM decoder 的梯度。把 Loss2 的 mapper 方案作为唯一优化信号，Loss1 用 TF 近似。

**黑盒 transfer（核心贡献之一）**：
- Universal adversarial prefix 在多个 source 模型（LLaVA + Qwen）的 ensemble 上训练
- 在 GPT-4V、Gemini-Vision 上测试 zero-shot transfer
- 评估指标：Transfer ASR = (zero-shot FHCR on target) / (white-box FHCR on source)

---

## 第三部分：完整实验清单

### 3.1 Category A：Phenomenon Validation（现象验证）——最高优先级

**实验 A1：DSA 广泛性大规模验证**

> 目标：证明 DSA 是广泛存在的、被现有 SOTA 攻击忽视的严重问题

- 模型：LLaVA-1.5-7B, LLaVA-1.5-13B, Qwen2.5-VL-7B, Qwen2.5-VL-32B, InternVL2-8B
- 攻击方法：GCG, AutoDAN, PAIR, JailBound, UltraBreak, Blank Image（你的 baseline）
- 数据集：HarmBench 标准集（400 prompts，覆盖 7 类 harm category）
- 指标：IBR（Initial Bypass Rate）, SCR（Self-Correction Rate given bypass）, FHCR（Full Harmful Completion Rate）
- **Evaluation**：Llama-Guard-2（替代 keyword matching，必须完成）
- **核心问题**：即使 SOTA 攻击的 IBR > 90%，FHCR 是否仍然显著低于 IBR？
- 预期结论：Qwen/InternVL2 的 baseline SCR = 89%/93%（你已有数据），SOTA 攻击下 SCR 不会降至 0，证明 DSA 是真实且严重的未解决问题

**实验 A2：DSA 触发机制验证（因果性）**

> 目标：区分 DSA 是由 safety signal intrinsic dynamics 触发，还是由 generated content semantic feedback 触发

- 设计：Forced Generation Probe（参考之前的 Gap-A 设计）
  - 用 teacher forcing 给定 harmful prefix（前 20 tokens 是有害内容的开始）
  - 分组 A：无任何 ablation，测量 SCR
  - 分组 B：从 t=20 开始 ablate layer 16，测量 SCR
  - 分组 C：从 t=20 开始 ablate random direction（等 norm，对照组），测量 SCR
- 预期：B 的 SCR 显著低于 A 和 C，证明 refusal direction 对 mid-sequence SC 有独立因果效力
- 注意：在多种 harmful prefix 上重复（内容 confound 控制）

**实验 A3：Refusal Direction Norm 对 SC Trigger 的预测性**

> 目标：建立 direction norm 上升 → SC 触发的时序因果证据

- 收集：50 条 full harmful sequences + 50 条 SC sequences（来自 Exp C / P0-A）
- 提取：每条 sequence 在 layer 16 的逐 step refusal direction norm 曲线 $\|h_t^{16} \cdot \hat{v}^{16}\|$
- 测量：norm 曲线对 SC 发生的 AUROC（≥0.80 则建立强预测关系）
- 额外：标记 SC token 的位置 $t_{sc}$，验证 norm spike 是否先于 $t_{sc}$ 出现（时序因果）

---

### 3.2 Category B：Mechanism Validation（机制验证）——高优先级

**实验 B1：Dynamic Rotation 跨模型验证（Exp 3B，尚未完成）**

> 目标：验证 dynamic rotation 是真实现象且在多种架构上存在

- 对 4 个模型（LLaVA, Qwen, InternVL2, 替换后的 LLaVA-v1.6-Mistral），在 text-only 和 mm 两种条件下分别提取逐步 pairwise cosine matrix
- 控制 content confound（teacher-forced fixed prefix）
- 预期：LLaVA 的 controlled min cos 在 mm 条件下更低（image modality 放大 rotation）

**实验 B2：Image Modality 对 Dynamic Rotation 的影响（澄清实验）**

> 目标：证明 image modality 的存在影响了 direction rotation 的幅度

- 设计：同一组 harmful prompts，text-only vs text+blank_image，提取逐 step direction 的 pairwise cos matrix
- 核心问题：mm 条件下 min pairwise cos 是否显著低于 text-only 条件？
- 若是：visual token 是 dynamic rotation 的 amplifier，是 sequence-level 攻击的机制基础

**实验 B3：Refusal Cone 维度验证**

> 目标：确定 Type I 模型是否真的单方向足够（mean-diff vs SVD cone）

- 对 LLaVA，测试 $k=1,2,3,5$ 维 cone ablation 的 ASR
- 对 InternVL2，测试 $k=1,2,3,5,10$ 维 cone ablation 的 ASR（诊断是否 linear subspace 可解）
- 预期：LLaVA $k=1$ 已接近 ceiling（89.7%），InternVL2 即使 $k=10$ 也无法显著提升

**实验 B4：LLM Backbone Crossover Hypothesis 强化验证**

> 目标：将 Type I 有效数据点从 1 增加到 3，强化因果主张

- 替换 InstructBLIP → LLaVA-v1.6-Mistral-7B（CLIP ViT-L + Mistral backbone）
- 新增 Llama-3.2-Vision-11B（CLIP ViT-H + Llama-3 backbone）
- 测量这两个模型的 crossover 深度，验证是否仍在 ~47% 相对深度
- 对照：Qwen2.5-VL-7B（自研 ViT + Qwen backbone）无 crossover

**实验 B5：Self-Correction 机制的三种命运——扩大规模验证**

> 目标：在 n=100 基础上确认 SC 对 ablation 的响应模式，并做统计显著性检验

- 当前数据：LLaVA SC 衰减型，Qwen SC 可摧毁型，InternVL2 SC 免疫型
- 扩大到 n=200，补充 LLaVA-32B 和 Qwen-32B
- 加入 Llama-Guard-2 评估（替代 keyword matching）

---

### 3.3 Category C：Attack Algorithm Validation（攻击算法验证）——核心贡献

**实验 C1：AGDA-V 白盒攻击主实验**

> 目标：验证 AGDA-V 在不同架构模型上的攻击效果

- 模型：LLaVA-7B, LLaVA-13B, Qwen2.5-VL-7B, InternVL2-8B
- 基线方法：GCG, AutoDAN, UltraBreak, JailBound, Blank Image attack
- 数据集：HarmBench（400 prompts）
- 指标：IBR, SCR, FHCR，+ Llama-Guard-2 quality score
- 关键比较：AGDA-V vs ablation（无 Loss2，纯 Loss1）——证明 geometric constraint 的贡献

**实验 C2：架构自适应策略消融实验（Ablation Study）**

> 目标：验证每个设计组件的贡献

| 配置 | Loss1 | Loss2 | Mapper | TF+AR |
|---|---|---|---|---|
| Full AGDA-V | ✓ | ✓ | ✓ | ✓ |
| w/o Loss2 | ✓ | ✗ | ✓ | ✓ |
| w/o Mapper（直接梯度） | ✓ | ✓ | ✗ | ✓ |
| TF only（无 AR） | ✓ | ✓ | ✓ | TF only |
| Static direction（非 dynamic） | ✓ | static | ✓ | ✓ |

**实验 C3：Universal vs Instance-level 对比**

> 目标：验证 universal adversarial prefix 的有效性和效率优势

- Universal：在 200 prompts 上训练 e*，在 200 held-out prompts 上测试
- Instance：对每个 test prompt 单独优化（oracle 上界）
- 指标：FHCR + 优化时间（seconds per attack）
- 预期：Universal 接近 Instance 的 80%+，但速度快 10-50x

**实验 C4：黑盒迁移性实验**

> 目标：证明 AGDA-V 具有 non-trivial 黑盒迁移能力

- Source 模型：LLaVA-7B + Qwen2.5-VL-7B（ensemble 训练 universal e*）
- Target 模型：LLaVA-13B, InternVL2-26B, GPT-4o（API，只能测 FHCR）, Gemini-1.5-Pro（API）
- 指标：Transfer FHCR
- 对照：Random image baseline，blank image baseline
- 预期：至少在 LLaVA-13B 上有 >50% FHCR（同架构迁移），GPT-4o 上有 >20%（跨架构迁移）

**实验 C5：Evaluation Protocol 标准化**

> 目标：替换 keyword matching，建立论文可信的 evaluation 基准

- 主 judge：Llama-Guard-2（HarmBench 标准）
- 辅助 judge：GPT-4o-based judge（用于 harmful content quality scoring，1-5 scale）
- 对比：keyword matching vs Llama-Guard-2 的 precision/recall，量化 false positive/negative
- 结论：在论文 Appendix 中报告 keyword matching 的偏差方向（false negative 偏多）

---

### 3.4 Category D：补充实验（Strengthening）——中等优先级

**实验 D1：Exp C 数据修正与扩展**

> 发现数据问题：full harmful completion rate 应为 4/8 而非 6/8，需要修正

- 重跑 Exp C，使用 Llama-Guard-2 做 evaluation
- 扩展到 n=100，4 个模型，6 种攻击方法
- 明确 IBR/SCR/FHCR 三个 metric 的统计置信区间（bootstrap CI）

**实验 D2：Layer 16 对语义能力的影响分析**

> 目标：验证"layer 16 是 refusal 最纯粹的层，对语义理解影响最小"这一 hypothesis

- 设计：ablate layer 16 后测试模型在 benign QA 任务上的性能（如 MMMU, VQAv2）
- 对比：ablate layer 12（浅层消融悖论层）和 ablate layer 24（深层）的语义影响
- 预期：layer 16 ablation 对 benign 任务影响最小，支持 narrow waist 的"功能纯粹性"解释

**实验 D3：多轮对话 Safety Geometry 初步验证**

> 目标：验证 safety geometry 在多轮对话中的演化（为 future work 铺垫）

- 设计：1-5 轮 benign 对话 + 第 N 轮 harmful request
- 测量：每轮结束后的 refusal direction norm 和 cos 变化
- 核心问题：随对话轮数增加，是否存在 safety signal 自然衰减的"滚雪球窗口"？

---

### 3.5 实验优先级与时间线（3-4 个月）

| 时间段 | 优先级 | 实验 | 依赖关系 |
|---|---|---|---|
| Week 1-2 | 🔴 P0 | A1（DSA 广泛性，替换 Llama-Guard-2） | 独立 |
| Week 1-2 | 🔴 P0 | B1（Dynamic Rotation 跨模型，Exp 3B） | 独立 |
| Week 2-3 | 🔴 P0 | B4（替换 InstructBLIP → LLaVA-v1.6-Mistral） | 独立 |
| Week 3-4 | 🔴 P0 | A2（DSA 触发机制，Gap-A 实验） | A1 完成后 |
| Week 3-4 | 🟡 P1 | A3（Norm AUROC） | B1 完成后 |
| Week 4-5 | 🟡 P1 | B3（Cone 维度验证） | B4 完成后 |
| Week 5-6 | 🟡 P1 | Mapper 训练 | B3 完成后 |
| Week 6-8 | 🔴 P0 | C1（AGDA-V 主实验） | Mapper + A2 完成后 |
| Week 8-9 | 🟡 P1 | C2（消融实验） | C1 完成后 |
| Week 9-10 | 🟡 P1 | C3（Universal vs Instance） | C1 完成后 |
| Week 10-12 | 🟡 P1 | C4（黑盒迁移） | C3 完成后 |
| Week 11-12 | 🟢 P2 | D1, D2 | 可并行 |
| Week 12+ | 🟢 P2 | D3（多轮对话） | future work |

---

## 第四部分：论文结构建议

```
1. Introduction
   - DSA 现象 + Qwen 89% / InternVL2 93% SC rate 作为 motivation 数据
   - 现有攻击的根本局限：忽视 mid-sequence safety dynamics
   - 贡献：三分类框架 + AGDA-V

2. Background & Related Work
   - Refusal direction（Arditi, Wollschläger）
   - VLM safety（ICET, TGA, JailBound）
   - Adversarial attacks（GCG, AutoDAN, UltraBreak）

3. Phenomenon: Delayed Safety Awareness (实验 A1)
   - 定义、大规模验证、与现有攻击的对比

4. Mechanism: Safety Geometry of VLMs (实验 B1-B5)
   - Amplitude Reversal
   - Dynamic Rotation
   - 三分类框架

5. Method: AGDA-V
   - 问题形式化
   - 三目标 loss
   - Mapper 设计
   - TF + AR 两阶段优化
   - 架构自适应策略

6. Experiments (实验 C1-C4)
   - 主实验
   - 消融实验
   - 迁移性

7. Discussion & Limitations
   - Type III 的根本困难
   - 白盒假设的局限
   - 防御启示

8. Conclusion
```

---

## 第五部分：待澄清的关键开放问题

以下问题是目前研究中尚未有明确答案的核心问题，需要通过实验或深入分析来解决：

1. **DSA 的双重触发机制**：safety signal intrinsic dynamics vs generated content semantic feedback，哪个更主导？实验 A2 是直接判决实验。

2. **Type III 模型（InternVL2）的真实机制**：mean-diff direction 失效是因为安全是 attention-level 的非线性机制，还是只是 linear subspace 维度不够？实验 B3 给出答案。

3. **Universal adversarial prefix 的 CLIP-to-hidden-state 映射**：mapper 能否准确近似 hidden state 的几何结构？这决定了 Loss2 的有效性。需要 mapper 的 validation 实验（MSE on held-out data）。

4. **末端复核机制（Qwen layer 26）的本质**：是独立的 safety head，还是 refusal direction 在深层的 self-reinforcing？需要 layer 26 的 probe 实验。

5. **黑盒迁移的下界**：ensemble source 模型的多样性（CLIP 系 + 自研 ViT）是否足以产生跨架构的 universal adversarial semantics？这是论文 claim 的最大不确定性来源。

---

*注：本文档所有实验设计基于 2026-04-09 的研究进展，部分实验细节（如 loss weight hyperparameter）需要通过 pilot experiment 确定。最终决策请与导师讨论对齐。*
