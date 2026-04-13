# 两篇论文精读分析

## Arditi et al. (NeurIPS 2024) —— 核心内容

- **方法**：采用 Difference-in-Means (DIM) 方法，在 post-instruction token 位置（即 chat template 中 `<|im_end|>` 之后的 token）提取单一的 refusal direction，并对所有层与所有 token 位置执行 directional ablation。
- **Qwen 相关的关键发现**：
  - Arditi 实验对象为纯文本 LLM（非 VLM）。在该方法下，Qwen 系列（1.8B/7B/14B/72B）的 ASR 达到 **79–84%**，为所有受测模型中最高。
  - Qwen 在启用/禁用 system prompt 时的 ASR 差异极小（**79.2% vs 74.8%**）；相比之下，LLaMA-2 的差异极为显著（**22.6% vs 79.9%**）。
- **§F.2 的解释**：Qwen 的 system prompt 内容较为简短，缺乏 LLaMA-2 中明确的安全指示。但作者明确指出：“这不能完全解释差异，可能反映了两类模型对系统级指令的响应方式存在根本不同。”换言之，原论文未能完全阐明 Qwen 对几何攻击高度敏感的根本原因。
- **评估方式的关键局限**：原研究采用两项指标：
  1. `refusal_score`：基于关键词匹配（如检测 "I'm sorry" 等）
  2. `safety_score`：基于 LlamaGuard-2 判定输出内容是否有害
  > ⚠️ **局限性说明**：该评估框架完全无法检测 **stealth refusal**。若模型以伦理反驳等形式进行隐性拒绝，既未触发拒绝关键词，LlamaGuard 亦会判定为安全内容。此类回复在原框架中将被误判为“成功越狱”，但实际并未生成有害信息。

---

## Wollschläger et al. (ICML 2025) —— 核心内容

- **方法升级：RDO（Refusal Direction Optimization）**
  通过梯度下降直接优化 refusal direction，以满足两项关键属性：
  - **Monotonic Scaling**：激活方向加减时，拒绝概率呈单调变化
  - **Surgical Ablation**：消融后模型可回答有害请求，且对无害请求的行为不受影响
- **损失函数**（与 3-loss 框架高度一致）：
  $$
  \mathcal{L} = \lambda_{\text{abl}} \mathcal{L}_{\text{ablation}} + \lambda_{\text{add}} \mathcal{L}_{\text{addition}} + \lambda_{\text{ret}} \mathcal{L}_{\text{retain}}
  $$
  其中 `retain loss` 采用 KL 散度，旨在严格控制消融 direction 时对无害行为产生的副作用。
- **Concept Cone（核心发现）**：Refusal 行为由多维 polyhedral cone 介导，而非单一方向。实验测得 cone 维度最高可达 5 维。关键在于：cone 内所有方向均满足相同的 refusal 属性，表明该 cone 是同一“拒绝概念”的几何延伸，而非多个独立机制的集合。
- **Representational Independence**：正交性（Orthogonality）不等同于独立性（Independence）。即使两个方向正交，消融其一仍可能通过非线性路径影响另一方向。该研究据此提出了更严格的独立性定义。
- **Qwen 2.5 的关键发现**：
  - **Figure 4**：Qwen 2.5 的 cone 性能随维度增加而下降的速度显著快于其他模型。小参数量模型（1.5B）的 cone lower bound 在 $k>2$ 后迅速崩溃，而大参数量模型（14B）表现更为稳定。
  - **Figure 16（附录）**：对 Qwen 2.5 系列而言，DIM 方向的副作用过大，原文指出 *“for the majority of estimated DIM directions, the side-effects are too high, rendering it an unsuccessful attack”*。RDO 的优势在于可通过 retain loss 控制副作用，但 Qwen 仍是所有受测模型中最难处理的对象。
- **TruthfulQA 副作用（Table 1）**：
  - **Qwen 2.5 7B**：DIM 消融后 TruthfulQA 得分由 63.1 降至 58.7（$-4.4$），RDO 优化后回升至 60.0（仍为 $-3.1$）
  - **Qwen 2.5 14B**：DIM 消融后由 70.8 降至 63.3（$-7.5$），RDO 优化后回升至 67.9（$-2.9$）
  > Qwen 的 TruthfulQA 副作用为所有模型中最严重，且随参数量增大而加剧。该副作用的本质在于：Qwen 的 DIM 方向并非仅编码拒绝语气，而是与伦理判断、事实性等高级语义能力高度纠缠。消融该方向不仅移除了拒绝行为，亦会破坏模型对“争议性内容”的审慎判断机制。

---

## 两篇文献对当前研究的核心启示（Insights）

### Insight 1：Stealth Refusal 的发现揭示了现有评估框架的系统性盲区
此为最具创新性的实证发现。Arditi 采用 keyword matching + LlamaGuard，Wollschläger 采用 StrongREJECT judge，二者均仅判定“是否生成有害内容”，而未探究“模型是否以非标准形式持续拒绝”。
- 在 Arditi 框架下，Qwen（LLM）的 ASR 为 **79–84%**；而在您的 VLM 实验中（FHCR_judge），ASR 仅为 **8–13%**。
- 差异归因假设：
  1. VLM 版 Qwen 的安全对齐强度高于纯文本 LLM 版
  2. 您的评估标准更为严格（LLM judge vs 关键词匹配）
  3. Qwen-VL 演化出更完善的 stealth refusal 机制
- **关键验证实验**：建议在您的 Qwen-VL 数据上复现 Arditi 的 keyword matching 评估。若 keyword-based ASR 较高（60–70%），而 FHCR_judge 仅 8–13%，即可实证 stealth refusal 为前述两篇文献共同遗漏的现象，构成您论文的核心贡献。

### Insight 2：Qwen 的 DIM 副作用问题直接印证了 stealth refusal 的生成机制
Wollschläger 指出 Qwen DIM 方向具有“过强的副作用”，消融操作将损害 TruthfulQA 表现（涵盖 misinformation, stereotypes, conspiracies 等类别）。这表明 Qwen 的 DIM 方向高度耦合了“拒绝语气”与“对敏感/争议内容的高级判断能力”。消融该方向后，模型将激活其他独立路径进行补偿，从而触发 stealth refusal。
- **理论映射**：Qwen 的 refusal cone 为与 `ethical stance` 高度耦合的几何结构。消融 cone 相当于部分削弱 Layer 1 的同时，强化了 Layer 2 的补偿激活。此现象并非源于独立的双层机制，而是 cone 自身结构复杂性的直接体现。

### Insight 3：RDO 方法更适配 VLM 场景，构成方法论升级路径
当前研究沿用 DIM 提取 refusal direction。Wollschläger 的 RDO 具备两项优势：
1. `retain loss` 可有效控制副作用（对高纠缠的 Qwen 模型尤为关键）
2. 可提取完整的 concept cone，而非仅保留 PC1
- **扩展方案**：将 RDO 迁移至 VLM 场景，引入 visual token 约束：
  $$
  \mathcal{L}_{\text{VLM-RDO}} = \lambda_{\text{abl}}\mathcal{L}_{\text{ablation}} + \lambda_{\text{add}}\mathcal{L}_{\text{addition}} + \lambda_{\text{ret}}\mathcal{L}_{\text{retain}} + \lambda_{\text{vis}}\mathcal{L}_{\text{visual-retain}}
  $$
  其中 $\mathcal{L}_{\text{visual-retain}}$ 为新增约束项，要求消融后模型在 visual token 处的表征不发生异常偏移。该设计可直接对应您发现的 Amplitude Reversal 现象，实现针对多模态场景的方法论创新。

### Insight 4：Representational Independence 理论可优化 Cone 实验设计
Wollschläger 已证明：正交性 $\neq$ 独立性。即使 cone 内的基向量相互正交，消融其一仍可能通过非线性路径激活其他向量。
- **对 P0 实验的指导**：执行 top-$k$ cone 消融时，不同 $k$ 值对应不同的干预范围，但方向间未必独立。若 $k=5$ 消融后 stealth refusal 仍存续，需首先排除“方向正交但不独立、消融操作产生交互影响”的可能性。
- **补充实验设计**：在 $k=3$ 或 $k=5$ 消融后，执行 activation addition（重新注入 refusal direction），观察是否可复现 stealth refusal。
  - 若可复现：支持“不完整 cone 消融”假说（stealth refusal 依赖 cone 内方向）
  - 若不可复现：支持“独立 Layer 2”假说（stealth refusal 通过独立路径激活）

### Insight 5：Qwen Cone 的低维衰减特性与您的三分类框架高度契合
Wollschläger 发现 Qwen 2.5 的 cone lower bound 随维度增加下降最快，原因为“较小的 residual stream dimension 导致方向更难正交”。
- **深层理论解释**：结合您的三分类框架，Qwen 属于 Type II（Late Gate）架构。其 refusal cone 并非集中于单一层级（区别于 LLaVA 的 Narrow Waist），而是分布于多层级之中。维度增加时，各维度在不同层级生效，层间非线性交互导致 cone lower bound 快速衰减。此现象与 Type II “多层分布、早期积累、晚期执行”的机制描述完全一致。
- **理论延伸**：Qwen 的 cone 低维性并非仅由 residual stream 维度限制所致，而是其 Late Gate 架构的几何表征。您的 VLM 实验可为此结论提供跨模态的直接验证。

---

## 综合对比：您的研究与现有文献的定位关系

| 维度 | Arditi (2024) | Wollschläger (2025) | 您的研究 |
|:---|:---|:---|:---|
| **模型类型** | 纯文本 LLM | 纯文本 LLM | VLM（多模态） |
| **方向提取** | DIM | RDO（梯度优化） | DIM（待升级为 RDO） |
| **Cone 维度** | 不考虑 | 考虑（最高 5D） | P0 正在验证 |
| **评估方式** | keyword + LlamaGuard | StrongREJECT judge | 双 judge 交叉验证 |
| **检测 stealth refusal** | ❌ 无法检测 | ❌ 无法检测 | ✅ 首次量化 |
| **Qwen 结果** | 高 ASR（79–84%） | DIM 副作用过高 | 低 FHCR_judge（8–13%） |
| **架构对比分析** | 无 | 无 | 三分类框架 |
| **Visual modality 分析** | 无 | 无 | Amplitude Reversal |

---

## 下一步实验建议（结合两篇文献）

### P0 实验补充（低成本，基于 Wollschläger 方法论）
- 在执行 top-$k$ DIM cone 消融的同时，平行测试 top-$k$ RDO cone 消融。
- **预期推论**：若 RDO 消融后 stealth refusal 的下降幅度更显著（因副作用更小，消融更彻底），则表明 DIM cone 中的 stealth refusal 成分部分源于 DIM 方法自身的表征纠缠，而非独立的 Layer 2。该对比实验可直接回应核心研究问题。

### P1 实验（P0 完成后执行）
- Arditi 曾分析 adversarial suffix 对 Qwen 1.8B 的影响，指出 suffix 会压制 refusal direction 在 last token 处的 cosine similarity（Figure 5）。
- **VLM 对应分析**：验证 blank image 与有害图像是否具备与 adversarial suffix 相同的压制效应。若成立，则可建立 `visual modality = multimodal adversarial suffix` 的机制等价性，为 Amplitude Reversal 现象提供底层机制支撑。

### P0 结果导向的论文叙事定位
- **若假说 A 成立（top-$k$ cone 消融基本消除 stealth refusal）**：
  > “现有研究（Arditi, Wollschläger）的评估框架存在系统性盲区，无法识别 stealth refusal。在 VLM 场景下，该盲区导致几何攻击的实际有效性被严重高估。通过引入 LLM judge 评估体系，我们证实 Type II 架构模型的高表观 ASR 实为 stealth refusal 的误判。在此基础上，本文扩展了 cone 消融方法，量化了 Type I 与 Type II 模型在视觉多模态条件下的真实安全脆弱性差异。”
- **若假说 B 成立（top-$k$ cone 消融无效）**：
  > “现有几何攻击方法（含 Wollschläger 的 RDO cone）对强对齐 VLM 存在根本性局限。本文首次系统揭示 Qwen-VL 存在独立于 refusal cone 之外的第二层安全机制（stealth refusal）。该机制在现有 LLM 安全文献中尚未被记录，且已通过跨 judge 交叉验证确认其真实性。”
- **结论**：最终叙事路线将严格依据 P0 实验结果确定。