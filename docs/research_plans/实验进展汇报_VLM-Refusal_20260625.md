# VLM-Refusal 实验进展汇报

> 汇报日期：2026-06-25　|　主攻模型：Qwen 系列（Qwen2.5-VL / Qwen3-VL-8B）
> 进展基线：2026-06-04（此后暂停，本次为重启前的回顾与规划）

---

## 一、一句话定位

**核心现象（Evidence Relocation）**：VLM 加入视觉模态后，模型其实"知道"输入有害（risk evidence 仍高度可读，AUC=1.000），但这个"知道"发生在**完全不同的位置和方向**上——不是"模型不知道"（感知失败），也不是"信号被压缩"（ReGap 的 separability collapse），而是**信号搬家了**。固定几何的安全干预因此打在了错误的位置。

**研究问题**：当视觉载体改变 risk evidence 的编码**层（layer）、方向（direction）、token 位置（position）**时，固定几何的安全干预失败，如何通过一个**可训练的 carrier-aware bridge** 把迁移后的 risk evidence 重新路由到行为控制点，恢复安全防御。

**两个核心假设**：
- **H1（sensor–gate 解耦）**：risk evidence 最佳可读位置（sensor）≠ 因果控制安全行为的位置（gate），尤其在 FigStep 等极端视觉载体下。
- **H2（动态桥接 > 固定干预）**：输入依赖的 sensor→gate 桥接，优于固定方向消融 / 固定子空间投影 / 全局 drift correction。

---

## 二、已完成工作回顾（已被数据坐实）

| # | 结论 | 关键证据 | 状态 |
|---|---|---|---|
| 1 | LLM 上固定方向干预有效且可复现 | Qwen2.5-7B DIM ablation：ASR_LG3=94.5%；SRR 底线 +4.7~9.4pp（n=128） | ✅ 强 |
| 2 | 同方法迁移到 VLM 大幅退化 | Qwen2.5-VL V-text：ASR_LG3 从 94.5%→57.0%（-37.5pp）；Gemma3 仅 10.2% | ✅ 强 |
| 3 | Risk evidence 没消失，而是迁移了 | 全部视觉条件 AUC=1.000；Qwen3-VL FigStep 最优层 15→1，token pos=-1 | ✅ 强 |
| 4 | 这是"重编码"而非"平移" | FigStep vs V-text 方向余弦仅 0.259（Qwen3-VL）/ 0.136（InternVL3）；层 0/1 vs 6/15 | ✅ 强（区别于 ReGap 的命门） |
| 5 | 探针有效性成立（非 artifact） | Phase 1.5 审计：held-out AUC≈1.0，permutation≈0.5，全部 PASS | ✅ 强 |
| 6 | 评估幻觉已识别 | Gemma3 V-text ASR_kw=96.9% 但 ASR_LG3=10.2%，keyword 虚高 9.5× | ✅ 已规避 |

**小结**：现象层（evidence relocation）已经打磨扎实，与 ReGap 的差异化论证清晰。**但因果层（gate 在哪、桥接是否有用）目前是空的**——这正是 reviewer 会重点攻击、也是方法创新所依附的部分。

---

## 三、待完成工作梳理（你之前忘了的部分，已帮你定位）

### 3.1 Motivation experiment 还需补充的部分（来自 Phase 1.5 审计的已知局限）

1. **`paired_id` group-split 验证**：当前 held-out 切分可能让同一 prompt 的不同视觉条件分散在 train/test 两侧，存在信息泄露嫌疑，需按 paired_id 做分组切分重跑 probe。
2. **Cross-category generalization 验证**：probe 是否只在见过的有害类别上有效，需做跨类别（held-out category）泛化测试，否则 AUC=1.000 会被质疑为类别记忆。
3. **行为层失败证据（最致命的缺口）**：目前只证明 evidence "可读且迁移"，但**没有 condition-specific behavior labels** 证明模型在 FigStep 上行为层面真的失败。reviewer 必问：既然 evidence 完全可读，你凭什么说模型没用它？→ 需要在同一批样本上把"AUC=1.000 可读"与"行为却 comply"并排呈现。
4. **数据集 confound（dataset artifact 风险）**：现有 harmful/harmless 来自不同数据源，需构造 `(q_i, v_i^safe, v_i^risk)` 三元组（文本相同、只改视觉载体），否则 AUC=1.000 可能被解释为"数据源可分"而非"风险可分"。

### 3.2 Idea 有效性验证实验（核心创新点，尚未开始）

1. **Sensor–Gate 解耦验证（验 H1，地基）**：用 activation patching/steering 做因果层扫描，找出干预最能改变安全策略的 (layer, token_pos)=gate，与 sensor locus 对比。→ **这是 Go/No-Go 节点**：若发现 sensor 与 gate 重合，carrier-aware bridge 的故事就要改写。
2. **Oracle Bridge vs Fixed Geometry（验 H2 上限）**：已知 sensor locus 直接 patch 到 gate，与 fixed direction / drift correction / VLM-Guard 比较。→ **Go/No-Go 标准：dynamic bridge 必须显著优于 ReGap-style drift correction**，否则问题退化为 ReGap 子问题。
3. **Learnable Bridge（Stage 2，建议暂缓）**：LoRA 只加在 sensor→gate 路径，训练目标 `L_total = L_safe_policy + L_utility + L_routing_reg`，其中 `L_routing_reg = -λ·corr(s(x), Δ(x))`。→ **依赖 H1 先成立**，不要在地基未验证前投入训练工程。

---

## 四、文献对标：与现有已发表工作的相似 / 可借鉴之处

### 4.1 最直接的"竞品/威胁"——必须作为 baseline 并明确区分

| 工作 | 出处 | 核心做法 | 与你的关系（差异化要点） |
|---|---|---|---|
| **CMRM** | arXiv:2410.09047 | 把多模态表示拉回 LLM 文本分布，恢复安全（LLaVA 不安全率 61.5%→3.15%） | **最大威胁**：已提出"representation gap 导致安全对齐迁移失效"。你的区分：它做**全局 drift 拉回**，假设同方向平移；你的 FigStep 数据是**换层换方向的重编码**，全局 drift 在几何上无意义。 |
| **ReGap** | arXiv:2605.18104 | 估计并自适应修正 modality drift，恢复 text-refusal separability（training-free） | **正面对手**：解决 separability collapse（同方向被压缩）。你的区分：你的可分性**没被压缩**（AUC=1.000），而是位置和方向变了。**必须作为核心 baseline**。 |
| **VLM-Guard** | arXiv:2502.10486 | 从 LLM 提取 safety direction，把 VLM 表示投影到其正交子空间 | 假设 LLM safety direction 可直接迁移；你的数据显示 cos(LLM,VLM)=0.671，迁移不充分。 |
| **TGA**（Cross-Modal Safety Mechanism Transfer） | arXiv:2410.12662，**ICLR 2025** | 检索相关文本引导视觉→LLM hidden-state 对齐，迁移安全机制 | **可借鉴**：已发现"特定 transformer 层的 hidden state 对安全机制关键"——直接支撑你的 **gate locus** 概念。但它把对齐目标锁死在 text 的层结构上；你主张 FigStep 在 layer 0/1 就可读，不该硬对齐回 text 的 layer 15。 |

### 4.2 给你启发 / 可直接复用的工作

| 工作 | 出处 | 可借鉴点 |
|---|---|---|
| **Arditi: Refusal mediated by a single direction** | arXiv:2406.11717，NeurIPS 2024 | 你的方法学源头。值得注意：它**提取方向用单层、施加干预需全层广播**——已隐含"读取位置≠控制位置"雏形，可作为 sensor–gate 解耦的引证支撑。 |
| **Geometry of Refusal / Concept Cones** | arXiv:2502.17420，ICML 2025 | **反驳单方向**，证明拒答由多维"概念锥"驱动、"正交≠干预下独立"。→ 为你的 **multi-locus probe**（多位置读取）提供理论依据，也说明单一固定方向必然不够。 |
| **CAST: Conditional Activation Steering** | arXiv:2409.05907，ICLR 2025 | **carrier-aware bridge 的 LLM 版前身**：按输入条件性地施加/撤回 steering，实现 selective refusal。→ 可直接借鉴其"条件判别器（何时注入）"设计，对应你的 Router 模块。 |
| **VLSBench（Visual Leakage）** | arXiv:2411.19939，ACL 2025 | **解决你 confound 的现成工具**：提供无视觉信息泄露（VSIL）的测试床。→ 可用来构造/校验你的 `(q, v^safe, v^risk)` 三元组数据集，挡住 dataset artifact 攻击。 |
| **FigStep** | arXiv:2311.05608，AAAI 2025 | 你的主攻攻击载体来源；其"视觉 embedding 缺乏安全对齐"的 embedding 分布分析可作为现象佐证与攻击基线。 |
| **NullSteer（null-space projection）** | arXiv:2603.22094，CVPR 2026 | 解决"steering 既增强 refusal 又导致 over-refusal"的矛盾，benign 子空间置零。→ 可借鉴其 utility 保持机制，对应你的 `L_utility`。 |

### 4.3 一句话提炼你在文献版图中的位置

> 现有"modality drift / subspace projection"家族（CMRM、ReGap、VLM-Guard、NullSteer）共享同一假设链：**视觉引入 drift → 压缩 refusal 可分性 → 安全失效**，干预手段都是"沿固定/估计方向把表示拉回或投影"。**你的差异化叙事**：在 FigStep 等极端载体下，失败不是"沿同一轴平移/压缩"，而是 risk evidence **整体换层换方向的重编码**——固定几何（含 drift correction）从原理上不够，需要 **carrier-aware 的动态 sensor→gate 桥接**。这个"重编码 vs 平移"的二分是你区别于全家族的核心卖点，但**必须用 behavioral gate 实验把因果链补上**才能立住。

---

## 五、下一步执行路线（建议汇报时按此讲）

按"先堵漏洞 → 再验核心 → 最后比基线"推进，不要先冲算法：

1. **【优先级 1】补行为证据 + 三元组数据集**：产出 condition-specific behavior labels（comply/refuse），并构造 `(q, v^safe, v^risk)` paired 数据集（可借 VLSBench 思路）。同时把 motivation 的 group-split / cross-category 两个审计窟窿补上。
2. **【优先级 2 · Go/No-Go】Sensor–Gate 解耦验证（验 H1）**：causal patching 找 gate，对比 sensor。预先写死判据：sensor≠gate 才支持桥接故事。
3. **【优先级 3 · Go/No-Go】Baseline 先行 + Oracle Bridge（验 H2 上限）**：先把 fixed direction / drift correction(ReGap) / VLM-Guard 在新数据集上全跑出来，再做 oracle bridge。dynamic bridge 必须显著优于 drift correction。
4. **【优先级 4 · 暂缓】Learnable Bridge（Stage 2 LoRA + L_routing_reg）**：仅在 H1 成立后启动。

**模型策略**：以 **Qwen3-VL-8B 为主模型**走完整因果链（其 FigStep 现象最清晰：层 15→1、cos=0.259），Qwen2.5-VL 作迁移/泛化验证，Gemma3 / InternVL3 仅作 cross-model 佐证。

---

## 六、汇报时可能被追问的问题（提前准备）

- **Q：你和 CMRM/ReGap 到底差在哪？** → 他们是"同方向平移/压缩"，我是"换层换方向重编码"；我会用 FigStep cos=0.259、层 15→1 的数据正面回应，并把 ReGap 设为必比 baseline。
- **Q：AUC=1.000 是不是数据源 artifact？** → 已通过 Phase 1.5 permutation 审计（≈0.5）排除探针 artifact；数据源 confound 正用三元组数据集解决（优先级 1）。
- **Q：evidence 既然可读，凭什么说模型行为失败？** → 这是当前最大缺口，优先级 1 的 behavior labels 专门补这一环。
- **Q：sensor≠gate 万一不成立？** → 已设为 Go/No-Go 节点；若重合则转向"为何固定方向仍失败"的纯诊断论文，叙事可退守。

---

## 附：文献来源

1. Arditi et al. Refusal in LMs Is Mediated by a Single Direction. NeurIPS 2024. https://arxiv.org/abs/2406.11717
2. Wollschläger et al. The Geometry of Refusal in LLMs: Concept Cones. ICML 2025. https://arxiv.org/abs/2502.17420
3. Xu et al. Cross-Modal Safety Mechanism Transfer (TGA). ICLR 2025. https://arxiv.org/abs/2410.12662
4. Liu et al. Unraveling and Mitigating Safety Alignment Degradation of VLMs (CMRM). 2024. https://arxiv.org/abs/2410.09047
5. VLM-Guard: Safeguarding VLMs via Fulfilling Safety Alignment Gap. 2025. https://arxiv.org/abs/2502.10486
6. ReGap: Safety Geometry Collapse in Multimodal LLMs and Adaptive Drift Correction. 2026. https://arxiv.org/abs/2605.18104
7. Lee et al. Programming Refusal with Conditional Activation Steering (CAST). ICLR 2025. https://arxiv.org/abs/2409.05907
8. Hu et al. VLSBench: Unveiling Visual Leakage in Multimodal Safety. ACL 2025. https://arxiv.org/abs/2411.19939
9. Gong et al. FigStep: Jailbreaking LVLMs via Typographic Visual Prompts. AAAI 2025. https://arxiv.org/abs/2311.05608
10. Zhu et al. Principled Steering via Null-space Projection (NullSteer). CVPR 2026. https://arxiv.org/abs/2603.22094
11. Wang et al. SIUO: Cross-modality Safety Alignment. NAACL 2025. https://arxiv.org/abs/2406.15279
12. Chakraborty et al. Cross-Modal Safety Alignment: Textual Unlearning. EMNLP 2024 Findings. https://arxiv.org/abs/2406.02575

> 注：部分 2025–2026 会议归属（如 ReGap 暂为 preprint、DAVSP/Gamma-Guard 等）以 arXiv 为准，正式引用前建议二次核对原文。
