# 后续研究方法探索：感知→行为路由（Sensor→Gate Routing）

> 日期：2026-06-25　|　主模型：Qwen3-VL-8B（主）/ Qwen2.5-VL（迁移验证）
> 定位：在已确认"risk evidence 随视觉载体换层换方向重编码"基础上，设计"如何把感知直接导向行为"的具体方法

---

## 〇、核心判断：把"感知→行为"拆成三个可独立设计的算子

师姐的三个想法不是三个互斥方案，而是同一条路由链上的**三个算子**。统一架构如下：

```
输入 (q, v)
   │
   ▼
[Router R(x)]  ──→ 决定从哪一层/哪些层读 risk evidence（应对 sensor 迁移）   ← 师姐想法2 (MoE 选层)
   │  p ∈ Δ^L（L 维，每层被选中的概率）
   ▼
[读取 s(x)]   ──→ s(x) = Σ_l p_l · probe_l(h_l(x))，得到风险分数/证据向量
   │
   ▼
[Gate G(s)]   ──→ 阈值决定"拒答模块"是否加载/强度多大                       ← 师姐想法1 (阈值 + LoRA 加载/卸载)
   │  g(x) = 是否触发 / 触发强度
   ▼
[Transport B] ──→ 把证据搬到 gate locus / 后几层 / 最后一层，诱导拒答        ← 师姐想法1 (后几层小 LoRA → 最后一层) + 想法3 (打标签)
   │
   ▼
安全行为输出（refuse / comply）
```

**好处**：三个算子可以单独消融（固定其中两个、只动一个），这正好满足你"多设计几个方法、并行跑、选最好"的需求——每个算子都有 2–3 个变体，组合成实验矩阵。

---

## 一、方法族 A：阈值门控的动态 LoRA（师姐想法1 的形式化）

**直觉**：拒答模块不应该一直开着（会伤 utility），而应该由 sensor 读到的风险分数 s(x) 通过一个阈值决定"加载/卸载"。LoRA 只加在后几层，把证据直接导向最后一层诱导拒答。

| 变体 | 门控机制 | 梯度处理 | 特点 |
|---|---|---|---|
| **A1 硬阈值** | s(x) > τ → 加载 LoRA；否则恒等 | 直通估计器 STE | 最贴近师姐"加载/卸载"原意，推理时干净，但训练不稳 |
| **A2 软门控** | g(x)=σ(α·(s(x)−τ))，LoRA 输出乘 g(x) | 全可微 | 训练稳定，τ/α 可学；推理时可再二值化 |
| **A3 载体相关阈值** | 先轻量分类器判 carrier 类型 → 用 τ_c | 可微 | 直接处理"FigStep 和 V-text 阈值不同"的问题 |

**关键设计点**：
- LoRA 加在哪几层？建议 `[gate_locus, L]`（从 gate 到最后一层），而不是盲目全层——这正是你 Stage 2 "只在 sensor→gate 路径加 LoRA"的落地。
- s(x) 在哪读？由 Router（方法族 B）给出，或先用固定 sensor locus 做简化版。
- 与 CAST（Conditional Activation Steering, ICLR'25）的区别：CAST 注入的是**固定 steering 向量**，我们注入的是**可训练 LoRA**，表达力更强、能学习载体相关的搬运。

---

## 二、方法族 B：MoE 层选择路由（师姐想法2 的形式化）

**直觉**：sensor locus 随载体迁移（FigStep→layer1，V-text→layer15）。与其固定一层，不如让一个 router 网络看输入、输出一个 L 维概率向量，**学会按载体动态选层**。

形式化：
```
Router:  R(x) → logits ∈ R^L → p = softmax(logits) ∈ Δ^L
读取:    s(x) = Σ_{l=1}^{L} p_l · probe_l(h_{l, pos}(x))
```
- **p_l 的物理含义**：第 l 层对"当前这条输入"携带多少决策相关的风险信号。FigStep 输入下 router 应把质量压在 layer 1，V-text 下压在 layer 15——这是可验证的诊断输出（直接对照你已有的 MIBD locus 表）。

| 变体 | 选择方式 | 防坍缩 | 特点 |
|---|---|---|---|
| **B1 软混合** | dense，所有层加权 | 不需要 | 最稳，可解释（看 p 分布） |
| **B2 Top-1 硬路由** | argmax 选一层 | 需 load-balancing loss | 最像 MoE，推理省，但易坍缩到单层 |
| **B3 载体条件路由** | router 额外输入 carrier embedding | 不需要 | 把"载体→层"先验注入，收敛快 |

**强诊断价值**：即使 B 不用于最终防御，它产出的 `p_l` 分布本身就是 H1（sensor 迁移）的可视化证据——可以画一张"载体 × 层"的 router 注意力热图放进论文 motivation。

---

## 三、方法族 C：Sensor→Gate 传输桥 + 打标签（师姐想法3 的核心）

师姐想法3 的真问题是：**"某一层输入什么、输出什么、物理含义、输出怎么打标签"**——也就是这个 bridge 到底学什么、用什么监督信号。这是整个方法能不能 train 起来的关键。

**桥的形式**：低秩线性/MLP，`B: h_sensor → δ_gate`，把 sensor 读到的证据映射成 gate 处的注入量。

**三种打标签方案（建议并行试）**：

| 方案 | 监督信号（标签） | 物理含义 | 优劣 |
|---|---|---|---|
| **C1 行为标签** | 最终输出 refuse/comply（L_safe_policy） | 端到端学"该拒就拒" | 直接，但 credit assignment 难、易过拟到模板 |
| **C2 蒸馏标签（推荐）** | 用**纯文本版本会拒答时**的 gate 隐状态作 teacher，让 bridge 把视觉输入的 gate 隐状态拉向它 | bridge 学"把视觉证据翻译成文本证据在 gate 处的表征" | 标签来自模型自身可拒答的反事实，信号干净 |
| **C3 自监督路由正则** | 无显式 gate 标签，最大化 corr(s(x), Δ(x)) | 鼓励"sensor 读到的风险"与"gate 处安全间隔"正相关 | 你文档里 L_routing_reg 的来源，无需额外标注 |

**层的物理含义梳理**（回答师姐的问题）：
- sensor 层 `h_ℓ`：输入=残差流状态，probe 输出=标量风险分/证据向量；含义"这里能线性读出有害性"。
- gate 层 `h_g`：输入=残差流状态，干预输出=安全策略翻转幅度；含义"在这里推一下能改变拒/答"。
- bridge：学习"证据空间→决策空间"的传输映射。**标签的本质 = 告诉 bridge 决策空间里"安全"长什么样**，C2 用反事实 teacher 给得最自然。

---

## 四、并行实验矩阵（先 inference-time 找上限，再训 LoRA）

执行你"先并行试、看哪个最有效、多跑几版选最好"的要求。**关键纪律：先做不训练的 inference-time 版本验证上限，再投入 LoRA 训练。**

**第 0 轮（inference-time oracle，0 训练成本，1–2 天出结论）**：
| 实验 | Router | Gate | Transport | 目的 |
|---|---|---|---|---|
| O1 固定 sensor + 直接 patch | 固定 locus | 无阈值 | 直接复制 | 上限 baseline |
| O2 oracle 选层 + patch | 用 MIBD 已知最优层 | 无 | 直接复制 | 验证"选对层"收益 |
| O3 + 阈值门控 | oracle 层 | 硬阈值 τ | 直接复制 | 验证门控对 utility 的保护 |

**第 1 轮（训练版，挑第 0 轮赢家做 LoRA）**：
| 实验 | 组合 | 对照 |
|---|---|---|
| T1 | A2 软门控 LoRA（固定层） | vs O3 |
| T2 | B1 软路由 + A2 门控 | vs T1（验证学习选层的增益） |
| T3 | B+A+C2 全桥（蒸馏标签） | vs T2、vs ReGap/VLM-Guard baseline |
| T4 | B+A+C3（自监督正则） | vs T3（验证是否能省掉标签） |

**Go/No-Go**：T3 必须显著优于 ReGap-style drift correction，否则问题退化为 ReGap 子问题。

---

## 五、Bench-first 数据策略 + 测试目标规划

你的诊断逻辑很对，我把它写成决策树：

```
先在 bench 上训一版
   ├─ bench 有效果 → 迁移到（更真实/更大的）数据集
   │        ├─ 数据集也有效 → 成立，扩展规模
   │        └─ 数据集没效果 → 数据分布差太远（domain gap），需补域内数据/做域适配
   └─ bench 没效果 → 是算法问题，回去改 Router/Gate/Transport 设计，不要先怪数据
```

**bench 与数据集分工（均为真实、已核实）**：
| 角色 | 候选 | 用途 |
|---|---|---|
| 训练/诊断 bench | **FigStep**（AAAI'25）+ 自建 `(q, v^safe, v^risk)` 三元组 | 现象最干净，先在这训一版 |
| 无泄露安全测试床 | **VLSBench**（ACL'25，无 VSIL） | 挡 dataset artifact，验证真·视觉风险 |
| 组合性风险 | **SIUO**（NAACL'25） | 测单模态安全、组合不安全的难例 |
| 规模化迁移 | **MM-SafetyBench**（ECCV'24） | bench 成立后做规模迁移 |
| Utility / 过拒 | MME / MMBench 等通用 VQA | 验证门控没伤正常能力（L_utility） |

**"想好未来在什么上测"——三层测试目标**：
1. **安全有效性**：ASR_LG3↓（用 LlamaGuard3 判，不用 keyword 防评估幻觉）在 FigStep / VLSBench / SIUO。
2. **能力保持**：通用 VQA 准确率、过拒率（benign 误拒）基本不掉。
3. **跨载体/跨模型泛化**：FigStep 训练 → V-text/V-blank/V-noise 测；Qwen3-VL 训练 → Qwen2.5-VL 测。

---

## 六、Motivation 关键一行重写 + 最有利的验证实验

**Motivation 最关键一行（建议定稿）**：
> 「VLM 安全失效的根因不是模型读不到风险，而是视觉载体把 risk evidence 搬到了一个决策机制读不到的位置；我们用一个载体感知的桥，把迁移后的证据动态送回行为控制点，恢复拒答。」

英文版（投稿用）：
> *In VLMs, safety fails not because risk is imperceptible, but because the visual carrier relocates risk evidence to a locus the decision mechanism does not read; we restore refusal by learning a carrier-aware bridge from the migrated sensor to the behavioral gate.*

**根据 motivation 放"最有利"的验证实验（一张三联图打穿）**：
在同一批 FigStep 样本上并排呈现——
1. 迁移后 sensor 处 **probe AUC=1.000**（证据可读）；
2. 模型行为却 **comply**（行为失败）；
3. **从 sensor patch 到 gate → 恢复 refuse**，而 fixed direction / drift correction **做不到**。

这三格直接对应"可读 → 没用上 → 桥接修复"，是支撑 motivation 最有力、最省篇幅的证据。

---

## 七、需要你拍板的设计开放点

1. **Router 粒度**：先做 B1 软混合（稳、可解释、出热图）还是直接上 B2 Top-1 硬 MoE（更像创新点但易坍缩）？我建议 B1 先行、B2 作为升级。
2. **打标签主线**：C2 蒸馏标签（需构造"纯文本会拒答"的反事实）还是 C3 自监督正则（省标注）作为主方法？我建议 C2 主、C3 作为消融证明"标签可省"。
3. **Gate 注入位置**：注入到单一 gate locus，还是注入到"gate→最后一层"整段（师姐说的后几层小 LoRA 直导最后一层）？这取决于实验 2 找到的 gate 是单层还是一段。
4. **第 0 轮是否先做**：是否同意先花 1–2 天做 inference-time oracle（O1–O3）确认上限，再决定投不投 LoRA 训练？

---

> 说明：本文档为方法设计提案，所有方法均为待验证设计，非已有结论；引用的 bench/方法（FigStep、VLSBench、SIUO、MM-SafetyBench、ReGap、VLM-Guard、CAST）均为已核实的真实工作。
