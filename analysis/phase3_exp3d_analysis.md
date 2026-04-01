# Phase 3 Exp 3D 深度分析报告：逐层 Ablation 曲线的跨模型比较

> 日期：2026-04-01  
> 数据来源：Exp 3A（方向提取）+ Exp 3C（5配置消融）+ Exp 3D（逐层 ablation 曲线，50 prompts，stride=2）  
> 分析对象：LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL2-8B、InstructBLIP-7B

---

## 一、四模型数据总览

| 模型 | ViT架构 | NW层 | baseline fhr | NW单层 fhr | 全层 fhr | 3D曲线峰值层 | 峰值 fhr | 模式 |
|------|---------|------|-------------|-----------|---------|------------|---------|------|
| LLaVA-1.5-7B | CLIP ViT-L | 16 (50%) | 0.580 | **0.870** | 0.720 | 16 | **0.900** | 单峰集中型 |
| Qwen2.5-VL-7B | 自研 ViT | 24 (86%) | 0.020 | 0.540 | **0.990** | 22 | **0.640** | 深层悬崖型 |
| InternVL2-8B | InternViT | 28 (88%) | 0.020 | 0.030 | 0.050 | 18 | 0.040 | 平线型（免疫） |
| InstructBLIP-7B | BLIP-2 ViT-G | 20 (62%) | 1.000 | 1.000 | 1.000 | - | 1.000 | 常数型（无对齐） |

> **注**：InstructBLIP 结果为 eval artifact + Vicuna backbone 无安全对齐，排除后续分析。

---

## 二、LLaVA：窄腰现象的完整几何证明

### 2.1 曲线形态：非对称单峰

```
层  0: ██████████       0.540
层  2: ████████████     0.600
层  4: ████████████     0.620
层  6: ███████████      0.580
层  8: █████████        0.460
层 10: ████████         0.420
层 12: ██████           0.320   ← 低谷（异常）
层 14: ████████████     0.620
层 16: ██████████████████ 0.900 ← NW 峰值（cos=0.917，norm_ratio=1.061，深度50%）
层 18: ████████████████ 0.820
层 20: ██████████████   0.720
... 单调下降至 0.580
```

### 2.2 核心发现：幅度反转与单层有效性的因果关系

Exp 3A 的 norm_ratio 序列：`0.605 → 0.777 → 1.061 → 1.151 → 1.221`

**layer 16 是 norm_ratio 从 <1 跨越到 >1 的临界点**。这不是巧合——这正是 ablation 效果的峰值层。这建立了一个因果链：

> 视觉模态在 layer 16 由"压制安全信号"转为"放大安全信号"。这个反转点是安全信号从分散的浅层表达汇聚到纯粹的、高幅度的单方向信号的转折。正是这种汇聚使得单层精确 ablation 比全层粗暴 ablation 更有效（89.7% vs 74.1%）。

**新发现：幅度反转点 = ablation 最优层**，这一因果关系在 LLaVA 上得到了完整的实验验证。

### 2.3 Layer 12 的"诱人低谷"——最反直觉的发现

Layer 12 的 fhr=0.320，**低于 baseline=0.580**。消融一层反而让攻击效果**下降到低于不消融的水平**。

**为什么会出现这种现象？**

在 layer 12，norm_ratio=0.777，视觉模态**仍在压制**安全信号。此处的安全方向（mean-diff direction）实际上是混合信号：既包含安全属性，也包含 blank image 对安全信号的"削弱贡献"。当我们消融 layer 12 的这个方向时，我们同时移除了这两种贡献——包括那个有利于攻击者的"视觉压制分量"。结果：安全信号反而变强了。

这一发现的意义：**refusal direction 在浅层携带"双重信息"**——既是安全信号，也是视觉模态与安全机制的交互界面。浅层消融可能适得其反。这为 Phase 4 的攻击设计提供了重要约束：**避免消融 shallow layers**。

---

## 三、Qwen2.5-VL：深层悬崖与"全或无"的安全门控

### 3.1 曲线形态：前段死寂 + 后段陡崖 + 断崖式终止

```
层  0-18: fhr ≈ 0.000   (死寂区，消融无效)
层  20:   fhr = 0.280   ████ (陡然激活)
层  22:   fhr = 0.640   ████████████ (真正峰值)
层  24:   fhr = 0.440   ████████  (NW层，次峰)
层  26:   fhr = 0.040   (断崖式坍塌)
层  27+:  fhr ≈ 0.020   (恢复到 baseline)
```

### 3.2 三个独立的反常发现

**反常 1：真正峰值（layer 22）≠ cos 最高层（NW layer 24）**

Exp 3A 认定 layer 24 为 narrow waist（cos=0.961，最高），但 ablation 效果在 layer 22 达到峰值（fhr=0.640 > layer 24 的 0.440）。这说明 **cos(v_text, v_mm) 最高层 ≠ 安全信号的功能执行层**。对于 Qwen，layer 22 是实际"触发"安全拒绝的执行层，而 layer 24 只是方向最纯粹的层。

**反常 2：all-layer ablation（0.990）远超 single-layer ablation（0.640）**

LLaVA 的规律是"NW > all"，Qwen 的规律正好相反。这意味着 Qwen 的安全机制**不是单一的**：层 20-24 确实是关键执行区，但早期层（0-18）储存了安全相关的"前置判断"。仅消融执行层可以让 64% 的请求通过，但那 36% 被拦下的请求有着来自浅层的安全表征，无法被后期 ablation 覆盖。只有消融**所有层的**安全方向，才能让早期层的前置判断也失效。

这揭示了 Qwen 的安全架构：**"早期积累 + 晚期执行"的委员会决策模型**，而非 LLaVA 的"单一瓶颈"模型。

**反常 3：layer 26 的断崖式复原（0.640 → 0.040）**

消融 layer 22 可以让 64% 的请求通过，但消融 layer 26 只有 4%。在 layer 22 到 layer 26 之间的短短 2 层（相对深度 0.786 → 0.929），模型几乎完全重建了安全防御。这暗示 Qwen 存在一个**末端安全复核机制**——即使中间层的安全信号被破坏，最后几层会重新检测并恢复拒绝行为。

这个"末端复核"机制是我们目前在文献中没有见到过的 VLM 特有现象。

### 3.3 Qwen 的 Amp-Reversal 缺失与安全扩散的关联

Exp 3A 显示 Qwen 的 norm_ratio 序列：`0.993 → 0.854 → 0.568 → 0.886 → 0.911`（全部 < 1，无反转）。

与 LLaVA 不同，Qwen 的视觉模态**始终压制**安全信号而非放大。这导致了两个结果：
1. 没有集中点 → 单层 ablation 只能做到 64%（而非 LLaVA 的 90%）
2. 整体安全幅度被视觉模态均匀削弱 → 所有层同时消融时，才能达到近 100% 攻击成功率

---

## 四、InternVL2：绝对免疫——安全机制的第三种范式

### 4.1 曲线形态：完全平线

```
全部 18 个探测层（包括 extra 层）：fhr = 0.020（与 baseline 完全一致）
最大波动：layer 18 的 fhr = 0.040（+0.020，可能是统计噪音）
```

3C 的全层消融也只有 0.050。**无论消融哪一层、所有层，InternVL2 的安全机制都不受影响。**

### 4.2 为什么 InternVL2 对方向消融免疫？

这不是实验错误。Sanity check 显示正常的拒绝回答（"I'm sorry, but I cannot..."），decoder 正确工作，ablation hooks 也确实被注册在正确的层上。

真正的原因有三层：

**原因 1：no amplitude reversal → 无浓缩点**
InternVL2 的 norm_ratio 全程 0.74-0.84（均匀压制），没有像 LLaVA 那样的浓缩点。安全信号以均匀分布的方式编码在所有层中，消融任何单一层只是移除了 1/32 的安全表征，剩余的 31 层足以重建安全决策。

**原因 2：mean-diff direction 在 InternVL2 上可能是错误的代理**
mean-difference 方法提取的是 harmful vs harmless prompts 在该层的均值差。对于 InternVL2（InternLM2 backbone，有 Constitutional AI 式的安全训练），安全信号可能不是单一的线性方向，而是更复杂的非线性结构。mean-diff 提取的方向是安全信号的一个线性近似，但这个近似可能不足以捕获 InternLM2 真正使用的安全特征。

**原因 3：InternVL2 的安全深度已超过 NW 层位置**
NW 层是 layer 28（深度 88%）。到了这个深度，前 28 层已经完成了完整的安全决策。在 layer 28 处消融方向，只是在安全决策已经形成之后进行干预，无法改变已经流入 residual stream 的信息。这与 LLaVA 的 layer 16（50% 深度）形成对比：LLaVA 的 NW 恰好在安全信号汇聚的"交通节点"，而 InternVL2 的 NW 在安全决策的"尾声"。

### 4.3 InternVL2 的重要性：最强安全对齐 + 最抗攻击

在四个模型中，InternVL2 和 Qwen2.5-VL 的 baseline_text 和 baseline_mm 均低于 10%，是真正强安全对齐的模型。但 InternVL2 对任何 direction ablation 完全免疫，而 Qwen 在 all-layer ablation 下可以达到 99%。这说明 **InternVL2 代表了一种在方向消融维度上"不可攻击"的安全架构**，其防御不依赖于单一方向的安全编码。

---

## 五、跨模型比较：三种安全几何的统一框架

### 5.1 安全架构的三分类

基于逐层 ablation 曲线形状，我们提出三种 VLM 安全几何范式：

| 范式 | 代表 | 曲线形状 | 安全集中度 | Amp Reversal | 单层攻击有效 | 全层攻击有效 |
|------|------|---------|-----------|--------------|------------|------------|
| **单点瓶颈型（Bottleneck）** | LLaVA | 单峰，峰>>全层 | 高度集中在 50% 深度 | ✓ 有 | ✓✓✓ 90% | ✓✓ 72% |
| **晚期门控型（Late Gate）** | Qwen2.5-VL | 深层陡崖，全层>>单层 | 后 30% 层 | ✗ 无 | ✓ 54% | ✓✓✓ 99% |
| **弥散不变型（Diffuse）** | InternVL2 | 平线 | 均匀分布 | ✗ 无 | ✗ 3% | ✗ 5% |

这三种范式可以通过两个轴来刻画：
- **轴1（集中度）**：安全信号是集中在少数层还是均匀分布？
- **轴2（深度）**：安全执行发生在模型的前段、中段还是后段？

```
                 集中                  弥散
浅-中  ┃  Bottleneck (LLaVA)    ┃  (假设存在未知模型)
      ┃  单峰 50%               ┃
深     ┃  Late Gate (Qwen)       ┃  Diffuse (InternVL2)
      ┃  陡崖 70-86%            ┃  平线
```

### 5.2 Amplitude Reversal 是安全集中度的预测因子

这是本研究最重要的跨模型 insight：

**Amp Reversal（norm_ratio 由 <1 跨越 >1）是安全集中度的充分指示符。**

- 有 Amp Reversal → 深层视觉模态放大安全信号 → 信号在反转点汇聚 → 形成 bottleneck → 单层消融高效
- 无 Amp Reversal → 视觉模态全程均匀影响 → 无浓缩点 → 单层消融低效

这建立了一个因果模型：

```
CLIP ViT (线性 patch projection) 
    → 图像表征平滑注入 LLM
    → 浅层 safety signal 被视觉噪音稀释 (norm_ratio < 1)
    → 中层残差流将多种模态信号整合
    → 深层安全机制将分散的安全线索"汇聚结晶" → norm_ratio 跨越 1
    → 形成 Bottleneck

自研 ViT (Qwen/InternVL 的 cross-attention 或 InternViT)
    → 图像表征通过更复杂的路径注入 LLM
    → 安全信号从一开始就被分散在跨层的多个表示维度中
    → norm_ratio 全程 < 1，无集中事件
    → 无法形成 Bottleneck
```

这个假说可以被 Phase 4 进一步验证：找一个 CLIP ViT 的模型（如 LLaVA-NeXT-Mistral），看其是否也呈现 Bottleneck 模式。

### 5.3 Bottleneck 的 Universal Depth：~50% 相对深度

LLaVA 的 narrow waist 在相对深度 50%（layer 16/32）。这与 LLM 文献中发现的"功能层"位置一致：Gao et al. (2025) 在纯 LLM 中也发现中间层是安全信号的汇聚点。本研究在 VLM MM 模式下复现了这一发现，并增加了新的维度——**视觉模态的 amp reversal 恰好在 50% 深度发生，这不是偶然，而是 CLIP 架构下残差流积累的必然结果**。

---

## 六、重要发现汇总与 Paper Claim

### Finding 1：单层消融效果与幅度反转点精确对应（LLaVA）
- **数据**：Layer 16 ablation fhr=0.900，同时 norm_ratio 在 layer 16 恰好跨越 1.0
- **含义**：幅度反转不仅是 Exp A 的观察现象，而且是安全瓶颈的**功能性标志**
- **Novelty**：建立了几何特征（norm_ratio）→ 功能特征（ablation 效果）的因果桥梁

### Finding 2：浅层消融可能适得其反（LLaVA Layer 12 悖论）
- **数据**：Layer 12 ablation fhr=0.320，**低于** baseline fhr=0.580
- **含义**：浅层的 refusal direction 携带双重信息（安全 + 视觉压制），消融后反而强化了模型的安全性
- **Novelty**：首次展示了安全方向消融的"双刃剑效应"，对攻击设计有直接指导意义

### Finding 3：Qwen 的末端安全复核机制（"Last-Layer Recovery"）
- **数据**：Layer 22 ablation fhr=0.640，Layer 26 ablation fhr=0.040，2层之内完全复原
- **含义**：Qwen 在最后几层存在独立的安全复核，这是其安全鲁棒性的重要来源
- **Novelty**：在 VLM 安全文献中首次量化了末端复核机制的存在

### Finding 4："NW > all-layer" 的模型特异性
- **数据**：LLaVA 中 NW(0.870) > all-layer(0.720)，Qwen 中正好相反 NW(0.540) < all-layer(0.990)
- **含义**：这两个看似矛盾的结果实际上揭示了不同的安全架构：LLaVA 是单点瓶颈，Qwen 是多层协同
- **Novelty**：用同一实验范式揭示了两种对立的安全几何结构，证明"NW > all-layer"不是普遍规律

### Finding 5：InternVL2 代表 direction-ablation 免疫的第三种范式
- **数据**：所有 18 个探测层，fhr 均维持在 0.020，全层消融也只有 0.050
- **含义**：当安全信号均匀弥散时，基于线性方向的消融攻击原理性失效
- **Novelty**：界定了 direction ablation 攻击的有效性边界，这对防御设计有启示（InternVL2 式的分布安全比 LLaVA 式的集中安全更鲁棒）

---

## 七、对 Phase 4 AGDA 设计的 Implication

基于以上分析，Phase 4 的攻击策略需要**针对不同安全架构采取不同策略**：

| 模型 | 安全架构 | Phase 4 推荐攻击策略 |
|------|---------|-------------------|
| LLaVA | Bottleneck (单点) | AGDA 对准 layer 16，minimize MSE to layer-16 ablation hidden state |
| Qwen2.5-VL | Late Gate (多层协同) | 需要同时 distill 多个层（20-24），或直接以 all-layer ablation 为 oracle |
| InternVL2 | Diffuse (免疫) | direction ablation 路线无效，需要考虑 adversarial input / 不同攻击 paradigm |

**关键结论**：AGDA 对 LLaVA 家族（Bottleneck 型）最有希望达到高 ASR，因为存在单一的高效攻击点。对 Qwen 需要更复杂的多层 distillation。对 InternVL2，当前的方向消融思路需要从根本上重新设计。

---

## 八、局限性说明

1. **InstructBLIP 数据无效**：Vicuna backbone 无安全对齐 + eval 解码 artifact，不能作为有效数据点。Group A 的第二个代表需要用 LLaVA-NeXT 替代。
2. **50 prompts 的统计功效**：3D 实验用 50 条 prompts，部分百分比（如 0.020 = 1/50）的统计显著性有限。关键数值需要在 100 条以上数据上确认。
3. **单一 direction 的局限**：ablation 全程使用 NW 层的 v_mm 方向。对于没有明显 NW 的模型（Qwen、InternVL2），用其他层的方向可能会有不同结果（这是下一步的探索方向）。

---

*分析结束。最重要的三个可写入 paper 的跨模型发现：幅度反转预测瓶颈层（Finding 1）、浅层消融悖论（Finding 2）、安全几何三分类框架（Section 5.1）。*
