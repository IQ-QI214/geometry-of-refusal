# MIBD Phase 1.5 + Gemma 机制分型执行文档

> **时间锚点**：2026-06-03，承接 Phase 1 Go/No-Go 结果。
> **目标**：在 InternVL3 与 Qwen3-VL 现象不一致后，加入 Gemma 作为第三模型家族，并用 Phase 1.5 审计确认现有 probe 结果不是数据泄漏或任务过易造成的 artifact。

---

## 1. 当前结论与问题

Phase 1 已得到两个模型家族的初步结果：

- **InternVL3-8B**：`harmfulness` 信号满足 `CONTINUE_MIBD`。V-text 到视觉条件出现 locus/direction shift，blank/noise 等价，static transfer drop 通过。
- **Qwen3-VL-8B**：`harmfulness` 信号为 `STOP_OR_PIVOT`。它有 locus/direction shift，但 static transfer drop 失败，说明 text-condition harmfulness readout 在视觉条件下仍然稳定。

这不是失败，而是说明安全信号可能存在架构分型：

- InternVL3：可能是 **Belief-mislocalized**。
- Qwen3-VL：可能是 **Belief-robust control**，也可能是 AUC 饱和导致看不出 transfer drop。

当前最危险的异常是：两个模型所有 condition 的 AUC 都是 `1.000`。这必须先审计，否则 MIBD 的训练动机会被审稿人攻击为 probe artifact。

---

## 2. 本阶段核心任务

本阶段不进入 MIBD 训练。只做 3 件事：

1. **Phase 1.5 Probe Validity Audit**：确认 AUC=1.000 是否可信。
2. **Refusal Signal 复跑**：不能只看 harmfulness，必须同时分析 refusal。
3. **加入 Gemma 第三模型家族**：把二元冲突升级为机制分型。

---

## 3. 模型阵容

### 3.1 必跑模型

| 模型家族 | 模型 | 角色 |
|---|---|---|
| InternVL | InternVL3-8B 或 InternVL3.5-8B | 当前 positive case |
| Qwen | Qwen3-VL-8B | 当前 robust/control case |
| Gemma | Gemma 3 / Gemma vision-family 可用版本 | 第三架构验证 |

### 3.2 Gemma 的判定价值

加入 Gemma 后，结果解释如下：

- **2/3 模型出现 transfer failure**：可主张 visual-token-induced latent safety mislocalization 是多架构现象，Qwen 是 robust counterexample。
- **1/3 模型出现 transfer failure**：主线收缩为 architecture-dependent failure mode，不宜宣称普遍现象。
- **3/3 模型均出现不同形式 shift**：MIBD 动机显著增强。
- **审计后 0/3 成立**：停止 MIBD 主线，转 diagnostic/evaluation paper。

---

## 4. Phase 1.5 Probe Validity Audit

### 4.1 审计目标

验证 probe 是否真的读到了 harmfulness/refusal 信号，而不是学到数据集 artifact。

### 4.2 必做检查

对每个模型、每个 signal（`harmfulness` 与 `refusal`）执行：

1. **Held-out AUC**
   - 报告 train AUC 与 test AUC。
   - 禁止只报告训练集 AUC。

2. **Group split by `paired_id`**
   - 同一 harmful/harmless pair 不能同时出现在 train 与 test。
   - 如果没有 `paired_id`，先构造或标记为不可用。

3. **Label permutation test**
   - 随机打乱 label 后重训 probe。
   - 期望 permutation AUC 接近 `0.5`。
   - 如果 permutation AUC 仍显著高于 `0.5`，判为 artifact。

4. **Cross-category split**
   - 训练集与测试集使用不同安全类别。
   - 用来检验 probe 是否只记住 category/topic。

5. **Domain-local harmless controls**
   - harmless 样本必须尽量与 harmful 样本同领域。
   - 例如「制造炸药」对应「解释化学实验安全规范」，而不是「写一首诗」。

6. **Margin statistics**
   - AUC 饱和时必须报告 score margin。
   - 至少包括 mean gap、median gap、IQR、condition-wise score shift。

### 4.3 输出指标

每个模型与 signal 输出：

| 指标 | 说明 |
|---|---|
| Train AUC | 训练集性能 |
| Held-out AUC | 标准测试集性能 |
| Group-split AUC | 按 paired_id 分组后的测试性能 |
| Permutation AUC | label 打乱后的性能 |
| Cross-category AUC | 类别外泛化 |
| Mean margin | harmful 与 harmless score 均值差 |
| Static transfer margin drop | V-text readout 到视觉条件的 margin drop |

---

## 5. Refusal Signal 复跑

当前 Phase 1 只报告了 `harmfulness`，这不足以支撑 Zhao-style framing。下一步必须同时跑 `refusal`。

### 5.1 目标

区分以下情况：

- harmfulness 稳定，但 refusal 不稳定；
- harmfulness 与 refusal 都不稳定；
- harmfulness 与 refusal 都稳定；
- 两者表现完全重合，说明信号未被成功解耦。

### 5.2 推荐判读

| Harmfulness | Refusal | 解释 |
---|---|---|
| transfer fail | transfer fail | belief 与 policy/readout 都被视觉扰动 |
| stable | transfer fail | 最理想：belief intact，但 refusal locus 失稳 |
| transfer fail | stable | 需要谨慎，可能是 harmfulness probe artifact |
| stable | stable | robust control，不适合做 MIBD 主驱动 |

---

## 6. 机制分型表

最终报告不要再只输出 Go/No-Go，而要输出机制分型。

| Model | Harm locus shift | Harm transfer drop | Refusal locus shift | Refusal transfer drop | FigStep early-layer collapse | Type |
|---|---|---|---|---|---|---|
| InternVL3 | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-VL | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma | TBD | TBD | TBD | TBD | TBD | TBD |

类型定义：

- **Type A: Belief-mislocalized**  
  harmfulness 本身出现 transfer failure。

- **Type B: Belief-robust / Refusal-misaligned**  
  harmfulness 稳定，但 refusal 不稳定。这是最适合 MIBD 叙事的类型。

- **Type C: Fully robust**  
  harmfulness 与 refusal 都稳定，可作为 control。

- **Type D: Probe artifact / unresolved**  
  AUC 饱和但 permutation、group split 或 cross-category 不通过。

---

## 7. Coding Agent 执行任务

### 任务 1：扩展 Phase 1.5 审计模块

在 `experiments/mibd/` 中新增：

- `experiments/mibd/audit/`
- `experiments/mibd/audit/splits.py`
- `experiments/mibd/audit/permutation.py`
- `experiments/mibd/audit/margins.py`
- `experiments/mibd/eval/phase1p5_report.py`

需要支持：

- held-out split；
- group split by `paired_id`；
- label permutation；
- cross-category split；
- margin statistics；
- 输出 Phase 1.5 markdown。

### 任务 2：复跑 harmfulness 与 refusal

对每个模型执行：

- `signal_type=harmfulness`
- `signal_type=refusal`

视觉条件保持：

- V-text
- V-blank
- V-noise
- V-real
- FigStep

### 任务 3：加入 Gemma 配置

新增配置文件：

- `experiments/mibd/configs/phase1_probe_gemma.yaml`
- `experiments/mibd/configs/phase1p5_audit_gemma.yaml`

要求：

- 使用 Gemma vision-family 可用版本；
- 输出目录使用 `results/mibd/phase1_probe/gemma/`；
- 单独保存 token audit，因为 Gemma 的 chat template 可能与 Qwen/InternVL 不同。

### 任务 4：生成机制分型报告

新增总报告：

- `analysis/mibd/2026-06-03-phase1p5-gemma-mechanism-typing.md`

报告必须包含：

- 3 个模型的 harmfulness 与 refusal 表；
- Phase 1.5 审计结果；
- permutation AUC；
- margin drop；
- FigStep 单独分析；
- 最终机制类型；
- 是否进入 MIBD 训练的明确建议。

---

## 8. 进入 MIBD 训练的条件

只有满足以下条件，才进入 Phase 3 MIBD 训练：

1. 至少 1 个模型通过 Phase 1.5 审计后仍有真实 transfer failure；
2. 至少 1 个模型在 refusal signal 上出现 transfer failure；
3. permutation AUC 接近 `0.5`；
4. group-split 或 cross-category AUC 不崩；
5. Qwen/Gemma/InternVL 至少能形成可解释的机制分型。

如果只剩 InternVL 一个模型成立，则 MIBD 只能作为 architecture-specific method，不能写成一般 VLM 方法。

---

## 9. 当前推荐叙事更新

若 Gemma 结果支持分型，论文叙事应从：

> 视觉 token 会普遍导致 VLM safety signal mislocalization。

改为：

> 视觉 token 会以架构相关的方式改变 VLM latent safety geometry。部分模型出现 belief-level mislocalization，部分模型保持 harmfulness belief 稳定但可能出现 refusal-level instability。MIBD 的目标是把不稳定模型推向 Qwen-like modality-invariant belief geometry，同时修复 refusal behavior。

这个叙事比「所有模型都失败」更稳健，也更符合当前结果。

