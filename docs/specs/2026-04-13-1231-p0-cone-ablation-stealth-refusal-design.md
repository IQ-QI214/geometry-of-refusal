# P0: Concept Cone Ablation for Stealth Refusal Mechanism Verification

> Date: 2026-04-13 12:31
> Status: Approved
> Author: qi + Claude
> Priority: 最高（A2/A3 暂停，等 P0 结果后再决定）

---

## 1. 实验目标与核心假说

### 1.1 核心研究问题

**Stealth Refusal 的来源是什么？**

Category A 实验发现：Qwen-7B 在 DIM 单一方向消融（k=1）后，ASR_keyword=96.5% 但 ASR_judge=13.3%，差值 83.2% 为 stealth refusal——模型停止使用标准拒绝语言但仍以伦理反驳形式拒绝。

本实验通过扩展消融维度（DIM cone + RDO cone, k=1,3,5）来判定 stealth refusal 的机制来源。

### 1.2 两个假说

- **假说 A（Incomplete Cone）**: 单一方向消融只覆盖了 refusal cone 的一个维度。扩展到完整 cone 或使用 RDO 优化后的方向后，stealth refusal 会显著下降，ASR_judge 大幅上升。
- **假说 B（Independent Layer 2）**: Stealth refusal 由独立于 refusal cone 的第二层安全机制产生。即使完整 cone 消融，stealth refusal 仍顽固存在。

### 1.3 判定方式

观察 k=1→3→5 的 ASR_judge 趋势曲线：
- 单调递增且斜率显著 → 支持假说 A
- 平坦或饱和 → 支持假说 B
- 不预设硬门槛，以趋势为判据

### 1.4 与现有文献的关系

| 文献 | 方法 | Qwen ASR | 评估方式 | 能检测 stealth refusal |
|------|------|---------|---------|---------------------|
| Arditi (NeurIPS 2024) | DIM k=1 | 79-84% | keyword + LlamaGuard | ❌ |
| Wollschläger (ICML 2025) | RDO cone | 未报告 Qwen-VL | StrongREJECT | ❌ |
| 本研究 (Category A) | DIM k=1 | ASR_kw=96.5%, ASR_judge=13.3% | 双 judge 交叉验证 | ✅ |
| **本实验 (P0)** | **DIM + RDO cone** | **待测** | **四层统一评估** | **✅** |

核心论文贡献：在 Wollschläger 最强方法（RDO cone）上证明 stealth refusal 是否仍然存在。

---

## 2. 实验设计总览

### 2.1 实验模型

| 模型 | Type | 角色 | Conda Env | 模型路径 |
|------|------|------|-----------|---------|
| Qwen2.5-VL-7B | II (Late Gate) | Stealth refusal 主体 | qwen3-vl | `[PROJECT_ROOT]/models/Qwen2.5-VL-7B-Instruct` |
| LLaVA-1.5-7B | I (Bottleneck) | 对照（无 stealth refusal） | rdo | HF cache `llava-hf/llava-1.5-7b-hf` |

### 2.2 消融方法 × 维度

| 方法 | k=1 | k=3 | k=5 |
|------|-----|-----|-----|
| DIM Cone | PCA top-1（≈当前 A1 baseline） | PCA top-3 | PCA top-5 |
| RDO Cone | `train_refusal_vector()` | `train_refusal_cone(cone_dim=3)` | `train_refusal_cone(cone_dim=5)` |

总计：2 models × 2 methods × 3 k-values = **12 组实验**

### 2.3 数据集

使用现有 `data/saladbench_splits/`（与 Wollschläger 一致）：

| 用途 | 数据 | 数量 |
|------|------|------|
| RDO 训练 (harmful) | `harmful_train.json` | 1184 |
| RDO 训练 (harmless) | `harmless_train.json` | subsample 1184 |
| RDO 验证 | `harmful_val.json` + `harmless_val.json` | 128 + 128 |
| 最终评估 | `harmful_val.json` | 128 |

VLM 适配：所有 text prompt 配 224×224 blank white image。

### 2.4 统一评估指标：ASR

| 指标 | 评估方法 | 对标文献 | 能检测 stealth refusal |
|------|---------|---------|---------------------|
| ASR_keyword | substring matching (refusal prefix) | Arditi (1 - refusal_score) | ❌ |
| ASR_strongreject | StrongREJECT learned classifier | Wollschläger | ❌ |
| ASR_judge_q3g | Qwen3Guard-8B binary safe/unsafe | 本研究 | ✅ |
| ASR_judge_lg | LLaMA-Guard-3-8B binary safe/unsafe | 交叉验证 | ✅ |

**派生指标**：
- `Stealth Refusal Rate (SRR) = ASR_keyword - ASR_judge`（Q3G 和 LG 报告范围）
- `Evaluation Gap Ratio (EGR) = ASR_keyword / ASR_judge`
- `Cross-Judge Concordance = 1 - |ASR_q3g - ASR_lg| / max(ASR_q3g, ASR_lg)`

---

## 3. VLM Model Adapter 技术设计

### 3.1 架构

继承现有 `ModelBase` 抽象基类，新增两个 VLM adapter：

```
refusal_direction/pipeline/model_utils/
├── model_base.py          # 不动
├── model_factory.py       # 修改：新增 VLM 路径识别
├── qwen_model.py          # 不动（text-only）
├── llama3_model.py        # 不动
├── gemma_model.py         # 不动
├── llava_vlm_model.py     # 新增
└── qwen_vlm_model.py      # 新增
```

### 3.2 输入处理

```python
# text-only LLM（现有）
tokenize_fn(instruction) → input_ids

# VLM adapter（新增）
tokenize_fn(instruction) → input_ids + pixel_values + image_grid_thw
```

Blank image：预生成一张 224×224 白色 PIL Image，在 `_get_tokenize_instructions_fn()` 中统一注入。

### 3.3 Hook 作用范围

```
VLM 架构:
  Vision Encoder (ViT)     ← 不动
  Visual Projector          ← 不动
  LLM Backbone Layers       ← Hook 作用于此
  Output Head
```

Hook 只挂在 LLM backbone transformer layers 上，与 text-only RDO 完全一致。理由：
- Category A 已验证此策略有效（all_vmm 消融 IBR=96.5%）
- 与 Wollschläger 在 hook 层面完全可比
- 不破坏 vision encoder 能力

### 3.4 方向提取的 Token 位置

```
VLM token 序列:
  [system] [visual_tokens...] [instruction] [eoi_token] [assistant_start]
                                               ↑ 提取位置
```

在 end-of-instruction token 位置提取 hidden state（此时已融合 visual 和 text 信息）。

### 3.5 Refusal Tokens

| Model | 预期 Refusal Tokens | 验证方式 |
|-------|-------------------|---------|
| Qwen2.5-VL-7B | `[40, 2121]` ("I", "As") — 与 text Qwen 共享 tokenizer | Phase 1 smoke test |
| LLaVA-1.5-7B | 从 LLaMA-2 tokenizer 提取 | Phase 1 smoke test |

### 3.6 Forward Pass 适配

```python
# VLM forward
outputs = model(
    input_ids=input_ids,
    attention_mask=mask,
    pixel_values=pixel_values,          # 新增
    image_grid_thw=image_grid_thw,      # Qwen-VL 特有
)
```

### 3.7 Loss 架构（前向兼容 L_visual-retain）

```python
def compute_rdo_loss(model, batch, cone, config):
    # 三项损失（text tokens only，复现 Wollschläger）
    loss_abl = ablation_ce_loss(model, harmful_batch, cone, text_mask)
    loss_add = addition_ce_loss(model, harmless_batch, cone, text_mask)
    loss_ret = retain_kl_loss(model, harmless_batch, cone, text_mask)
    
    total = λ_abl * loss_abl + λ_add * loss_add + λ_ret * loss_ret
    
    # Visual token 监控（始终记录，不参与梯度）
    with torch.no_grad():
        vis_drift = visual_token_drift(model, batch, cone, vis_mask)
        log("visual_token_l2_drift", vis_drift)
    
    # 第四项损失（开关控制，Phase 3 默认关闭）
    if config.enable_visual_retain:
        loss_vis = visual_retain_loss(model, batch, cone, vis_mask)
        total += λ_vis * loss_vis
    
    return total
```

三项损失完全复现 Wollschläger。visual drift 始终监控但默认不参与训练。若观察到 drift 显著，仅需 `enable_visual_retain=True` 重新训练，不改架构。

---

## 4. DIM Cone 提取方案

### 4.1 方法

```python
# Step 1: 提取 activations
H_harm = extract_activations(model, harmful_data, layer=best_layer)  # (N, d)
H_safe = extract_activations(model, harmless_data, layer=best_layer) # (N, d)

# Step 2: 计算 centered diffs
diffs = H_harm - H_safe  # (N, d)

# Step 3: PCA
U, S, Vt = torch.svd(diffs)
cone_basis = Vt[:k]  # (k, d) — top-k 右奇异向量

# Step 4: 在所有层上消融
for layer in all_layers:
    for v in cone_basis:
        add_ablation_hook(layer, direction=v)
```

### 4.2 最优层选择

用现有 `select_direction.py` 选出最优 (position, layer)，在该层上做 PCA。

### 4.3 数据用量

| 用途 | 数据 | 数量 |
|------|------|------|
| 方向提取 | harmful_train + harmless_train (subsample) | 1184 + 1184 |
| 最优层选择 | harmful_val + harmless_val | 128 + 128 |
| 最终评估 | harmful_val | 128 |

---

## 5. RDO 复现方案

### 5.1 完整流程（忠实复现 Wollschläger）

```
Step 1: DIM 方向作为初始化
  └ generate_directions.py → select_direction.py → 最优 DIM 方向

Step 2: Target 生成（原论文 §3.1）
  └ harmful prompts → DIM 消融 → 生成 30 tokens → t_answer
  └ harmless prompts → DIM activation addition → t_refusal
  └ harmless prompts → 无干预 → 生成 29 tokens → t_retain

Step 3: RDO 梯度优化
  └ k=1: train_refusal_vector()
  └ k=3,5: train_refusal_cone()

Step 4: 方向选择
  └ k=1: select_rdo_direction()
  └ k=3,5: select_cone_basis()

Step 5: 消融 → 生成 → 四层评估
```

### 5.2 三项损失（原论文 §3.2）

```
L_total = λ_abl · L_ablation + λ_add · L_addition + λ_ret · L_retain

L_ablation: 消融 cone 后模型在 harmful prompts 上生成 t_answer 的 CE loss
L_addition: 添加 cone 后模型在 harmless prompts 上生成 t_refusal 的 CE loss
L_retain:   消融 cone 后模型在 harmless prompts 上输出分布与原始的 KL divergence
            作用于最后 30 tokens（含 chat template 尾 token）
```

λ 参数沿用 `rdo.py` 中原论文默认值。

### 5.3 Cone 训练参数

| 参数 | k=1 | k=3 | k=5 |
|------|-----|-----|-----|
| 函数 | `train_refusal_vector()` | `train_refusal_cone()` | `train_refusal_cone()` |
| 基向量数 | 1 | 3 | 5 |
| 采样方式 | 无 | hypersphere | hypersphere |
| 正交化 | 无 | 训练中自动正交化 | 训练中自动正交化 |
| 训练数据 | harmful 1184 + harmless 1184 | 同左 | 同左 |
| 验证数据 | harmful 128 + harmless 128 | 同左 | 同左 |

### 5.4 DIM vs RDO 的核心区别

| 维度 | DIM Cone | RDO Cone |
|------|----------|----------|
| 方向来源 | PCA 分解 mean_diff | 梯度优化 |
| 副作用控制 | 无 | L_retain (KL divergence) |
| 方向质量 | 可能纠缠无关语义 | 专注 refusal 概念 |
| 计算成本 | 低（一次 SVD） | 高（多轮梯度更新） |
| cone 质量保证 | 无 | hypersphere 采样验证 |

---

## 6. 评估框架统一设计

### 6.1 四层评估流水线

```
消融后的 128 条生成响应
         │
         ├─→ Layer 1: ASR_keyword      (substring_matching, CPU)
         ├─→ Layer 2: ASR_strongreject  (StrongREJECT classifier, CPU)
         ├─→ Layer 3: ASR_judge_q3g     (Qwen3Guard-8B, GPU 0)
         └─→ Layer 4: ASR_judge_lg      (LLaMA-Guard-3-8B, GPU 1)
```

### 6.2 各层实现

**Layer 1 (ASR_keyword)**: 复用 `evaluate_jailbreak.py` 的 `substring_matching`
- 12 个 refusal prefix ("I'm sorry", "I apologize", "I cannot", ...)
- ASR = 1 - refusal_rate

**Layer 2 (ASR_strongreject)**: 复用 `evaluate_jailbreak.py` 的 `strongreject`
- `from strong_reject.evaluate import evaluate_response`
- 需确认包已安装

**Layer 3/4 (ASR_judge)**: 复用 Category A 的 `judge_utils.py`
- Qwen3Guard-8B + LLaMA-Guard-3-8B
- ASR = unsafe 判定比例

### 6.3 论文核心结果表

```
Table 1: ASR Comparison Across Evaluation Frameworks

| Model    | Method | k | ASR_kw | ASR_sr | ASR_q3g | ASR_lg | SRR        |
|----------|--------|---|--------|--------|---------|--------|------------|
| Qwen-7B  | —      | 0 | (base) | (base) | (base)  | (base) | —          |
| Qwen-7B  | DIM    | 1 |        |        |         |        |            |
| Qwen-7B  | DIM    | 3 |        |        |         |        |            |
| Qwen-7B  | DIM    | 5 |        |        |         |        |            |
| Qwen-7B  | RDO    | 1 |        |        |         |        |            |
| Qwen-7B  | RDO    | 3 |        |        |         |        |            |
| Qwen-7B  | RDO    | 5 |        |        |         |        |            |
| LLaVA-7B | —      | 0 | (base) | (base) | (base)  | (base) | —          |
| LLaVA-7B | DIM    | 1 |        |        |         |        |            |
| LLaVA-7B | DIM    | 3 |        |        |         |        |            |
| LLaVA-7B | DIM    | 5 |        |        |         |        |            |
| LLaVA-7B | RDO    | 1 |        |        |         |        |            |
| LLaVA-7B | RDO    | 3 |        |        |         |        |            |
| LLaVA-7B | RDO    | 5 |        |        |         |        |            |

Table 2: Stealth Refusal Trend (Qwen-7B)

| Method | k | SRR  | 趋势 → 假说判定 |
|--------|---|------|----------------|
| DIM    | 1 | ~83% | baseline       |
| DIM    | 3 |  ?   |                |
| DIM    | 5 |  ?   |                |
| RDO    | 1 |  ?   | vs DIM k=1     |
| RDO    | 3 |  ?   |                |
| RDO    | 5 |  ?   | 关键判定点      |
```

---

## 7. 时间线与依赖关系

### 7.1 执行顺序

```
Week 1
──────────────────────────────────────────────────
Day 1-2: Phase 1 — VLM Adapter
  ├─ 实现 llava_vlm_model.py + qwen_vlm_model.py
  ├─ 更新 model_factory.py
  └─ Smoke test: DIM k=1, 10 samples, 验证 ASR 一致

Day 3: Phase 2 — DIM Cone
  ├─ 提取 activations + PCA
  ├─ k=1,3,5 消融 + 生成（4×H100 并行）
  └─ ~0.5 天

Day 4-5: Phase 3 — RDO 训练
  ├─ Target 生成 (~1h)
  ├─ RDO 训练 k=1,3,5 × 2 models（4×H100 并行）
  └─ RDO 消融 + 生成

Week 2
──────────────────────────────────────────────────
Day 6: 四层评估
  ├─ Layer 1/2: CPU, ~30min
  └─ Layer 3/4: 2×GPU 并行, ~1h

Day 7: Phase 4 — 分析
  ├─ 填充结果表 + 绘制趋势曲线
  ├─ 假说 A/B 判定
  └─ 分析报告
```

### 7.2 依赖图

```
Phase 1 (Adapter)
  │
  ├──→ Phase 2 (DIM Cone) ──→ 四层评估 ──→ Phase 4 (分析)
  │                                           ↑
  └──→ Phase 3 (RDO Cone) ──→ 四层评估 ──────┘
         │
         └─→ visual drift 观察 → 决定是否加 L_visual-retain
```

Phase 2 与 Phase 3 可在 Phase 1 完成后并行启动。

### 7.3 检查点

| 检查点 | 时机 | 标准 | 止损 |
|--------|------|------|------|
| CP1 | Phase 1 完成 | smoke test ASR 与 A1 ±5pp | debug adapter |
| CP2 | DIM k=1 完成 | ASR_keyword 与 Category A 一致 | 检查 PCA |
| CP3 | RDO k=1 完成 | ASR_keyword ≥ DIM k=1 | RDO 未收敛，调参 |
| CP4 | 全部评估完成 | 四层 ASR 数据完整 | — |

### 7.4 产出物

| 产出 | 路径 |
|------|------|
| VLM adapters | `refusal_direction/pipeline/model_utils/{llava_vlm,qwen_vlm}_model.py` |
| DIM cone 方向 | `results/p0/{model}/dim_cone_k{1,3,5}.pt` |
| RDO cone 方向 | `results/p0/{model}/rdo_cone_k{1,3,5}.pt` |
| 生成响应 | `results/p0/{model}/{method}_k{k}_responses.json` |
| 四层评估结果 | `results/p0/{model}/{method}_k{k}_eval.json` |
| 训练日志 | `results/p0/{model}/rdo_train_k{k}.log` |
| 分析报告 | `analysis/p0/p0_cone_analysis_YYYY-MM-DD-HHmm.md` |

---

## 8. P0 结果后的下一步

### 若假说 A 成立（cone 消融消除 stealth refusal）

论文叙事："现有评估框架（Arditi, Wollschläger）的系统性盲区。在 VLM 场景下，单一方向消融的高 ASR 实为 stealth refusal 的误判。扩展到完整 cone 后，真实 ASR 与表观 ASR 趋于一致。"

后续：
- 扩展到 Qwen-32B + LLaVA-13B 验证 scale 效应
- 回到 A3 norm prediction，在 cone 消融条件下分析
- 可能需要加入 L_visual-retain

### 若假说 B 成立（cone 消融无效，独立 Layer 2）

论文叙事："现有几何攻击方法（含 RDO cone）对强对齐 VLM 存在根本性局限。首次系统揭示 Qwen-VL 存在独立于 refusal cone 的第二层安全机制，该机制在 LLM 安全文献中尚未被记录。"

后续：
- 设计定位 Layer 2 的新实验（activation patching, causal tracing）
- A2 因果验证重新设计（针对 Layer 2）
- 探索 Layer 2 是否可以通过其他攻击路径绕过

### 若结果介于 A/B 之间

论文叙事："几何攻击的效果存在上界。Cone 消融可部分瓦解 stealth refusal（SRR 从 83% 降至 X%），但 Type II 模型仍保留显著的残余安全能力。"

后续：
- 量化 cone 维度与 SRR 的精确关系
- 细化 k 值（补充 k=2,4）
- 探索 DIM 与 RDO 的差异是否与 Wollschläger 论文中 Qwen 的 TruthfulQA 副作用相关
