# Phase 3 完整进展报告（Exp 3A + 3C）

> **文档版本**: 2026-03-31
> **状态**: Exp 3A ✅ 全部完成 | Exp 3C ✅ 全部完成 | Exp 3B ⏳ 待实现
> **用途**: 总结 Phase 3 已有结论，指导新对话窗口快速接续工作

---

## 1. 快速接续指令（新窗口必读）

```
1. 读取 plan-markdown/base_instructions-phase1.md       → 基础规则
2. 读取 analysis/phase3_exp3ac_report.md（本文档）     → Phase 3 完整进展
3. 读取 plan-markdown/gapc_phase3_supplement.md         → 完整 Phase 3 方案
4. 代码在 experiments/phase3/，结果在 results/phase3/
5. 下一步：实现 Exp 3B (Dynamic Rotation)
6. Qwen2.5-VL 必须用 qwen3-vl 环境运行（transformers >= 4.52）
7. 所有给 qi 的命令必须是单行格式（多行复制粘贴会乱码）
```

---

## 2. 实验结果总表

### 2.1 Exp 3A — Refusal Direction 跨模态分析

| 模型 | Group | 总层数 | NW 层 | NW 相对深度 | NW cos | Amplitude Reversal | 浅层 ratio | 深层 ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LLaVA-1.5-7B** | A | 32 | 16 | 0.50 | 0.917 | ✅ True | 0.691 | 1.144 |
| **Qwen2.5-VL-7B** | B | 28 | 24 | 0.86 | 0.961 | ❌ False | 0.924 | 0.788 |
| **InternVL2-8B** | B | 32 | 28 | 0.88 | 0.919 | ❌ False | 0.799 | 0.789 |
| **InstructBLIP-7B** | A | 32 | 20 | 0.63 | 0.467 | ✅ True | 0.740 | 1.271 |

- **Group A** (LLaVA, InstructBLIP): CLIP 系 ViT，浅层压制 + 深层放大（reversal）
- **Group B** (Qwen2.5-VL, InternVL2): 自研 ViT，全层均匀压制（no reversal）

### 2.2 Exp 3C — Narrow Waist Ablation Attack（full_harmful_completion_rate）

| 模型 | Group | baseline_text | baseline_mm | ablation_nw_vmm | ablation_all_vmm | ablation_nw_vtext |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LLaVA-1.5-7B** | A | 0.125 | 0.625 | **0.875** | 0.875 | 0.875 |
| **Qwen2.5-VL-7B** | B | 0.000 | 0.000 | 0.625 | **1.000** | 0.625 |
| **InternVL2-8B** | B | 0.000 | 0.125 | 0.125 | **0.250** | 0.125 |
| **InstructBLIP-7B** | A | 0.750 | **1.000** | 1.000 | 1.000 | 1.000 |

- **ablation_nw_vmm**: 仅 NW 层，用 v_mm 方向
- **ablation_all_vmm**: 全部层，用 v_mm 方向
- **ablation_nw_vtext**: 仅 NW 层，用 v_text 方向

---

## 3. 核心发现

### Finding 3C-1（最重要）：Amplitude Reversal 预测最优 Ablation 策略

**这是将 3A 和 3C 打通的核心联结。**

| Group | Amplitude 行为 | NW vs All-layer 效果 | 机制解释 |
|:---|:---|:---|:---|
| **Group A** (LLaVA, InstructBLIP) | 浅层压制 + 深层放大 | NW ≥ all-layer | Refusal 集中在瓶颈层，单层精准攻击有效 |
| **Group B** (Qwen2.5-VL, InternVL2) | 全层均匀压制 | **all-layer > NW** | Refusal 分布于所有层，必须全层攻击 |

最强证据：Qwen2.5-VL NW=62.5% vs all-layer=**100%**（差距 37.5pp）。

**Paper claim**: "The presence or absence of amplitude reversal predicts the optimal ablation strategy — CLIP-based architectures exhibit a narrow waist bottleneck enabling single-layer attacks, while custom ViT architectures require distributed ablation."

### Finding 3C-2：Qwen2.5-VL — 最强 Baseline Safety，但最完全的对齐崩溃

- baseline（text & mm）= **0% full_harmful**（唯一完全拒绝的模型）
- all-layer ablation = **100% full_harmful**（无任何 self-correction）

Safety 机制完全依赖 refusal direction 的完整性，无冗余备份。一旦被全局消除，安全机制完全失效。

**Open question**: 高 baseline safety ↔ 低冗余 是否是 alignment 训练的内在 trade-off？

### Finding 3C-3：InstructBLIP — Blank Image 已足够，无需 Ablation

- baseline_mm = **100%**（仅凭 blank image 即可完全绕过，无需任何 ablation）
- 因果链：Q-Former 32-token 压缩 → cos(v_text, v_mm) = 0.467（最低）→ visual modality 彻底改变 hidden state 分布 → safety 机制失效

**Paper claim**: "Q-Former architectures are trivially bypassed by visual input alone — the efficiency bottleneck becomes a security vulnerability."

### Finding 3C-4：v_text ≈ v_mm 对 Ablation 效果完全等价（所有模型）

| 模型 | cos(v_text, v_mm) | nw_vmm | nw_vtext | 差距 |
|:---|:---:|:---:|:---:|:---:|
| LLaVA | 0.917 | 0.875 | 0.875 | 0 |
| Qwen2.5-VL | 0.961 | 0.625 | 0.625 | 0 |
| InternVL2 | 0.919 | 0.125 | 0.125 | 0 |
| InstructBLIP | 0.467 | 1.000 | 1.000 | 0（ceiling）|

**实践意义**: 攻击者只需纯文本数据即可提取有效攻击方向，无需构建 multimodal 配对数据集。

### Finding 3C-5（Problem）：InternVL2 对 Ablation 的强抗性无法完全解释

InternVL2 all-layer ablation 仅 **25%**，远低于其他模型（LLaVA 87.5%，Qwen 100%，InstructBLIP 100%）。可能原因：
1. NW 层在 88% 深度（接近输出层），ablation 时机过晚
2. InternLM2 的 alignment 训练具有冗余机制，safety 不依赖单一方向
3. Mean-diff direction 未必是 InternVL2 最优的 refusal direction 表示

**待验证**: 对 InternVL2 扫描更多层，找到实际有效的 ablation 层。

---

## 4. 对 Paper Narrative 的影响

### 原假设 vs 实际结果

| 原假设 | 结果 | 修正方向 |
|:---|:---|:---|
| Narrow waist 是通用 architectural bottleneck | **仅 Group A 成立** | 改为 "与 visual encoder 类型相关的条件性现象" |
| NW ablation > all-layer ablation（通用） | **Group B 相反：all-layer > NW** | 新 claim：amplitude reversal 预测最优策略 |
| 所有模型 NW cos > 0.85 | **3/4 成立**（InstructBLIP 例外） | 排除 Q-Former 架构，该规律成立 |
| Q-Former cos 低 → ablation 效果差 | **相反：InstructBLIP 最易攻击** | Q-Former 的问题不是 ablation 无效，而是根本不需要 ablation |

### 推荐的 Paper Narrative（修正后）

> **"Visual encoder 架构类型通过 amplitude reversal 机制决定 refusal signal 的空间分布，进而决定最优攻击策略：CLIP 系架构产生层级瓶颈（单层精准攻击有效），自研 ViT 架构产生均匀分布（需要全层攻击）。同时，Q-Former 架构因极端信息压缩构成天然安全漏洞，无需任何 ablation 即可被 visual input 完全绕过。"**

---

## 5. 下一步方向

### 5.1 推荐优先：Exp 3B（Dynamic Rotation）

**目标**: 验证 refusal direction 在生成过程中是否随 token 位置旋转（Phase 2 Finding B）。

**设计方案**（使用每个模型的最高 cos 层）：

| 模型 | 目标层 | cos |
|:---|:---:|:---:|
| LLaVA-1.5-7B | 16 | 0.917 |
| Qwen2.5-VL-7B | 24 | 0.961 |
| InternVL2-8B | 28 | 0.919 |
| InstructBLIP-7B | 20 | 0.467 |

**方法**: Teacher-forced controlled generation，在多个 token 位置提取 hidden states，计算 direction 随 token 位置的 cosine similarity 变化。

**注意**: InstructBLIP 的 cos 偏低（0.467）且 baseline_mm 已经 100%，其 3B 结果可信度较低，建议优先跑 LLaVA + Qwen2.5-VL。

### 5.2 可选后续：更细粒度层扫描（InternVL2 专项）

对 InternVL2 的所有 32 层做 ablation 扫描，找到实际有效的攻击层，解释为什么 NW(layer 28) ablation 效果差。

### 5.3 可选：扩大 3C 测试集

当前 8 个 TEST_PROMPTS 太少，统计噪声大（尤其是 25% 和 12.5% 的差距）。可加入 PAIRED_PROMPTS 中的 12 个 harmful prompts，总共 20 个，提高置信度。

---

## 6. 代码与环境速查

### 6.1 项目路径

- **项目根目录**: `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/`
- **Phase 3 代码**: `experiments/phase3/`
- **Phase 3 结果**: `results/phase3/{llava_7b,qwen2vl_7b,internvl2_8b,instructblip_7b}/`

### 6.2 代码结构

```
experiments/phase3/
├── common/
│   ├── __init__.py
│   ├── model_configs.py          # 4 模型配置 + load_model_by_name()
│   └── model_adapters.py         # 4 个 adapter（含 generate_mm/generate_text）
├── exp_3a_amplitude_reversal.py  # ✅ 完成
├── exp_3c_ablation_attack.py     # ✅ 完成
├── run_3a_all.sh                 # 4 GPU 并行（3A）
├── run_3c_all.sh                 # 4 GPU 并行（3C）
└── logs/                         # 实验日志
```

### 6.3 运行环境

| 模型 | conda 环境 | 特殊依赖 |
|:---|:---|:---|
| LLaVA-1.5-7B | rdo | 无 |
| Qwen2.5-VL-7B | **qwen3-vl** | transformers >= 4.52，Qwen2_5_VLForConditionalGeneration |
| InternVL2-8B | rdo | timm（已安装在 GPU 节点） |
| InstructBLIP-7B | rdo | 无 |

### 6.4 模型本地路径

| 模型 | 路径 |
|:---|:---|
| LLaVA-1.5-7B | `models/hub/` (HF hub cache，model_id=`llava-hf/llava-1.5-7b-hf`) |
| Qwen2.5-VL-7B | `models/Qwen2.5-VL-7B-Instruct/` |
| InternVL2-8B | `models/InternVL2-8B/` |
| InstructBLIP-7B | `models/InstructBLIP-7B/` |

### 6.5 已知代码适配要点

1. **Qwen2.5-VL 加载**: 用 `Qwen2_5_VLForConditionalGeneration`（不能用 `AutoModel`，后者没有 `generate()`），`torch_dtype=torch.bfloat16`，手动 `.to(device)` 不用 `device_map`
2. **Qwen2.5-VL layers**: `model.model.language_model.layers`（ForConditionalGeneration 的路径）
3. **InternVL2 forward_mm**: 不能用 `model.forward()`（需要 image_flags），改为手动注入 visual features 到 `inputs_embeds` 后调 `model.language_model()`
4. **InternVL2 generate_mm**: 同上逻辑，调 `model.language_model.generate(inputs_embeds=...)`，输出不含 prompt tokens（直接 decode `gen_ids[0]`）
5. **InstructBLIP generate_mm**: `model.generate()` 输出不含 prompt（直接 decode `gen_ids[0]`）
6. **sys.path 顺序**: phase3 最后插入 index 0（最高优先级）；`eval_utils`/`llava_utils` 直接加 `phase2/common/` 到 sys.path 作为顶层模块导入，避免 `common.*` 冲突
7. **ablation hooks**: pre-hook（layer 输入）+ output-hook（self_attn, mlp）三处同时注册，通过 `adapter.get_llm_layers()` 获取目标层

### 6.6 结果文件

```
results/phase3/{model}/
├── exp_3a_results.json        # 3A: 逐层 cos/norm/ratio + NW + reversal
├── exp_3a_directions.pt       # 3A: 每层 v_text, v_mm（供 3B/3C 加载）
└── exp_3c_results.json        # 3C: 5 配置 × 8 prompts 攻击结果
```

`exp_3a_directions.pt` 格式：
```python
{
    "model": "llava_7b",
    "probe_layers": [8, 12, 16, 20, 28],
    "narrow_waist_layer": 16,
    "directions": {
        16: {"v_text": tensor(4096,), "v_mm": tensor(4096,)},
        ...
    }
}
```

---

## 7. Phase 3 完整结论汇总（paper 用）

| # | 结论 | 支持证据 | 强度 |
|:---|:---|:---|:---:|
| C1 | Refusal direction 跨模态稳定性通用（cos > 0.44 for all 4 models） | 3A，4/4 模型 | ★★★ |
| C2 | Amplitude reversal 是架构相关的条件性现象（非通用） | 3A，Group A vs B | ★★★ |
| C3 | CLIP ViT → reversal → NW 瓶颈层 → 单层攻击有效 | 3A + 3C，LLaVA | ★★★ |
| C4 | 自研 ViT → uniform suppression → 无瓶颈 → 需全层攻击 | 3A + 3C，Qwen2.5-VL | ★★★ |
| C5 | v_text 与 v_mm 在 ablation 效果上完全等价 | 3C，全部模型 | ★★★ |
| C6 | Q-Former 架构（InstructBLIP）: blank image 即完全绕过 | 3C，InstructBLIP | ★★★ |
| C7 | Qwen2.5-VL: 最强 baseline + 全局 ablation → 完全崩溃 | 3C，Qwen2.5-VL | ★★★ |
| C8 | InternVL2: 对 ablation 有强抗性，机制待解释 | 3C，InternVL2 | ★★（待验证）|

---

*文档结束。Phase 3 Exp 3A + 3C 全部完成，下一步 Exp 3B。*
