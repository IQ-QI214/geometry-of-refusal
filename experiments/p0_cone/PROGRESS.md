# P0 Cone Ablation — 执行进度日志

> **Plan**: `docs/plans/2026-04-13-1231-p0-cone-ablation-implementation.md`
> **Spec**: `docs/specs/2026-04-13-1231-p0-cone-ablation-stealth-refusal-design.md`
> **项目根目录**: `[PROJECT_ROOT]/geometry-of-refusal/`

---

## 环境约束（新 Agent 必读）

1. **当前 Claude 运行在 CPU 联网实例，没有 GPU。**
   - 所有需要 GPU 的命令不能在 bash 工具里执行
   - 写好脚本后，以代码块形式把完整命令交给 qi 在 GPU 节点手动运行

2. **文件系统权限：只能改 `zhujiaqi` 目录**
   - ✅ 可以读写：`[PROJECT_ROOT]/`
   - ❌ 其他目录只读，任何写/创建/删除操作必须先询问 qi

3. **离线包安装**:
   ```bash
   # CPU 节点（有网）下载
   conda run -n <env> pip download <pkg> --dest /inspire/hdd/global_user/wenming-253108090054/pip_wheels --no-cache-dir
   # GPU 节点（无网）安装
   conda run -n <env> pip install --no-index --find-links /inspire/hdd/global_user/wenming-253108090054/pip_wheels <pkg>
   ```

4. **称呼用户为 qi**

---

## 任务状态总览

| Task | 描述 | 状态 |
|------|------|------|
| 1 | LLaVA VLM Model Adapter | ✅ 完成 |
| 2 | Qwen VLM Model Adapter | ✅ 完成 |
| 3 | Update Model Factory | ✅ 完成 |
| 4 | Smoke Test VLM Adapters | ✅ 完成 |
| 5 | DIM Direction Extraction + PCA Cone | ✅ 完成（两模型） |
| 6 | DIM Cone Ablation + Generation | ✅ 完成（两模型，共 6 组） |
| 7 | RDO Training for VLM | ✅ 完成（两模型，共 6 组 cone） |
| 8 | RDO Cone Ablation + Generation | ⚠️ 部分完成（见下表） |
| 9 | Four-Layer ASR Evaluation Pipeline | ✅ 脚本完成（等 LLaVA RDO 响应生成后可运行） |
| 10 | Shell Scripts | ⏳ 待开始 |
| 11 | Analysis Report | ⏳ 等实验结果 |

---

## 实验目的与设计框架

### 研究问题
VLM（LLaVA、Qwen2.5-VL）的拒绝机制是否存在 **stealth refusal**：即激活方向消融后，关键词检测显示"绕过"，但语义上模型仍然拒绝回答有害内容？

### 假说
- **假说 A**：DIM cone 不完整，增大 k 能持续提升 ASR_judge（语义绕过率），说明拒绝机制集中在可捕获的子空间
- **假说 B**：存在独立的 stealth refusal 层，即使 k=5 完全消融，ASR_judge 仍低于 ASR_keyword（语义上仍在拒绝）

### 方法框架
两种方向提取方法 × 三种 cone 维度 × 两个模型：

| 方法 | 描述 |
|------|------|
| **DIM** | PCA 对 harmful/harmless 激活均值差做 SVD，取前 k 个主成分 |
| **RDO** | Wollschläger et al. ICML 2025，三损失优化：L_ablation + L_addition + L_retain |

四层 ASR 评估（Task 9 待跑）：

| 层级 | 方法 | 含义 |
|------|------|------|
| Layer 1 | keyword | 响应不以拒绝词开头 |
| Layer 2 | StrongREJECT score | 响应对有害请求实际有帮助 |
| Layer 3 | Qwen3Guard-8B | 模型级安全检测 |
| Layer 4 | LLaMA-Guard-3-8B | 模型级安全检测 |

**SRR（Stealth Refusal Rate）** = ASR_keyword − ASR_judge（>0 说明存在 stealth refusal）

---

## DIM 方向选择结果

| 模型 | 选中层 | pos | steering_score | 备注 |
|------|--------|-----|----------------|------|
| LLaVA-7B | layer 22 | -1 | — | 32 层中约 70% 深度，语言推理主导区 |
| Qwen2.5-VL-7B | layer 16 | -5 | 1.484 | 28 层中约 57% 深度；层 15-18 均有正 steering_score |

---

## Task 6 & 8：Keyword ASR 汇总（已有结果）

> **说明**：bypass_rate = 未以拒绝词开头的响应比例（粗糙 Layer 1 指标，不代表语义绕过）

### LLaVA-7B（DIM，Task 6 完成；RDO ablation 运行中）

| 方法 | k | n | refusals | bypass_rate(kw) |
|------|---|---|----------|-----------------|
| DIM | 1 | 128 | 59 | 0.539 |
| DIM | 3 | 128 | 67 | 0.477 |
| DIM | 5 | 128 | 45 | 0.648 |
| RDO | 1 | — | — | ⏳ running |
| RDO | 3 | — | — | ⏳ running |
| RDO | 5 | — | — | ⏳ running |

**LLaVA DIM 分析**：
- 非单调（k=1 > k=3 < k=5）：PCA 成分 2-3 破坏 VLM 跨模态对齐，产生退化输出（重复 token），keyword 检测误判为拒绝
- k=1 的 69 个 bypass 中，约 40 个是 token collapse（degenerate），21 个有实质语义内容
- 高 keyword bypass 中大量为退化噪声，SRR 预期较小

### Qwen2.5-VL-7B（DIM Task 6 完成；RDO 部分完成）

| 方法 | k | n | refusals | bypass_rate(kw) |
|------|---|---|----------|-----------------|
| DIM | 1 | 128 | 50 | 0.609 |
| DIM | 3 | 128 | 0 | **1.000** |
| DIM | 5 | 128 | 0 | **1.000** |
| RDO | 1 | 128 | 58 | 0.547 |
| RDO | 3 | 128 | 46 | 0.641 |
| RDO | 5 | — | — | ⏳ running |

**Qwen DIM 分析**：
- k=3、k=5 达到 100% keyword bypass，且**响应内容真实（非退化）**，模型确实在讨论有害话题
- 但响应风格是"reframing/deflection"（如"这个短语表示..."），可能是语义 stealth refusal，需要 judge 层判定
- DIM k=3 vs RDO k=3（64%）：RDO bypass 率更低，说明 RDO 优化后的 cone 方向对 Qwen 安全机制的消融效果反而不如 PCA 成分的纯粹方向
- 这个反转本身就是有意义的发现，需要 Task 9 judge 层来解释

---

## 已知问题 & 坑点汇总

| # | 问题 | 解决方案 |
|---|------|---------|
| 1 | LLaVA `model.config.num_hidden_layers` 顶层不存在 | `_load_model` 中 patch：`model.config.num_hidden_layers = model.config.text_config.num_hidden_layers` |
| 2 | Qwen VLM backbone 层路径 | `model.model.language_model.layers`（非 `model.model.layers`） |
| 3 | `generate_directions.py` 原来不传 `pixel_values` | 已 patch 为 `forward_kwargs` 模式，不需要再改 |
| 4 | `select_direction.py` 同样需要传 `pixel_values` | 已 patch，不需要再改 |
| 5 | `_get_eoi_toks` 不要动态解析模板 | 硬编码：LLaVA `"\nASSISTANT:"`，Qwen `"<\|im_end\|>\n<\|im_start\|>assistant\n"` |
| 6 | `qwen3-vl` env 缺少 `jaxtyping` | 已通过 pip_wheels 安装 |
| 12 | `select_direction` steering 过滤器对 VLM 全失效 | `induce_refusal_threshold=None` + 缓存快速路径（Commit `2419f17`）|
| 13 | LLaVA `device_map="auto"` 扩散到多 GPU | 改为 `device_map={"": "cuda:0"}`；选卡用 `CUDA_VISIBLE_DEVICES=N`（Commit `3b8bf92`）|
| 14 | RDO hooks 不能用 in-place op | `x -= proj` → `x_new = x - proj`（版本号冲突） |
| 15 | RDO direction 不能 `.detach()` | `direction` 必须保留 grad_fn；`x.detach() @ d_cast` 是安全的 |

---

## 结果文件目录

```
results/p0_cone/
├── llava_7b/
│   ├── dim/
│   │   ├── dim_cone_k{1,3,5}.pt        ✅
│   │   ├── dim_metadata.json            ✅  {"layer":22,"pos":-1}
│   │   ├── dim_k{1,3,5}_responses.json  ✅  各 128 条
│   │   └── dim_singular_values.pt       ✅
│   └── rdo/
│       ├── targets/{harmful,harmless}_targets.json  ✅
│       ├── rdo_cone_k{1,3,5}.pt         ✅
│       ├── rdo_cone_k{1,3,5}_log.json   ✅
│       └── rdo_k{1,3,5}_responses.json  ⏳ running
├── qwen2vl_7b/
│   ├── dim/
│   │   ├── dim_cone_k{1,3,5}.pt        ✅
│   │   ├── dim_metadata.json            ✅  {"layer":16,"pos":-5}
│   │   ├── dim_k{1,3,5}_responses.json  ✅  各 128 条
│   │   └── dim_singular_values.pt       ✅
│   └── rdo/
│       ├── targets/{harmful,harmless}_targets.json  ✅
│       ├── rdo_cone_k{1,3,5}.pt         ✅
│       ├── rdo_cone_k{1,3,5}_log.json   ✅
│       ├── rdo_k1_responses.json         ✅  128 条
│       ├── rdo_k3_responses.json         ✅  128 条
│       └── rdo_k5_responses.json         ⏳ running
```

---

## 关键路径上已修改的共享文件（不需要再改）

| 文件 | 修改内容 | Commit |
|------|---------|--------|
| `refusal_direction/pipeline/submodules/generate_directions.py` | `get_mean_activations` 改用 `forward_kwargs` | `6514a2c` |
| `refusal_direction/pipeline/submodules/select_direction.py` | `get_refusal_scores` + `get_last_position_logits` 同样修复 | `6514a2c` |
| `refusal_direction/pipeline/model_utils/llava_vlm_model.py` | device_map 固定单卡 | `3b8bf92` |

---

*最后更新: 2026-04-14，Task 6+7+8（部分）完成后*
