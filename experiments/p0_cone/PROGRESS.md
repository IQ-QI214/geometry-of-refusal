# P0 Cone Ablation — 执行进度日志

> **Plan**: `docs/plans/2026-04-13-1231-p0-cone-ablation-implementation.md`
> **Spec**: `docs/specs/2026-04-13-1231-p0-cone-ablation-stealth-refusal-design.md`
> **项目根目录**: `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/`
> **执行方式**: `superpowers:subagent-driven-development`，Task 顺序执行，每 Task 含 spec review + code quality review

---

## 快速恢复指引（新窗口打开时必读）

1. 读本文件了解进度
2. 读 plan 文件了解下一个 Task 的完整内容
3. 检查对应文件是否已存在：`ls refusal_direction/pipeline/model_utils/` 和 `ls experiments/p0_cone/`
4. 从上次未完成的 Task 继续，使用 `superpowers:subagent-driven-development` skill

**当前状态（2026-04-13）**: Tasks 1-3 ✅ 完成，Task 4 进行中（脚本已写，等待 GPU 运行）

---

## 任务状态总览

| Task | 描述 | 状态 | 完成日期 |
|------|------|------|---------|
| 1 | LLaVA VLM Model Adapter | ✅ 完成 | 2026-04-13 |
| 2 | Qwen VLM Model Adapter | ✅ 完成 | 2026-04-13 |
| 3 | Update Model Factory | ✅ 完成 | 2026-04-13 |
| 4 | Smoke Test VLM Adapters (CP1) | 🔄 脚本已写，等待 GPU | — |
| 5 | DIM Direction Extraction + PCA Cone | ⏳ 待开始 | — |
| 6 | DIM Cone Ablation + Generation | ⏳ 待开始 | — |
| 7 | RDO Training for VLM | ⏳ 待开始 | — |
| 8 | RDO Cone Ablation + Generation | ⏳ 待开始 | — |
| 9 | Four-Layer ASR Evaluation Pipeline | ⏳ 待开始 | — |
| 10 | Shell Scripts for Execution | ⏳ 待开始 | — |
| 11 | Analysis Report | ⏳ 待开始 | — |

---

## Task 详细记录

---

### Task 1: LLaVA VLM Model Adapter ✅

**文件**: `refusal_direction/pipeline/model_utils/llava_vlm_model.py`
**Commits**: `dd85eb4` (初始实现), `6514a2c` (修复 3 个 critical bugs)

**完成内容**:
- 创建 `LlavaVLMModel(ModelBase)` 类，支持 `LlavaForConditionalGeneration`
- `tokenize_instructions_llava_vlm` 使用 336×336 空白 PIL 图像
- 覆盖 `generate_completions` 以传递 `pixel_values` 和 `image_sizes`
- 验证 LLaVA refusal token: `[306]` ("I" in LLaMA-2 tokenizer) ✅

**⚠️ 关键坑点（后续 Adapter 必看）**:

1. **`_get_eoi_toks` 不要动态解析模板**
   - ❌ 错误做法：`apply_chat_template(...).split("X")[-1]` —— 脆弱且在 VLM 模板中不可靠
   - ✅ 正确做法：硬编码后缀字符串，如 `"\nASSISTANT:"` 再 `.encode()`
   - 参考: `_LLAVA_EOI_SUFFIX = "\nASSISTANT:"`

2. **VLM 复合 config 需要 patch**（LLaVA 特有）
   - `LlavaForConditionalGeneration.config` 是 `LlavaConfig`，`num_hidden_layers`/`hidden_size` 在 `model.config.text_config` 而不是 `model.config`
   - ✅ 修复：在 `_load_model` 中 `return` 前加：
     ```python
     model.config.num_hidden_layers = model.config.text_config.num_hidden_layers
     model.config.hidden_size = model.config.text_config.hidden_size
     ```
   - **Qwen2.5-VL 是否也需要这个 patch？** 需要验证（Qwen2VLConfig 可能直接在顶层有这些属性）

3. **pipeline 的 `generate_directions.py` 和 `select_direction.py` 已更新**
   - 原代码调用 `model(input_ids=..., attention_mask=...)` — VLM 需要 `pixel_values`
   - 已修复为 `forward_kwargs` dict 模式，自动传递 `pixel_values`/`image_grid_thw`/`image_sizes`
   - ✅ 这两个文件已 patch，**后续 Task 不需要再改**

4. **Pylance import 警告是虚假警告**
   - `torch`, `transformers`, `jaxtyping`, `PIL` 都是 conda env 里的包，Pylance 找不到是因为 IDE 没有激活 conda env，**不是真正的错误**

---

### Task 2: Qwen VLM Model Adapter 🔄

**文件**: `refusal_direction/pipeline/model_utils/qwen_vlm_model.py`
**状态**: 进行中

**预计坑点**（从 Task 1 学到的教训）:
- `_get_eoi_toks`：plan 已硬编码 `"<|im_end|>\n<|im_start|>assistant\n"`，直接使用即可 ✅
- `model.config`：`Qwen2_5_VLForConditionalGeneration` 使用 `Qwen2VLConfig`，需在实现后验证 `model.config.num_hidden_layers` 是否可直接访问（若不可则需同样 patch）
- `_load_model` 使用 `.to("cuda:0")` 而不是 `device_map="auto"`（避免 accelerate 问题，spec §3.2）
- 不需要 `cache_dir`（Qwen 使用绝对路径加载）
- `tokenize_instructions_qwen_vlm` 中 outputs 处理：`idx = len(prompts)` 在 append 前计算，索引正确

---

### Task 3: Update Model Factory ⏳

**文件**: `refusal_direction/pipeline/model_utils/model_factory.py`
**预计坑点**:
- VLM 路径检测要在 text-only Qwen 检测之前（`Qwen2.5-VL` 包含 `qwen`，会被 text-only 路径截获）
- 按 plan 中的代码完整替换文件内容

---

### Task 4: Smoke Test VLM Adapters (CP1) ⏳

**文件**: `experiments/p0_cone/smoke_test_adapters.py`
**运行命令**:
```bash
# LLaVA (rdo env, cuda:0)
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run --no-capture-output -n rdo \
    python experiments/p0_cone/smoke_test_adapters.py --model llava_7b --device cuda:0

# Qwen (qwen3-vl env, cuda:1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run --no-capture-output -n qwen3-vl \
    python experiments/p0_cone/smoke_test_adapters.py --model qwen2vl_7b --device cuda:1
```
**CP1 通过标准**: 5/5 检查通过，generation 产生文本输出
**预计坑点**:
- LLaVA 的 `get_mean_diff` 调用需要 `pipeline/submodules/generate_directions.py` 已包含 `forward_kwargs` 修复（已完成）
- Qwen 若 `model.config.num_hidden_layers` 报 AttributeError，需在 `qwen_vlm_model.py` 中加 patch

---

### Task 5: DIM Direction Extraction + PCA Cone ⏳

**文件**: `experiments/p0_cone/exp_p0_dim_extract.py`
**输出**: `results/p0/{model}/dim_cone_k{1,3,5}.pt`, `dim_metadata.json`, `dim_singular_values.pt`
**运行环境**: llava_7b → rdo env; qwen2vl_7b → qwen3-vl env

---

### Task 6: DIM Cone Ablation + Generation ⏳

**文件**: `experiments/p0_cone/exp_p0_dim_ablate.py`
**输出**: `results/p0/{model}/dim_k{1,3,5}_responses.json` (各含 128 条)
**CP2**: `dim_k1` 的 ASR_keyword 需与 Category A 中该模型的结果一致（±5pp）

---

### Task 7: RDO Training for VLM ⏳

**文件**: `exp_p0_rdo_targets.py` + `exp_p0_rdo_train.py`
**注意**:
- 需先运行 `exp_p0_rdo_targets.py` 再运行 `exp_p0_rdo_train.py`
- RDO 训练使用 PyTorch hooks（非 nnsight），因为 nnsight 不兼容 VLM 的 multi-modal 输入
- CP3: RDO k=1 的 ASR_keyword ≥ DIM k=1

---

### Task 8: RDO Cone Ablation + Generation ⏳

**文件**: `exp_p0_rdo_ablate.py`
**输出**: `results/p0/{model}/rdo_k{1,3,5}_responses.json`

---

### Task 9: Four-Layer ASR Evaluation Pipeline ⏳

**文件**: `common/eval_pipeline.py` + `exp_p0_evaluate.py`
**注意**:
- judge_utils.py 从 `experiments/category_a/common/` 复用，需确认路径正确
- StrongREJECT 包需确认在 conda env 中已安装
- CP4: 所有 12 组实验的四层 ASR 数据完整

---

### Task 10: Shell Scripts ⏳

**文件**: `run_p0_phase2.sh`, `run_p0_phase3.sh`, `run_p0_evaluate.sh`

---

### Task 11: Analysis Report ⏳

**文件**: `analysis/p0/p0_cone_analysis_YYYY-MM-DD-HHmm.md`
**核心判定**: k=1→3→5 ASR_judge 趋势 → 假说 A（递增）vs B（平坦）

---

## 关键路径上已修改的共享文件

以下文件在 Task 1 中已修改，**后续 Task 无需再改**：

| 文件 | 修改内容 |
|------|---------|
| `refusal_direction/pipeline/submodules/generate_directions.py` | `get_mean_activations` 改用 `forward_kwargs` 传递 VLM 额外输入 |
| `refusal_direction/pipeline/submodules/select_direction.py` | `get_refusal_scores` 和 `get_last_position_logits` 同样修复 |

---

*最后更新: 2026-04-13，Task 1 完成后*
