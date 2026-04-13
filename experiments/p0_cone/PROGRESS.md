# P0 Cone Ablation — 执行进度日志

> **Plan**: `docs/plans/2026-04-13-1231-p0-cone-ablation-implementation.md`
> **Spec**: `docs/specs/2026-04-13-1231-p0-cone-ablation-stealth-refusal-design.md`
> **项目根目录**: `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/`
> **执行方式**: `superpowers:subagent-driven-development`，Task 顺序执行

---

## ⚡ 新窗口 Agent 必读（关键约束）

### 环境约束

1. **当前 Claude 运行在 CPU 联网实例，没有 GPU。**
   - 所有需要 GPU 的命令（模型加载、推理、训练）不能在 bash 工具里执行
   - 写好脚本后，以代码块形式把完整命令交给用户（qi）在 GPU 节点手动运行
   - 等待 qi 贴回运行结果，再根据结果继续

2. **文件系统权限：只能改 `zhujiaqi` 目录**
   - ✅ 可以读写：`/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/`
   - ❌ 其他目录（包括 pip_wheels、models 等）：只读，任何写/创建/删除操作都必须先询问 qi
   - pip wheel 共享目录（只读）：`/inspire/hdd/global_user/wenming-253108090054/pip_wheels`

3. **离线包安装流程（CPU→GPU）**:
   ```bash
   # 在 CPU 节点（有网）下载 wheel 到共享目录（需先问 qi 是否可以）
   conda run -n <env> pip download <package> \
       --dest /inspire/hdd/global_user/wenming-253108090054/pip_wheels \
       --no-cache-dir

   # 在 GPU 节点（无网）离线安装
   conda run -n <env> pip install \
       --no-index \
       --find-links /inspire/hdd/global_user/wenming-253108090054/pip_wheels \
       <package>
   ```

4. **称呼用户为 qi**

### 恢复工作流程

```bash
# 1. 在项目根目录确认文件状态
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
ls refusal_direction/pipeline/model_utils/     # 确认 adapter 文件存在
ls experiments/p0_cone/                        # 确认 p0 实验文件

# 2. 查看 git log 了解已提交的内容
git log --oneline -10

# 3. 读本文件 + plan 文件，从下一个未完成 Task 开始
```

### 已知问题 & 坑点汇总（必看，避免重复踩坑）

| # | 问题 | 解决方案 |
|---|------|---------|
| 1 | LLaVA 的 `model.config.num_hidden_layers` 在顶层不存在 | 在 `_load_model` 中 patch：`model.config.num_hidden_layers = model.config.text_config.num_hidden_layers`（已做） |
| 2 | Qwen VLM 架构：backbone 层路径 **不是** `model.model.layers` | 正确路径：`model.model.language_model.layers`（已验证，28 层）|
| 3 | `generate_directions.py` 原来不传 `pixel_values` | 已 patch 为 `forward_kwargs` 模式，**不需要再改** |
| 4 | `select_direction.py` 同样需要传 `pixel_values` | 已 patch，**不需要再改** |
| 5 | `_get_eoi_toks` 不要动态解析模板 | 必须硬编码后缀字符串（LLaVA: `"\nASSISTANT:"`，Qwen: `"<\|im_end\|>\n<\|im_start\|>assistant\n"`） |
| 6 | `qwen3-vl` env 缺少 `jaxtyping` | 已通过 pip_wheels 安装解决 |
| 7 | Pylance import 警告（torch/transformers/jaxtyping/PIL）| 假警告，conda env 里有包，IDE 没激活，**无需处理** |
| 8 | Qwen `from_pretrained` 的 `torch_dtype=` | 应改为 `dtype=`（已修复） |
| 9 | `temperature=0` + `do_sample=False` 的警告 | 无害，后续若清理可只在 `do_sample=True` 时设置 temperature |
| 10 | LLaVA 模型路径用 snapshot 绝对路径更可靠 | `/inspire/hdd/.../models/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| 11 | smoke test 里 Qwen 即使消融也拒绝 | 正常现象：smoke test 用的是随机方向，不是最优方向；是 stealth refusal 的早期信号 |

---

## 任务状态总览

| Task | 描述 | 状态 | 完成日期 |
|------|------|------|---------|
| 1 | LLaVA VLM Model Adapter | ✅ 完成 | 2026-04-13 |
| 2 | Qwen VLM Model Adapter | ✅ 完成 | 2026-04-13 |
| 3 | Update Model Factory | ✅ 完成 | 2026-04-13 |
| 4 | Smoke Test VLM Adapters (CP1) | ✅ 完成 | 2026-04-13 |
| 5 | DIM Direction Extraction + PCA Cone | ⏳ 待开始（脚本未写） | — |
| 6 | DIM Cone Ablation + Generation | ⏳ 待开始 | — |
| 7 | RDO Training for VLM | ⏳ 待开始 | — |
| 8 | RDO Cone Ablation + Generation | ⏳ 待开始 | — |
| 9 | Four-Layer ASR Evaluation Pipeline | ⏳ 待开始 | — |
| 10 | Shell Scripts for Execution | ⏳ 待开始 | — |
| 11 | Analysis Report | ⏳ 待开始（等实验结果） | — |

**执行依赖**：Task 5 和 Task 7 可在 Task 4 后并行写脚本，但 Task 7 需要 Task 5 的 DIM 方向作为初始化。实际执行顺序：5→6→7→8→9→10→11。

---

## 已完成 Task 详细记录

---

### Task 1: LLaVA VLM Model Adapter ✅

**文件**: `refusal_direction/pipeline/model_utils/llava_vlm_model.py`
**Commits**: `dd85eb4` (初始实现), `6514a2c` (修复 3 个 critical bugs)

**关键设计**:
- `LlavaVLMModel(ModelBase)`，`LlavaForConditionalGeneration`
- Blank image: 336×336 白色 PIL
- EOI suffix 硬编码：`_LLAVA_EOI_SUFFIX = "\nASSISTANT:"`
- Refusal tokens: `[306]`（"I" in LLaMA-2 tokenizer，已验证）
- Config patch（LLaVA 特有）：`model.config.num_hidden_layers = model.config.text_config.num_hidden_layers`
- `generate_completions` 覆盖：传递 `pixel_values` + `image_sizes`

**同步修改的共享文件**（已完成，不需要再动）:
- `refusal_direction/pipeline/submodules/generate_directions.py` → `get_mean_activations` 改用 `forward_kwargs`
- `refusal_direction/pipeline/submodules/select_direction.py` → `get_refusal_scores` + `get_last_position_logits` 同样修复

---

### Task 2: Qwen VLM Model Adapter ✅

**文件**: `refusal_direction/pipeline/model_utils/qwen_vlm_model.py`
**Commit**: `b61eb35`

**关键设计**:
- `QwenVLMModel(ModelBase)`，`Qwen2_5_VLForConditionalGeneration`
- Blank image: 336×336 白色 PIL
- EOI suffix 硬编码：`_QWEN_VLM_EOI_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"`
- Refusal tokens: `[40, 2121]`（["I", "As"]，与 text-only Qwen 共享 tokenizer）
- 模型加载：`dtype=dtype`（非 `torch_dtype`）+ `.to("cuda:0")`（非 `device_map="auto"`）
- Backbone 路径（经过验证）：`model.model.language_model.layers`（28 层）
- `generate_completions` 覆盖：传递 `pixel_values` + `image_grid_thw`
- **不需要** config patch（`Qwen2VLConfig` 直接有 `num_hidden_layers=28`, `hidden_size=3584`）

**Plan 中的错误（已修正）**:
- Plan 写 `model.model.layers`，实际应为 `model.model.language_model.layers`
- `_get_orthogonalization_mod_fn` 和 `_get_act_add_mod_fn` 均已使用正确路径

---

### Task 3: Update Model Factory ✅

**文件**: `refusal_direction/pipeline/model_utils/model_factory.py`
**Commit**: `2dfe47f`

VLM 检测（`llava-hf`, `qwen2.5-vl`）在 text-only Qwen 检测之前，防止路径截获。

---

### Task 4: Smoke Test VLM Adapters (CP1) ✅

**文件**: `experiments/p0_cone/smoke_test_adapters.py`
**Commit**: `e86e9d6`

**CP1 结果（已通过）**:

| 检查项 | LLaVA-7B | Qwen2.5-VL-7B |
|--------|---------|---------------|
| 层数 | 32 | 28 |
| hidden_size | 4096 | 3584 |
| EOI tokens 数量 | 6 个 | 5 个 |
| pixel_values | ✅ `[2,3,336,336]` | ✅ `[1152,1176]` |
| image_grid_thw | N/A | `[[1,24,24],[1,24,24]]` |
| mean_diff shape | `[6,32,4096]` | `[5,28,3584]` |
| mean_diff NaN | False ✓ | False ✓ |
| 消融后 generation | 生成有害内容 ✓ | **仍然拒绝**（预期！） |

**重要发现**: Qwen 用随机方向消融后仍输出 "I'm sorry, but I cannot..."。
这是 stealth refusal 的早期信号，说明 Qwen 的安全机制对随机方向免疫，需要最优方向（Task 5 目标）才能有效消融。

---

## 待完成 Task 指引

---

### Task 5: DIM Direction Extraction + PCA Cone ⏳

**脚本**: `experiments/p0_cone/exp_p0_dim_extract.py`（待写）
**完整代码**: 见 plan `## Task 5` 节

**输出文件**（运行后应存在）:
```
results/p0/llava_7b/dim_cone_k1.pt      # shape: (1, 4096)
results/p0/llava_7b/dim_cone_k3.pt      # shape: (3, 4096)
results/p0/llava_7b/dim_cone_k5.pt      # shape: (5, 4096)
results/p0/llava_7b/dim_metadata.json   # {"pos": int, "layer": int}
results/p0/llava_7b/dim_singular_values.pt
results/p0/qwen2vl_7b/...               # 同上，d_model=3584
```

**运行命令（写完脚本后交给 qi 运行）**:
```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

# LLaVA (rdo env, cuda:0)，约 10-20 分钟
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n rdo \
    python experiments/p0_cone/exp_p0_dim_extract.py --model llava_7b --device cuda:0 \
    > results/p0/llava_7b/dim_extract.log 2>&1 &
echo "LLaVA DIM PID: $!"

# Qwen (qwen3-vl env, cuda:1)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup conda run --no-capture-output -n qwen3-vl \
    python experiments/p0_cone/exp_p0_dim_extract.py --model qwen2vl_7b --device cuda:1 \
    > results/p0/qwen2vl_7b/dim_extract.log 2>&1 &
echo "Qwen DIM PID: $!"
```

**CP2 验证**:
```bash
# 检查输出文件是否存在
ls results/p0/llava_7b/dim_cone_k*.pt results/p0/llava_7b/dim_metadata.json
ls results/p0/qwen2vl_7b/dim_cone_k*.pt results/p0/qwen2vl_7b/dim_metadata.json

# 检查 shape
python -c "
import torch
for m in ['llava_7b', 'qwen2vl_7b']:
    for k in [1,3,5]:
        t = torch.load(f'results/p0/{m}/dim_cone_k{k}.pt')
        print(f'{m} k={k}: {t.shape}')
"
```

---

### Task 6: DIM Cone Ablation + Generation ⏳

**脚本**: `experiments/p0_cone/exp_p0_dim_ablate.py`（待写）
**完整代码**: 见 plan `## Task 6` 节
**依赖**: Task 5 输出的 `dim_cone_k*.pt` 文件

**输出**: `results/p0/{model}/dim_k{1,3,5}_responses.json`（各含 128 条）

---

### Task 7: RDO Training for VLM ⏳

**脚本**: `exp_p0_rdo_targets.py` + `exp_p0_rdo_train.py`（待写）
**完整代码**: 见 plan `## Task 7` 节
**依赖**: Task 5 输出的 `dim_cone_k1.pt`（用作 RDO 初始化）

**注意**:
- 必须先运行 `exp_p0_rdo_targets.py` 生成 targets，再运行训练
- RDO 使用 PyTorch hooks（非 nnsight），因为 nnsight 不兼容 VLM multi-modal 输入
- 三项损失：`L_ablation + L_addition + L_retain`（忠实复现 Wollschläger）
- CP3: RDO k=1 的 ASR_keyword ≥ DIM k=1

---

### Task 8: RDO Cone Ablation + Generation ⏳

**脚本**: `exp_p0_rdo_ablate.py`（待写）
**依赖**: Task 7 输出的 `rdo_cone_k*.pt`

---

### Task 9: Four-Layer ASR Evaluation Pipeline ⏳

**脚本**: `common/eval_pipeline.py` + `exp_p0_evaluate.py`（待写）
**注意**:
- 四层评估：keyword / StrongREJECT / Qwen3Guard-8B / LLaMA-Guard-3-8B
- judge_utils 复用自 `experiments/category_a/common/judge_utils.py`
- CP4: 所有 12 组实验评估完整
- 运行前确认 `strong_reject` 包已安装（否则 ASR_sr = -1）

---

### Task 10: Shell Scripts ⏳

**脚本**: `run_p0_phase2.sh`, `run_p0_phase3.sh`, `run_p0_evaluate.sh`
参考 `experiments/category_a/` 下的 run 脚本格式

---

### Task 11: Analysis Report ⏳

**文件**: `analysis/p0/p0_cone_analysis_YYYY-MM-DD-HHmm.md`
**核心判定**: k=1→3→5 ASR_judge 趋势 → 假说 A（递增）vs 假说 B（平坦）

---

## 关键路径上已修改的共享文件

**后续 Task 不需要再改这些文件：**

| 文件 | 修改内容 | Commit |
|------|---------|--------|
| `refusal_direction/pipeline/submodules/generate_directions.py` | `get_mean_activations` 改用 `forward_kwargs` 传递 VLM 额外输入（pixel_values 等） | `6514a2c` |
| `refusal_direction/pipeline/submodules/select_direction.py` | `get_refusal_scores` + `get_last_position_logits` 同样修复 | `6514a2c` |

---

*最后更新: 2026-04-13，Task 4 完成后*
