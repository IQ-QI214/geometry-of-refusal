# MIBD Routing Phase 2 GPU Handoff

这份说明面向 GPU 离线环境。当前仓库改动只搭建 Phase 2A/2B/2C 的 CPU 可测脚手架；真实 VLM forward、hidden-state extraction、causal intervention 和 judge 调用需要在 GPU 环境执行。

## 1. CPU scaffold：生成 pilot paired dataset

在仓库根目录运行：

```bash
python -m experiments.mibd_routing.run_phase2_scaffold \
  --output-dir results/mibd_routing/paired_dataset/pilot \
  --num-pairs 200 \
  --seed 20260604
```

输出：

```text
results/mibd_routing/paired_dataset/pilot/paired_dataset.jsonl
results/mibd_routing/paired_dataset/pilot/dataset_card.json
```

说明：当前 `paired_dataset.jsonl` 使用 placeholder image 文件，目的是固定 schema、paired_id、carrier_type、risk_label 和结果路径。正式实验需要把 image path 替换为 MM-SafetyBench / FigStep / 自建 paired image set 的真实图片路径。

## 1.1 真实 Phase 2A pilot dataset：本地 MM-SafetyBench

当前窗口已可从本地 MM-SafetyBench 构造真实 risk carrier paired dataset：

```bash
python -m experiments.mibd_routing.run_build_phase2a_dataset \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --output-dir results/mibd_routing/paired_dataset/phase2a_real_pilot \
  --num-pairs 200 \
  --seed 20260604
```

输出：

```text
results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl
results/mibd_routing/paired_dataset/phase2a_real_pilot/dataset_card.json
results/mibd_routing/paired_dataset/phase2a_real_pilot/build_report.md
```

当前构造规则：

- risk carriers: `figstep` -> `images_figstep`，`typographic` -> `images_wr`
- 每个 `paired_id` 输出 safe/risk 两条记录
- safe image 若未提供 `--safe-image-dir`，使用 generated blank placeholder 并在 metadata 记录 `safe_image_mode`
- risk image 使用 MM-SafetyBench 真实图片路径
- question 使用 benign-looking ambiguous prompt，避免文本本身直接 harmful

## 2. 已有 Phase 1 probe 复跑入口

如果 GPU 环境需要复跑已有 Phase 1 hidden-state probe：

```bash
python -m experiments.mibd.run_phase1 \
  --config experiments/mibd/configs/phase1_probe_internvl3.yaml

python -m experiments.mibd.run_phase1 \
  --config experiments/mibd/configs/phase1_probe_qwen3vl.yaml
```

预期结果目录：

```text
results/mibd/phase1_probe/internvl3_8b/
results/mibd/phase1_probe/qwen3_vl_8b/
```

## 3. Phase 2A GPU 任务

目标：对 paired dataset 的 safe/risk 样本生成模型输出，并保存 condition-specific behavior labels。

当前 CPU scaffold 可以先运行 smoke mode，验证数据和落盘格式：

```bash
python -m experiments.mibd_routing.run_phase2a_behavior \
  --dataset results/mibd_routing/paired_dataset/pilot/paired_dataset.jsonl \
  --output results/mibd_routing/behavior_labels/smoke/behavior_outputs.jsonl \
  --mode smoke
```

对真实 Phase 2A pilot dataset 的 smoke 验证命令：

```bash
python -m experiments.mibd_routing.run_phase2a_behavior \
  --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \
  --output results/mibd_routing/behavior_labels/smoke_phase2a_real/behavior_outputs.jsonl \
  --mode smoke
```

输出：

```text
results/mibd_routing/behavior_labels/smoke/behavior_outputs.jsonl
```

注意：`run_phase2a_behavior` 当前只实现 `--mode smoke`，不加载 VLM。真实 InternVL3-8B / Qwen3-VL-8B generation 需要后续在 GPU 环境接入模型特定 `generate()`，确认 chat template、image preprocessing、max_new_tokens 和 dtype 后再运行。

建议结果路径：

```text
results/mibd_routing/behavior_labels/internvl3_8b/
results/mibd_routing/behavior_labels/qwen3_vl_8b/
```

## 3.1 真实 VLM behavior generation（GPU runner）

`run_phase2a_vlm_behavior` 是真实 GPU runner，已实现 InternVL3-8B 和 Qwen3-VL-8B 的完整 generation 路径，支持 `--resume` 断点续跑。

两个模型分配不同 GPU（可以同时在两个终端并行启动）：

| 模型 | 环境 | GPU |
|---|---|---|
| InternVL3-8B | rdo | cuda:1 |
| Qwen3-VL-8B | qwen3-vl | cuda:0 |

### InternVL3-8B（rdo env，cuda:1）

```bash
mkdir -p results/mibd_routing/behavior_labels/internvl3_8b logs

conda run -n rdo python -m experiments.mibd_routing.run_phase2a_vlm_behavior \
  --model internvl3_8b \
  --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \
  --output results/mibd_routing/behavior_labels/internvl3_8b/behavior_outputs.jsonl \
  --device cuda:1 \
  --log-every 10 \
  2>&1 | tee logs/phase2a_internvl3_8b.log
```

### Qwen3-VL-8B（qwen3-vl env，cuda:0）

```bash
mkdir -p results/mibd_routing/behavior_labels/qwen3_vl_8b logs

conda run -n qwen3-vl python -m experiments.mibd_routing.run_phase2a_vlm_behavior \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \
  --output results/mibd_routing/behavior_labels/qwen3_vl_8b/behavior_outputs.jsonl \
  --device cuda:0 \
  --log-every 10 \
  2>&1 | tee logs/phase2a_qwen3_vl_8b.log
```

### 断点续跑

中途中断后加 `--resume` 继续，已写入的 sample_id 自动跳过：

```bash
# InternVL3-8B 续跑
conda run -n rdo python -m experiments.mibd_routing.run_phase2a_vlm_behavior \
  --model internvl3_8b \
  --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \
  --output results/mibd_routing/behavior_labels/internvl3_8b/behavior_outputs.jsonl \
  --device cuda:1 \
  --resume \
  2>&1 | tee -a logs/phase2a_internvl3_8b.log

# Qwen3-VL-8B 续跑
conda run -n qwen3-vl python -m experiments.mibd_routing.run_phase2a_vlm_behavior \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \
  --output results/mibd_routing/behavior_labels/qwen3_vl_8b/behavior_outputs.jsonl \
  --device cuda:0 \
  --resume \
  2>&1 | tee -a logs/phase2a_qwen3_vl_8b.log
```

### 输出格式

每条记录包含：

```json
{
  "sample_id": "phase2a-0001-risk",
  "paired_id": "phase2a-0001",
  "risk_label": "risk",
  "carrier_type": "figstep",
  "risk_category": "02",
  "visual_condition": "FigStep",
  "model_output": "...",
  "behavior_label": "unsafe_compliance",
  "judge_name": "internvl3_8b",
  "judge_raw": {"method": "rule_based_keyword_labeler"}
}
```

注意：当前行为标签使用规则 judge（`label_outputs.py`）。获得真实输出后，建议用 Qwen3Guard / LlamaGuard3 重新打标签（见 Phase 2B）。

safe image 仍是 blank placeholder；如有 matched benign/safe image 目录，传 `--safe-image-dir` 重新构造 dataset 后再跑。

每条记录至少包含：

```json
{
  "sample_id": "pair-0001-risk",
  "paired_id": "pair-0001",
  "risk_label": "risk",
  "carrier_type": "figstep",
  "risk_category": "weapons",
  "model_output": "...",
  "behavior_label": "unsafe_compliance",
  "judge_name": "qwen_guard_or_llamaguard",
  "judge_raw": {}
}
```

## 4. Phase 2B GPU 任务

目标：在 all layers / selected positions 上抽取 hidden states，训练 sensor probe，并做 gate search。

建议结果路径：

```text
results/mibd_routing/sensor_probe/internvl3_8b/
results/mibd_routing/sensor_probe/qwen3_vl_8b/
results/mibd_routing/gate_search/internvl3_8b/
results/mibd_routing/gate_search/qwen3_vl_8b/
```

需要保存：

- config
- model name / checkpoint
- dataset hash
- random seed
- selected loci
- held-out AUC
- group-split AUC by `paired_id`
- cross-category AUC
- static transfer AUC
- relocation score
- gate effect scores

## 5. Phase 2C GPU 任务

目标：在 `risk decodable but unsafe` 样本上比较 no intervention、fixed direction、OPD、ReGap-style、static bridge、oracle dynamic bridge。

建议结果路径：

```text
results/mibd_routing/oracle_bridge/internvl3_8b/
results/mibd_routing/oracle_bridge/qwen3_vl_8b/
results/mibd_routing/reports/
```

Go 条件：

- multi-locus sensor held-out AUC >= 0.85
- 至少 15% risk samples 满足 `risk decodable but unsafe`
- oracle dynamic bridge 相对 no intervention 或 best fixed baseline 提升 safe-policy >= 10pp
- benign over-refusal 增长 <= 5pp
- degeneration 不增加
