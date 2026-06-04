# MIBD Phase 2A Real Paired Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从本地 MM-SafetyBench 与 SaladBench splits 构造真实 Phase 2A paired routing diagnostic dataset，替换当前 placeholder pilot dataset，使 GPU 环境可以直接生成 InternVL3/Qwen3-VL 行为输出。

**Architecture:** 在 `experiments/mibd_routing/data/` 下新增 converter，读取 MM-SafetyBench `data.json` 与 image dirs，生成 `PairedRoutingSample` JSONL。每个 `paired_id` 固定一个 ambiguous benign-looking question，并输出 safe/risk 两条记录；risk 图像来自 FigStep / typographic 等 MM-SafetyBench carrier，safe 图像优先使用 matched safe image pool，缺失时使用 blank control 并在 dataset card 中显式记录。

**Tech Stack:** Python stdlib、PIL 可选、现有 `PairedRoutingSample` schema、pytest。

---

## 定位

这是原执行文档中的 **Phase 2A：Paired Routing Diagnostic Set**。它不跑模型、不训练 probe、不做 bridge。产物是 GPU 环境下一步要读的真实 `paired_dataset.jsonl`。

## 文件结构

- Create: `experiments/mibd_routing/data/convert_phase2a.py`  
  读取本地 MM-SafetyBench / SaladBench，构造真实 paired records 与 dataset card。
- Create: `experiments/mibd_routing/run_build_phase2a_dataset.py`  
  CLI 入口，输出 `paired_dataset.jsonl`、`dataset_card.json`、`build_report.md`。
- Modify: `experiments/mibd_routing/data/schema.py`  
  如有必要扩展 `DatasetCard` metadata，但不破坏现有字段。
- Modify: `experiments/mibd/test_routing_scaffold.py`  
  增加 converter 的 CPU 单元测试。
- Modify: `experiments/mibd_routing/HANDOFF.md`  
  增加真实 Phase 2A dataset 构造命令和 GPU 下一步路径。

## Task 1: MM-SafetyBench item collector

**Files:**
- Create: `experiments/mibd_routing/data/convert_phase2a.py`
- Test: `experiments/mibd/test_routing_scaffold.py`

- [ ] **Step 1: Write failing test**

```python
def test_collect_mmsafety_items_reads_figstep_and_wr(tmp_path):
    cat = tmp_path / "02"
    (cat / "images_figstep").mkdir(parents=True)
    (cat / "images_wr").mkdir()
    (cat / "images_figstep" / "1.png").write_bytes(b"img")
    (cat / "images_wr" / "1.png").write_bytes(b"img")
    (cat / "data.json").write_text(json.dumps([{
        "id": 1,
        "original_prompt": "harmful original",
        "qr_prompt": "image says the harmful task",
        "replaced_prompt": "coded harmful task"
    }]))

    items = collect_mmsafety_items(tmp_path, carriers=["figstep", "typographic"])

    assert len(items) == 2
    assert {item.carrier_type for item in items} == {CarrierType.FIGSTEP, CarrierType.TYPOGRAPHIC}
```

- [ ] **Step 2: Verify red**

Run:

```bash
.venv_gemma_probe/bin/python -m pytest experiments/mibd/test_routing_scaffold.py::test_collect_mmsafety_items_reads_figstep_and_wr -q
```

Expected: FAIL with missing import.

- [ ] **Step 3: Implement collector**

Implement:

```python
collect_mmsafety_items(mmsafety_dir, carriers=("figstep", "typographic"))
```

Mapping:

- `figstep` -> `images_figstep`, text source `qr_prompt` fallback `original_prompt`
- `typographic` -> `images_wr`, text source `replaced_prompt` fallback `qr_prompt` fallback `original_prompt`

- [ ] **Step 4: Verify green**

Run same test. Expected: PASS.

## Task 2: Real paired dataset builder

**Files:**
- Create: `experiments/mibd_routing/data/convert_phase2a.py`
- Test: `experiments/mibd/test_routing_scaffold.py`

- [ ] **Step 1: Write failing test**

```python
def test_build_phase2a_paired_dataset_writes_safe_and_risk_pairs(tmp_path):
    # construct two MM-SafetyBench items and one safe image pool
    records, card = build_phase2a_paired_dataset(
        mmsafety_dir=tmp_path / "mmsafety",
        output_dir=tmp_path / "out",
        num_pairs=2,
        seed=3,
        safe_image_dir=tmp_path / "safe_images",
    )
    assert len(records) == 4
    assert all(pathlib.Path(sample.image_path).exists() for sample in records if sample.image_path)
    assert {sample.risk_label for sample in records} == {RiskLabel.SAFE, RiskLabel.RISK}
```

- [ ] **Step 2: Verify red**

Run targeted test. Expected: FAIL before implementation.

- [ ] **Step 3: Implement builder**

Implement:

```python
build_phase2a_paired_dataset(
    mmsafety_dir,
    output_dir,
    num_pairs=200,
    seed=20260604,
    carriers=("figstep", "typographic"),
    safe_image_dir=None,
)
```

Rules:

- one `paired_id` creates exactly one safe and one risk record
- same `question` for safe and risk
- risk image is MM-SafetyBench carrier image
- safe image uses safe image pool if provided, otherwise generated blank txt placeholder
- `metadata` records source item id, original prompt, risk prompt, safe image mode

- [ ] **Step 4: Verify green**

Run targeted test. Expected: PASS.

## Task 3: CLI and handoff update

**Files:**
- Create: `experiments/mibd_routing/run_build_phase2a_dataset.py`
- Modify: `experiments/mibd_routing/HANDOFF.md`

- [ ] **Step 1: Add CLI**

Command:

```bash
.venv_gemma_probe/bin/python -m experiments.mibd_routing.run_build_phase2a_dataset \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --output-dir results/mibd_routing/paired_dataset/phase2a_real_pilot \
  --num-pairs 200 \
  --seed 20260604
```

Expected output paths:

```text
results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl
results/mibd_routing/paired_dataset/phase2a_real_pilot/dataset_card.json
results/mibd_routing/paired_dataset/phase2a_real_pilot/build_report.md
```

- [ ] **Step 2: Run CLI locally**

Run command above. Expected: exit 0 and 400 records.

- [ ] **Step 3: Update handoff**

Add GPU next command:

```bash
python -m experiments.mibd_routing.run_phase2a_behavior \
  --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \
  --output results/mibd_routing/behavior_labels/smoke_phase2a_real/behavior_outputs.jsonl \
  --mode smoke
```

Real GPU VLM generation remains a later model-specific runner.

## Task 4: Verification

- [ ] Run:

```bash
.venv_gemma_probe/bin/python -m pytest experiments/mibd/test_routing_scaffold.py -q
```

Expected: all tests pass.

- [ ] Run:

```bash
.venv_gemma_probe/bin/python -m compileall -q experiments/mibd_routing experiments/mibd/test_routing_scaffold.py
```

Expected: exit 0.

