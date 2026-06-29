# MIBD Routing V2 Offline GPU Runbook

This runbook is for an offline GPU environment with local model checkpoints and
no network access. It intentionally does not require real benign images for the
first pass; safe samples will use blank placeholders unless you populate
`data/mibd_routing_v2/benign_safe_images`.

Without real benign images, treat results as H1/H2 diagnostics only:

- usable: risk evidence readability, visual-carrier relocation, behavior labels
  on risk carriers
- not final: benign over-refusal, utility preservation, H3 Go/No-Go

## 0. Build V2 Dataset

```bash
.venv_gemma_probe/bin/python -m experiments.mibd_routing_v2.run_build_phase2a_dataset \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --safe-image-dir data/mibd_routing_v2/benign_safe_images \
  --output-dir results/mibd_routing_v2/paired_dataset/phase2a_matched_v2 \
  --num-pairs 200 \
  --seed 20260604
```

Inspect:

```bash
sed -n '1,80p' results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/build_report.md
```

If no benign images are present, `safe image modes` will show
`generated_blank_placeholder`; that is acceptable for this first GPU pass.

## 1. GPU Smoke Behavior Generation

Run 6 samples first.

### Qwen3-VL-8B

```bash
mkdir -p results/mibd_routing_v2/behavior_labels/qwen3_vl_8b logs

conda run -n qwen3-vl python -m experiments.mibd_routing_v2.run_phase2a_vlm_behavior \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output results/mibd_routing_v2/behavior_labels/qwen3_vl_8b/behavior_outputs.smoke.jsonl \
  --device cuda:0 \
  --max-samples 6 \
  --log-every 1 \
  2>&1 | tee logs/mibd_routing_v2_qwen3_behavior_smoke.log
```

### InternVL3-8B

```bash
mkdir -p results/mibd_routing_v2/behavior_labels/internvl3_8b logs

conda run -n rdo python -m experiments.mibd_routing_v2.run_phase2a_vlm_behavior \
  --model internvl3_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output results/mibd_routing_v2/behavior_labels/internvl3_8b/behavior_outputs.smoke.jsonl \
  --device cuda:0 \
  --max-samples 6 \
  --log-every 1 \
  2>&1 | tee logs/mibd_routing_v2_internvl3_behavior_smoke.log
```

## 2. Full Behavior Generation

After the smoke output looks sane, run full generation. If you have two GPUs,
this can run in parallel with the full probe pass as long as they use different
devices.

```bash
mkdir -p results/mibd_routing_v2/behavior_labels/qwen3_vl_8b logs

conda run -n qwen3-vl python -m experiments.mibd_routing_v2.run_phase2a_vlm_behavior \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output results/mibd_routing_v2/behavior_labels/qwen3_vl_8b/behavior_outputs.jsonl \
  --device cuda:1 \
  --log-every 10 \
  2>&1 | tee logs/mibd_routing_v2_qwen3_behavior.log
```

```bash
mkdir -p results/mibd_routing_v2/behavior_labels/internvl3_8b logs

conda run -n rdo python -m experiments.mibd_routing_v2.run_phase2a_vlm_behavior \
  --model internvl3_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output results/mibd_routing_v2/behavior_labels/internvl3_8b/behavior_outputs.jsonl \
  --device cuda:1 \
  --log-every 10 \
  2>&1 | tee logs/mibd_routing_v2_internvl3_behavior.log
```

Resume after interruption by adding `--resume` and using the same output path.

## 3. Hidden Extraction + Probe Summary

Start with a small layer set and 20 samples.

### Qwen3-VL-8B Smoke

```bash
mkdir -p results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_smoke logs

conda run -n qwen3-vl python -m experiments.mibd_routing_v2.run_phase2b_extract_probe \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output-dir results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_smoke \
  --device cuda:0 \
  --layers 0,4,8,12,16,20,24,28,32,35 \
  --positions=-1 \
  --max-samples 20 \
  2>&1 | tee logs/mibd_routing_v2_qwen3_probe_smoke.log
```

### InternVL3-8B Smoke

```bash
mkdir -p results/mibd_routing_v2/sensor_probe/internvl3_8b_smoke logs

conda run -n rdo python -m experiments.mibd_routing_v2.run_phase2b_extract_probe \
  --model internvl3_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output-dir results/mibd_routing_v2/sensor_probe/internvl3_8b_smoke \
  --device cuda:0 \
  --layers 0,4,8,12,16,20,24,27 \
  --positions=-1 \
  --max-samples 20 \
  2>&1 | tee logs/mibd_routing_v2_internvl3_probe_smoke.log
```

Outputs:

```text
hidden_states.npz
probe_summary.json
```

Check:

```bash
python - <<'PY'
import json
from pathlib import Path
for p in Path("results/mibd_routing_v2/sensor_probe").glob("*/probe_summary.json"):
    data = json.loads(p.read_text())
    print("\n", p)
    for cond, rows in data["conditions"].items():
        if rows:
            best = max(rows, key=lambda r: r["subspace_auc"])
            print(cond, best)
PY
```

## 4. Full Probe Pass

Only after smoke passes, remove `--max-samples`. Keep selected layers first; all
layers can be run later if memory/time is acceptable.

### Qwen3-VL-8B Full Probe

```bash
mkdir -p results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_loci10 logs

conda run -n qwen3-vl python -m experiments.mibd_routing_v2.run_phase2b_extract_probe \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output-dir results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_loci10 \
  --device cuda:0 \
  --layers 0,4,8,12,16,20,24,28,32,35 \
  --positions=-1 \
  2>&1 | tee logs/mibd_routing_v2_qwen3_probe_loci10.log
```

### InternVL3-8B Full Probe

```bash
mkdir -p results/mibd_routing_v2/sensor_probe/internvl3_8b_loci8 logs

conda run -n rdo python -m experiments.mibd_routing_v2.run_phase2b_extract_probe \
  --model internvl3_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output-dir results/mibd_routing_v2/sensor_probe/internvl3_8b_loci8 \
  --device cuda:0 \
  --layers 0,4,8,12,16,20,24,27 \
  --positions=-1 \
  2>&1 | tee logs/mibd_routing_v2_internvl3_probe_loci8.log
```

## 5. Two-GPU Parallel Recipes

Run these in two separate terminals. They save logs and show progress at the
same time.

### Qwen3: Probe on GPU 0, Behavior on GPU 1

Terminal A:

```bash
mkdir -p results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_loci10 logs

conda run -n qwen3-vl python -m experiments.mibd_routing_v2.run_phase2b_extract_probe \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output-dir results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_loci10 \
  --device cuda:0 \
  --layers 0,4,8,12,16,20,24,28,32,35 \
  --positions=-1 \
  2>&1 | tee logs/mibd_routing_v2_qwen3_probe_loci10.log
```

Terminal B:

```bash
mkdir -p results/mibd_routing_v2/behavior_labels/qwen3_vl_8b logs

conda run -n qwen3-vl python -m experiments.mibd_routing_v2.run_phase2a_vlm_behavior \
  --model qwen3_vl_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output results/mibd_routing_v2/behavior_labels/qwen3_vl_8b/behavior_outputs.jsonl \
  --device cuda:1 \
  --log-every 10 \
  2>&1 | tee logs/mibd_routing_v2_qwen3_behavior.log
```

### Cross-Model: Qwen3 Probe on GPU 0, InternVL Behavior on GPU 1

Terminal A is the Qwen3 probe command above.

Terminal B:

```bash
mkdir -p results/mibd_routing_v2/behavior_labels/internvl3_8b logs

conda run -n rdo python -m experiments.mibd_routing_v2.run_phase2a_vlm_behavior \
  --model internvl3_8b \
  --dataset results/mibd_routing_v2/paired_dataset/phase2a_matched_v2/paired_dataset.jsonl \
  --output results/mibd_routing_v2/behavior_labels/internvl3_8b/behavior_outputs.jsonl \
  --device cuda:1 \
  --log-every 10 \
  2>&1 | tee logs/mibd_routing_v2_internvl3_behavior.log
```
