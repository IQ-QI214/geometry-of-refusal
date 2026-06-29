# MIBD Routing V2 Benign Safe Image Pool Setup

This note tracks the local benign image pool expected by the v2 Phase 2A
dataset builder. The actual image files live under `data/`, which is ignored by
git.

## Local Directory

```text
data/mibd_routing_v2/benign_safe_images/
  01/
  02/
  ...
  13/
```

Images directly under `data/mibd_routing_v2/benign_safe_images/` are treated as
a global fallback pool. Category subdirectories are preferred.

Supported suffixes: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.

## Current Pilot Category Demand

The existing `phase2a_real_pilot` risk-side distribution is:

| Category | Risk pairs |
|---|---:|
| `01` | 16 |
| `02` | 25 |
| `03` | 8 |
| `04` | 22 |
| `05` | 11 |
| `06` | 24 |
| `07` | 6 |
| `08` | 22 |
| `09` | 23 |
| `10` | 12 |
| `11` | 8 |
| `12` | 11 |
| `13` | 12 |

For the first pilot, aim for at least 3-5 visually diverse benign images per
category. Do not use MM-SafetyBench risk images, generated blank controls,
analysis figures, or images containing harmful instructions.

## Build Command

After populating the category folders:

```bash
.venv_gemma_probe/bin/python -m experiments.mibd_routing_v2.run_build_phase2a_dataset \
  --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench \
  --safe-image-dir data/mibd_routing_v2/benign_safe_images \
  --output-dir results/mibd_routing_v2/paired_dataset/phase2a_matched_v2 \
  --num-pairs 200 \
  --seed 20260604
```

Inspect `build_report.md` after the run. The target is that most safe samples
use `category_matched`, with minimal `global_fallback` and no
`generated_blank_placeholder`.
