# CaRoB v3 Audit Handoff: Problems Found and Next Steps

Date: 2026-06-30

## Current Decision

Do not proceed to router training.

The v3 extraction results are complete for both models, but the current data and
evaluation protocol do not isolate the CaRoB target signal. The next research
step is to fix the evaluation/data audit path before any new GPU training run.

## Confirmed Findings

### 1. Same-set AUC is not a valid Go/No-Go gate

The original `probe_summary.json` and `run_offline_oracle.py` fit a
mean-difference probe and evaluate it on the same hidden-state rows. In
4096/3584-dimensional hidden states with about 100 samples per class, this can
inflate AUC.

New audit tool:

- `experiments/mibd_routing_v2/eval/run_probe_audit.py`

New test coverage:

- `experiments/mibd_routing_v2/tests/test_offline_eval.py::TestProbeAudit`

Verification:

- `conda run -n qwen3-vl python -m pytest experiments/mibd_routing_v2/tests/test_offline_eval.py -q`
- Result: `21 passed`

### 2. The evaluator itself is not completely broken

Permutation held-out AUC is near chance:

- Qwen3-VL-8B: `mean_permutation_heldout_auc = 0.509692`
- InternVL3-8B: `mean_permutation_heldout_auc = 0.502371`

This means the AUC implementation and mean-difference audit are not trivially
returning 1.0 for arbitrary labels.

### 3. The real v3 labels remain too easy on held-out data

Leakage-aware audit:

- Qwen3-VL-8B:
  - `mean_within_heldout_auc = 0.970175`
  - `mean_cross_carrier_auc = 0.9825`
  - `heldout_cross_carrier_transfer_drop = -0.012324`
- InternVL3-8B:
  - `mean_within_heldout_auc = 0.946501`
  - `mean_cross_carrier_auc = 0.958358`
  - `heldout_cross_carrier_transfer_drop = -0.011856`

Interpretation: labels are highly separable, but the signal transfers across
carriers rather than exposing carrier-specific layer re-encoding. This is not
yet the CaRoB motivation signal.

### 4. v3 data fixed blank placeholders but introduced/kept other visual confounds

Pair integrity is good:

- `pairs = 200`
- `pair_problems = 0`
- labels balanced: `safe = 200`, `risk = 200`
- carrier/visual/category label counts are paired and balanced

But visual controls are still not distribution-matched:

- Risk images: all `500x500`
- Safe images: all `336x336`
- Safe/risk pixel standard deviation after resizing to 336:
  - safe mean std: `27.33`
  - risk mean std: `73.16`
- Safe carrier filename match:
  - matched: `110`
  - mismatched: `90`

The current neutral renderer therefore removes the blank-image confound, but not
the renderer/source/layout/contrast confound.

### 5. Current hidden-state archive cannot support pair-level splitting

`hidden_states.npz` manifest fields:

- `key`
- `visual_condition`
- `layer`
- `position`
- `label`
- `shape`

It does not store:

- `sample_id`
- `paired_id`
- `carrier_type`
- `risk_category`
- original row index

Therefore true pair-level split cannot be recovered from the current npz files.
Future extraction must save row metadata.

## Remaining Risks / Things Not Yet Ruled Out

1. The last-token position (`--positions=-1`) may be dominated by answer-format,
   image-token/template, or instruction-tail effects rather than semantic risk.
2. The safe neutral text is much shorter and generated from a small fixed phrase
   bank; risk text length/layout varies by MM-SafetyBench carrier.
3. The current safe renderer may use different font, margins, stroke density,
   and layout from MM-SafetyBench `images_wr` / `images_figstep`.
4. Cross-carrier transfer may be high because the probe is reading generic
   "risk-source image vs neutral-renderer image" artifacts, not harmfulness.
5. Current v3 hidden states are useful for post-hoc diagnosis, but not sufficient
   for definitive pair-level claims.

## Tomorrow's Recommended Plan

### Step 1: Make the gate protocol explicit

Replace any training gate based on `check_saturation` or same-set oracle with:

1. within-carrier held-out AUC
2. permutation held-out AUC
3. held-out cross-carrier transfer
4. carrier-matched vs mismatched subgroup report
5. pair-level split once metadata is available

### Step 2: Add extraction metadata

Modify hidden-state saving so every array row can be traced back to:

- `sample_id`
- `paired_id`
- `risk_label`
- `carrier_type`
- `visual_condition`
- `risk_category`
- `image_path`
- `counterpart_image_path`
- original dataset row index

This is required before a rigorous pair-level audit.

### Step 3: Build a data audit report before v4 extraction

For any candidate dataset, generate a CPU-only report with:

- safe/risk image size
- carrier match rate
- image pixel mean/std distribution
- text length/layout proxy
- category balance
- pair integrity
- source/renderer distribution

Do not run GPU extraction until this report is clean.

### Step 4: Design v4 data controls

Minimum v4 requirements:

- risk `figstep` -> safe `neutral_figstep`
- risk `typographic` -> safe `neutral_typographic`
- safe canvas size matches risk images (`500x500` unless model-specific reason)
- safe renderer approximates MM-SafetyBench font/margin/layout more closely
- safe text length is matched to risk carrier text length bins
- metadata records chosen safe carrier and neutral phrase/text

Stronger option:

- Re-render both risk and safe text through the same local renderer, so the only
  intended difference is harmful vs neutral content.

### Step 5: Only then rerun dual-GPU extraction

Once v4 passes CPU data audit:

- GPU0: Qwen3-VL-8B extraction
- GPU1: InternVL3-8B extraction

Then run leakage-aware audit before any router training.

## Bottom Line

The main issue is not just one bug. It is a chain:

1. original gate used same-set AUC;
2. v3 controls are still visually/source-distribution mismatched;
3. current hidden-state archives lack row metadata for pair-level split.

Fix those three before using any result as a CaRoB Go/No-Go signal.
