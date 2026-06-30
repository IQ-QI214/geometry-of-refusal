# MIBD Routing V2 Run: 20260630_125933_phase2a_v3_dual_model

## Purpose

Rebuild Phase 2A with category-matched neutral safe images, extract hidden states for Qwen3-VL-8B and InternVL3-8B on two GPUs, then run saturation and offline-oracle gates.

## Inputs

- Dataset: `dataset/paired_dataset.jsonl`
- Dataset report: `dataset/build_report.md`
- Safe image mode: `category_matched` for 200 paired IDs

## GPU Allocation

- GPU0: Qwen3-VL-8B extraction
- GPU1: InternVL3-8B extraction

## Outputs

- `sensor_probe/qwen3_vl_8b_v3/hidden_states.npz`
- `sensor_probe/qwen3_vl_8b_v3/probe_summary.json`
- `sensor_probe/internvl3_8b_v3/hidden_states.npz`
- `sensor_probe/internvl3_8b_v3/probe_summary.json`
- `offline_oracle/qwen3_vl_8b_v3.json`
- `offline_oracle/internvl3_8b_v3.json`

## Gate Result

No-Go for CaRoB training, but not because the original saturation guard is a
valid scientific gate. The original `probe_summary` and `offline_oracle` used
same-set probe fitting/evaluation, which can overstate AUC in high-dimensional
hidden states. This run has therefore been re-audited with held-out and
permutation baselines.

Original same-set summaries remain saturated at early layers:

- Qwen3-VL-8B: FigStep and V-real have layer 0/4 AUC = 1.0.
- InternVL3-8B: FigStep and V-real have layer 0/4 AUC = 1.0.

Offline oracle remains trivial:

- Qwen3-VL-8B: within oracle AUC = 1.0, cross oracle AUC = 1.0, transfer drop = 0.0.
- InternVL3-8B: within oracle AUC = 1.0, cross oracle AUC = 1.0, transfer drop = 0.0.

Leakage-aware probe audit:

- `offline_oracle/qwen3_vl_8b_v3_probe_audit.json`
  - mean within held-out AUC = 0.970175
  - mean cross-carrier AUC = 0.9825
  - held-out transfer drop = -0.012324
  - mean permutation held-out AUC = 0.509692
- `offline_oracle/internvl3_8b_v3_probe_audit.json`
  - mean within held-out AUC = 0.946501
  - mean cross-carrier AUC = 0.958358
  - held-out transfer drop = -0.011856
  - mean permutation held-out AUC = 0.502371

## Interpretation

The evaluation machinery is not completely broken: permutation held-out AUC is
near chance. However, the real labels remain highly separable on held-out data,
and cross-carrier transfer does not drop. This means the current dataset is
still dominated by an easy label signal that transfers across carriers, rather
than the CaRoB target signal of carrier-specific layer re-encoding. Do not train
the router on this run.

## Next Step

First fix the evaluation protocol and dataset audit path:

1. Treat same-set AUC only as a smoke metric.
2. Use held-out AUC, permutation held-out AUC, and held-out cross-carrier
   transfer as the gate.
3. Report carrier-matched vs carrier-mismatched safe controls separately.

Then build a v4 dataset where safe controls are not only category-matched, but
also carrier-matched at the per-sample carrier level and closer in text
length/layout to the risk carrier images. Re-extract hidden states and rerun the
leakage-aware audit before any router training.
