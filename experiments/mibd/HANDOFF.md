# MIBD Phase 1 Framework Handoff

This directory contains the CPU-testable scaffold for the MIBD Phase 1
mislocalization experiments.

## What is implemented

- YAML config loading and experiment matrix expansion.
- Canonical MIBD sample schema.
- Token-position audit helper for validating relative positions such as `-5`.
- Mean-difference direction, projection scores, cosine similarity, and binary AUC.
- Phase 1 Go/No-Go report builder.
- CLI entrypoints:
  - `python -m experiments.mibd.inspect_config --config experiments/mibd/configs/phase1_probe.yaml`
  - `python -m experiments.mibd.make_phase1_report --input <summary.json> --output <report.md>`

## What remains for the next implementation pass

1. Add dataset loaders for HarmBench, StrongREJECT, SALAD-Bench, MM-SafetyBench,
   SafeBench, FigStep, and MMJ-Bench using the `MIBDSample` schema.
2. Add Qwen3-VL and InternVL3.5 adapters that expose:
   - input preparation for `V-text`, `V-blank`, `V-noise`, `V-real`, `FigStep`;
   - hidden-state extraction over a layer-token grid;
   - token audit output for each prompt template.
3. Implement harmfulness and refusal probe training over extracted hidden states.
4. Emit a Phase 1 JSON summary compatible with `make_phase1_report.py`.
5. Only if Phase 1 returns `CONTINUE_MIBD`, implement router and MIBD training.

## Expected Phase 1 summary schema

```json
{
  "model_id": "Qwen/Qwen3-VL-8B-Instruct",
  "signal_type": "harmfulness",
  "results": [
    {"visual_condition": "V-text", "layer": 17, "token_pos": -5, "auc": 0.91}
  ],
  "condition_cosines": {
    "V-text|V-blank": 0.52,
    "V-blank|V-noise": 0.93
  },
  "static_transfer_auc": {
    "V-text|V-blank": 0.71
  }
}
```

