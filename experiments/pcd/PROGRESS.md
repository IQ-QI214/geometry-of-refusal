# PCD — Progress Tracker

**Spec**: `docs/superpowers/specs/2026-04-22-1500-pcd-pipeline-verification-design.md`
**Plan**: `docs/superpowers/plans/2026-04-22-1500-pcd-pipeline-verification-implementation.md`

---

## Environment

| Session | Role |
|---|---|
| CPU (Claude) | Code writing, file edits |
| GPU (qi, 4×H100) | All model-loading and experiment runs |

| Conda Env | Use for | Transformers |
|---|---|---|
| `rdo` | Qwen2.5-7B, general pipeline, pytest | 4.47.0 |
| `qwen3-vl` | Qwen2.5-VL AND Gemma-3 | 4.57.3 |

> **Note**: Gemma-3 MUST use `qwen3-vl`. The `rdo` env (transformers 4.47) lacks Gemma-3 support.

### Model Paths

| Model | Path |
|---|---|
| Gemma-3-4B-it | `/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it` |
| Qwen2.5-VL-7B | `/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct` |
| Qwen2.5-7B | `/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct` |

---

## Task Status

| Task | Description | Status | Notes |
|:-:|---|:-:|---|
| 1 | Download Gemma-3-4B + verify α | ✅ Done | α verdict below |
| 2 | Qwen VLM adapter image_mode + verify β | ✅ Done | β = PASS |
| 3 | Gemma-3 adapters + factory | ✅ Done | Path bug fixed in e922d4c |
| 4 | Arditi templates + judge + tests | ✅ Done | 6/6 tests pass |
| 5 | Bootstrap stability utility | ✅ Done | 3/3 tests pass |
| 6 | Three experiment entry scripts | ✅ Done | Import checks pass |
| 7 | run_all.sh + PROGRESS.md + HANDOFF.md | ✅ Done | This file |
| 8 | E2E smoke test n=4 | ⏳ Pending | GPU required |
| 9 | Layer sweep (6 conditions) | ⏳ Pending | GPU required |
| 10 | Analyze sweep results, pick layers | ⏳ Pending | Depends on Task 9 |
| 11 | Ablation runs | ⏳ Pending | Depends on Task 10 |
| 12 | RDO runs | ⏳ Pending | Depends on Task 10 |
| 13 | Judge evaluation | ⏳ Pending | Depends on Tasks 11-12 |
| 14 | Findings write-up + matrix | ⏳ Pending | Depends on Task 13 |

---

## Key Decisions

### α — Gemma backbone equivalence: PASS (effective)

- **Verdict**: L ≡ V-text for Gemma is VALID
- **Evidence**: `verify_alpha.py` returns KEY_MISMATCH, but the mismatch is purely key naming:
  - MM path (`model.language_model.state_dict()`): `layers.X.*`, `embed_tokens.weight`, `norm.weight` (444 keys)
  - SA path (`Gemma3ForCausalLM.state_dict()`): same keys with `model.` prefix + `lm_head.weight` (445 keys)
  - The transformer weights are **bit-identical**; `model.` prefix and `lm_head.weight` are the only differences
- **Consequence**: Run a single Gemma text-mode condition; report it as both "L" and "V-text" in `pcd_8x6_matrix.json`
- **Gemma num_layers**: 34 (layers.0 through layers.33)

### β — Qwen V-text mode: PASS

- **Verdict**: `pixel_values=None` forward works correctly in Qwen2.5-VL
- **Evidence**: `verify_beta.py` returns `BETA_RESULT: PASS`; generated text coherent
- **Consequence**: Proceed with V-text condition as clean text-only path; no 1×1 pixel fallback needed

### γ — Gemma adapter smoke: PASS ✅

- **VLM mode**: `VLM_GENERATE_SMOKE: PASS`, num layers: 34
- **TEXT mode**: `TEXT_GENERATE_SMOKE: PASS`, num layers: 34
- **verify_alpha re-run**: `ALPHA_RESULT: PASS` — hash `5c97fca4c73e8f70` matches; weights bit-identical
- **All Step C issues resolved**: path bug (e922d4c), accelerate import (9ddda74), bfloat16 numpy (6ff3621)

### H0 — Pipeline consistency null hypothesis: TBD

- Pending Task 9 layer sweep results
- H0: refusal direction is consistent across modality conditions (cos-sim ≥ threshold)

---

## Known Issues

*(none currently open)*
