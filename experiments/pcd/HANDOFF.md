# PCD — Handoff for New Session

**Last updated**: 2026-04-22 (Tasks 1-7 complete)
**Last commit**: `6af79d4` — "pcd: add orchestrator + PROGRESS + HANDOFF scaffolds (Task 7)"

---

## Quick Start for New Session

**Read this file only** — it contains everything needed. The spec/plan files are at:
- Spec: `docs/superpowers/specs/2026-04-22-1500-pcd-pipeline-verification-design.md`
- Plan: `docs/superpowers/plans/2026-04-22-1500-pcd-pipeline-verification-implementation.md`

---

## Task Status

| Task | Description | Status | Notes |
|:-:|---|:-:|---|
| 1 | Download Gemma-3-4B + verify α | ✅ Done | α = PASS (effective) |
| 2 | Qwen VLM adapter image_mode + verify β | ✅ Done | β = PASS |
| 3 | Gemma-3 adapters + factory | ✅ Done | Path bug fixed in e922d4c |
| 4 | Arditi templates + judge + tests | ✅ Done | 6/6 tests pass |
| 5 | Bootstrap stability utility | ✅ Done | 3/3 tests pass |
| 6 | Three experiment entry scripts | ✅ Done | Import checks pass |
| 7 | run_all.sh + PROGRESS.md + HANDOFF.md | ✅ Done | — |
| 8 | E2E smoke test n=4 | ⏳ Pending | GPU required |
| 9–14 | Stage B (sweep, ablate, evaluate, findings) | ⏳ Pending | — |

**Next actions** (in order):
1. **GPU Step C re-run** — verify Gemma adapter works after e922d4c fix (see command below)
2. **Task 8 E2E smoke** — n=4 on Qwen V-text (see command below)

---

## Next GPU Commands

### Step C: Gemma adapter smoke (verify fix)

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/refusal_direction
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl bash -c \
  "PYTHONPATH=. python ../experiments/pcd/smoke/smoke_gemma3_adapters.py \
   --model_path /inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it \
   --mode generate 2>&1 | tee ../experiments/pcd/smoke/smoke_gemma3_generate.log"
```

Expected: `VLM_GENERATE_SMOKE: PASS`, `TEXT_GENERATE_SMOKE: PASS`, `num layers: 34`

### Task 8: E2E smoke (n=4, Qwen V-text)

Follow plan §8.2 — create tiny val set, run sweep → ablate → evaluate on Qwen V-text.
(Detailed commands TBD once Task 8 is coded.)

---

## Verified GPU Results

### α — Gemma backbone equivalence: PASS (effective)

- MM keys (444): `layers.X.*`, `embed_tokens.weight`, `norm.weight`
- SA keys (445): same with `model.` prefix + `lm_head.weight`
- Transformer weights are **bit-identical**; mismatch is purely key naming
- **Decision**: L ≡ V-text for Gemma. Run single text-mode condition; report as both "L" and "V-text" in matrix.
- **Gemma num_layers**: 34

### β — Qwen V-text mode: PASS

- `pixel_values=None` forward works; output coherent ("Baking a cake is a delightful process...")
- **Decision**: Proceed. V-text condition is cleanly text-only; no 1×1 pixel fallback needed.

### γ — Gemma adapter smoke: FIXED (was FAIL)

- Was: `AttributeError: 'Gemma3TextModel' object has no attribute 'model'`
- Root cause: `f3ea391` used wrong path `language_model.model.layers`
- Fix: Reverted to `language_model.layers` in commit `e922d4c`
- Needs: GPU Step C re-run to confirm fix

---

## Environment

| Env | Use for | Transformers |
|---|---|---|
| `rdo` | Qwen2.5-7B, general pipeline, pytest | 4.47.0 |
| `qwen3-vl` | Qwen2.5-VL AND Gemma-3 | 4.57.3 |

> **Critical**: Gemma-3 MUST use `qwen3-vl`. The `rdo` env lacks Gemma-3 support.

Model paths:
- Gemma-3-4B-it: `/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it`
- Qwen2.5-VL-7B: `/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct`
- Qwen2.5-7B: `/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct`

Project root: `/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/`
Run scripts from: `refusal_direction/` with `PYTHONPATH=.`

---

## Commit History

| Commit | What |
|---|---|
| `12a9542` | Tasks 1-4: qwen_vlm adapter, gemma3 adapters, factory, arditi judge, smoke scripts |
| `f3ea391` | α script prefix fix (correct) + gemma3 path fix (WRONG — introduced regression) |
| `031e085` | Task 5: bootstrap.py + test_bootstrap.py (3/3 pass) |
| `16aff72` | Task 6: exp_pcd_layer_sweep.py, exp_pcd_ablate.py, exp_pcd_evaluate.py |
| `e922d4c` | Fix gemma3_vlm_model backbone path (language_model.layers, no .model wrapper) |
| `6af79d4` | Task 7: run_all.sh + PROGRESS.md + HANDOFF.md scaffolds |

---

*Created: 2026-04-22 | Updated: 2026-04-22 (Task 7 complete)*
