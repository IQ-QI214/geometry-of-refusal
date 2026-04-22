# PCD — Handoff for New Session

**Last updated**: 2026-04-22 (Task 8 sweep PASS; ablate + evaluate still needed)
**Last commit**: `2420bcd` — "pcd: fix jaxtyping stub — use subscriptable _JaxStub class instead of any"

---

## Quick Start for New Session

**Read this file only** — it contains everything needed. Reference files:
- Spec: `docs/superpowers/specs/2026-04-22-1500-pcd-pipeline-verification-design.md`
- Plan: `docs/superpowers/plans/2026-04-22-1500-pcd-pipeline-verification-implementation.md`

---

## Task Status

| Task | Description | Status | Notes |
|:-:|---|:-:|---|
| 1 | Download Gemma-3-4B + verify α | ✅ Done | α = PASS |
| 2 | Qwen VLM adapter image_mode + verify β | ✅ Done | β = PASS |
| 3 | Gemma-3 adapters + factory | ✅ Done | All fixes confirmed |
| 4 | Arditi templates + judge + tests | ✅ Done | 6/6 tests pass |
| 5 | Bootstrap stability utility | ✅ Done | 3/3 tests pass |
| 6 | Three experiment entry scripts | ✅ Done | — |
| 7 | run_all.sh + PROGRESS.md + HANDOFF.md | ✅ Done | — |
| 8 | E2E smoke test n=4 | 🔶 Partial | Sweep ✅; ablate + evaluate ⏳ |
| 9–14 | Stage B (sweep, ablate, evaluate, findings) | ⏳ Pending | — |

**Next action**: Complete Task 8 — run ablate then evaluate on the sweep result already saved.

---

## Task 8: Remaining Commands (ablate + evaluate)

The sweep already ran and saved `best_layer.json` to `results/pcd/smoke/qwen_vtext_sweep/`.
Result: `layer=16, pos=-5, filter_passed=True` ✅

### Step 2: Ablate + Generate

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/refusal_direction
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl bash -c \
  "PYTHONPATH=. python ../experiments/pcd/exp_pcd_ablate.py \
    --model_name qwen2.5-vl-7b \
    --model_path /inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct \
    --condition V-text \
    --sweep_dir ../results/pcd/smoke/qwen_vtext_sweep \
    --data_dir ../experiments/pcd/smoke/e2e_work/splits \
    --output_dir ../results/pcd/smoke/qwen_vtext_ablate \
    --n_val 4 \
  2>&1 | tee ../results/pcd/smoke/e2e_ablate.log"
```

Expected: `=== Ablation Complete ===` with `dim_responses.json` saved.

### Step 3: Evaluate (kw + arditi only for smoke)

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl bash -c \
  "PYTHONPATH=. python ../experiments/pcd/exp_pcd_evaluate.py \
    --responses_json ../results/pcd/smoke/qwen_vtext_ablate/dim_responses.json \
    --model_name qwen2.5-vl-7b \
    --model_path /inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct \
    --output_json ../results/pcd/smoke/qwen_vtext_eval.json \
    --layers kw arditi \
  2>&1 | tee ../results/pcd/smoke/e2e_eval.log"
```

Expected: JSON with `asr_keyword` and `arditi_refusal_rate` fields (values don't matter — just no errors).

### After Task 8 passes

1. Commit results + update PROGRESS.md Task 8 → ✅
2. **Suggest new session** for Stage B (Tasks 9–14 are all GPU-heavy experiments)
3. See plan §8.3 for the commit command

---

## Verified GPU Results

### α — Gemma backbone equivalence: PASS ✅
- Hash `5c97fca4c73e8f70` matches between MM and SA (normalized keys)
- **Decision**: L ≡ V-text for Gemma; run single text-mode condition

### β — Qwen V-text mode: PASS ✅
- `pixel_values=None` forward works correctly

### γ — Gemma adapter smoke: PASS ✅
- `VLM_GENERATE_SMOKE: PASS`, `TEXT_GENERATE_SMOKE: PASS`, `num layers: 34`
- All prior path bugs fixed

### Task 8 sweep — Qwen V-text: PASS ✅
- `layer=16, pos=-5, filter_passed=True`
- Matches P0 baseline layer 16

---

## Known Environment Issues (all FIXED in code)

| Issue | Fix | Commit |
|---|---|---|
| `jaxtyping` not in qwen3-vl GPU env | All `from jaxtyping import` wrapped in try/except with `_JaxStub` | `2420bcd` |
| `wandb` / `seaborn` not installed | Same try/except pattern | `c67b763` |
| `model_factory` didn't know `qwen2.5-vl-7b` | Added to `_MODEL_NAME_MAP` | `13c8304` |
| `select_direction` used `config.num_hidden_layers` (fails for Gemma3 nested config) | Replaced with `len(model_block_modules)` | `13c8304` |
| `Gemma3Model.generate_completions` missing → hit `accelerate` import | Added override | `9ddda74` |

> **Note**: If GPU machine still lacks jaxtyping, the `_JaxStub` stub makes it work without installation.

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

Run scripts from: `refusal_direction/` with `PYTHONPATH=.`

---

## Commit History (this session)

| Commit | What |
|---|---|
| `7444f5d` | HANDOFF updated with GPU A/B/C results, Tasks 5-6 done |
| `a5ce84e` | Fix gemma3_vlm_model backbone path |
| `e922d4c` | Fix stale comment in gemma3_vlm_model |
| `6af79d4` | Task 7: run_all.sh + PROGRESS + HANDOFF |
| `9ddda74` | Fix Gemma3Model missing generate_completions + verify_alpha key normalization |
| `6ff3621` | Fix verify_alpha bfloat16 numpy |
| `bae8231` | GPU Step C PASS recorded |
| `8f5e8d4` | wandb optional import |
| `1b65805` | seaborn optional import |
| `13c8304` | model_factory Qwen entries + select_direction portability |
| `bd503c1` | Smoke splits (n=8 train, n=4 val) |
| `c67b763` | All jaxtyping/wandb/seaborn imports optional |
| `2420bcd` | Fix jaxtyping stub → subscriptable _JaxStub |

---

*Created: 2026-04-22 | Updated: 2026-04-22 (Task 8 sweep PASS)*
