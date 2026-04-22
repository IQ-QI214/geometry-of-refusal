# PCD — Handoff for New Session

**Last updated**: 2026-04-22 (post-Tasks 1-4, GPU now available)
**Last commit**: `12a9542` — "pcd: Stage A Tasks 1-4 CPU side — adapters, Arditi judge, smoke scripts"

---

## Quick Start for New Session

**Read these files in order before doing anything:**

1. **Spec**: `docs/superpowers/specs/2026-04-22-1500-pcd-pipeline-verification-design.md`
2. **Plan**: `docs/superpowers/plans/2026-04-22-1500-pcd-pipeline-verification-implementation.md`
3. **This file** (current progress + GPU decisions)

**Then**: Use `superpowers:subagent-driven-development` to continue from Task 5.

---

## Current Task Status

| Task | Description | Status | Notes |
|:-:|---|:-:|---|
| 1 | Download Gemma-3-4B + verify α | 🟡 GPU pending | Model at `/inspire/.../models/gemma-3-4b-it`; script ready |
| 2 | Qwen VLM adapter image_mode + verify β | 🟡 GPU pending | Code done; beta script ready |
| 3 | Gemma-3 adapters + factory | ✅ Done | CPU smoke passed; GPU generate pending |
| 4 | Arditi templates + judge + tests | ✅ Done | 6/6 tests pass |
| 5 | Bootstrap stability utility | ⏳ Not started | Pure CPU, run tests in `rdo` env |
| 6 | Three experiment entry scripts | ⏳ Not started | CPU; reads P0 pipeline internals |
| 7 | run_all.sh + PROGRESS.md + HANDOFF.md | ⏳ Not started | CPU scaffold |
| 8 | E2E smoke test n=4 | ⏳ Not started | GPU required |
| 9 | Step 0 bootstrap H0 | ⏳ Not started | GPU required |
| 10–14 | Stage B (GPU runs, evaluate, findings) | ⏳ Not started | — |

**Next CPU tasks**: 5 → 6 → 7 (all CPU-only, can proceed without GPU)
**GPU tasks ready to run now**: Tasks 1 GPU, 2 GPU, 3 GPU smoke, then Task 8

---

## GPU Steps Ready to Execute (in order)

### GPU Step A: Verify α (Task 1)
```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl \
  python experiments/pcd/smoke/verify_alpha.py \
  --model_path /inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it \
  2>&1 | tee experiments/pcd/smoke/verify_alpha.log
cat experiments/pcd/smoke/verify_alpha.log
```
Expected: `ALPHA_RESULT: PASS` (weights equal) or `STANDALONE_LOAD_FAILED` (no CausalLM class) — both mean L ≡ V-text collapse is valid.
If `ALPHA_RESULT: FAIL`: keep Gemma L and V-text as separate conditions (adds ~3 GPU·h).

### GPU Step B: Verify β (Task 2)
```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl \
  python experiments/pcd/smoke/verify_beta.py \
  --model_path /inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct \
  2>&1 | tee experiments/pcd/smoke/verify_beta.log
cat experiments/pcd/smoke/verify_beta.log
```
Expected: `BETA_RESULT: PASS`. If `FAIL`: fallback to 1×1 pixel image as "near-text-only".

### GPU Step C: Gemma-3 adapter smoke (Task 3)
```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/refusal_direction
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n qwen3-vl bash -c \
  "PYTHONPATH=. python ../experiments/pcd/smoke/smoke_gemma3_adapters.py \
   --model_path /inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it \
   --mode generate 2>&1 | tee ../experiments/pcd/smoke/smoke_gemma3_generate.log"
```
Expected: `VLM_GENERATE_SMOKE: PASS` and `TEXT_GENERATE_SMOKE: PASS` with num_layers printed.
**Record the num_layers output** — needed to confirm `model.language_model.layers` path is correct.

**After GPU Steps A-C**: report results to new Claude session so it can record α/β verdicts and continue.

---

## Critical Environment Notes

| Env | Use for | Transformers version |
|---|---|---|
| `rdo` | Qwen2.5-7B, LLaVA, general pipeline, pytest | 4.47.0 (no Gemma-3 support) |
| `qwen3-vl` | Qwen2.5-VL, Gemma-3, all guard models | 4.57.3 ✅ |

**All Gemma-3 scripts must use `qwen3-vl` env.**

---

## What Was Built (commit 12a9542)

### Modified files
- `refusal_direction/pipeline/model_utils/qwen_vlm_model.py`
  - `tokenize_instructions_qwen_vlm` now accepts `image_mode: Literal["text","blank","noise"]` (default `"blank"` for backward compat)
  - `_make_noise_image(seed)` helper added
  - `generate_completions` already conditionally handles `pixel_values` — no change needed

### New files
- `refusal_direction/pipeline/model_utils/gemma3_vlm_model.py` — `Gemma3VLMModel`, three image modes
- `refusal_direction/pipeline/model_utils/gemma3_model.py` — `Gemma3Model` (text-only, CausalLM or submodule fallback)
- `refusal_direction/pipeline/model_utils/model_factory.py` — `construct_model_base(model_path, model_name=None)`; name-based dispatch for `"gemma-3-4b-it-vlm"` and `"gemma-3-4b-it"`
- `experiments/pcd/download_gemma3.sh` — download script (model already at `/models/gemma-3-4b-it`)
- `experiments/pcd/smoke/verify_alpha.py` — α verification
- `experiments/pcd/smoke/verify_beta.py` — β verification (also exercises adapter path)
- `experiments/pcd/smoke/smoke_gemma3_adapters.py` — `--mode tokenization` (CPU✅) / `--mode generate` (GPU)
- `experiments/pcd/common/__init__.py` — package marker
- `experiments/pcd/common/arditi_templates.py` — `ARDITI_REFUSAL_TEXT_SAMPLES`, `ARDITI_REFUSAL_TOKENS`, `build_templates`, `normalize_model_name`
- `experiments/pcd/common/arditi_judge.py` — `arditi_refusal_match`, `arditi_joint_asr`, `compute_arditi_metrics`
- `experiments/pcd/common/test_arditi.py` — 6/6 tests pass (`PYTHONPATH=. pytest`)

---

## Remaining CPU Tasks (5–7) — Summary for New Session

### Task 5: Bootstrap stability utility
- Create `experiments/pcd/common/bootstrap.py` + `test_bootstrap.py`
- Tests are CPU-only (no model load); run with `PYTHONPATH=. conda run -n rdo pytest`
- The `bootstrap_stability()` function takes a loaded model — tests only test `split_prompts()`
- Add `__main__` CLI block so it can be called as a script in Task 9 GPU step

### Task 6: Three experiment entry scripts
- `experiments/pcd/exp_pcd_layer_sweep.py` — Step 1 (layer×pos sweep)
- `experiments/pcd/exp_pcd_ablate.py` — Step 2 (ablate + generate)
- `experiments/pcd/exp_pcd_evaluate.py` — Step 3 (4-judge + Arditi)
- **Before writing**: read existing P0 scripts to understand exact function signatures:
  ```bash
  ls experiments/p0_cone/
  grep -n "def generate_directions\|def select_direction" refusal_direction/pipeline/submodules/*.py
  grep -n "def evaluate_keyword\|def evaluate_strongreject\|def evaluate_qwen3guard\|def evaluate_llamaguard" experiments/p0_cone/common/*.py
  ```
- These scripts import from `experiments.p0_cone.common.eval_pipeline` — verify that module exists before writing `exp_pcd_evaluate.py`

### Task 7: run_all.sh + PROGRESS.md + HANDOFF.md scaffold
- `run_all.sh` is a GPU orchestration script; the `run_ablate/rdo/judge` functions are stubs filled by Tasks 11-13
- `PROGRESS.md` tracks task status (use this file's table as source of truth)
- `HANDOFF.md` is this file — update it after Task 7 is done

---

## Key Decisions (pending verification)

- **α verdict**: _pending GPU Step A_ — determines if Gemma L≡V-text collapse is valid
- **β verdict**: _pending GPU Step B_ — determines if V-text is truly no-image
- **H0 verdict**: _pending Task 9 (bootstrap)_
- **Gemma num_layers**: _pending GPU Step C_ — confirms `model.language_model.layers` path

---

## Known Issues / Notes

1. `experiments/pcd/smoke/verify_beta.py` uses `sys.path.insert` to import the pipeline — when running from `refusal_direction/` with `PYTHONPATH=.` this is redundant. Can be run with: `cd refusal_direction && PYTHONPATH=. python ../experiments/pcd/smoke/verify_beta.py`
2. `Gemma3VLMModel._get_model_block_modules` returns `self.model.language_model.layers` — needs GPU Step C to confirm this path is correct. If wrong, update before Task 10 layer sweep.
3. When running experiment scripts, always set `PYTHONPATH=<project_root>/refusal_direction` or run from within that directory.
4. `download_gemma3.sh` was moved from `scripts/` to `experiments/pcd/` because `.gitignore` ignores `scripts/`.
