# Category A Run Guide

> GPU node, project root: `[PROJECT_ROOT]/geometry-of-refusal`

## Env Mapping

| Env | transformers | Models | Use for |
|-----|-------------|--------|---------|
| `rdo` | 4.47.0 | LLaVA-7B, LLaVA-13B, InternVL2-8B | A1/A2/A3 gen for these models |
| `qwen3-vl` | 4.57.3 | Qwen-7B, Qwen-32B, Qwen3Guard, LlamaGuard3 | A1/A2/A3 gen for Qwen + all judge eval |

**Rule**: LLaVA/InternVL2 use `rdo`, Qwen models use `qwen3-vl`, judge eval uses `qwen3-vl`.

---

## Step 0: Install sklearn (both envs, one-time)

```bash
# rdo
conda activate rdo
pip install --no-index --find-links /inspire/hdd/global_user/wenming-253108090054/pip_wheels/sklearn_rdo scikit-learn

# qwen3-vl
conda activate qwen3-vl
pip install --no-index --find-links /inspire/hdd/global_user/wenming-253108090054/pip_wheels/sklearn_qwen3vl scikit-learn
```

Verify:
```bash
conda activate rdo && python -c "from sklearn.metrics import roc_auc_score; print('sklearn OK in rdo')"
conda activate qwen3-vl && python -c "from sklearn.metrics import roc_auc_score; print('sklearn OK in qwen3-vl')"
```

---

## Step 1: Create result directories

```bash
cd [PROJECT_ROOT]/geometry-of-refusal
mkdir -p results/category_a/{llava_7b,llava_13b,qwen2vl_7b,qwen2vl_32b,internvl2_8b}
mkdir -p results/phase3/{llava_13b,qwen2vl_32b}
```

---

## Step 2: Smoke Test A3 (re-run, sklearn now installed)

A1 smoke already passed. Re-run A3:

```bash
conda activate rdo
cd [PROJECT_ROOT]/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python experiments/category_a/exp_a3_norm_prediction.py \
    --model llava_7b --device cuda:0 --n_prompts 10 --max_new_tokens 50
```

Verify:
```bash
python -c "
import json
with open('results/category_a/llava_7b/a3_norm_prediction.json') as f:
    d = json.load(f)
print('n_prompts:', d['n_prompts'])
seq0 = d['sequences'][0]
print('norms length:', len(seq0['norms']))
print('norms[:5]:', seq0['norms'][:5])
print('AUROC max_norm:', d['analysis'].get('auroc_max_norm'))
"
```

Clean up after smoke test:
```bash
rm -f results/category_a/llava_7b/a1_*_saladbench.json
rm -f results/category_a/llava_7b/a1_progress_saladbench.json
rm -f results/category_a/llava_7b/a3_norm_prediction.json
```

---

## Step 3: Phase 0 — Extract directions for new models

### 3a: LLaVA-13B (rdo env, GPU 0, ~26GB)

```bash
conda activate rdo
cd [PROJECT_ROOT]/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/phase3/exp_3a_amplitude_reversal.py \
    --model llava_13b --device cuda:0 \
    > results/phase3/llava_13b/exp_3a.log 2>&1 &
echo "LLaVA-13B PID: $!"
```

### 3b: Qwen-32B (qwen3-vl env, GPU 3, ~64GB)

```bash
conda activate qwen3-vl
cd [PROJECT_ROOT]/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/phase3/exp_3a_amplitude_reversal.py \
    --model qwen2vl_32b --device cuda:3 \
    > results/phase3/qwen2vl_32b/exp_3a.log 2>&1 &
echo "Qwen-32B PID: $!"
```

Monitor:
```bash
tail -f results/phase3/llava_13b/exp_3a.log
tail -f results/phase3/qwen2vl_32b/exp_3a.log
```

Verify:
```bash
ls -la results/phase3/llava_13b/exp_3a_directions.pt
ls -la results/phase3/qwen2vl_32b/exp_3a_directions.pt
```

---

## Step 4: Phase 1 — A1 Generation (5 models, 572 prompts)

### 4a: rdo models (LLaVA-7B, LLaVA-13B, InternVL2-8B)

```bash
conda activate rdo
cd [PROJECT_ROOT]/geometry-of-refusal

# GPU 0: LLaVA-7B
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a1_dsa_validation.py \
    --model llava_7b --device cuda:0 --resume \
    > results/category_a/llava_7b/a1_gen.log 2>&1 &
echo "LLaVA-7B PID: $!"

# GPU 1: LLaVA-13B
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a1_dsa_validation.py \
    --model llava_13b --device cuda:1 --resume \
    > results/category_a/llava_13b/a1_gen.log 2>&1 &
echo "LLaVA-13B PID: $!"

# GPU 2: InternVL2-8B
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a1_dsa_validation.py \
    --model internvl2_8b --device cuda:2 --resume \
    > results/category_a/internvl2_8b/a1_gen.log 2>&1 &
echo "InternVL2 PID: $!"
```

### 4b: qwen3-vl models (Qwen-7B, Qwen-32B)

```bash
conda activate qwen3-vl
cd [PROJECT_ROOT]/geometry-of-refusal

# GPU 2: Qwen-7B (after InternVL2 finishes, or use GPU 2 if free)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a1_dsa_validation.py \
    --model qwen2vl_7b --device cuda:2 --resume \
    > results/category_a/qwen2vl_7b/a1_gen.log 2>&1 &
echo "Qwen-7B PID: $!"

# GPU 3: Qwen-32B
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a1_dsa_validation.py \
    --model qwen2vl_32b --device cuda:3 --resume \
    > results/category_a/qwen2vl_32b/a1_gen.log 2>&1 &
echo "Qwen-32B PID: $!"
```

Monitor all:
```bash
tail -f results/category_a/*/a1_gen.log
```

---

## Step 5: A3 Norm Prediction (after LLaVA-7B A1 or in parallel)

```bash
conda activate rdo
cd [PROJECT_ROOT]/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a3_norm_prediction.py \
    --model llava_7b --device cuda:0 \
    > results/category_a/llava_7b/a3.log 2>&1 &
echo "A3 PID: $!"
```

If GPU 0 busy with A1, use another free GPU (e.g., `--device cuda:1`).

---

## Step 6: A2 Causality (after A1 completes for each model)

### 6a: LLaVA-7B (rdo)

```bash
conda activate rdo
cd [PROJECT_ROOT]/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a2_dsa_causality.py \
    --model llava_7b --device cuda:0 \
    > results/category_a/llava_7b/a2.log 2>&1 &
echo "A2 LLaVA-7B PID: $!"
```

### 6b: InternVL2-8B (rdo)

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a2_dsa_causality.py \
    --model internvl2_8b --device cuda:2 \
    > results/category_a/internvl2_8b/a2.log 2>&1 &
echo "A2 InternVL2 PID: $!"
```

### 6c: Qwen-7B (qwen3-vl)

```bash
conda activate qwen3-vl
cd [PROJECT_ROOT]/geometry-of-refusal
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup python experiments/category_a/exp_a2_dsa_causality.py \
    --model qwen2vl_7b --device cuda:1 \
    > results/category_a/qwen2vl_7b/a2.log 2>&1 &
echo "A2 Qwen-7B PID: $!"
```

---

## Step 7: Qwen3Guard Judge (qwen3-vl env, after A1 completes)

```bash
conda activate qwen3-vl
cd [PROJECT_ROOT]/geometry-of-refusal

for MODEL in llava_7b llava_13b qwen2vl_7b qwen2vl_32b internvl2_8b; do
    echo "=== Judging $MODEL ==="
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python experiments/category_a/exp_a1_judge.py \
        --model $MODEL --judge qwen3guard --dataset saladbench --device cuda:0
done
```

---

## Step 8: Verify Results

```bash
# A1 main metrics
python -c "
import json, glob
for path in sorted(glob.glob('results/category_a/*/a1_baseline_mm_saladbench.json')):
    with open(path) as f:
        d = json.load(f)
    m = d['metrics_kw']
    print(f'{d[\"model\"]}: IBR={m[\"initial_bypass_rate\"]:.3f} SCR={m[\"self_correction_rate_overall\"]:.3f} FHCR_kw={m[\"full_harmful_completion_rate\"]:.3f}')
"

# A3 AUROC
python -c "
import json
with open('results/category_a/llava_7b/a3_norm_prediction.json') as f:
    d = json.load(f)
a = d['analysis']
print('AUROC max_norm:', a.get('auroc_max_norm'))
print('AUROC mean_norm:', a.get('auroc_mean_norm'))
print('Spike precedes SC:', a.get('spike_precedes_sc_rate'))
"

# A2 causality
python -c "
import json
for model in ['llava_7b', 'qwen2vl_7b', 'internvl2_8b']:
    path = f'results/category_a/{model}/a2_causality.json'
    with open(path) as f:
        d = json.load(f)
    print(f'\n=== {model} (strategy={d[\"ablation_strategy\"]}) ===')
    for gname, gdata in d['groups'].items():
        m = gdata['metrics']
        print(f'  {gname:<25} SCR={m[\"self_correction_rate_overall\"]:.3f}  FHCR={m[\"full_harmful_completion_rate\"]:.3f}')
"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: sklearn` | Run Step 0 (install from wheel) |
| `cannot import Qwen2_5_VLForConditionalGeneration` | Use `qwen3-vl` env, not `rdo` |
| `OSError: couldn't connect to huggingface.co` for LLaVA-13B | Model now at local path, pull latest code |
| `past_key_values` deprecation warning in A3 | Harmless warning, safe to ignore |
| OOM on Qwen-32B | Ensure GPU 3 is exclusively free (~64GB needed) |
| A2 "A1 results not found" | A1 must complete first for that model's ablation config |
| A1 interrupted mid-run | Re-run with `--resume`, picks up from last checkpoint |
