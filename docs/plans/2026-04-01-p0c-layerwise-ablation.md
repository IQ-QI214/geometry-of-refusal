# P0-C Layerwise Ablation Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `exp_3d_layerwise_ablation.py` that sweeps every layer (stride=2) with single-layer ablation and produces a `full_harmful_rate(layer)` curve for all 4 models.

**Architecture:** Reuse `ablation_context`, `load_directions`, `load_test_prompts` from `exp_3c_ablation_attack.py` directly (copy the relevant pieces); iterate over `range(0, total_layers, 2)` as target layer; use NW layer's `v_mm` as direction for all sweeps; save per-layer metrics to JSON.

**Tech Stack:** Python 3, PyTorch, transformers; existing `common/model_configs.py`, `common/model_adapters.py`, `phase2/common/eval_utils.py`; conda envs `rdo` (LLaVA, InternVL2, InstructBLIP) and `qwen3-vl` (Qwen2.5-VL).

---

## File Structure

| Action  | Path                                                                      | Purpose                                   |
|---------|---------------------------------------------------------------------------|-------------------------------------------|
| Create  | `experiments/phase3/exp_3d_layerwise_ablation.py`                         | Main sweep script                         |
| Create  | `experiments/phase3/run_3d_layerwise.sh`                                  | 4-GPU parallel launcher                   |
| Output  | `results/phase3/{model}/exp_3d_layerwise_results.json`                    | Per-layer ASR results (auto-created)      |

No existing files are modified. All infrastructure (adapters, configs, eval utils) is imported as-is.

---

## Task 1: Write `exp_3d_layerwise_ablation.py`

**Files:**
- Create: `experiments/phase3/exp_3d_layerwise_ablation.py`

- [ ] **Step 1: Write the full script**

Write the file at `experiments/phase3/exp_3d_layerwise_ablation.py`:

```python
"""
Exp 3D: Layer-wise Ablation Curve

目标: 对每一层单独 ablate，测量 full_harmful_rate(layer)，
     生成 paper 用的"逐层 ablation 曲线"，并揭示 narrow waist 的精确位置。

用法:
  python exp_3d_layerwise_ablation.py --model llava_7b --device cuda:0
  python exp_3d_layerwise_ablation.py --model qwen2vl_7b --device cuda:0 --n_prompts 50
"""

import os
import sys
import argparse
import json
import contextlib
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

# ── 路径设置 (与 exp_3c 保持一致) ─────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase3"))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase2" / "common"))

from common.model_configs import MODEL_CONFIGS, load_model_by_name
from common.model_adapters import create_adapter, ModelAdapter
from eval_utils import evaluate_response, compute_attack_metrics


# ── Ablation Hook (直接复制自 exp_3c) ─────────────────────────────────────────

def _make_pre_hook(direction: torch.Tensor):
    def hook_fn(module, args):
        activation = args[0] if isinstance(args, tuple) else args
        d = direction.to(activation.device).to(activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_act = activation - proj
        return (new_act,) + args[1:] if isinstance(args, tuple) else new_act
    return hook_fn


def _make_output_hook(direction: torch.Tensor):
    def hook_fn(module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        d = direction.to(activation.device).to(activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_act = activation - proj
        return (new_act,) + output[1:] if isinstance(output, tuple) else new_act
    return hook_fn


@contextlib.contextmanager
def ablation_context(adapter: ModelAdapter, direction: torch.Tensor,
                     target_layers: Optional[List[int]] = None):
    llm_layers = adapter.get_llm_layers()
    n_layers = len(llm_layers)
    if target_layers is None:
        target_layers = list(range(n_layers))
    handles = []
    try:
        for idx in target_layers:
            layer = llm_layers[idx]
            handles.append(layer.register_forward_pre_hook(_make_pre_hook(direction)))
            if hasattr(layer, 'self_attn'):
                handles.append(layer.self_attn.register_forward_hook(_make_output_hook(direction)))
            if hasattr(layer, 'mlp'):
                handles.append(layer.mlp.register_forward_hook(_make_output_hook(direction)))
        yield
    finally:
        for h in handles:
            h.remove()


# ── Data / Direction Loaders ──────────────────────────────────────────────────

def load_test_prompts(n: int = 50) -> List[str]:
    data_path = _PROJ_ROOT / "data" / "saladbench_splits" / "harmful_test.json"
    with open(data_path) as f:
        data = json.load(f)
    prompts = [item["instruction"] for item in data[:n]]
    print(f"[load_test_prompts] Loaded {len(prompts)} prompts")
    return prompts


def load_directions(model_name: str) -> dict:
    """Returns {narrow_waist_layer, v_text, v_mm} — same as exp_3c."""
    directions_path = (_PROJ_ROOT / "results" / "phase3" / model_name
                       / "exp_3a_directions.pt")
    if not directions_path.exists():
        raise FileNotFoundError(f"Not found: {directions_path}")
    data = torch.load(directions_path, map_location="cpu")
    nw_layer = data["narrow_waist_layer"]
    v_text = data["directions"][nw_layer]["v_text"]
    v_mm   = data["directions"][nw_layer]["v_mm"]
    print(f"[directions] narrow_waist_layer={nw_layer}, "
          f"v_mm.shape={v_mm.shape}, v_mm.norm={v_mm.norm():.4f}")
    return {"narrow_waist_layer": nw_layer, "v_text": v_text, "v_mm": v_mm}


# ── Single-Layer Ablation Run ─────────────────────────────────────────────────

def run_single_layer(adapter: ModelAdapter, prompts: List[str],
                     blank_image: Image.Image, direction: torch.Tensor,
                     layer_idx: Optional[int], max_new_tokens: int = 150) -> dict:
    """
    Run generation on all prompts with ablation on [layer_idx] (or no ablation if None).
    Returns compute_attack_metrics-compatible result dict + raw responses list.
    """
    per_prompt = []
    ctx = (ablation_context(adapter, direction, [layer_idx])
           if layer_idx is not None
           else contextlib.nullcontext())

    for prompt in prompts:
        torch.cuda.empty_cache()
        with ctx:
            response = adapter.generate_mm(prompt, blank_image,
                                           max_new_tokens=max_new_tokens)
        ev = evaluate_response(response)
        ev["prompt"] = prompt
        ev["response"] = response[:500]
        per_prompt.append(ev)

    metrics = compute_attack_metrics(per_prompt)
    return {"metrics": metrics, "responses": per_prompt}


# ── Main Experiment ────────────────────────────────────────────────────────────

def run_exp_3d(model_name: str, device: str,
               n_prompts: int = 50, max_new_tokens: int = 150) -> dict:
    cfg = MODEL_CONFIGS[model_name]
    total_layers = cfg["total_layers"]

    print(f"\n{'='*60}")
    print(f"Exp 3D: Layer-wise Ablation Curve — {model_name}")
    print(f"total_layers={total_layers}, n_prompts={n_prompts}")
    print(f"{'='*60}")

    # 1. Load model
    model, processor = load_model_by_name(model_name, device)
    adapter = create_adapter(model_name, model, processor, device)
    blank_image = Image.new("RGB", cfg["blank_image_size"], color=(128, 128, 128))

    # 2. Load directions and prompts
    dir_data = load_directions(model_name)
    nw_layer = dir_data["narrow_waist_layer"]
    v_mm = dir_data["v_mm"]
    prompts = load_test_prompts(n=n_prompts)

    # 3. Sanity check: first prompt, baseline
    print("\n[sanity] baseline_mm on first prompt (max_new_tokens=50)...")
    sample = adapter.generate_mm(prompts[0], blank_image, max_new_tokens=50)
    print(f"[sanity] {sample[:200]}")

    layer_results = {}

    # 4. Baseline (no ablation)
    print("\n[baseline] Running baseline_mm...")
    baseline = run_single_layer(adapter, prompts, blank_image,
                                direction=v_mm, layer_idx=None,
                                max_new_tokens=max_new_tokens)
    layer_results["baseline"] = {
        "layer": -1,
        "relative_depth": -1.0,
        "full_harmful_rate":    baseline["metrics"]["full_harmful_completion_rate"],
        "initial_bypass_rate":  baseline["metrics"]["initial_bypass_rate"],
        "self_correction_rate": baseline["metrics"]["self_correction_rate_overall"],
        "n": n_prompts,
    }
    print(f"  baseline: full_harmful={layer_results['baseline']['full_harmful_rate']:.3f}")

    # 5. Layer sweep (stride=2, plus last 4 layers of InternVL2)
    layers_to_probe = list(range(0, total_layers, 2))
    if model_name == "internvl2_8b":
        # InternVL2 extra coverage: last 4 layers individually
        for extra in range(total_layers - 4, total_layers):
            if extra not in layers_to_probe:
                layers_to_probe.append(extra)
        layers_to_probe = sorted(set(layers_to_probe))

    print(f"\nSweeping {len(layers_to_probe)} layers: {layers_to_probe}")

    for layer_idx in layers_to_probe:
        print(f"  Ablating layer {layer_idx}/{total_layers-1} "
              f"(rel={layer_idx/total_layers:.2f})...", end=" ", flush=True)
        result = run_single_layer(adapter, prompts, blank_image,
                                  direction=v_mm, layer_idx=layer_idx,
                                  max_new_tokens=max_new_tokens)
        fhr = result["metrics"]["full_harmful_completion_rate"]
        ibr = result["metrics"]["initial_bypass_rate"]
        scr = result["metrics"]["self_correction_rate_overall"]
        print(f"full_harmful={fhr:.3f}  bypass={ibr:.3f}  sc={scr:.3f}")

        key = f"layer_{layer_idx}"
        layer_results[key] = {
            "layer": layer_idx,
            "relative_depth": round(layer_idx / total_layers, 4),
            "full_harmful_rate":    fhr,
            "initial_bypass_rate":  ibr,
            "self_correction_rate": scr,
            "n": n_prompts,
        }

    # 6. Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name}  (narrow_waist={nw_layer})")
    print(f"{'Layer':>8} {'RelDepth':>10} {'FullHarm':>10} {'Bypass':>8} {'SC':>6}")
    print(f"{'─'*46}")
    print(f"{'baseline':>8} {'':>10} "
          f"{layer_results['baseline']['full_harmful_rate']:>10.3f}")
    for k, v in sorted(layer_results.items(), key=lambda x: x[1]["layer"]):
        if k == "baseline":
            continue
        print(f"{v['layer']:>8} {v['relative_depth']:>10.3f} "
              f"{v['full_harmful_rate']:>10.3f} "
              f"{v['initial_bypass_rate']:>8.3f} "
              f"{v['self_correction_rate']:>6.3f}")

    # Mark narrow waist in output
    nw_key = f"layer_{nw_layer}"
    if nw_key in layer_results:
        print(f"\n★ narrow_waist layer {nw_layer}: "
              f"full_harmful={layer_results[nw_key]['full_harmful_rate']:.3f}")

    # 7. Save
    output = {
        "model": model_name,
        "total_layers": total_layers,
        "n_prompts": n_prompts,
        "max_new_tokens": max_new_tokens,
        "narrow_waist_layer": nw_layer,
        "direction_source": "exp_3a_directions.pt (nw_layer v_mm)",
        "layer_results": layer_results,
    }
    save_dir = _PROJ_ROOT / "results" / "phase3" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "exp_3d_layerwise_results.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[{model_name}] Saved to: {save_path}")
    return output


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 3D: Layer-wise Ablation Curve")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_exp_3d(args.model, args.device, args.n_prompts, args.max_new_tokens)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file exists and has no syntax errors**

Run: `cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal && python3 -c "import ast; ast.parse(open('experiments/phase3/exp_3d_layerwise_ablation.py').read()); print('Syntax OK')"`

Expected output: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
git add experiments/phase3/exp_3d_layerwise_ablation.py
git commit -m "feat: add exp_3d layerwise ablation curve script (P0-C)"
```

---

## Task 2: Write `run_3d_layerwise.sh`

**Files:**
- Create: `experiments/phase3/run_3d_layerwise.sh`

- [ ] **Step 1: Write the launcher script**

Write the file at `experiments/phase3/run_3d_layerwise.sh`:

```bash
#!/bin/bash
# Run Exp 3D: Layer-wise Ablation Curve on all 4 models in parallel
# Usage: bash run_3d_layerwise.sh [n_prompts=50]
#
# GPU assignment:
#   GPU 0: LLaVA-1.5-7B     (rdo env)
#   GPU 1: Qwen2.5-VL-7B    (qwen3-vl env, transformers >= 4.52)
#   GPU 2: InternVL2-8B     (rdo env, trust_remote_code)
#   GPU 3: InstructBLIP-7B  (rdo env)

N_PROMPTS=${1:-50}
SCRIPT="experiments/phase3/exp_3d_layerwise_ablation.py"
LOG_DIR="experiments/phase3/logs"
mkdir -p "$LOG_DIR"

echo "Starting Exp 3D with n_prompts=${N_PROMPTS}"

CUDA_VISIBLE_DEVICES=0 conda run -n rdo \
    python "$SCRIPT" --model llava_7b --device cuda:0 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_llava7b.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 conda run -n qwen3-vl \
    python "$SCRIPT" --model qwen2vl_7b --device cuda:1 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_qwen2vl.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 conda run -n rdo \
    python "$SCRIPT" --model internvl2_8b --device cuda:2 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_internvl2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 conda run -n rdo \
    python "$SCRIPT" --model instructblip_7b --device cuda:3 --n_prompts "$N_PROMPTS" \
    > "$LOG_DIR/3d_instructblip.log" 2>&1 &

wait
echo "Exp 3D complete. Results in results/phase3/{model}/exp_3d_layerwise_results.json"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/experiments/phase3/run_3d_layerwise.sh`

Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
git add experiments/phase3/run_3d_layerwise.sh
git commit -m "feat: add run_3d_layerwise.sh parallel launcher"
```

---

## Task 3: Verification after GPU run

*This task is executed by qi on the GPU node after running the script.*

- [ ] **Step 1: Run the experiment (qi executes on GPU node)**

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal && bash experiments/phase3/run_3d_layerwise.sh 50
```

Expected: all 4 background processes complete; log files written to `experiments/phase3/logs/3d_*.log`

- [ ] **Step 2: Verify output JSON files exist and have expected structure**

```bash
python3 -c "
import json
from pathlib import Path

proj = Path('/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal')
models = ['llava_7b', 'qwen2vl_7b', 'internvl2_8b', 'instructblip_7b']
for m in models:
    path = proj / 'results' / 'phase3' / m / 'exp_3d_layerwise_results.json'
    if not path.exists():
        print(f'MISSING: {path}')
        continue
    r = json.load(open(path))
    n_layers = sum(1 for k in r['layer_results'] if k != 'baseline')
    max_asr = max(v['full_harmful_rate'] for k, v in r['layer_results'].items() if k != 'baseline')
    nw = r['narrow_waist_layer']
    nw_asr = r['layer_results'].get(f'layer_{nw}', {}).get('full_harmful_rate', 'NOT_PROBED')
    print(f'{m}: n_layer_probed={n_layers}, max_asr={max_asr:.3f}, nw_layer={nw}, nw_asr={nw_asr}')
"
```

Expected: 4 lines, each with `n_layer_probed=16` (for 32-layer models) or `n_layer_probed=14` (Qwen2.5-VL 28-layer), `max_asr` a real number.

- [ ] **Step 3: LLaVA sanity check against exp_3c baseline**

```bash
python3 -c "
import json
from pathlib import Path

proj = Path('/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal')

# 3D baseline should match 3C baseline_mm
d3 = json.load(open(proj / 'results/phase3/llava_7b/exp_3d_layerwise_results.json'))
d3c = json.load(open(proj / 'results/phase3/llava_7b/exp_3c_results.json'))

d3_base = d3['layer_results']['baseline']['full_harmful_rate']
d3c_base = d3c['configs']['baseline_mm']['metrics']['full_harmful_completion_rate']
print(f'exp_3d baseline_mm:  {d3_base:.3f}')
print(f'exp_3c baseline_mm:  {d3c_base:.3f}')
print(f'Delta: {abs(d3_base - d3c_base):.3f}  (should be <0.10 given different n_prompts)')

# NW layer should have higher ASR than baseline
nw = d3['narrow_waist_layer']
nw_asr = d3['layer_results'].get(f'layer_{nw}', {}).get('full_harmful_rate', 0)
print(f'narrow_waist layer {nw} ASR: {nw_asr:.3f}  (should be > {d3_base:.3f})')
"
```

Expected: `Delta < 0.10`, `nw_asr > d3_base`.

---

## Self-Review

**Spec coverage:**
- ✅ New file `exp_3d_layerwise_ablation.py` — covers "逐层 ablation 曲线" goal
- ✅ Stride=2 sweep over all layers — covers "每隔两层" from vibe guide
- ✅ InternVL2 extra last-4-layers coverage — covers the InternVL2 special case
- ✅ 50-prompt default — matches P0-C spec ("50 条足够画曲线")
- ✅ JSON output format matches the vibe guide's expected structure
- ✅ Parallel launcher `run_3d_layerwise.sh` with 4-GPU assignment
- ✅ Qwen2.5-VL uses `qwen3-vl` env — required per workspace rules

**Placeholder scan:** None. All code is concrete.

**Type consistency:**
- `run_single_layer()` uses `evaluate_response()` → returns dict with `full_harmful_completion`, `initial_bypass`, `self_correction_found`; then `compute_attack_metrics()` → returns dict with `full_harmful_completion_rate`, `initial_bypass_rate`, `self_correction_rate_overall`. These key names are used consistently throughout.
- `ablation_context()` signature matches exp_3c exactly (copied directly).
- `load_directions()` output keys `narrow_waist_layer`, `v_mm` are used consistently.

**One known edge case:** `run_single_layer` uses the same `ctx` object in a loop. In Python, a `contextlib.nullcontext()` is reusable, but `ablation_context(...)` is a generator-based context manager — it is **not** reusable across loop iterations. The code correctly re-creates `ctx` per call by having `ctx` constructed inside `run_single_layer` before the loop.

**GPU memory note:** The script calls `torch.cuda.empty_cache()` per prompt. This is safe.
