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

    # Register hooks once around the whole loop (more efficient + correct)
    if layer_idx is not None:
        ctx = ablation_context(adapter, direction, [layer_idx])
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        for prompt in prompts:
            torch.cuda.empty_cache()
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
