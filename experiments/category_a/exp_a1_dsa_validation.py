"""
Exp A1: DSA Widespread Validation

5 models × 5 configs × N prompts (572 SaladBench or 400 HarmBench).
Generates full responses, evaluates with keyword matching.
Qwen3Guard/LlamaGuard3 evaluation done separately by exp_a1_judge.py.

Usage:
  python exp_a1_dsa_validation.py --model llava_7b --device cuda:0
  python exp_a1_dsa_validation.py --model qwen2vl_32b --device cuda:0 --dataset saladbench
  python exp_a1_dsa_validation.py --model llava_7b --device cuda:0 --resume
"""

import os
import sys
import argparse
import json
import contextlib
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase2" / "common"))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "category_a"))
# phase3 inserted last → becomes sys.path[0] → 'common' resolves to phase3/common
# (which has model_configs.py and model_adapters.py)
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase3"))

from common.model_configs import MODEL_CONFIGS, load_model_by_name
from common.model_adapters import create_adapter, ModelAdapter
from eval_utils import evaluate_response, compute_attack_metrics
# Import data_utils directly from category_a/common (avoids 'common' namespace conflict)
import importlib.util as _ilu
_du_spec = _ilu.spec_from_file_location(
    "cat_a_data_utils",
    _PROJ_ROOT / "experiments" / "category_a" / "common" / "data_utils.py")
_du_mod = _ilu.module_from_spec(_du_spec)
_du_spec.loader.exec_module(_du_mod)
load_dataset = _du_mod.load_dataset


# ── Ablation hooks (copied from exp_3c for self-containment) ────────────────

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
                handles.append(layer.self_attn.register_forward_hook(
                    _make_output_hook(direction)))
            if hasattr(layer, 'mlp'):
                handles.append(layer.mlp.register_forward_hook(
                    _make_output_hook(direction)))
        yield
    finally:
        for h in handles:
            h.remove()


# ── Direction loading ────────────────────────────────────────────────────────

def load_directions(model_name: str) -> dict:
    """Load Exp 3A directions. Returns {narrow_waist_layer, v_text, v_mm}."""
    directions_path = (_PROJ_ROOT / "results" / "phase3" / model_name
                       / "exp_3a_directions.pt")
    if not directions_path.exists():
        raise FileNotFoundError(
            f"Directions not found: {directions_path}. Run Exp 3A first.")
    data = torch.load(directions_path, map_location="cpu", weights_only=False)
    nw_layer = data["narrow_waist_layer"]
    v_text = data["directions"][nw_layer]["v_text"]
    v_mm = data["directions"][nw_layer]["v_mm"]
    print(f"[directions] NW layer={nw_layer}, "
          f"v_text norm={v_text.norm():.4f}, v_mm norm={v_mm.norm():.4f}")
    return {"narrow_waist_layer": nw_layer, "v_text": v_text, "v_mm": v_mm}


# ── Generation ───────────────────────────────────────────────────────────────

def generate_response(adapter: ModelAdapter, prompt: str, image,
                      direction: Optional[torch.Tensor],
                      target_layers: Optional[List[int]],
                      max_new_tokens: int = 200) -> str:
    torch.cuda.empty_cache()
    if direction is not None:
        ctx = ablation_context(adapter, direction, target_layers)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        if image is not None:
            return adapter.generate_mm(prompt, image, max_new_tokens=max_new_tokens)
        else:
            return adapter.generate_text(prompt, max_new_tokens=max_new_tokens)


# ── Resume support ───────────────────────────────────────────────────────────

def load_progress(progress_path: Path) -> Dict[str, int]:
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {}


def save_progress(progress_path: Path, progress: Dict[str, int]):
    with open(progress_path, "w") as f:
        json.dump(progress, f)


def load_partial_results(result_path: Path) -> List[Dict]:
    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
            # Handle both raw list and dict-with-responses format
            if isinstance(data, list):
                return data
            return data.get("responses", [])
    return []


def save_partial_results(result_path: Path, results: List[Dict]):
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_a1(model_name: str, device: str, dataset_name: str = "saladbench",
           max_new_tokens: int = 200, resume: bool = False, n_prompts_limit: int = 0):

    cfg = MODEL_CONFIGS[model_name]
    total_layers = cfg["total_layers"]
    prompts_data = load_dataset(dataset_name)
    prompts = [item["instruction"] for item in prompts_data]
    if n_prompts_limit > 0:
        prompts = prompts[:n_prompts_limit]
    n_prompts = len(prompts)

    save_dir = _PROJ_ROOT / "results" / "category_a" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    progress_path = save_dir / f"a1_progress_{dataset_name}.json"

    print(f"\n{'='*60}")
    print(f"Exp A1: DSA Validation — {model_name}")
    print(f"Dataset: {dataset_name} ({n_prompts} prompts)")
    print(f"{'='*60}")

    # Load model
    model, processor = load_model_by_name(model_name, device)
    adapter = create_adapter(model_name, model, processor, device)
    blank_image = Image.new("RGB", cfg["blank_image_size"], color=(128, 128, 128))

    # Load directions
    dir_data = load_directions(model_name)
    nw_layer = dir_data["narrow_waist_layer"]
    v_text = dir_data["v_text"]
    v_mm = dir_data["v_mm"]
    all_layers = list(range(total_layers))

    # Configs: (name, image, direction, target_layers)
    configs = [
        ("baseline_text",      None,        None,   None),
        ("baseline_mm",        blank_image, None,   None),
        ("ablation_nw_vmm",    blank_image, v_mm,   [nw_layer]),
        ("ablation_all_vmm",   blank_image, v_mm,   all_layers),
        ("ablation_nw_vtext",  blank_image, v_text, [nw_layer]),
    ]

    progress = load_progress(progress_path) if resume else {}

    for config_name, image, direction, target_layers in configs:
        result_path = save_dir / f"a1_{config_name}_{dataset_name}.json"
        start_idx = progress.get(config_name, 0) if resume else 0

        if start_idx >= n_prompts:
            print(f"\n[{config_name}] Already complete, skipping.")
            continue

        existing = load_partial_results(result_path) if resume and start_idx > 0 else []
        print(f"\n{'─'*50}")
        print(f"Config: {config_name} (starting from idx={start_idx})")

        for i in range(start_idx, n_prompts):
            prompt = prompts[i]
            response = generate_response(
                adapter, prompt, image, direction, target_layers, max_new_tokens)

            eval_result = evaluate_response(response)
            eval_result["prompt"] = prompt
            eval_result["response"] = response  # full response, no truncation
            eval_result["prompt_idx"] = i
            existing.append(eval_result)

            status = ("FULL_HARMFUL" if eval_result["full_harmful_completion"]
                      else ("bypassed" if eval_result["initial_bypass"] else "refused"))
            if eval_result["self_correction_found"]:
                status += f"+sc@{eval_result['self_correction_pos_chars']}"

            if (i + 1) % 50 == 0 or i == n_prompts - 1:
                print(f"  [{i+1}/{n_prompts}] last={prompt[:40]}... → {status}")
                save_partial_results(result_path, existing)
                progress[config_name] = i + 1
                save_progress(progress_path, progress)

        # Final metrics
        metrics = compute_attack_metrics(existing)
        print(f"  IBR={metrics['initial_bypass_rate']:.3f}  "
              f"SCR={metrics['self_correction_rate_overall']:.3f}  "
              f"FHCR_kw={metrics['full_harmful_completion_rate']:.3f}")

        # Save with metrics header
        final = {"config": config_name, "model": model_name,
                 "dataset": dataset_name, "n_prompts": n_prompts,
                 "metrics_kw": metrics, "responses": existing}
        with open(result_path, "w") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        progress[config_name] = n_prompts
        save_progress(progress_path, progress)

    print(f"\n[{model_name}] A1 complete. Results in {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Exp A1: DSA Validation")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="saladbench",
                        choices=["saladbench", "harmbench"])
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_prompts", type=int, default=0,
                        help="Limit number of prompts (0 = use all). Use 10 for smoke test.")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_a1(args.model, args.device, args.dataset, args.max_new_tokens, args.resume, args.n_prompts)


if __name__ == "__main__":
    main()
