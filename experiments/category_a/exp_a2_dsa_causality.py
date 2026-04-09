"""
Exp A2: DSA Causality — Forced Generation Probe

Teacher-force harmful prefix (20 tokens), then free-generate under 3 conditions:
  Group A (control): no ablation
  Group B (ablation): architecture-optimal ablation with refusal direction
  Group C (random):   same layer strategy as B, but random direction (equal norm)

Group C mirrors Group B's layer strategy exactly — only the direction differs.

Usage:
  python exp_a2_dsa_causality.py --model llava_7b --device cuda:0
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

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "category_a"))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase2" / "common"))
# phase3 inserted last → becomes sys.path[0] → 'common' resolves to phase3/common
# (which has model_configs.py and model_adapters.py, not category_a/common)
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase3"))

from common.model_configs import MODEL_CONFIGS, load_model_by_name
from common.model_adapters import create_adapter, ModelAdapter
from eval_utils import evaluate_response, compute_attack_metrics


# ── Architecture-aware ablation config ───────────────────────────────────────

# Type I (Bottleneck): NW layer only
# Type II (Late Gate): all layers
# Type III (Diffuse): all layers
ABLATION_STRATEGY = {
    "llava_7b":     "nw",    # Type I
    "llava_13b":    "nw",    # Type I
    "qwen2vl_7b":   "all",   # Type II
    "qwen2vl_32b":  "all",   # Type II
    "internvl2_8b": "all",   # Type III
}

# Which A1 config to source harmful prefixes from
PREFIX_SOURCE_CONFIG = {
    "llava_7b":     "ablation_nw_vmm",
    "llava_13b":    "ablation_nw_vmm",
    "qwen2vl_7b":   "ablation_all_vmm",
    "qwen2vl_32b":  "ablation_all_vmm",
    "internvl2_8b": "baseline_mm",  # ablation ineffective, use baseline successes
}


# ── Generation-only ablation hook ────────────────────────────────────────────

class GenerationOnlyAblationHook:
    """Only ablate during autoregressive steps (seq_len==1), not during prefill."""

    def __init__(self, direction: torch.Tensor):
        self.direction = direction

    def __call__(self, module, args):
        activation = args[0] if isinstance(args, tuple) else args
        if activation.shape[1] > 1:  # prefill phase
            return args
        d = self.direction.to(activation.device, activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_act = activation - proj
        return (new_act,) + args[1:] if isinstance(args, tuple) else new_act


class GenerationOnlyOutputHook:
    """Output hook variant: only ablate during generation steps."""

    def __init__(self, direction: torch.Tensor):
        self.direction = direction

    def __call__(self, module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        if activation.shape[1] > 1:  # prefill
            return output
        d = self.direction.to(activation.device, activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_act = activation - proj
        return (new_act,) + output[1:] if isinstance(output, tuple) else new_act


@contextlib.contextmanager
def generation_only_ablation_context(
    adapter: ModelAdapter, direction: torch.Tensor,
    target_layers: List[int]
):
    """Register generation-only ablation hooks on specified layers."""
    llm_layers = adapter.get_llm_layers()
    handles = []
    try:
        for idx in target_layers:
            layer = llm_layers[idx]
            handles.append(layer.register_forward_pre_hook(
                GenerationOnlyAblationHook(direction)))
            if hasattr(layer, 'self_attn'):
                handles.append(layer.self_attn.register_forward_hook(
                    GenerationOnlyOutputHook(direction)))
            if hasattr(layer, 'mlp'):
                handles.append(layer.mlp.register_forward_hook(
                    GenerationOnlyOutputHook(direction)))
        yield
    finally:
        for h in handles:
            h.remove()


# ── Direction loading ────────────────────────────────────────────────────────

def load_directions(model_name: str) -> dict:
    directions_path = (_PROJ_ROOT / "results" / "phase3" / model_name
                       / "exp_3a_directions.pt")
    if not directions_path.exists():
        raise FileNotFoundError(
            f"Directions not found: {directions_path}. Run Exp 3A first.")
    data = torch.load(directions_path, map_location="cpu")
    nw_layer = data["narrow_waist_layer"]
    v_mm = data["directions"][nw_layer]["v_mm"]
    return {"narrow_waist_layer": nw_layer, "v_mm": v_mm}


# ── Harmful prefix extraction ────────────────────────────────────────────────

def extract_harmful_prefixes(
    model_name: str, tokenizer, n_prefix_tokens: int = 20,
    dataset_name: str = "saladbench",
) -> List[Dict]:
    """
    Extract (prompt, prefix_token_ids) pairs from A1 successful harmful completions.
    """
    config_name = PREFIX_SOURCE_CONFIG[model_name]
    result_path = (_PROJ_ROOT / "results" / "category_a" / model_name
                   / f"a1_{config_name}_{dataset_name}.json")

    if not result_path.exists():
        raise FileNotFoundError(
            f"A1 results not found: {result_path}. Run A1 first.")

    with open(result_path) as f:
        data = json.load(f)

    responses = data.get("responses", data)
    pairs = []
    for r in responses:
        if r.get("full_harmful_completion", False):
            response_text = r["response"]
            tokens = tokenizer.encode(response_text, add_special_tokens=False)
            if len(tokens) >= n_prefix_tokens:
                pairs.append({
                    "prompt": r["prompt"],
                    "prefix_ids": tokens[:n_prefix_tokens],
                    "prefix_text": tokenizer.decode(tokens[:n_prefix_tokens]),
                })

    print(f"[prefixes] Found {len(pairs)} harmful prefixes from {config_name}")
    if len(pairs) < 50:
        print(f"[WARNING] Only {len(pairs)} pairs available (target: 50)")

    return pairs


# ── Main experiment ──────────────────────────────────────────────────────────

def run_a2(model_name: str, device: str, max_new_tokens: int = 180,
           n_prefix_tokens: int = 20, seed: int = 42):

    cfg = MODEL_CONFIGS[model_name]
    total_layers = cfg["total_layers"]
    strategy = ABLATION_STRATEGY[model_name]

    save_dir = _PROJ_ROOT / "results" / "category_a" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exp A2: DSA Causality — {model_name}")
    print(f"Ablation strategy: {strategy}")
    print(f"{'='*60}")

    # Load model
    model, processor = load_model_by_name(model_name, device)
    adapter = create_adapter(model_name, model, processor, device)
    blank_image = Image.new("RGB", cfg["blank_image_size"], color=(128, 128, 128))

    # Load directions
    dir_data = load_directions(model_name)
    nw_layer = dir_data["narrow_waist_layer"]
    v_mm = dir_data["v_mm"]

    # Determine target layers based on architecture type
    if strategy == "nw":
        target_layers = [nw_layer]
    else:  # "all"
        target_layers = list(range(total_layers))

    # Random direction (same norm as v_mm)
    torch.manual_seed(seed)
    random_dir = torch.randn_like(v_mm)
    random_dir = random_dir / random_dir.norm() * v_mm.norm()

    # Get tokenizer
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    # Extract harmful prefixes from A1 results
    pairs = extract_harmful_prefixes(model_name, tokenizer, n_prefix_tokens)

    # 3 groups: (name, direction, layers)
    groups = [
        ("group_a_control",  None,       None),
        ("group_b_ablation", v_mm,       target_layers),
        ("group_c_random",   random_dir, target_layers),
    ]

    all_results = {}

    for group_name, direction, layers in groups:
        print(f"\n{'─'*50}")
        print(f"Group: {group_name}")

        group_responses = []

        for j, pair in enumerate(pairs):
            prompt = pair["prompt"]
            prefix_ids = pair["prefix_ids"]

            # Build input: chat template + harmful prefix
            mm_inputs = adapter.prepare_mm_inputs(prompt, blank_image)
            input_ids = mm_inputs["input_ids"]

            # Append prefix tokens
            prefix_tensor = torch.tensor([prefix_ids], device=input_ids.device)
            full_input_ids = torch.cat([input_ids, prefix_tensor], dim=1)
            mm_inputs["input_ids"] = full_input_ids

            # Extend attention_mask if present
            if "attention_mask" in mm_inputs:
                prefix_mask = torch.ones(
                    1, len(prefix_ids),
                    device=mm_inputs["attention_mask"].device,
                    dtype=mm_inputs["attention_mask"].dtype,
                )
                mm_inputs["attention_mask"] = torch.cat(
                    [mm_inputs["attention_mask"], prefix_mask], dim=1)

            # Generate with conditional ablation
            if direction is not None:
                ctx = generation_only_ablation_context(adapter, direction, layers)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                with torch.no_grad():
                    pad_id = (tokenizer.eos_token_id
                              if hasattr(tokenizer, 'eos_token_id') else 2)
                    gen_ids = adapter.model.generate(
                        **mm_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )

            # Decode only new tokens (after full input)
            new_ids = gen_ids[0][full_input_ids.shape[1]:]
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
            full_response = pair["prefix_text"] + continuation

            eval_result = evaluate_response(full_response)
            eval_result["prompt"] = prompt
            eval_result["prefix_text"] = pair["prefix_text"]
            eval_result["continuation"] = continuation[:300]
            eval_result["pair_idx"] = j
            group_responses.append(eval_result)

            if (j + 1) % 10 == 0:
                sc_count = sum(1 for r in group_responses if r["self_correction_found"])
                print(f"  [{j+1}/{len(pairs)}] SC so far: {sc_count}/{j+1}")

        metrics = compute_attack_metrics(group_responses)
        all_results[group_name] = {
            "metrics": metrics,
            "responses": group_responses,
        }

        print(f"  SCR={metrics['self_correction_rate_overall']:.3f}  "
              f"FHCR={metrics['full_harmful_completion_rate']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"A2 Summary — {model_name} (strategy={strategy})")
    for gn, gd in all_results.items():
        m = gd["metrics"]
        print(f"  {gn:<25} SCR={m['self_correction_rate_overall']:.3f}  "
              f"FHCR={m['full_harmful_completion_rate']:.3f}")

    # Save
    output = {
        "model": model_name,
        "ablation_strategy": strategy,
        "target_layers": target_layers,
        "narrow_waist_layer": nw_layer,
        "n_prefix_tokens": n_prefix_tokens,
        "n_pairs": len(pairs),
        "seed": seed,
        "groups": {gn: {"metrics": gd["metrics"], "responses": gd["responses"]}
                   for gn, gd in all_results.items()},
    }
    output_path = save_dir / "a2_causality.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp A2: DSA Causality")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--n_prefix_tokens", type=int, default=20)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_a2(args.model, args.device, args.max_new_tokens, args.n_prefix_tokens)


if __name__ == "__main__":
    main()
