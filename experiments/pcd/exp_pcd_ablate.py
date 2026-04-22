"""
PCD Experiment: Ablation + Generation (Task 6, Script 2)

Loads best_layer.json from --sweep_dir, re-extracts the direction at
(best_layer, best_pos), applies orthogonal weight ablation, then generates
completions on harmful_val (n=128 by default).

Saves dim_responses.json to --output_dir.

Usage (from project root with PYTHONPATH=refusal_direction):
  python experiments/pcd/exp_pcd_ablate.py \
      --model_name qwen2vl_7b \
      --model_path /path/to/Qwen2.5-VL-7B \
      --condition V-blank \
      --sweep_dir results/pcd/qwen2vl_7b/V-blank/sweep \
      --output_dir results/pcd/qwen2vl_7b/V-blank
"""

import sys
import os
import json
import argparse
import functools
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../"))
_REFUSAL_DIR = os.path.join(_PROJECT_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import get_mean_diff

_CONDITION_IMAGE_MODE = {
    "V-text":  "text",
    "V-blank": "blank",
    "V-noise": "noise",
}


def _override_image_mode(model_base, image_mode: str):
    """Re-bind tokenize_instructions_fn with the given image_mode."""
    model_base.tokenize_instructions_fn = functools.partial(
        model_base.tokenize_instructions_fn,
        image_mode=image_mode,
    )


def main():
    parser = argparse.ArgumentParser(
        description="PCD ablation + generation for one condition"
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--condition", required=True,
                        choices=["L", "V-text", "V-blank", "V-noise"])
    parser.add_argument("--sweep_dir", required=True,
                        help="Directory containing best_layer.json (and mean_diffs.pt)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save dim_responses.json")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--n_val", type=int, default=128,
                        help="Number of harmful_val prompts to generate on")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = args.data_dir or os.path.join(_PROJECT_ROOT, "data", "saladbench_splits")

    print(f"=== PCD Ablation: model={args.model_name}  condition={args.condition} ===")
    print(f"  sweep_dir  : {args.sweep_dir}")
    print(f"  output_dir : {args.output_dir}")

    # -----------------------------------------------------------------------
    # Load best_layer.json
    # -----------------------------------------------------------------------
    best_layer_path = os.path.join(args.sweep_dir, "best_layer.json")
    if not os.path.exists(best_layer_path):
        raise FileNotFoundError(
            f"best_layer.json not found at {best_layer_path}. "
            "Run exp_pcd_layer_sweep.py first."
        )
    with open(best_layer_path) as f:
        best_info = json.load(f)
    best_layer = int(best_info["layer"])
    best_pos   = int(best_info["pos"])
    print(f"  Loaded best_layer.json: layer={best_layer}, pos={best_pos}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    def _load(fname):
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            return json.load(f)

    harmful_train_raw  = _load("harmful_train.json")
    harmless_train_raw = _load("harmless_train.json")
    harmful_val_raw    = _load("harmful_val.json")

    # Balance train instructions (strings)
    n_train = min(len(harmful_train_raw), len(harmless_train_raw))
    harmful_train_inst  = [d["instruction"] for d in harmful_train_raw[:n_train]]
    harmless_train_inst = [d["instruction"] for d in harmless_train_raw[:n_train]]

    # Val: keep full dicts for generate_completions (needs 'category' key)
    harmful_val_dataset = harmful_val_raw[:args.n_val]
    print(f"  Train: {len(harmful_train_inst)} harmful / {len(harmless_train_inst)} harmless")
    print(f"  Val  : {len(harmful_val_dataset)} prompts")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("[1/3] Loading model ...")
    model_base = construct_model_base(args.model_path, args.model_name)

    # Override image_mode for VLM conditions
    image_mode = _CONDITION_IMAGE_MODE.get(args.condition)
    if image_mode is not None:
        print(f"  Overriding tokenize_instructions_fn with image_mode='{image_mode}'")
        _override_image_mode(model_base, image_mode)

    # -----------------------------------------------------------------------
    # Step 1: Re-extract direction at best (pos, layer) using full train data
    #
    # get_mean_diff returns mean_diffs of shape (n_positions, n_layers, d_model)
    # where n_positions corresponds to the positions list passed.
    # We need ONE position and ONE layer, so we pass positions=[best_pos] and
    # then index [0, best_layer] to get the direction vector.
    # -----------------------------------------------------------------------
    print("[2/3] Re-extracting direction at best (pos, layer) ...")
    mean_diffs = get_mean_diff(
        model_base.model,
        model_base.tokenizer,
        harmful_train_inst,
        harmless_train_inst,
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        batch_size=args.batch_size,
        positions=[best_pos],   # single position → shape (1, n_layers, d_model)
    )
    # mean_diffs shape: (1, n_layers, d_model)
    direction = mean_diffs[0, best_layer].float()          # (d_model,)
    direction = direction / direction.norm().clamp(min=1e-8)
    direction = direction.to(model_base.model.device)
    print(f"  direction shape: {direction.shape}, norm={direction.norm().item():.4f}")

    # -----------------------------------------------------------------------
    # Step 2: Apply orthogonal weight ablation
    #
    # _get_orthogonalization_mod_fn(direction) returns a fn(model) that
    # modifies model weights in-place (orthogonalizes w.r.t. direction).
    # -----------------------------------------------------------------------
    print("[3/3] Applying orthogonal weight ablation ...")
    orthogonalize_fn = model_base._get_orthogonalization_mod_fn(direction)
    orthogonalize_fn(model_base.model)
    print("  Ablation applied.")

    # -----------------------------------------------------------------------
    # Step 3: Generate completions on harmful_val
    # generate_completions expects dataset of dicts with 'instruction' and 'category'
    # -----------------------------------------------------------------------
    print(f"  Generating on {len(harmful_val_dataset)} harmful_val prompts ...")
    # Temporarily set max_batch_size if not set
    if model_base.max_batch_size is None:
        model_base.max_batch_size = args.gen_batch_size

    completions = model_base.generate_completions(
        dataset=harmful_val_dataset,
        max_new_tokens=args.max_new_tokens,
    )

    # -----------------------------------------------------------------------
    # Build and save response JSON
    # -----------------------------------------------------------------------
    responses_out = []
    for i, comp in enumerate(completions):
        responses_out.append({
            "instruction": comp.get("prompt") or harmful_val_dataset[i]["instruction"],
            "category":    comp.get("category") or harmful_val_dataset[i].get("category", ""),
            "response":    comp.get("response", ""),
        })

    result = {
        "model":     args.model_name,
        "condition": args.condition,
        "layer":     best_layer,
        "pos":       best_pos,
        "n_prompts": len(responses_out),
        "responses": responses_out,
    }

    out_path = os.path.join(args.output_dir, "dim_responses.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(responses_out)} responses to {out_path}")

    print("=== Ablation Complete ===")


if __name__ == "__main__":
    main()
