"""
PCD Experiment: Layer x Position Sweep (Task 6, Script 1)

Runs the DIM layer×pos sweep for one (model, condition) pair.
Saves mean_diffs.pt and best_layer.json to --output_dir.

Conditions:
  L        — language-only (no image)
  V-text   — VLM with image_mode='text' (no image pixel block)
  V-blank  — VLM with image_mode='blank' (white image)
  V-noise  — VLM with image_mode='noise' (random noise image)

Usage (from project root with PYTHONPATH=refusal_direction):
  python experiments/pcd/exp_pcd_layer_sweep.py \
      --model_name qwen2vl_7b \
      --model_path /path/to/Qwen2.5-VL-7B \
      --condition V-blank \
      --output_dir results/pcd/qwen2vl_7b/V-blank/sweep
"""

import sys
import os
import json
import argparse
import functools
import torch

# Allow running from project root with PYTHONPATH=refusal_direction
# (also handle direct invocation from any CWD)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../"))
_REFUSAL_DIR = os.path.join(_PROJECT_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction

# --------------------------------------------------------------------------- #
# Condition → image_mode mapping                                              #
# --------------------------------------------------------------------------- #
# For VLM conditions, we override tokenize_instructions_fn using functools.partial
# with the appropriate image_mode kwarg.  For L (language-only), no override.
_CONDITION_IMAGE_MODE = {
    "V-text":  "text",
    "V-blank": "blank",
    "V-noise": "noise",
}


def _override_image_mode(model_base, image_mode: str):
    """Re-bind tokenize_instructions_fn with the given image_mode.

    The existing fn was built via functools.partial(...) in _get_tokenize_instructions_fn.
    We create a new partial that adds/overrides the image_mode kwarg.
    """
    model_base.tokenize_instructions_fn = functools.partial(
        model_base.tokenize_instructions_fn,
        image_mode=image_mode,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="PCD layer×pos sweep for one (model, condition) pair"
    )
    parser.add_argument("--model_name", required=True,
                        help="Logical model name passed to construct_model_base "
                             "(e.g. 'qwen2vl_7b', 'gemma-3-4b-it-vlm')")
    parser.add_argument("--model_path", required=True,
                        help="Filesystem path to model weights")
    parser.add_argument("--condition", required=True,
                        choices=["L", "V-text", "V-blank", "V-noise"],
                        help="Condition label")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save mean_diffs.pt + best_layer.json")
    parser.add_argument("--data_dir",
                        default=None,
                        help="Path to saladbench_splits directory "
                             "(default: <project_root>/data/saladbench_splits)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--select_batch_size", type=int, default=8)
    parser.add_argument("--induce_refusal_threshold", type=float, default=0.0)
    parser.add_argument("--kl_threshold", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Data paths
    # -----------------------------------------------------------------------
    data_dir = args.data_dir or os.path.join(_PROJECT_ROOT, "data", "saladbench_splits")

    print(f"=== PCD Layer Sweep: model={args.model_name}  condition={args.condition} ===")
    print(f"  output_dir : {args.output_dir}")
    print(f"  data_dir   : {data_dir}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    def _load(fname):
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            return json.load(f)

    harmful_train  = [d["instruction"] for d in _load("harmful_train.json")]
    harmless_train = [d["instruction"] for d in _load("harmless_train.json")]
    harmful_val    = [d["instruction"] for d in _load("harmful_val.json")]
    harmless_val   = [d["instruction"] for d in _load("harmless_val.json")]

    # Balance train sets
    n_train = min(len(harmful_train), len(harmless_train))
    harmful_train  = harmful_train[:n_train]
    harmless_train = harmless_train[:n_train]

    print(f"  Train: {len(harmful_train)} harmful / {len(harmless_train)} harmless")
    print(f"  Val  : {len(harmful_val)} harmful / {len(harmless_val)} harmless")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("[1/3] Loading model...")
    model_base = construct_model_base(args.model_path, args.model_name)
    print(f"  Layers     : {len(model_base.model_block_modules)}")
    print(f"  hidden_size: {getattr(model_base.model.config, 'hidden_size', getattr(getattr(model_base.model.config, 'text_config', None), 'hidden_size', '?'))}")
    print(f"  EOI toks   : {model_base.eoi_toks}")

    # -----------------------------------------------------------------------
    # Override image_mode for VLM conditions
    # -----------------------------------------------------------------------
    image_mode = _CONDITION_IMAGE_MODE.get(args.condition)
    if image_mode is not None:
        print(f"  Overriding tokenize_instructions_fn with image_mode='{image_mode}'")
        _override_image_mode(model_base, image_mode)

    # -----------------------------------------------------------------------
    # Step 1: generate_directions (mean_diffs)
    #   Signature: generate_directions(model_base, harmful, harmless, artifact_dir)
    #   Returns  : mean_diffs  shape=(n_eoi_toks, n_layers, d_model)
    #   Also saves: artifact_dir/mean_diffs.pt
    # -----------------------------------------------------------------------
    mean_diffs_cache = os.path.join(args.output_dir, "mean_diffs.pt")
    if os.path.exists(mean_diffs_cache):
        print(f"[2/3] Loading cached mean_diffs from {mean_diffs_cache} ...")
        mean_diffs = torch.load(mean_diffs_cache, map_location="cpu")
    else:
        print("[2/3] Generating mean_diffs (generate_directions) on train set ...")
        mean_diffs = generate_directions(
            model_base,
            harmful_train,
            harmless_train,
            args.output_dir,  # artifact_dir — generate_directions saves mean_diffs.pt here
        )
    print(f"  mean_diffs shape: {mean_diffs.shape}")

    # -----------------------------------------------------------------------
    # Step 2: select_direction
    #   Signature: select_direction(model_base, harmful, harmless,
    #                               candidate_directions, artifact_dir,
    #                               kl_threshold, induce_refusal_threshold, ...)
    #   Returns  : (pos, layer, direction)
    #   pos is a NEGATIVE integer, e.g. -1 or -2
    # -----------------------------------------------------------------------
    print("[3/3] Selecting best (pos, layer) via select_direction on val set ...")
    filter_passed = True
    try:
        best_pos, best_layer, best_direction = select_direction(
            model_base,
            harmful_val,
            harmless_val,
            mean_diffs,
            artifact_dir=args.output_dir,
            kl_threshold=args.kl_threshold,
            induce_refusal_threshold=args.induce_refusal_threshold,
            batch_size=args.select_batch_size,
        )
    except AssertionError as exc:
        # "All scores have been filtered out!" — fall back to best unfiltered
        print(f"  WARNING: select_direction filter assertion: {exc}")
        print("  Falling back: loading direction_evaluations.json for best unfiltered score.")
        evals_path = os.path.join(args.output_dir, "direction_evaluations.json")
        with open(evals_path) as f:
            evals = json.load(f)
        # Sort by lowest refusal_score (best ablation)
        evals_sorted = sorted(evals, key=lambda e: e["refusal_score"])
        best_entry = evals_sorted[0]
        best_pos   = int(best_entry["position"])
        best_layer = int(best_entry["layer"])
        filter_passed = False
        print(f"  Fallback selected: pos={best_pos}, layer={best_layer}")

    print(f"  Selected: pos={best_pos}, layer={best_layer}, filter_passed={filter_passed}")

    # -----------------------------------------------------------------------
    # Save best_layer.json
    # -----------------------------------------------------------------------
    best_layer_info = {
        "layer":         int(best_layer),
        "pos":           int(best_pos),
        "filter_passed": filter_passed,
        "condition":     args.condition,
    }
    out_path = os.path.join(args.output_dir, "best_layer.json")
    with open(out_path, "w") as f:
        json.dump(best_layer_info, f, indent=2)
    print(f"  Saved best_layer.json: {best_layer_info}")

    print("=== Layer Sweep Complete ===")


if __name__ == "__main__":
    main()
