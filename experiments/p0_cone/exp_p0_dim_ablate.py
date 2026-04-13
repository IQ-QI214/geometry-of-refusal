"""
P0 Phase 3: DIM cone ablation + generation for VLMs.

For each (model, k) combination:
1. Load the saved dim_cone_k{k}.pt cone basis (shape: k x d_model)
2. Build ablation hooks for ALL k directions across ALL layers
3. Generate responses on the harmful_val set with those hooks active
4. Save responses to results/p0_cone/{model}/dim_k{k}_responses.json

Usage:
  python experiments/p0_cone/exp_p0_dim_ablate.py --model llava_7b   --k 1
  python experiments/p0_cone/exp_p0_dim_ablate.py --model qwen2vl_7b --k 3
"""
import sys
import os
import json
import torch
import argparse

# Offline mode — must be set before any HF imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set up path so pipeline imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "llava_7b": (
        "/inspire/hdd/global_user/wenming-253108090054/models/hub/"
        "models--llava-hf--llava-1.5-7b-hf/snapshots/"
        "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
    ),
    "qwen2vl_7b": (
        "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct"
    ),
}


def build_ablation_hooks(model_base, cone_basis):
    """
    Build (fwd_pre_hooks, fwd_hooks) that ablate every direction in cone_basis
    from every layer, delegating to get_all_direction_ablation_hooks per direction.

    cone_basis: (k, d_model) tensor — each row is one basis direction.
    """
    all_pre_hooks = []
    all_fwd_hooks = []
    for i in range(cone_basis.shape[0]):
        # .clone() guards against the nonlocal-mutation pattern inside hook closures
        pre, fwd = get_all_direction_ablation_hooks(model_base, cone_basis[i].clone())
        all_pre_hooks.extend(pre)
        all_fwd_hooks.extend(fwd)
    return all_pre_hooks, all_fwd_hooks


def main():
    parser = argparse.ArgumentParser(
        description="DIM cone ablation + generation for P0 cone ablation experiment"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_PATHS.keys()), required=True,
        help="Which model to run: llava_7b or qwen2vl_7b",
    )
    parser.add_argument(
        "--k", type=int, choices=[1, 3, 5], required=True,
        help="Number of PCA cone directions to ablate",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root (two levels above this file)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
    results_dir = os.path.join(project_root, "results", "p0_cone", args.model)
    os.makedirs(results_dir, exist_ok=True)

    cone_path = os.path.join(results_dir, f"dim_cone_k{args.k}.pt")
    output_path = os.path.join(results_dir, f"dim_k{args.k}_responses.json")
    data_path = os.path.join(
        project_root, "data", "saladbench_splits", "harmful_val.json"
    )

    print(f"=== DIM Cone Ablation: {args.model}, k={args.k} ===")
    print(f"  Cone file : {cone_path}")
    print(f"  Output    : {output_path}")

    # Skip if already done
    if os.path.exists(output_path):
        print(f"  Output already exists — skipping. Remove to re-run.")
        return

    # -----------------------------------------------------------------------
    # Load eval data
    # -----------------------------------------------------------------------
    print("[1/4] Loading harmful_val data...")
    with open(data_path) as f:
        harmful_val = json.load(f)

    # generate_completions expects dicts with "instruction" and "category" keys
    dataset = [
        {"instruction": item["instruction"], "category": item.get("category", "")}
        for item in harmful_val
    ]
    print(f"  Loaded {len(dataset)} prompts.")

    # -----------------------------------------------------------------------
    # Load cone basis
    # -----------------------------------------------------------------------
    print("[2/4] Loading cone basis...")
    if not os.path.exists(cone_path):
        raise FileNotFoundError(
            f"Cone file not found: {cone_path}\n"
            f"Run exp_p0_dim_extract.py first to generate it."
        )
    cone_basis = torch.load(cone_path, map_location="cpu")  # (k, d_model)
    assert cone_basis.ndim == 2 and cone_basis.shape[0] == args.k, (
        f"Expected cone_basis shape ({args.k}, d_model), got {cone_basis.shape}"
    )
    print(f"  cone_basis shape: {cone_basis.shape}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("[3/4] Loading model...")
    model_base = construct_model_base(MODEL_PATHS[args.model])
    n_layers = model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size
    print(f"  Layers     : {n_layers}")
    print(f"  hidden_size: {d_model}")
    assert cone_basis.shape[1] == d_model, (
        f"cone_basis d_model mismatch: file has {cone_basis.shape[1]}, "
        f"model has {d_model}. Wrong cone file loaded?"
    )

    # -----------------------------------------------------------------------
    # Build ablation hooks (k directions x n_layers x 3 hook points)
    # -----------------------------------------------------------------------
    print("[4/4] Building ablation hooks and generating completions...")
    fwd_pre_hooks, fwd_hooks = build_ablation_hooks(model_base, cone_basis)
    print(f"  Pre-hooks  : {len(fwd_pre_hooks)}  (k={args.k} dirs x {n_layers} layers)")
    print(f"  Fwd-hooks  : {len(fwd_hooks)}  (k={args.k} dirs x {n_layers} layers x 2 [attn+mlp])")

    # generate_completions handles VLM kwargs (pixel_values etc.) internally
    completions = model_base.generate_completions(
        dataset=dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=args.max_new_tokens,
    )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "model": args.model,
        "method": "dim",
        "k": args.k,
        "n_prompts": len(completions),
        "max_new_tokens": args.max_new_tokens,
        "responses": completions,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  Saved {len(completions)} responses to: {output_path}")


if __name__ == "__main__":
    main()
