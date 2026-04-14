"""
P0 Phase 3 (Step 2): Ablate RDO cone directions and generate responses.

Identical structure to exp_p0_dim_ablate.py but loads rdo_cone_k{k}.pt.
Saves to results/p0_cone/{model}/rdo_k{k}_responses.json

Requires: exp_p0_rdo_train.py must have run first.

Usage:
  python experiments/p0_cone/exp_p0_rdo_ablate.py --model llava_7b   --k 1
  python experiments/p0_cone/exp_p0_rdo_ablate.py --model llava_7b   --k 3
  python experiments/p0_cone/exp_p0_rdo_ablate.py --model llava_7b   --k 5
  python experiments/p0_cone/exp_p0_rdo_ablate.py --model qwen2vl_7b --k 1
"""
import sys
import os
import json
import torch
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../refusal_direction"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)

MODEL_PATHS = {
    "llava_7b": "/inspire/hdd/global_user/wenming-253108090054/models/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "qwen2vl_7b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct",
}


def build_ablation_hooks(model_base, cone_basis):
    """Build ablation hooks for all k directions on all layers."""
    n_layers = model_base.model.config.num_hidden_layers
    fwd_pre_hooks = []
    fwd_hooks = []
    for i in range(cone_basis.shape[0]):
        direction = cone_basis[i].to(
            device=model_base.model.device, dtype=model_base.model.dtype
        )
        for l in range(n_layers):
            fwd_pre_hooks.append((
                model_base.model_block_modules[l],
                get_direction_ablation_input_pre_hook(direction=direction),
            ))
            fwd_hooks.append((
                model_base.model_attn_modules[l],
                get_direction_ablation_output_hook(direction=direction),
            ))
            fwd_hooks.append((
                model_base.model_mlp_modules[l],
                get_direction_ablation_output_hook(direction=direction),
            ))
    return fwd_pre_hooks, fwd_hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
    parser.add_argument("--k", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    results_dir = os.path.join(project_root, "results", "p0_cone", args.model)
    os.makedirs(results_dir, exist_ok=True)

    cone_path = os.path.join(results_dir, f"rdo_cone_k{args.k}.pt")
    output_path = os.path.join(results_dir, f"rdo_k{args.k}_responses.json")
    data_path = os.path.join(project_root, "data", "saladbench_splits", "harmful_val.json")

    print(f"=== RDO Cone Ablation: {args.model}, k={args.k} ===")
    print(f"  Cone file : {cone_path}")

    # Protect existing results
    if os.path.exists(output_path):
        tag = datetime.now().strftime("%Y%m%d_%H%M")
        root, ext = os.path.splitext(output_path)
        output_path = f"{root}_{tag}{ext}"
        print(f"  [PROTECT] Output exists → saving to: {output_path}")
    else:
        print(f"  Output    : {output_path}")

    if not os.path.exists(cone_path):
        raise FileNotFoundError(
            f"RDO cone not found: {cone_path}\n"
            "Run exp_p0_rdo_train.py --cone_dim {k} first."
        )

    # Load cone
    cone_basis = torch.load(cone_path, weights_only=True, map_location="cpu")
    if cone_basis.dim() == 1:
        cone_basis = cone_basis.unsqueeze(0)
    assert cone_basis.shape[0] == args.k, (
        f"Expected {args.k} basis vectors, got {cone_basis.shape[0]}"
    )
    print(f"  cone_basis shape: {cone_basis.shape}")

    # Load eval data
    harmful_val = json.load(open(data_path))
    dataset = [
        {"instruction": item["instruction"], "category": item.get("category", "")}
        for item in harmful_val
    ]
    print(f"  Eval prompts: {len(dataset)}")

    # Load model
    model_base = construct_model_base(MODEL_PATHS[args.model])
    d_model = model_base.model.config.hidden_size
    assert cone_basis.shape[1] == d_model, (
        f"Cone d_model {cone_basis.shape[1]} ≠ model d_model {d_model}"
    )

    # Build hooks and generate
    fwd_pre_hooks, fwd_hooks = build_ablation_hooks(model_base, cone_basis)
    print(f"  Hooks: {len(fwd_pre_hooks)} pre, {len(fwd_hooks)} fwd")

    completions = model_base.generate_completions(
        dataset=dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=args.max_new_tokens,
    )

    output = {
        "model": args.model,
        "method": "rdo",
        "k": args.k,
        "n_prompts": len(completions),
        "max_new_tokens": args.max_new_tokens,
        "responses": completions,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  Saved {len(completions)} responses → {output_path}")


if __name__ == "__main__":
    main()
