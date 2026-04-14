"""
P0 Phase 3 (Step 0): Generate RDO training targets for VLMs.

Must run BEFORE exp_p0_rdo_train.py.

For each model:
1. harmful_targets: ablate DIM k=1 direction on all layers → model speaks freely
                    save first 30-token completion as "ablation" target
2. harmless_targets:
   - "addition": add DIM k=1 direction at add_layer → model should refuse
   - "retain"  : no hooks → model's baseline response

Usage:
  python experiments/p0_cone/exp_p0_rdo_targets.py --model llava_7b   --device cuda:0
  python experiments/p0_cone/exp_p0_rdo_targets.py --model qwen2vl_7b --device cuda:0
"""
import sys
import os
import json
import torch
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../refusal_direction"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_all_direction_ablation_hooks,
    get_activation_addition_input_pre_hook,
)

MODEL_PATHS = {
    "llava_7b": "/inspire/hdd/global_user/wenming-253108090054/models/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "qwen2vl_7b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct",
}

NUM_TARGET_TOKENS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    model_dir = os.path.join(project_root, "results", "p0_cone", args.model)
    dim_dir = os.path.join(model_dir, "dim")       # DIM cone files live here
    targets_dir = os.path.join(model_dir, "rdo", "targets")  # RDO targets go here
    os.makedirs(targets_dir, exist_ok=True)

    harmful_targets_path = os.path.join(targets_dir, "harmful_targets.json")
    harmless_targets_path = os.path.join(targets_dir, "harmless_targets.json")

    if os.path.exists(harmful_targets_path) and os.path.exists(harmless_targets_path):
        print(f"Targets already exist in {targets_dir}, skipping.")
        return

    print(f"=== RDO Target Generation: {args.model} ===")

    # Load DIM direction and metadata (from dim/ subdir)
    dim_direction = torch.load(
        os.path.join(dim_dir, "dim_cone_k1.pt"), weights_only=True, map_location="cpu"
    )
    if dim_direction.dim() == 2:
        dim_direction = dim_direction[0]  # (d_model,)

    meta = json.load(open(os.path.join(dim_dir, "dim_metadata.json")))
    add_layer = meta["layer"]
    alpha = dim_direction.norm().item()
    print(f"  DIM direction norm (alpha): {alpha:.4f}, add_layer: {add_layer}")

    # Load model
    print("[1/4] Loading model...")
    model_base = construct_model_base(MODEL_PATHS[args.model])

    # Move direction to model device/dtype for hooks
    direction_dev = dim_direction.to(device=model_base.model.device, dtype=model_base.model.dtype)

    # Load data
    data_dir = os.path.join(project_root, "data", "saladbench_splits")
    harmful_train = json.load(open(os.path.join(data_dir, "harmful_train.json")))
    harmless_train = json.load(open(os.path.join(data_dir, "harmless_train.json")))
    n = min(len(harmful_train), len(harmless_train))
    harmful_train = harmful_train[:n]
    harmless_train = harmless_train[:n]
    harmful_inst = [d["instruction"] for d in harmful_train]
    harmless_inst = [d["instruction"] for d in harmless_train]
    print(f"  Data: {n} harmful, {n} harmless")

    # -----------------------------------------------------------------------
    # Step 1: harmful_targets (ablation)
    # -----------------------------------------------------------------------
    if not os.path.exists(harmful_targets_path):
        print("[2/4] Generating harmful targets (ablation hooks, DIM k=1)...")
        fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction_dev)
        dataset = [{"instruction": inst, "category": "harmful"} for inst in harmful_inst]
        completions = model_base.generate_completions(
            dataset=dataset,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=NUM_TARGET_TOKENS,
        )
        harmful_targets = [
            {"prompt": c["prompt"], "ablation": c["response"]}
            for c in completions
        ]
        with open(harmful_targets_path, "w") as f:
            json.dump(harmful_targets, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(harmful_targets)} harmful targets → {harmful_targets_path}")
    else:
        print("[2/4] harmful_targets already exist, skipping.")

    # -----------------------------------------------------------------------
    # Step 2: harmless_targets (addition + retain)
    # -----------------------------------------------------------------------
    if not os.path.exists(harmless_targets_path):
        print("[3/4] Generating harmless targets (addition hooks, DIM k=1)...")
        coeff = torch.tensor(alpha, device=model_base.model.device, dtype=model_base.model.dtype)
        add_fwd_pre_hooks = [
            (model_base.model_block_modules[add_layer],
             get_activation_addition_input_pre_hook(vector=direction_dev, coeff=coeff))
        ]
        dataset = [{"instruction": inst, "category": "harmless"} for inst in harmless_inst]
        addition_completions = model_base.generate_completions(
            dataset=dataset,
            fwd_pre_hooks=add_fwd_pre_hooks,
            fwd_hooks=[],
            max_new_tokens=NUM_TARGET_TOKENS,
        )

        print("[4/4] Generating harmless targets (retain, no hooks)...")
        retain_completions = model_base.generate_completions(
            dataset=dataset,
            fwd_pre_hooks=[],
            fwd_hooks=[],
            max_new_tokens=NUM_TARGET_TOKENS,
        )

        harmless_targets = []
        for add_c, ret_c in zip(addition_completions, retain_completions):
            # Take only up to first sentence for addition target (matches original rdo.py)
            addition_text = add_c["response"].split(".")[0] if add_c["response"] else ""
            harmless_targets.append({
                "prompt": add_c["prompt"],
                "addition": addition_text,
                "retain": ret_c["response"],
            })

        with open(harmless_targets_path, "w") as f:
            json.dump(harmless_targets, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(harmless_targets)} harmless targets → {harmless_targets_path}")
    else:
        print("[3/4] harmless_targets already exist, skipping.")

    # Sanity check
    ht = json.load(open(harmful_targets_path))
    ht2 = json.load(open(harmless_targets_path))
    print(f"\n=== Target Generation Complete ===")
    print(f"  harmful_targets : {len(ht)}, example ablation: '{ht[0]['ablation'][:80]}'")
    print(f"  harmless_targets: {len(ht2)}, example addition: '{ht2[0]['addition'][:80]}'")


if __name__ == "__main__":
    main()
