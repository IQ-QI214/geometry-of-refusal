"""
P0 Phase 2: Extract DIM directions and compute PCA cone for VLMs.

For each model:
1. Extract mean_diffs (harmful - harmless activations) on train set
2. Select best (position, layer) via refusal score evaluation on val set
3. Save dim_cone_k1.pt as the best direction (1D mean_diff direction)
4. Extract per-sample activation diffs at best layer for PCA
5. Run SVD on centered diffs -> top-k PCA basis vectors
6. Save dim_cone_k{1,3,5}.pt, dim_metadata.json, dim_singular_values.pt

Usage:
  python experiments/p0_cone/exp_p0_dim_extract.py --model llava_7b   --device cuda:0
  python experiments/p0_cone/exp_p0_dim_extract.py --model qwen2vl_7b --device cuda:1
"""
import sys
import os
import json
import torch
import argparse

# Set up path so pipeline imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../refusal_direction"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from pipeline.utils.hook_utils import add_hooks

# Absolute paths required for offline execution
MODEL_PATHS = {
    "llava_7b": "/inspire/hdd/global_user/wenming-253108090054/models/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "qwen2vl_7b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct",
}


def extract_individual_diffs(
    model_base, harmful_instructions, harmless_instructions,
    best_layer, best_pos, batch_size=16
):
    """
    Extract per-sample activation diffs at the best (pos, layer) for PCA.

    best_pos is a NEGATIVE index (e.g. -1, -2) returned by select_direction.
    We use it directly as a tensor index — Python negative indexing works on tensors.

    The hook is registered as a forward_pre_hook on model_block_modules[best_layer].
    It captures input[0][:, best_pos, :] (shape: [batch, d_model]).
    """
    n_harmful = len(harmful_instructions)
    n_harmless = len(harmless_instructions)
    n_samples = min(n_harmful, n_harmless)

    d_model = model_base.model.config.hidden_size

    harmful_acts = torch.zeros((n_samples, d_model), dtype=torch.float64)
    harmless_acts = torch.zeros((n_samples, d_model), dtype=torch.float64)

    def make_capture_hook(storage, idx_offset):
        """Create a pre-hook that captures activation at best_pos for each sample."""
        def hook_fn(module, input):
            # input[0]: (batch_size, seq_len, d_model)
            activation = input[0].clone().to(torch.float64)
            bs = activation.shape[0]
            for b in range(bs):
                global_idx = idx_offset + b
                if global_idx < n_samples:
                    # best_pos is negative — valid Python/PyTorch negative indexing
                    storage[global_idx] = activation[b, best_pos, :]
        return hook_fn

    def run_forward_capture(instructions, storage):
        """Run forward passes and capture activations into storage."""
        for i in range(0, n_samples, batch_size):
            batch = instructions[i : i + batch_size]
            inputs = model_base.tokenize_instructions_fn(instructions=batch)

            # Build forward_kwargs exactly as generate_directions.py does:
            # always pass input_ids + attention_mask, then any VLM extras
            forward_kwargs = {
                "input_ids": inputs.input_ids.to(model_base.model.device),
                "attention_mask": inputs.attention_mask.to(model_base.model.device),
            }
            for _key in ("pixel_values", "image_grid_thw", "image_sizes"):
                if _key in inputs and inputs[_key] is not None:
                    _val = inputs[_key]
                    forward_kwargs[_key] = (
                        _val.to(model_base.model.device) if hasattr(_val, "to") else _val
                    )

            hook = make_capture_hook(storage, i)
            fwd_pre_hooks = [(model_base.model_block_modules[best_layer], hook)]

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model_base.model(**forward_kwargs)

    print(f"    Extracting harmful activations ({n_samples} samples)...")
    run_forward_capture(harmful_instructions[:n_samples], harmful_acts)

    print(f"    Extracting harmless activations ({n_samples} samples)...")
    run_forward_capture(harmless_instructions[:n_samples], harmless_acts)

    diffs = harmful_acts - harmless_acts  # (N, d_model)
    return diffs


def compute_pca_cone(diffs, k_values=(1, 3, 5)):
    """
    Compute PCA on individual diffs via SVD. Returns:
      - cones: dict mapping k -> (k, d_model) normalized basis tensor
      - singular_values: 1D tensor of all singular values
    """
    # Center diffs
    diffs_centered = diffs - diffs.mean(dim=0, keepdim=True)

    # SVD: Vt rows are right singular vectors (principal components)
    U, S, Vt = torch.linalg.svd(diffs_centered, full_matrices=False)

    total_variance = (S ** 2).sum()

    cones = {}
    for k in k_values:
        cone_basis = Vt[:k].clone()  # (k, d_model)
        # Normalize each basis vector (they should already be unit vectors from SVD,
        # but normalize explicitly for numerical safety)
        norms = cone_basis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cone_basis = cone_basis / norms
        cones[k] = cone_basis

        explained_var = (S[:k] ** 2).sum() / total_variance
        print(f"  k={k}: explained variance = {explained_var.item():.4f}")

    return cones, S


def main():
    parser = argparse.ArgumentParser(
        description="Extract DIM directions + PCA cone for P0 cone ablation experiment"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_PATHS.keys()), required=True,
        help="Which model to run: llava_7b or qwen2vl_7b"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Target CUDA device (used implicitly via model loading)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for activation extraction (reduce if OOM)"
    )
    parser.add_argument(
        "--select_batch_size", type=int, default=8,
        help="Batch size for select_direction (smaller to reduce OOM during eval)"
    )
    args = parser.parse_args()

    # Determine save directory relative to project root
    # Script is at experiments/p0_cone/; project root is two levels up
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
    save_dir = os.path.join(project_root, "results", "p0_cone", args.model)
    os.makedirs(save_dir, exist_ok=True)

    data_dir = os.path.join(project_root, "data", "saladbench_splits")

    print(f"=== DIM Extraction + PCA Cone: {args.model} ===")
    print(f"  Save dir: {save_dir}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("[0/4] Loading model...")
    model_base = construct_model_base(MODEL_PATHS[args.model])
    print(f"  Layers: {len(model_base.model_block_modules)}")
    print(f"  hidden_size: {model_base.model.config.hidden_size}")
    print(f"  EOI toks: {model_base.eoi_toks} (len={len(model_base.eoi_toks)})")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("[0/4] Loading data...")
    with open(os.path.join(data_dir, "harmful_train.json")) as f:
        harmful_train = json.load(f)
    with open(os.path.join(data_dir, "harmless_train.json")) as f:
        harmless_train = json.load(f)
    with open(os.path.join(data_dir, "harmful_val.json")) as f:
        harmful_val = json.load(f)
    with open(os.path.join(data_dir, "harmless_val.json")) as f:
        harmless_val = json.load(f)

    # Balance train sets
    n_train = min(len(harmful_train), len(harmless_train))
    harmful_train = harmful_train[:n_train]
    harmless_train = harmless_train[:n_train]

    harmful_train_inst = [d["instruction"] for d in harmful_train]
    harmless_train_inst = [d["instruction"] for d in harmless_train]
    harmful_val_inst = [d["instruction"] for d in harmful_val]
    harmless_val_inst = [d["instruction"] for d in harmless_val]

    print(f"  Train: {len(harmful_train_inst)} harmful, {len(harmless_train_inst)} harmless")
    print(f"  Val:   {len(harmful_val_inst)} harmful, {len(harmless_val_inst)} harmless")

    # -----------------------------------------------------------------------
    # Step 1: Generate mean_diffs on the full train set
    # -----------------------------------------------------------------------
    artifact_dir = os.path.join(save_dir, "dim_directions")
    os.makedirs(artifact_dir, exist_ok=True)
    mean_diffs_cache = os.path.join(artifact_dir, "mean_diffs.pt")

    if os.path.exists(mean_diffs_cache):
        print(f"[1/4] Loading cached mean_diffs from {mean_diffs_cache} ...")
        mean_diffs = torch.load(mean_diffs_cache, map_location="cpu")
    else:
        print("[1/4] Extracting mean_diffs (generate_directions) on train set...")
        mean_diffs = generate_directions(
            model_base, harmful_train_inst, harmless_train_inst, artifact_dir
        )

    # mean_diffs shape: (n_eoi_toks, n_layers, d_model)
    print(f"  mean_diffs shape: {mean_diffs.shape}")
    assert not mean_diffs.isnan().any(), "mean_diffs contains NaN!"

    # -----------------------------------------------------------------------
    # Step 2: Select best (pos, layer) via select_direction on val set
    # -----------------------------------------------------------------------
    print("[2/4] Selecting best direction on val set...")
    selection_dir = os.path.join(save_dir, "dim_selection")
    os.makedirs(selection_dir, exist_ok=True)

    evals_cache = os.path.join(selection_dir, "direction_evaluations.json")
    if os.path.exists(evals_cache):
        # Fast path: all KL/ablation/steering scores already computed.
        # Apply relaxed filter (induce_refusal_threshold=None for VLMs) and pick best.
        print(f"  [cache hit] Loading scores from {evals_cache} ...")
        import math
        evals = json.load(open(evals_cache))
        n_layers = len(model_base.model_block_modules)
        kl_threshold = 0.1
        prune_layer_percentage = 0.20
        max_ok_layer = int(n_layers * (1.0 - prune_layer_percentage))

        filtered = []
        for e in evals:
            pos = e["position"]
            layer = e["layer"]
            rs = e["refusal_score"]
            ss = e["steering_score"]
            kl = e["kl_div_score"]
            if any(math.isnan(v) for v in [rs, ss, kl]):
                continue
            if layer >= max_ok_layer:
                continue
            if kl > kl_threshold:
                continue
            # induce_refusal_threshold=None → steering filter disabled for VLMs
            filtered.append((-rs, pos, layer))

        assert len(filtered) > 0, "Still no valid directions after relaxed filter!"
        filtered.sort(key=lambda x: x[0], reverse=True)
        _, best_pos, best_layer = filtered[0]
        best_direction = mean_diffs[best_pos, best_layer].to(model_base.model.device)
        print(f"  [fast path] Selected from cache: pos={best_pos}, layer={best_layer}")
    else:
        best_pos, best_layer, best_direction = select_direction(
            model_base,
            harmful_val_inst,
            harmless_val_inst,
            mean_diffs,
            artifact_dir=selection_dir,
            batch_size=args.select_batch_size,
            induce_refusal_threshold=None,  # VLM: steering filter disabled
        )

    # best_pos is a NEGATIVE integer (e.g. -1, -2)
    print(f"  Selected: pos={best_pos}, layer={best_layer}")
    print(f"  best_direction shape: {best_direction.shape}")

    # Save metadata
    metadata = {"pos": int(best_pos), "layer": int(best_layer)}
    with open(os.path.join(save_dir, "dim_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved dim_metadata.json: {metadata}")

    # -----------------------------------------------------------------------
    # Step 3: Extract individual per-sample activation diffs at best layer
    # -----------------------------------------------------------------------
    print("[3/4] Extracting per-sample activation diffs for PCA...")
    diffs = extract_individual_diffs(
        model_base,
        harmful_train_inst,
        harmless_train_inst,
        best_layer=best_layer,
        best_pos=best_pos,
        batch_size=args.batch_size,
    )
    print(f"  diffs shape: {diffs.shape}")
    assert not diffs.isnan().any(), "diffs contains NaN!"

    # -----------------------------------------------------------------------
    # Step 4: Compute PCA cones (k=1, 3, 5)
    # -----------------------------------------------------------------------
    print("[4/4] Computing PCA cones (k=1, 3, 5)...")
    k_values = [1, 3, 5]
    cones, singular_values = compute_pca_cone(diffs, k_values=k_values)

    for k, basis in cones.items():
        out_path = os.path.join(save_dir, f"dim_cone_k{k}.pt")
        # Save as float32 for downstream compatibility
        torch.save(basis.float(), out_path)
        print(f"  Saved dim_cone_k{k}.pt, shape={basis.shape}")

    sv_path = os.path.join(save_dir, "dim_singular_values.pt")
    torch.save(singular_values.float(), sv_path)
    print(f"  Saved dim_singular_values.pt, n_components={singular_values.shape[0]}")

    print("=== DIM Extraction Complete ===")
    print(f"Outputs in: {save_dir}")
    print("Files created:")
    for fname in [
        "dim_cone_k1.pt", "dim_cone_k3.pt", "dim_cone_k5.pt",
        "dim_metadata.json", "dim_singular_values.pt",
    ]:
        fpath = os.path.join(save_dir, fname)
        exists = os.path.exists(fpath)
        print(f"  {'OK' if exists else 'MISSING'}: {fname}")


if __name__ == "__main__":
    main()
