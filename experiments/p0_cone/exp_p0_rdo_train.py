"""
P0 Phase 3: RDO cone training for VLMs using standard PyTorch hooks.

Faithful reproduction of Wollschläger et al. (ICML 2025):
  L = λ_abl * L_ablation + λ_add * L_addition + λ_ret * L_retain
  - L_ablation: CE loss on harmful prompt with ablation hooks active (model should speak freely)
  - L_addition: CE loss on harmless prompt with addition hook active (model should refuse)
  - L_retain  : KL(baseline || ablated) on harmless prompt (ablation shouldn't distort behavior)

Key differences from original rdo.py:
  - Uses standard PyTorch forward hooks instead of nnsight (VLM multi-modal compatibility)
  - Inherits ModelBase infrastructure (tokenize_instructions_fn handles pixel_values etc.)
  - For cone_dim=1: no hypersphere sampling, optimize basis vector directly
  - For cone_dim>1: Gram-Schmidt orthonormalization after each optimizer step

Requires: exp_p0_rdo_targets.py must have run first.

Usage:
  python experiments/p0_cone/exp_p0_rdo_train.py --model llava_7b   --cone_dim 1
  python experiments/p0_cone/exp_p0_rdo_train.py --model llava_7b   --cone_dim 3
  python experiments/p0_cone/exp_p0_rdo_train.py --model llava_7b   --cone_dim 5
  python experiments/p0_cone/exp_p0_rdo_train.py --model qwen2vl_7b --cone_dim 1
"""
import sys
import os
import json
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../refusal_direction"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_activation_addition_input_pre_hook,
)

MODEL_PATHS = {
    "llava_7b": "/inspire/hdd/global_user/wenming-253108090054/models/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "qwen2vl_7b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct",
}

# Training hyperparams (matches Wollschläger defaults)
DEFAULTS = {
    "lr": 1e-2,
    "epochs": 1,
    "batch_size": 1,
    "effective_batch_size": 16,
    "patience": 5,
    "n_lr_reduce": 2,
    "ablation_lambda": 1.0,
    "addition_lambda": 0.2,
    "retain_lambda": 1.0,
    "n_sample": 8,          # cone_dim>1: samples from hypersphere per step
    "num_target_tokens": 30,
}


# ---------------------------------------------------------------------------
# RefusalCone: trainable k-dim subspace
# ---------------------------------------------------------------------------

class RefusalCone(nn.Module):
    def __init__(self, cone_dim: int, d_model: int, init_direction=None, alpha: float = 1.0):
        super().__init__()
        self.cone_dim = cone_dim
        self.alpha = alpha

        if init_direction is not None and cone_dim == 1:
            init = init_direction.unsqueeze(0).float()
        elif init_direction is not None:
            # Warm-start: first basis = DIM direction, rest random
            init = torch.randn(cone_dim, d_model)
            init[0] = init_direction.float()
        else:
            init = torch.randn(cone_dim, d_model)

        self.basis = nn.Parameter(init)

    def orthonormalize_(self):
        """In-place Gram-Schmidt orthonormalization of basis rows."""
        with torch.no_grad():
            for i in range(self.cone_dim):
                for j in range(i):
                    self.basis.data[i] -= (
                        self.basis.data[i] @ self.basis.data[j]
                    ) * self.basis.data[j]
                norm = self.basis.data[i].norm()
                if norm > 1e-8:
                    self.basis.data[i] /= norm

    def get_direction(self, idx: int = 0) -> torch.Tensor:
        """Return normalized basis vector idx, scaled by alpha."""
        b = self.basis[idx]
        return (b / (b.norm() + 1e-8) * self.alpha).to(self.basis.dtype)

    def sample_direction(self) -> torch.Tensor:
        """Sample a random unit direction from the cone, scaled by alpha."""
        if self.cone_dim == 1:
            return self.get_direction(0)
        # Positive hypersphere sampling (matches rdo.py sample_hypersphere_gaussian)
        coeffs = torch.randn(self.cone_dim, device=self.basis.device).abs()
        coeffs = coeffs / (coeffs.norm() + 1e-8)
        basis_norm = self.basis / (self.basis.norm(dim=-1, keepdim=True) + 1e-8)
        direction = coeffs @ basis_norm
        direction = direction / (direction.norm() + 1e-8)
        return (direction * self.alpha).to(self.basis.dtype)


# ---------------------------------------------------------------------------
# Loss helpers (all use standard PyTorch hooks)
# ---------------------------------------------------------------------------

def _build_forward_kwargs(tokenized, model):
    """Build model() kwargs from tokenized batch, adding VLM extras."""
    kwargs = {
        "input_ids": tokenized.input_ids.to(model.device),
        "attention_mask": tokenized.attention_mask.to(model.device),
    }
    for key in ("pixel_values", "image_grid_thw", "image_sizes"):
        if key in tokenized and tokenized[key] is not None:
            val = tokenized[key]
            kwargs[key] = val.to(model.device) if hasattr(val, "to") else val
    return kwargs


def compute_ce_loss_with_hooks(model, model_base, tokenized, target_ids, fwd_pre_hooks, fwd_hooks):
    """CE loss on target_ids[-len:] with intervention hooks."""
    kwargs = _build_forward_kwargs(tokenized, model)

    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        outputs = model(**kwargs)

    # Causal LM: shift by 1
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = target_ids[:, 1:].contiguous().to(model.device)
    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
    return loss


def compute_kl_retain_loss(model, model_base, tokenized, direction, n_retain=30):
    """KL(baseline || ablated) on last n_retain token positions."""
    n_layers = model.config.num_hidden_layers
    kwargs = _build_forward_kwargs(tokenized, model)

    with torch.no_grad():
        baseline_logits = model(**kwargs).logits[:, -n_retain:, :].detach()

    fwd_pre_hooks = [
        (model_base.model_block_modules[l], get_direction_ablation_input_pre_hook(direction))
        for l in range(n_layers)
    ]
    fwd_hooks = [
        (model_base.model_attn_modules[l], get_direction_ablation_output_hook(direction))
        for l in range(n_layers)
    ] + [
        (model_base.model_mlp_modules[l], get_direction_ablation_output_hook(direction))
        for l in range(n_layers)
    ]

    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        ablated_logits = model(**kwargs).logits[:, -n_retain:, :]

    # KL(P_baseline || P_ablated) token-averaged
    p_base = torch.softmax(baseline_logits.float(), dim=-1)
    log_p_abl = torch.log_softmax(ablated_logits.float(), dim=-1)
    kl = (p_base * (torch.log(p_base + 1e-8) - log_p_abl)).sum(dim=-1).mean()
    return kl


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(model_base, harmful_targets, harmless_targets, cone, add_layer, cfg, save_path):
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    n_train = min(len(harmful_targets), len(harmless_targets))
    grad_accum = cfg["effective_batch_size"] // cfg["batch_size"]

    optimizer = torch.optim.AdamW(cone.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), weight_decay=0.0)

    best_loss = float("inf")
    patience_ctr = 0
    lr_reduce_ctr = 0
    log = []
    step = 0

    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n_train).tolist()
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for idx_in_epoch, data_idx in enumerate(tqdm(perm, desc=f"Epoch {epoch}")):
            # Sample direction
            if cone.cone_dim == 1:
                direction = cone.get_direction(0).to(model.device)
            else:
                direction = cone.sample_direction().to(model.device)

            # ----------------------------------------------------------------
            # Ablation loss: harmful prompt + ablation hooks → should output
            #                harmful content (ablation target)
            # ----------------------------------------------------------------
            ht = harmful_targets[data_idx]
            harm_prompt = ht["prompt_clean"]
            harm_target = ht["ablation"]
            tokenized_harm_full = model_base.tokenize_instructions_fn(
                instructions=[harm_prompt], outputs=[harm_target]
            )
            tokenized_harm_prompt_only = model_base.tokenize_instructions_fn(
                instructions=[harm_prompt]
            )
            prompt_len = tokenized_harm_prompt_only.input_ids.shape[1]
            target_ids = tokenized_harm_full.input_ids.clone()
            target_ids[0, :prompt_len] = -100

            abl_pre_hooks = [
                (model_base.model_block_modules[l], get_direction_ablation_input_pre_hook(direction))
                for l in range(n_layers)
            ]
            abl_fwd_hooks = [
                (model_base.model_attn_modules[l], get_direction_ablation_output_hook(direction))
                for l in range(n_layers)
            ] + [
                (model_base.model_mlp_modules[l], get_direction_ablation_output_hook(direction))
                for l in range(n_layers)
            ]
            loss_abl = compute_ce_loss_with_hooks(
                model, model_base, tokenized_harm_full, target_ids, abl_pre_hooks, abl_fwd_hooks
            )

            # ----------------------------------------------------------------
            # Addition loss: harmless prompt + addition hook → should refuse
            #                (addition target)
            # ----------------------------------------------------------------
            ht2 = harmless_targets[data_idx]
            harm_less_prompt = ht2["prompt_clean"]
            harm_less_target = ht2["addition"]
            coeff = torch.tensor(cone.alpha, device=model.device, dtype=model.dtype)
            tokenized_hl_full = model_base.tokenize_instructions_fn(
                instructions=[harm_less_prompt], outputs=[harm_less_target]
            )
            tokenized_hl_prompt_only = model_base.tokenize_instructions_fn(
                instructions=[harm_less_prompt]
            )
            hl_prompt_len = tokenized_hl_prompt_only.input_ids.shape[1]
            hl_target_ids = tokenized_hl_full.input_ids.clone()
            hl_target_ids[0, :hl_prompt_len] = -100

            add_pre_hooks = [
                (model_base.model_block_modules[add_layer],
                 get_activation_addition_input_pre_hook(vector=direction, coeff=coeff))
            ]
            loss_add = compute_ce_loss_with_hooks(
                model, model_base, tokenized_hl_full, hl_target_ids, add_pre_hooks, []
            )

            # ----------------------------------------------------------------
            # Retain loss: harmless prompt + ablation → KL should stay low
            # ----------------------------------------------------------------
            retain_target = ht2["retain"]
            tokenized_retain = model_base.tokenize_instructions_fn(
                instructions=[harm_less_prompt], outputs=[retain_target]
            )
            loss_ret = compute_kl_retain_loss(
                model, model_base, tokenized_retain, direction,
                n_retain=cfg["num_target_tokens"]
            )

            total = (
                cfg["ablation_lambda"] * loss_abl
                + cfg["addition_lambda"] * loss_add
                + cfg["retain_lambda"] * loss_ret
            ) / grad_accum
            total.backward()

            epoch_loss += total.item() * grad_accum
            n_steps += 1
            step += 1

            if step % grad_accum == 0:
                # Project gradient to tangent of sphere (remove radial component)
                with torch.no_grad():
                    for i in range(cone.cone_dim):
                        b = cone.basis.data[i]
                        g = cone.basis.grad[i] if cone.basis.grad is not None else None
                        if g is not None:
                            cone.basis.grad.data[i] -= (g @ b) * b
                torch.nn.utils.clip_grad_norm_(cone.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()
                cone.orthonormalize_()

                avg_loss = epoch_loss / n_steps
                log_entry = {
                    "step": step,
                    "avg_loss": avg_loss,
                    "loss_abl": loss_abl.item(),
                    "loss_add": loss_add.item(),
                    "loss_ret": loss_ret.item(),
                }
                log.append(log_entry)

                if n_steps % 50 == 0:
                    print(
                        f"  step={step} avg={avg_loss:.4f} "
                        f"abl={loss_abl.item():.4f} add={loss_add.item():.4f} ret={loss_ret.item():.4f}"
                    )

        avg_epoch = epoch_loss / max(n_steps, 1)
        print(f"Epoch {epoch} avg loss: {avg_epoch:.4f}")

        if avg_epoch < best_loss:
            best_loss = avg_epoch
            patience_ctr = 0
            torch.save(cone.get_normalized_basis_(), save_path)
            print(f"  [best] saved → {save_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                if lr_reduce_ctr >= cfg["n_lr_reduce"]:
                    print("  Early stopping.")
                    break
                lr_reduce_ctr += 1
                old_lr = optimizer.param_groups[0]["lr"]
                optimizer.param_groups[0]["lr"] /= 10
                print(f"  LR {old_lr:.2e} → {optimizer.param_groups[0]['lr']:.2e}")
                patience_ctr = 0

    # Save final regardless
    final_basis = cone.get_normalized_basis_()
    torch.save(final_basis, save_path)
    return log, final_basis


# Patch RefusalCone with helper method
def _get_normalized_basis(self):
    with torch.no_grad():
        basis = self.basis.clone().float()
        for i in range(self.cone_dim):
            for j in range(i):
                basis[i] -= (basis[i] @ basis[j]) * basis[j]
            basis[i] = basis[i] / (basis[i].norm() + 1e-8)
    return basis
RefusalCone.get_normalized_basis_ = _get_normalized_basis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
    parser.add_argument("--cone_dim", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = DEFAULTS.copy()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    base_dir = os.path.join(project_root, "results", "p0_cone", args.model)
    targets_dir = os.path.join(base_dir, "rdo_targets")

    # Output path with protection
    save_path = os.path.join(base_dir, f"rdo_cone_k{args.cone_dim}.pt")
    if os.path.exists(save_path):
        tag = datetime.now().strftime("%Y%m%d_%H%M")
        root, ext = os.path.splitext(save_path)
        save_path = f"{root}_{tag}{ext}"
        print(f"  [PROTECT] Output exists → {save_path}")

    print(f"=== RDO Training: {args.model}, cone_dim={args.cone_dim} ===")
    print(f"  Save: {save_path}")

    # Load targets
    harmful_targets_raw = json.load(open(os.path.join(targets_dir, "harmful_targets.json")))
    harmless_targets_raw = json.load(open(os.path.join(targets_dir, "harmless_targets.json")))

    # Normalize key: original rdo.py stores "prompt" not "prompt_clean"
    for t in harmful_targets_raw:
        t["prompt_clean"] = t.get("prompt_clean", t.get("prompt", ""))
    for t in harmless_targets_raw:
        t["prompt_clean"] = t.get("prompt_clean", t.get("prompt", ""))

    n = min(len(harmful_targets_raw), len(harmless_targets_raw))
    harmful_targets = harmful_targets_raw[:n]
    harmless_targets = harmless_targets_raw[:n]
    print(f"  Training samples: {n}")

    # Load DIM direction
    dim_dir = torch.load(
        os.path.join(base_dir, "dim_cone_k1.pt"), weights_only=True, map_location="cpu"
    )
    if dim_dir.dim() == 2:
        dim_dir = dim_dir[0]
    meta = json.load(open(os.path.join(base_dir, "dim_metadata.json")))
    add_layer = meta["layer"]
    alpha = dim_dir.norm().item()
    print(f"  add_layer={add_layer}, alpha={alpha:.4f}")

    # Load model
    print("Loading model...")
    model_base = construct_model_base(MODEL_PATHS[args.model])
    d_model = model_base.model.config.hidden_size

    # Init cone
    init_dir = dim_dir.to(model_base.model.device, dtype=torch.float32)
    cone = RefusalCone(args.cone_dim, d_model, init_direction=init_dir, alpha=alpha)
    cone = cone.to(model_base.model.device)

    # Train
    log, final_basis = train(
        model_base, harmful_targets, harmless_targets,
        cone, add_layer, cfg, save_path
    )

    # Save training log
    log_path = save_path.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n=== RDO Training Complete ===")
    print(f"  cone shape: {final_basis.shape}")
    print(f"  best ckpt : {save_path}")
    print(f"  train log : {log_path}")


if __name__ == "__main__":
    main()
