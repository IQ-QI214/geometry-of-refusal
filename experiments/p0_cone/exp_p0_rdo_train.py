"""
P0 Phase 3: RDO cone training for VLMs using standard PyTorch hooks.

Faithful reproduction of Wollschläger et al. (ICML 2025):
  L = λ_abl * L_ablation + λ_add * L_addition + λ_ret * L_retain

Key differences from original rdo.py:
  - Uses standard PyTorch forward hooks instead of nnsight (VLM multi-modal compatibility)
  - direction passed to hooks is .detach()-ed; gradients flow only through cone.basis → loss → backward
  - Non-inplace activation ops inside hooks to avoid version-mismatch in backward

Output directory: results/p0_cone/{model}/rdo/

Requires: exp_p0_rdo_targets.py must have run first (targets in results/p0_cone/{model}/targets/).

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
import contextlib
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../refusal_direction"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

MODEL_PATHS = {
    "llava_7b": "/inspire/hdd/global_user/wenming-253108090054/models/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "qwen2vl_7b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct",
}

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
    "num_target_tokens": 30,
}


# ---------------------------------------------------------------------------
# Non-inplace hook factories (critical: avoids backward version mismatch)
# Direction must be .detach()-ed before passing to these hooks.
# ---------------------------------------------------------------------------

def make_ablation_pre_hook(direction: torch.Tensor):
    """Pre-hook: project out `direction` from layer input. NON-inplace."""
    d = direction.detach()

    def hook_fn(module, input):
        x = input[0] if isinstance(input, tuple) else input
        d_norm = d / (d.norm() + 1e-8)
        d_cast = d_norm.to(dtype=x.dtype, device=x.device)
        # Non-inplace: x - proj
        proj = (x @ d_cast).unsqueeze(-1) * d_cast
        x_new = x - proj
        if isinstance(input, tuple):
            return (x_new, *input[1:])
        return x_new
    return hook_fn


def make_ablation_out_hook(direction: torch.Tensor):
    """Post-hook: project out `direction` from layer output. NON-inplace."""
    d = direction.detach()

    def hook_fn(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        d_norm = d / (d.norm() + 1e-8)
        d_cast = d_norm.to(dtype=x.dtype, device=x.device)
        proj = (x @ d_cast).unsqueeze(-1) * d_cast
        x_new = x - proj
        if isinstance(output, tuple):
            return (x_new, *output[1:])
        return x_new
    return hook_fn


def make_addition_pre_hook(direction: torch.Tensor, coeff: float):
    """Pre-hook: add `coeff * direction` to layer input. NON-inplace."""
    d = direction.detach()
    c = coeff

    def hook_fn(module, input):
        x = input[0] if isinstance(input, tuple) else input
        d_cast = d.to(dtype=x.dtype, device=x.device)
        x_new = x + c * d_cast
        if isinstance(input, tuple):
            return (x_new, *input[1:])
        return x_new
    return hook_fn


def build_full_ablation_hooks(model_base, direction):
    """Build ablation hooks for ALL layers (pre + attn-out + mlp-out)."""
    n = model_base.model.config.num_hidden_layers
    pre = [(model_base.model_block_modules[l], make_ablation_pre_hook(direction)) for l in range(n)]
    fwd = [(model_base.model_attn_modules[l], make_ablation_out_hook(direction)) for l in range(n)]
    fwd += [(model_base.model_mlp_modules[l], make_ablation_out_hook(direction)) for l in range(n)]
    return pre, fwd


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _fwd_kwargs(tokenized, model):
    kw = {
        "input_ids": tokenized.input_ids.to(model.device),
        "attention_mask": tokenized.attention_mask.to(model.device),
    }
    for key in ("pixel_values", "image_grid_thw", "image_sizes"):
        if key in tokenized and tokenized[key] is not None:
            v = tokenized[key]
            kw[key] = v.to(model.device) if hasattr(v, "to") else v
    return kw


def ce_loss_with_hooks(model, model_base, tokenized, target_ids, pre_hooks, fwd_hooks):
    kw = _fwd_kwargs(tokenized, model)
    with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=fwd_hooks):
        logits = model(**kw).logits
    shift = logits[:, :-1, :].contiguous()
    labels = target_ids[:, 1:].contiguous().to(model.device)
    return nn.CrossEntropyLoss(ignore_index=-100)(
        shift.view(-1, shift.size(-1)), labels.view(-1)
    )


def kl_retain_loss(model, model_base, tokenized, direction, n_retain=30):
    kw = _fwd_kwargs(tokenized, model)
    with torch.no_grad():
        baseline = model(**kw).logits[:, -n_retain:, :].detach()
    pre_hooks, fwd_hooks = build_full_ablation_hooks(model_base, direction)
    with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=fwd_hooks):
        ablated = model(**kw).logits[:, -n_retain:, :]
    p = torch.softmax(baseline.float(), dim=-1)
    log_q = torch.log_softmax(ablated.float(), dim=-1)
    return (p * (torch.log(p + 1e-8) - log_q)).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# RefusalCone
# ---------------------------------------------------------------------------

class RefusalCone(nn.Module):
    def __init__(self, cone_dim, d_model, init_direction=None, alpha=1.0):
        super().__init__()
        self.cone_dim = cone_dim
        self.alpha = alpha
        if init_direction is not None and cone_dim == 1:
            init = init_direction.unsqueeze(0).float()
        elif init_direction is not None:
            init = torch.randn(cone_dim, d_model)
            init[0] = init_direction.float()
        else:
            init = torch.randn(cone_dim, d_model)
        self.basis = nn.Parameter(init)

    def orthonormalize_(self):
        with torch.no_grad():
            for i in range(self.cone_dim):
                for j in range(i):
                    self.basis.data[i] -= (self.basis.data[i] @ self.basis.data[j]) * self.basis.data[j]
                n = self.basis.data[i].norm()
                if n > 1e-8:
                    self.basis.data[i] /= n

    def get_direction(self, idx=0):
        """Normalized + alpha-scaled basis vector. DETACHED for hook use."""
        b = self.basis[idx]
        return (b / (b.norm() + 1e-8) * self.alpha).to(self.basis.dtype)

    def get_normalized_basis(self):
        """Return orthonormalized basis as plain tensor (for saving)."""
        with torch.no_grad():
            basis = self.basis.clone().float()
            for i in range(self.cone_dim):
                for j in range(i):
                    basis[i] -= (basis[i] @ basis[j]) * basis[j]
                n = basis[i].norm()
                if n > 1e-8:
                    basis[i] /= n
        return basis


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model_base, harmful_targets, harmless_targets, cone, add_layer, cfg, save_path):
    model = model_base.model
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

        for data_idx in tqdm(perm, desc=f"Epoch {epoch}"):
            # Direction for this step — DETACHED so hooks don't pollute grad graph
            direction = cone.get_direction(0).detach().to(model.device)

            # ----------------------------------------------------------------
            # Ablation loss
            # ----------------------------------------------------------------
            ht = harmful_targets[data_idx]
            tok_full = model_base.tokenize_instructions_fn(
                instructions=[ht["prompt_clean"]], outputs=[ht["ablation"]]
            )
            tok_prompt = model_base.tokenize_instructions_fn(
                instructions=[ht["prompt_clean"]]
            )
            plen = tok_prompt.input_ids.shape[1]
            tgt = tok_full.input_ids.clone()
            tgt[0, :plen] = -100

            abl_pre, abl_fwd = build_full_ablation_hooks(model_base, direction)
            loss_abl = ce_loss_with_hooks(model, model_base, tok_full, tgt, abl_pre, abl_fwd)

            # ----------------------------------------------------------------
            # Addition loss
            # ----------------------------------------------------------------
            ht2 = harmless_targets[data_idx]
            tok_full_hl = model_base.tokenize_instructions_fn(
                instructions=[ht2["prompt_clean"]], outputs=[ht2["addition"]]
            )
            tok_prompt_hl = model_base.tokenize_instructions_fn(
                instructions=[ht2["prompt_clean"]]
            )
            hl_plen = tok_prompt_hl.input_ids.shape[1]
            tgt_hl = tok_full_hl.input_ids.clone()
            tgt_hl[0, :hl_plen] = -100

            add_pre = [(model_base.model_block_modules[add_layer],
                        make_addition_pre_hook(direction, cone.alpha))]
            loss_add = ce_loss_with_hooks(model, model_base, tok_full_hl, tgt_hl, add_pre, [])

            # ----------------------------------------------------------------
            # Retain loss (KL)
            # ----------------------------------------------------------------
            tok_retain = model_base.tokenize_instructions_fn(
                instructions=[ht2["prompt_clean"]], outputs=[ht2["retain"]]
            )
            loss_ret = kl_retain_loss(
                model, model_base, tok_retain, direction, cfg["num_target_tokens"]
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
                # Project gradient to tangent of sphere
                with torch.no_grad():
                    if cone.basis.grad is not None:
                        for i in range(cone.cone_dim):
                            b = cone.basis.data[i]
                            g = cone.basis.grad[i]
                            cone.basis.grad.data[i] = g - (g @ b) * b
                torch.nn.utils.clip_grad_norm_(cone.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()
                cone.orthonormalize_()
                torch.cuda.empty_cache()

                avg = epoch_loss / n_steps
                log.append({
                    "step": step,
                    "avg_loss": avg,
                    "loss_abl": loss_abl.item(),
                    "loss_add": loss_add.item(),
                    "loss_ret": loss_ret.item(),
                })
                if (step // grad_accum) % 10 == 0:
                    print(f"  step={step} avg={avg:.4f} "
                          f"abl={loss_abl.item():.4f} add={loss_add.item():.4f} ret={loss_ret.item():.4f}")

        avg_epoch = epoch_loss / max(n_steps, 1)
        print(f"Epoch {epoch} avg loss: {avg_epoch:.4f}")

        if avg_epoch < best_loss:
            best_loss = avg_epoch
            patience_ctr = 0
            torch.save(cone.get_normalized_basis(), save_path)
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

    # Always save final
    final = cone.get_normalized_basis()
    torch.save(final, save_path)
    return log, final


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
    parser.add_argument("--cone_dim", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = DEFAULTS.copy()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    base_dir = os.path.join(project_root, "results", "p0_cone", args.model)

    # --- New directory structure ---
    # targets/   ← generated by exp_p0_rdo_targets.py
    # dim/       ← DIM cone files (k1,k3,k5), metadata, singular values, selection plots
    # rdo/       ← RDO cone files, training logs
    # responses/ ← all *_responses.json (dim and rdo)
    targets_dir = os.path.join(base_dir, "targets")
    rdo_dir = os.path.join(base_dir, "rdo")
    os.makedirs(rdo_dir, exist_ok=True)

    save_path = os.path.join(rdo_dir, f"rdo_cone_k{args.cone_dim}.pt")
    if os.path.exists(save_path):
        tag = datetime.now().strftime("%Y%m%d_%H%M")
        root, ext = os.path.splitext(save_path)
        save_path = f"{root}_{tag}{ext}"
        print(f"  [PROTECT] Output exists → {save_path}")

    print(f"=== RDO Training: {args.model}, cone_dim={args.cone_dim} ===")
    print(f"  Output: {save_path}")

    # Load targets
    harmful_targets_raw = json.load(open(os.path.join(targets_dir, "harmful_targets.json")))
    harmless_targets_raw = json.load(open(os.path.join(targets_dir, "harmless_targets.json")))
    # Normalize key name
    for t in harmful_targets_raw:
        t.setdefault("prompt_clean", t.get("prompt", ""))
    for t in harmless_targets_raw:
        t.setdefault("prompt_clean", t.get("prompt", ""))

    n = min(len(harmful_targets_raw), len(harmless_targets_raw))
    print(f"  Training samples: {n}")

    # Load DIM direction
    dim_dir = os.path.join(base_dir, "dim")
    dim_direction = torch.load(
        os.path.join(dim_dir, "dim_cone_k1.pt"), weights_only=True, map_location="cpu"
    )
    if dim_direction.dim() == 2:
        dim_direction = dim_direction[0]
    meta = json.load(open(os.path.join(dim_dir, "dim_metadata.json")))
    add_layer = meta["layer"]
    alpha = dim_direction.norm().item()
    print(f"  add_layer={add_layer}, alpha={alpha:.4f}")

    # Load model
    print("Loading model...")
    model_base = construct_model_base(MODEL_PATHS[args.model])
    d_model = model_base.model.config.hidden_size

    # Init cone
    init_dir = dim_direction.to(model_base.model.device, dtype=torch.float32)
    cone = RefusalCone(args.cone_dim, d_model, init_direction=init_dir, alpha=alpha)
    cone = cone.to(model_base.model.device)

    # Train
    log, final_basis = train(
        model_base, harmful_targets_raw[:n], harmless_targets_raw[:n],
        cone, add_layer, cfg, save_path
    )

    log_path = save_path.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n=== RDO Training Complete ===")
    print(f"  cone : {save_path}  shape={final_basis.shape}")
    print(f"  log  : {log_path}")


if __name__ == "__main__":
    main()
