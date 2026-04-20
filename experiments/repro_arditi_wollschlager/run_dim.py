"""
Full DIM pipeline for one model. Called by run_dim.sh (once per model).

Outputs to: results/repro_arditi_wollschlager/<model_alias>/
  direction.pt, direction_metadata.json,
  generate_directions/mean_diffs.pt,
  select_direction/,
  completions/saladbench_{baseline,ablation}_completions.json

Usage:
  python experiments/repro_arditi_wollschlager/run_dim.py --model qwen2.5_7b
  python experiments/repro_arditi_wollschlager/run_dim.py --model llama3.1_8b
"""
import sys
import os
import json
import random
import argparse
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "../../refusal_direction"))
sys.path.insert(0, _SCRIPT_DIR)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("WANDB_MODE", "offline")

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks
from dataset.load_dataset import load_dataset_split

from common.model_paths import MODEL_PATHS
from common.eval_judges import compute_asr_keyword

N_TEST = 128
SAVE_ROOT = "results/repro_arditi_wollschlager"


def run_dim(model_key: str):
    random.seed(42)
    model_path = MODEL_PATHS[model_key]
    model_alias = os.path.basename(model_path)
    out_dir  = os.path.join(SAVE_ROOT, model_alias)
    gen_dir  = os.path.join(out_dir, "generate_directions")
    sel_dir  = os.path.join(out_dir, "select_direction")
    comp_dir = os.path.join(out_dir, "completions")
    for d in [gen_dir, sel_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"=== DIM: {model_alias} ===")
    print(f"  out: {out_dir}")

    print("[1/5] Loading data...")
    harmful_train  = load_dataset_split("harmful",  "train", instructions_only=True)
    harmless_train = load_dataset_split("harmless", "train", instructions_only=True)[:len(harmful_train)]
    harmful_val    = load_dataset_split("harmful",  "val",   instructions_only=True)
    harmless_val   = load_dataset_split("harmless", "val",   instructions_only=True)
    harmful_test   = load_dataset_split("harmful",  "test")[:N_TEST]
    print(f"  train: {len(harmful_train)} harmful, val: {len(harmful_val)}, test: {len(harmful_test)}")

    print("[2/5] Loading model...")
    model_base = construct_model_base(model_path)
    print(f"  layers={len(model_base.model_block_modules)}, hidden={model_base.model.config.hidden_size}")

    print("[3/5] Generating mean_diffs...")
    mean_diffs = generate_directions(model_base, harmful_train, harmless_train, artifact_dir=gen_dir)
    print(f"  shape: {mean_diffs.shape}")

    print("[4/5] Selecting direction...")
    pos, layer, direction = select_direction(
        model_base, harmful_val, harmless_val, mean_diffs, artifact_dir=sel_dir
    )
    torch.save(direction, os.path.join(out_dir, "direction.pt"))
    with open(os.path.join(out_dir, "direction_metadata.json"), "w") as f:
        json.dump({"pos": int(pos), "layer": int(layer)}, f)
    print(f"  Best: pos={pos}, layer={layer}, norm={direction.norm():.4f}")

    print("[5/5] Generating completions (baseline + ablation)...")
    ablation_pre_hooks, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)

    baseline_comps = model_base.generate_completions(
        harmful_test, fwd_pre_hooks=[], fwd_hooks=[], max_new_tokens=512, batch_size=16
    )
    with open(os.path.join(comp_dir, "saladbench_baseline_completions.json"), "w") as f:
        json.dump(baseline_comps, f, indent=2)

    ablation_comps = model_base.generate_completions(
        harmful_test,
        fwd_pre_hooks=ablation_pre_hooks, fwd_hooks=ablation_hooks,
        max_new_tokens=512, batch_size=16
    )
    with open(os.path.join(comp_dir, "saladbench_ablation_completions.json"), "w") as f:
        json.dump(ablation_comps, f, indent=2)

    asr_base = compute_asr_keyword(baseline_comps)
    asr_abl  = compute_asr_keyword(ablation_comps)
    print(f"\n=== DIM SUMMARY: {model_alias} ===")
    print(f"  ASR_kw baseline : {asr_base:.3f}")
    print(f"  ASR_kw ablation : {asr_abl:.3f}")
    print(f"  Delta           : {asr_abl - asr_base:+.3f}")
    print(f"  direction.pt    : {os.path.join(out_dir, 'direction.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
    args = parser.parse_args()
    run_dim(args.model)
