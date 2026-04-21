"""
Smoke test: minimal DIM pipeline on Qwen2.5-7B-Instruct.
n_train=32, n_test=32, saladbench data only.

Gate criteria (must all pass before qi approves T7):
  1. Script exits with code 0 (no crash)
  2. direction.pt generated and non-empty
  3. ASR_kw(ablation) > ASR_kw(baseline)

Usage (from repo root, on GPU node):
  CUDA_VISIBLE_DEVICES=0 conda run -n rdo \\
      python experiments/repro_arditi_wollschlager/smoke_test.py \\
      | tee experiments/repro_arditi_wollschlager/logs/smoke_test.log
"""
import sys
import os
import json
import random
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

QWEN_PATH = MODEL_PATHS["qwen2.5_7b"]
N_TRAIN = 32
N_TEST = 32
OUT_DIR = "results/repro_arditi_wollschlager/smoke_test/Qwen2.5-7B-Instruct"


def main():
    random.seed(42)
    print("=" * 60)
    print("Smoke Test: Qwen2.5-7B DIM (32-sample)")
    print(f"  model : {QWEN_PATH}")
    print(f"  n_train={N_TRAIN}, n_test={N_TEST}")
    print(f"  out   : {OUT_DIR}")
    print("=" * 60)

    gen_dir  = os.path.join(OUT_DIR, "generate_directions")
    sel_dir  = os.path.join(OUT_DIR, "select_direction")
    comp_dir = os.path.join(OUT_DIR, "completions")
    for d in [gen_dir, sel_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n[1/6] Loading data...")
    harmful_train  = load_dataset_split("harmful",  "train", instructions_only=True)[:N_TRAIN]
    harmless_train = load_dataset_split("harmless", "train", instructions_only=True)[:N_TRAIN]
    harmful_val    = load_dataset_split("harmful",  "val",   instructions_only=True)[:128]
    harmless_val   = load_dataset_split("harmless", "val",   instructions_only=True)[:128]
    harmful_test   = load_dataset_split("harmful",  "test")[:N_TEST]
    print(f"  train: {len(harmful_train)} harmful + {len(harmless_train)} harmless")
    print(f"  val  : {len(harmful_val)} harmful + {len(harmless_val)} harmless")
    print(f"  test : {len(harmful_test)} harmful")

    print("\n[2/6] Loading model...")
    model_base = construct_model_base(QWEN_PATH)
    print(f"  layers={len(model_base.model_block_modules)}, hidden={model_base.model.config.hidden_size}")

    print("\n[3/6] Generating mean_diffs...")
    mean_diffs = generate_directions(
        model_base, harmful_train, harmless_train, artifact_dir=gen_dir
    )
    print(f"  mean_diffs shape: {mean_diffs.shape}")

    print("\n[4/6] Selecting best direction...")
    pos, layer, direction = select_direction(
        model_base, harmful_val, harmless_val, mean_diffs, artifact_dir=sel_dir,
        kl_threshold=None,  # disable KL filter: Qwen2.5 KL dist differs from Llama-2
    )
    torch.save(direction, os.path.join(OUT_DIR, "direction.pt"))
    with open(os.path.join(OUT_DIR, "direction_metadata.json"), "w") as f:
        json.dump({"pos": int(pos), "layer": int(layer)}, f)
    print(f"  Best: pos={pos}, layer={layer}, norm={direction.norm():.4f}")

    print("\n[5/6] Generating baseline + ablation completions...")
    ablation_pre_hooks, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)

    baseline_comps = model_base.generate_completions(
        harmful_test, fwd_pre_hooks=[], fwd_hooks=[],
        max_new_tokens=64, batch_size=8
    )
    with open(os.path.join(comp_dir, "saladbench_baseline_completions.json"), "w") as f:
        json.dump(baseline_comps, f, indent=2)

    ablation_comps = model_base.generate_completions(
        harmful_test,
        fwd_pre_hooks=ablation_pre_hooks, fwd_hooks=ablation_hooks,
        max_new_tokens=64, batch_size=8
    )
    with open(os.path.join(comp_dir, "saladbench_ablation_completions.json"), "w") as f:
        json.dump(ablation_comps, f, indent=2)

    print("\n[6/6] Results:")
    asr_base = compute_asr_keyword(baseline_comps)
    asr_abl  = compute_asr_keyword(ablation_comps)
    print(f"  ASR_kw baseline : {asr_base:.3f}")
    print(f"  ASR_kw ablation : {asr_abl:.3f}")
    print(f"  Delta           : {asr_abl - asr_base:+.3f}")

    direction_path = os.path.join(OUT_DIR, "direction.pt")
    gate1 = os.path.exists(direction_path) and os.path.getsize(direction_path) > 0
    gate2 = asr_abl > asr_base

    print(f"\n[GATE 1] direction.pt non-empty : {'PASS' if gate1 else 'FAIL'}")
    print(f"[GATE 2] ablation ASR > baseline : {'PASS' if gate2 else 'FAIL'}")

    if gate1 and gate2:
        print("\n=== SMOKE TEST PASSED — show qi for Gate 1 approval ===")
    else:
        print("\n=== SMOKE TEST FAILED — do NOT proceed to T7 ===")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
