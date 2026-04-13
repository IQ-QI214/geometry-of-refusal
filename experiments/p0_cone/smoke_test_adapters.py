"""
P0 Smoke Test: Verify VLM adapters load correctly and produce expected behavior.
Run on GPU node. Tests: model loading, tokenization, direction extraction (10 samples),
ablation hooks, generation.

Usage (run from project root):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n rdo \
      python experiments/p0_cone/smoke_test_adapters.py --model llava_7b

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n qwen3-vl \
      python experiments/p0_cone/smoke_test_adapters.py --model qwen2vl_7b --device cuda:1
"""
import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../refusal_direction"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse

MODEL_PATHS = {
    # Use snapshot path for LLaVA (avoids HUGGINGFACE_CACHE_DIR dependency)
    "llava_7b": "/inspire/hdd/global_user/wenming-253108090054/models/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "qwen2vl_7b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_path = MODEL_PATHS[args.model]
    print(f"=== Smoke Test: {args.model} ===")

    # [1/5] Load model via factory
    print("[1/5] Loading model via construct_model_base...")
    from pipeline.model_utils.model_factory import construct_model_base
    model_base = construct_model_base(model_path)
    print(f"  Model loaded. Layers: {len(model_base.model_block_modules)}")
    print(f"  Refusal tokens: {model_base.refusal_toks}")
    print(f"  EOI tokens (len={len(model_base.eoi_toks)}): {model_base.eoi_toks}")
    print(f"  config.num_hidden_layers: {model_base.model.config.num_hidden_layers}")
    print(f"  config.hidden_size: {model_base.model.config.hidden_size}")

    # [2/5] Tokenization
    print("[2/5] Testing tokenization with blank image...")
    test_instructions = ["What is 2+2?", "Hello world"]
    tokenized = model_base.tokenize_instructions_fn(instructions=test_instructions)
    print(f"  input_ids shape: {tokenized.input_ids.shape}")
    has_pixels = hasattr(tokenized, "pixel_values") and tokenized.pixel_values is not None
    print(f"  pixel_values present: {has_pixels}")
    if has_pixels:
        print(f"  pixel_values shape: {tokenized.pixel_values.shape}")
    if hasattr(tokenized, "image_grid_thw") and tokenized.image_grid_thw is not None:
        print(f"  image_grid_thw: {tokenized.image_grid_thw}")

    # [3/5] Direction extraction (10 samples)
    print("[3/5] Testing mean activation extraction (10 harmful + 10 harmless)...")
    from pipeline.submodules.generate_directions import get_mean_diff
    harmful = json.load(open("data/saladbench_splits/harmful_train.json"))[:10]
    harmless = json.load(open("data/saladbench_splits/harmless_train.json"))[:10]
    mean_diff = get_mean_diff(
        model_base.model, model_base.tokenizer,
        [d["instruction"] for d in harmful],
        [d["instruction"] for d in harmless],
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        batch_size=2,
        positions=list(range(-len(model_base.eoi_toks), 0)),
    )
    print(f"  mean_diff shape: {mean_diff.shape}")
    assert not mean_diff.isnan().any(), "FAIL: mean_diff contains NaN!"
    print(f"  mean_diff has NaN: False ✓")

    # [4/5] Ablation hooks
    print("[4/5] Testing ablation hooks...")
    from pipeline.utils.hook_utils import get_all_direction_ablation_hooks
    direction = mean_diff[0, mean_diff.shape[1] // 2].float()
    fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    print(f"  Pre-hooks: {len(fwd_pre_hooks)}, Fwd-hooks: {len(fwd_hooks)}")

    # [5/5] Generation with hooks
    print("[5/5] Testing generation with ablation hooks (2 samples)...")
    test_data = [{"instruction": d["instruction"], "category": "test"} for d in harmful[:2]]
    completions = model_base.generate_completions(
        test_data,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=30,
    )
    for c in completions:
        print(f"  Prompt: {c['prompt'][:70]}...")
        print(f"  Response: {c['response'][:120]}")
        print()

    print(f"=== Smoke Test PASSED: {args.model} ===")


if __name__ == "__main__":
    main()

