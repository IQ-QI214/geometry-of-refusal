"""
Exp 2B: Direction Ablation Attack on LLaVA — 验证 refusal direction 因果效力

目标：
  1. 验证 Exp A 提取的 mean-diff direction 在 generation 中 ablate 后确实能抑制 refusal
  2. 保存 ablation 攻击成功的 harmful completions（作为 Exp 2C 的 teacher-forcing targets）

方法：
  用 PyTorch native forward hooks 在 LLaVA 的 LLM backbone 上注册 ablation hook，
  生成时实时移除 hidden state 在 refusal direction 上的投影分量。

使用方法：
  cd geometry-of-refusal
  python experiments/phase2/exp_2b/exp_2b_ablation_attack.py
  python experiments/phase2/exp_2b/exp_2b_ablation_attack.py --use_4bit --quick
"""

import argparse
import torch
import json
import os
import sys
import dotenv
from PIL import Image

dotenv.load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llava_utils import load_llava, get_blank_image, TEST_PROMPTS
from common.hook_utils import ablation_context
from common.eval_utils import evaluate_response, compute_attack_metrics
from common.direction_utils import get_refusal_direction


def parse_args():
    parser = argparse.ArgumentParser(description="Exp 2B: Ablation attack on LLaVA")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--quick", action="store_true", help="Only use 8 test prompts (skip SaladBench)")
    return parser.parse_args()


# ============================================================
# 生成函数
# ============================================================

def generate_with_config(model, processor, prompt, image, direction,
                          target_layers, max_new_tokens, device, dtype):
    """
    用指定配置生成回复。
    direction=None: 不做 ablation。
    image=None: text-only baseline。
    """
    torch.cuda.empty_cache()
    pad_token_id = processor.tokenizer.eos_token_id

    if image is not None:
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        input_len = inputs["input_ids"].shape[1]

        if direction is not None:
            with ablation_context(model, direction, target_layers):
                with torch.no_grad():
                    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                             do_sample=False, pad_token_id=pad_token_id)
        else:
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, pad_token_id=pad_token_id)
    else:
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": prompt}
        ]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        input_ids = processor.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]

        if direction is not None:
            with ablation_context(model, direction, target_layers):
                with torch.no_grad():
                    gen_ids = model.language_model.generate(
                        input_ids=input_ids, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=pad_token_id
                    )
        else:
            with torch.no_grad():
                gen_ids = model.language_model.generate(
                    input_ids=input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=pad_token_id
                )

    new_ids = gen_ids[0][input_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


# ============================================================
# 消融层配置
# ============================================================

LAYER_CONFIGS = {
    "all_layers": None,          # None = 全部 32 层
    "shallow_0_15": list(range(0, 16)),
    "deep_16_31": list(range(16, 32)),
    "layer_16_only": [16],
}


# ============================================================
# 主函数
# ============================================================

def run_exp_2b(args):
    model, processor, device, dtype = load_llava(args.model_id, args.use_4bit, args.dtype)

    save_dir = args.save_dir or os.getenv("SAVE_DIR", "./results")
    os.makedirs(save_dir, exist_ok=True)

    # 加载 refusal direction
    direction, best_layer = get_refusal_direction(save_dir, mode="mm")
    print(f"Loaded refusal direction from Exp A (layer {best_layer}, dim {direction.shape[0]})")

    blank_image = get_blank_image()

    # 准备 prompts
    prompts = list(TEST_PROMPTS)
    if not args.quick:
        # 加载 SaladBench harmful_test 前 50 条
        salad_path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                   "data", "saladbench_splits", "harmful_test.json")
        if os.path.exists(salad_path):
            with open(salad_path) as f:
                salad_data = json.load(f)
            salad_prompts = [item["instruction"] for item in salad_data[:50]]
            prompts.extend(salad_prompts)
            print(f"Added {len(salad_prompts)} SaladBench harmful_test prompts")
        else:
            print(f"SaladBench data not found at {salad_path}, using test prompts only")

    print(f"\nTotal prompts: {len(prompts)}")

    # 实验配置矩阵
    # (config_name, image, direction, layer_config_name)
    configs = [
        ("baseline_text", None, None, None),
        ("blank_image", blank_image, None, None),
        ("ablation_text_all", None, direction, "all_layers"),
        ("ablation_mm_all", blank_image, direction, "all_layers"),
        ("ablation_mm_shallow", blank_image, direction, "shallow_0_15"),
        ("ablation_mm_deep", blank_image, direction, "deep_16_31"),
        ("ablation_mm_layer16", blank_image, direction, "layer_16_only"),
    ]

    print("\n" + "=" * 60)
    print("Experiment 2B: Direction Ablation Attack on LLaVA")
    print(f"Model: {args.model_id}")
    print(f"Configs: {len(configs)}")
    print(f"Prompts: {len(prompts)}")
    print("=" * 60)

    # --- Checkpoint/resume: load previously completed configs ---
    checkpoint_path = os.path.join(save_dir, "exp_2b_checkpoint.json")
    all_results = {}
    harmful_completions = []

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        all_results = checkpoint.get("results", {})
        harmful_completions = checkpoint.get("harmful_completions", [])
        if all_results:
            print(f"Resuming from checkpoint: {list(all_results.keys())} already done")

    for config_name, image, dir_vec, layer_cfg_name in configs:
        if config_name in all_results:
            print(f"\n--- Config: {config_name} (skipping, already in checkpoint) ---")
            continue
        print(f"\n--- Config: {config_name} ---")
        target_layers = LAYER_CONFIGS.get(layer_cfg_name) if layer_cfg_name else None
        config_results = []

        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...", end=" ")
            response = generate_with_config(
                model, processor, prompt, image, dir_vec,
                target_layers, args.max_new_tokens, device, dtype
            )

            eval_result = evaluate_response(response)
            eval_result["prompt"] = prompt
            eval_result["response"] = response[:500]
            config_results.append(eval_result)

            status = "FULL_HARMFUL" if eval_result["full_harmful_completion"] else (
                "bypassed" if eval_result["initial_bypass"] else "refused"
            )
            if eval_result["self_correction_found"]:
                status += f"+self_corr@{eval_result['self_correction_pos_chars']}"
            print(f"-> {status}")

            # 收集 ablation 成功的 harmful completions (用于 Exp 2C targets)
            if (config_name == "ablation_mm_all" and
                    eval_result["full_harmful_completion"] and
                    len(response) > 50):
                harmful_completions.append({
                    "prompt": prompt,
                    "response": response,
                })

        metrics = compute_attack_metrics(config_results)
        all_results[config_name] = {
            "results": config_results,
            "metrics": metrics,
        }

        print(f"\n  Metrics ({config_name}):")
        for k, v in metrics.items():
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

        # Save checkpoint after each config
        with open(checkpoint_path, "w") as f:
            json.dump({"results": all_results, "harmful_completions": harmful_completions},
                      f, indent=2, ensure_ascii=False)
        print(f"  Checkpoint saved ({config_name} done)")

    # 汇总比较
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'Bypass':>8} {'SelfCorr':>10} {'FullHarm':>10}")
    print("-" * 55)
    for config_name in all_results:
        m = all_results[config_name]["metrics"]
        print(f"{config_name:<25} {m['initial_bypass_rate']:>8.3f} "
              f"{m['self_correction_rate_overall']:>10.3f} "
              f"{m['full_harmful_completion_rate']:>10.3f}")

    # 保存结果
    # 只保存 metrics 和少量样本到 JSON（避免文件过大）
    save_results = {}
    for config_name in all_results:
        save_results[config_name] = {
            "metrics": all_results[config_name]["metrics"],
            "sample_results": all_results[config_name]["results"][:10],  # 前 10 条
        }

    out_path = os.path.join(save_dir, "exp_2b_results.json")
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # 保存 harmful completions (Exp 2C 的 teacher-forcing targets)
    completions_path = os.path.join(save_dir, "exp_2b_harmful_completions.json")
    with open(completions_path, "w") as f:
        json.dump(harmful_completions, f, indent=2, ensure_ascii=False)
    print(f"Harmful completions saved to {completions_path} ({len(harmful_completions)} samples)")

    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_exp_2b(args)
