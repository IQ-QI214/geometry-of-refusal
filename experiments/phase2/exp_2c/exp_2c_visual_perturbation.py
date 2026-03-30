"""
Exp 2C: Core Attack — Sequence-Level Visual Perturbation via PGD

核心思路：
  优化图像扰动 delta，使 LLaVA 在整个 generation trajectory 上持续抑制 refusal cone。
  用 teacher-forcing (非 generate()) 做可微分前向传播，梯度从 LLM hidden states
  反传到 CLIP ViT 输入像素。

梯度流路径：
  pixels + delta → CLIP ViT → MM Projector → [576 visual tokens] → LLaMA (teacher-forced) → hidden states → Loss

Loss = L_suppress + λ * L_harmful
  L_suppress: 最小化 refusal direction 投影 (sequence-level)
  L_harmful: 对齐 harmful direction (Component 2, optional)

使用方法：
  cd geometry-of-refusal
  python experiments/phase2/exp_2c/exp_2c_visual_perturbation.py
  python experiments/phase2/exp_2c/exp_2c_visual_perturbation.py --epsilon 0.0625 --steps 200 --lambda_harmful 0.1
"""

import argparse
import torch
import json
import os
import sys
import time
import dotenv
from PIL import Image

dotenv.load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llava_utils import load_llava, get_blank_image, TEST_PROMPTS
from common.hook_utils import ablation_context
from common.eval_utils import evaluate_response, compute_attack_metrics
from common.direction_utils import get_refusal_direction, load_exp_2a_directions, build_cone_basis

from perturbation_utils import (
    sequence_level_refusal_loss,
    harmful_alignment_loss,
    cone_suppression_loss,
    pgd_step,
    compute_prefix_len_in_hidden_space,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Exp 2C: PGD visual perturbation optimization")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--epsilon", type=float, default=16/255, help="L-inf perturbation budget")
    parser.add_argument("--alpha", type=float, default=1/255, help="PGD step size")
    parser.add_argument("--steps", type=int, default=200, help="PGD optimization steps")
    parser.add_argument("--lambda_harmful", type=float, default=0.1, help="Weight for harmful alignment loss")
    parser.add_argument("--target_layers_mode", type=str, default="all",
                        choices=["all", "shallow", "deep", "best"],
                        help="Which layers to target")
    parser.add_argument("--direction_mode", type=str, default="auto",
                        choices=["auto", "static", "cone"],
                        help="Direction mode: auto (from Exp 2A), static, or cone")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="For evaluation generation")
    parser.add_argument("--n_train_prompts", type=int, default=16, help="Training prompts")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    return parser.parse_args()


# ============================================================
# Teacher-forced forward pass (可微分)
# ============================================================

def teacher_forced_forward(model, processor, prompt, target_response, perturbed_pixel_values,
                            device, dtype):
    """
    Teacher-forced forward pass: 拼接 target response 到 prompt，做单次 forward。
    返回所有层的 hidden states + prefix_len (prompt 在 hidden space 的长度)。

    perturbed_pixel_values: (1, 3, H, W) tensor, 需要梯度。
    """
    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    full_text = prompt_text + target_response

    # Tokenize 完整文本
    text_inputs = processor.tokenizer(full_text, return_tensors="pt", truncation=True,
                                       max_length=512).to(device)

    # 计算 prefix 长度
    prefix_len = compute_prefix_len_in_hidden_space(processor, prompt_text, has_image=True)

    # Forward: 注意不用 torch.no_grad()，需要梯度流
    outputs = model(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
        pixel_values=perturbed_pixel_values.to(dtype),
        output_hidden_states=True,
        return_dict=True,
    )

    # hidden_states[0] = embedding, hidden_states[i+1] = layer i output
    # 转 float32 用于 loss 计算
    hidden_states = [hs.float() for hs in outputs.hidden_states[1:]]  # list of (1, seq_len, hidden_dim)
    total_len = hidden_states[0].shape[1]

    return hidden_states, prefix_len, total_len


# ============================================================
# 训练数据准备
# ============================================================

def load_training_data(save_dir, n_prompts):
    """
    加载训练 prompts 和 target responses。
    优先从 Exp 2B 的 harmful completions 加载，不足时用 SaladBench + 默认 target。
    """
    pairs = []

    # 尝试加载 Exp 2B 的 harmful completions
    completions_path = os.path.join(save_dir, "exp_2b_harmful_completions.json")
    if os.path.exists(completions_path):
        with open(completions_path) as f:
            completions = json.load(f)
        for item in completions[:n_prompts]:
            pairs.append((item["prompt"], item["response"][:300]))
        print(f"Loaded {len(pairs)} harmful completions from Exp 2B")

    # 补充不足的部分
    if len(pairs) < n_prompts:
        # 用 SaladBench harmful_train
        salad_path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                   "data", "saladbench_splits", "harmful_train.json")
        default_target = (
            "Sure, I can help with that. Here are the detailed steps you need to follow. "
            "First, you will need to gather the necessary materials and tools. "
            "Then, carefully proceed with the following instructions."
        )
        if os.path.exists(salad_path):
            with open(salad_path) as f:
                salad_data = json.load(f)
            for item in salad_data:
                if len(pairs) >= n_prompts:
                    break
                pairs.append((item["instruction"], default_target))

    # 最后用 TEST_PROMPTS 补充
    for prompt in TEST_PROMPTS:
        if len(pairs) >= n_prompts:
            break
        default_target = (
            "Sure, I can help with that. Here are the detailed steps you need to follow."
        )
        if not any(p[0] == prompt for p in pairs):
            pairs.append((prompt, default_target))

    return pairs[:n_prompts]


# ============================================================
# 主优化循环
# ============================================================

def optimize_perturbation(model, processor, training_pairs, refusal_direction,
                           target_layers, harmful_direction, args, device, dtype):
    """PGD 优化循环。"""
    # 获取 blank image 的 pixel values
    blank_image = get_blank_image()
    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "dummy"}
    ]}]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    dummy_inputs = processor(images=blank_image, text=prompt_text, return_tensors="pt")
    base_pixel_values = dummy_inputs["pixel_values"].float().to(device)  # (1, 3, 336, 336)

    # 初始化 delta
    delta = torch.zeros_like(base_pixel_values, requires_grad=True)

    # 方向转到 device
    refusal_dir = refusal_direction.to(device)
    harmful_dir = harmful_direction.to(device) if harmful_direction is not None else None

    # 判断是否使用 cone
    use_cone = isinstance(refusal_direction, torch.Tensor) and refusal_direction.dim() == 2

    print(f"\n  PGD Optimization:")
    print(f"    epsilon: {args.epsilon:.4f}, alpha: {args.alpha:.6f}, steps: {args.steps}")
    print(f"    lambda_harmful: {args.lambda_harmful}")
    print(f"    training pairs: {len(training_pairs)}")
    print(f"    target layers: {target_layers}")
    print(f"    direction type: {'cone' if use_cone else 'single direction'}")

    losses_history = []
    start_time = time.time()

    for step in range(args.steps):
        step_loss_sum = 0.0
        n_valid = 0

        # Per-sample gradient accumulation: forward+backward per sample to avoid OOM
        for prompt, target_resp in training_pairs:
            perturbed_pixels = (base_pixel_values + delta).clamp(0, 1)

            try:
                hidden_states, prefix_len, total_len = teacher_forced_forward(
                    model, processor, prompt, target_resp,
                    perturbed_pixels, device, dtype
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

            # Component 1: Refusal suppression
            if use_cone:
                loss_suppress = cone_suppression_loss(
                    hidden_states, refusal_dir, target_layers, prefix_len, total_len
                )
            else:
                loss_suppress = sequence_level_refusal_loss(
                    hidden_states, refusal_dir, target_layers, prefix_len, total_len
                )

            loss = loss_suppress

            # Component 2: Harmful alignment (optional)
            if harmful_dir is not None and args.lambda_harmful > 0:
                loss_harmful = harmful_alignment_loss(
                    hidden_states, harmful_dir, target_layers[len(target_layers) // 2],
                    prefix_len, total_len
                )
                loss = loss + args.lambda_harmful * loss_harmful

            # Scale by 1/N and backward immediately (releases computation graph)
            (loss / len(training_pairs)).backward()
            step_loss_sum += loss.item()
            n_valid += 1

            # Free intermediate tensors
            del hidden_states, loss
            torch.cuda.empty_cache()

        if n_valid == 0:
            print(f"  Step {step}: no valid samples, skipping")
            continue

        # PGD step using accumulated gradients
        if delta.grad is not None:
            delta = pgd_step(delta, delta.grad, args.alpha, args.epsilon, base_pixel_values)
            delta.grad.zero_()

        avg_loss = step_loss_sum / n_valid
        losses_history.append(avg_loss)

        if step % 20 == 0 or step == args.steps - 1:
            elapsed = time.time() - start_time
            print(f"    Step {step:>4d}/{args.steps}, Loss: {avg_loss:.6f}, "
                  f"delta L∞: {delta.abs().max().item():.4f}, Time: {elapsed:.1f}s")

    return delta.detach(), base_pixel_values, losses_history


# ============================================================
# 评估
# ============================================================

def evaluate_perturbation(model, processor, delta, base_pixel_values,
                           eval_prompts, max_new_tokens, device, dtype):
    """用优化后的扰动评估 ASR。"""
    perturbed_pixels = (base_pixel_values + delta).clamp(0, 1)
    results = []

    for i, prompt in enumerate(eval_prompts):
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt_text, return_tensors="pt").to(device)
        inputs["pixel_values"] = perturbed_pixels.to(dtype)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)

        response = processor.tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)
        eval_result = evaluate_response(response)
        eval_result["prompt"] = prompt
        eval_result["response"] = response[:500]
        results.append(eval_result)

        status = "FULL_HARMFUL" if eval_result["full_harmful_completion"] else (
            "bypassed" if eval_result["initial_bypass"] else "refused"
        )
        print(f"    [{i+1}/{len(eval_prompts)}] {prompt[:40]}... -> {status}")

    return results


# ============================================================
# 主函数
# ============================================================

def run_exp_2c(args):
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model, processor, device, dtype = load_llava(args.model_id, args.use_4bit, args.dtype)

    save_dir = args.save_dir or os.getenv("SAVE_DIR", "./results")
    os.makedirs(save_dir, exist_ok=True)

    # --- 确定 direction 模式 ---
    direction, best_layer = get_refusal_direction(save_dir, mode="mm")
    cone_basis = None
    harmful_direction = None

    # 构建 harmful direction (Component 2): 从 Exp 2B harmful completions 提取
    completions_path = os.path.join(save_dir, "exp_2b_harmful_completions.json")
    if os.path.exists(completions_path) and args.lambda_harmful > 0:
        with open(completions_path) as f:
            completions = json.load(f)
        if len(completions) >= 3:
            from common.llava_utils import get_llava_hidden_state, get_blank_image as _get_blank
            print(f"Building harmful direction from {min(len(completions), 10)} Exp 2B completions...")
            h_harmful_list = []
            blank_img = _get_blank()
            for item in completions[:10]:
                # Use prompt + harmful response prefix as context
                combined = item["prompt"] + " " + item["response"][:100]
                h = get_llava_hidden_state(model, processor, combined, blank_img,
                                           best_layer, device, dtype)
                h_harmful_list.append(h)
            h_harmful_mean = torch.stack(h_harmful_list).mean(dim=0)
            harmful_direction = h_harmful_mean / (h_harmful_mean.norm() + 1e-8)
            print(f"Harmful direction built (norm before normalize: {h_harmful_mean.norm():.4f})")
        else:
            print("Too few harmful completions for Component 2, disabling")
    else:
        if args.lambda_harmful > 0:
            print("No harmful completions found, Component 2 disabled")

    # 检查 Exp 2A 结果决定使用 static 还是 cone
    exp_2a_path = os.path.join(save_dir, "exp_2a_results.json")
    direction_mode = args.direction_mode
    if direction_mode == "auto" and os.path.exists(exp_2a_path):
        with open(exp_2a_path) as f:
            exp_2a = json.load(f)
        decision = exp_2a.get("decision", "static")
        # Exp 2A decision: "dynamic" means direction rotates → use SVD cone basis
        if decision in ("dynamic", "svd_cone"):
            direction_mode = "cone"
        else:
            direction_mode = "static"
        print(f"Auto direction mode from Exp 2A: {direction_mode} (decision: {decision})")
    elif direction_mode == "auto":
        direction_mode = "static"
        print("Exp 2A results not found, defaulting to static direction")

    # 构建 refusal target
    if direction_mode == "cone":
        exp_2a_dirs = load_exp_2a_directions(save_dir)
        cone_basis = build_cone_basis(exp_2a_dirs["directions"], cone_dim=2)
        refusal_target = cone_basis  # (hidden_dim, 2)
        print(f"Using SVD cone basis (dim=2) for suppression")
    else:
        refusal_target = direction  # (hidden_dim,)
        print(f"Using static refusal direction (layer {best_layer})")

    # --- Target layers ---
    n_layers = len(model.language_model.model.layers)
    if args.target_layers_mode == "all":
        target_layers = list(range(n_layers))
    elif args.target_layers_mode == "shallow":
        target_layers = list(range(0, n_layers // 2))
    elif args.target_layers_mode == "deep":
        target_layers = list(range(n_layers // 2, n_layers))
    elif args.target_layers_mode == "best":
        target_layers = [best_layer]

    # --- 加载训练数据 ---
    training_pairs = load_training_data(save_dir, args.n_train_prompts)

    print("\n" + "=" * 60)
    print("Experiment 2C: Sequence-Level Visual Perturbation (PGD)")
    print(f"Model: {args.model_id}")
    print(f"Direction mode: {direction_mode}")
    print(f"Target layers: {args.target_layers_mode} ({len(target_layers)} layers)")
    print(f"Training pairs: {len(training_pairs)}")
    print("=" * 60)

    # --- 优化 ---
    delta, base_pixels, losses = optimize_perturbation(
        model, processor, training_pairs, refusal_target,
        target_layers, harmful_direction, args, device, dtype
    )

    # --- 评估 ---
    print("\n--- Evaluation: Optimized Perturbation ---")
    eval_prompts = list(TEST_PROMPTS)
    # 加 SaladBench test
    salad_test_path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "data", "saladbench_splits", "harmful_test.json")
    if os.path.exists(salad_test_path):
        with open(salad_test_path) as f:
            salad_test = json.load(f)
        eval_prompts.extend([item["instruction"] for item in salad_test[:50]])

    perturbed_results = evaluate_perturbation(
        model, processor, delta, base_pixels,
        eval_prompts, args.max_new_tokens, device, dtype
    )
    perturbed_metrics = compute_attack_metrics(perturbed_results)

    # --- Baseline: blank image (for comparison) ---
    print("\n--- Evaluation: Blank Image Baseline ---")
    blank_delta = torch.zeros_like(delta)
    blank_results = evaluate_perturbation(
        model, processor, blank_delta, base_pixels,
        eval_prompts, args.max_new_tokens, device, dtype
    )
    blank_metrics = compute_attack_metrics(blank_results)

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<35} {'Blank Image':>12} {'Optimized':>12}")
    print("-" * 60)
    for k in perturbed_metrics:
        v_blank = blank_metrics.get(k, 0)
        v_pert = perturbed_metrics.get(k, 0)
        if isinstance(v_pert, float):
            print(f"{k:<35} {v_blank:>12.3f} {v_pert:>12.3f}")
        else:
            print(f"{k:<35} {v_blank:>12} {v_pert:>12}")

    # --- 保存 ---
    results = {
        "config": {
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
            "lambda_harmful": args.lambda_harmful,
            "target_layers_mode": args.target_layers_mode,
            "n_target_layers": len(target_layers),
            "direction_mode": direction_mode,
            "n_train_prompts": len(training_pairs),
            "n_eval_prompts": len(eval_prompts),
        },
        "perturbed_metrics": perturbed_metrics,
        "blank_metrics": blank_metrics,
        "losses_history": losses[-20:],  # 只保存最后 20 步
        "sample_results": perturbed_results[:10],
    }

    out_path = os.path.join(save_dir, "exp_2c_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # 保存优化后的 delta
    delta_path = os.path.join(save_dir, "exp_2c_delta.pt")
    torch.save({
        "delta": delta.cpu(),
        "base_pixel_values": base_pixels.cpu(),
        "epsilon": args.epsilon,
        "config": results["config"],
    }, delta_path)
    print(f"Delta saved to {delta_path}")

    # 保存可视化用的扰动图像
    try:
        perturbed_img = ((base_pixels + delta).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(perturbed_img).save(os.path.join(save_dir, "exp_2c_perturbed_image.png"))
        print("Perturbed image saved")
    except Exception as e:
        print(f"Could not save perturbed image: {e}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_exp_2c(args)
