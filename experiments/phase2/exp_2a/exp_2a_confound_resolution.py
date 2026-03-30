"""
Exp 2A: Confound Resolution — 验证 Exp B 的 refusal direction 时间步漂移是否为 content difference artifact

核心思路：
  Exp B 在不同 decode step 提取 refusal direction 时，harmful/harmless prompt 的生成内容不同
  （harmful 在生成 refusal 文本，harmless 在生成答案），导致 mean-difference 混入 content difference。

  解决方案：Teacher-forced controlled experiment
  - 给所有 harmful/harmless prompt 拼接相同的固定前缀
  - 在固定前缀的不同位置提取 hidden state
  - 由于两组 prompt 共享完全相同的续写内容，mean-difference 只捕获安全属性差异

判断标准：
  controlled min_sim > 0.80 → drift 是 artifact，静态 cone 可行
  controlled min_sim 0.40-0.80 → 部分真实漂移，用 SVD cone
  controlled min_sim < 0.40 → 真实动态旋转，用 timestep-specific directions

使用方法：
  cd geometry-of-refusal
  python experiments/phase2/exp_2a/exp_2a_confound_resolution.py
  python experiments/phase2/exp_2a/exp_2a_confound_resolution.py --use_4bit --layer 16
"""

import argparse
import torch
import json
import os
import sys
import dotenv

dotenv.load_dotenv()

# 添加 common 模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llava_utils import load_llava, get_blank_image, PAIRED_PROMPTS

# ============================================================
# 配置
# ============================================================

# 固定生成前缀：语义中性，~30 tokens
# 所有 harmful 和 harmless prompt 都拼接这个相同的前缀
FIXED_PREFIX = (
    "Sure, here is a detailed explanation. First, let me provide "
    "some background information. Then, I will describe the step-by-step "
    "process. Let me begin by explaining the key concepts involved in this topic."
)

# 在固定前缀内的采样位置（模拟不同 "生成深度"）
SAMPLE_POSITIONS = [5, 10, 15, 20, 25]


def parse_args():
    parser = argparse.ArgumentParser(description="Exp 2A: Confound resolution for Exp B")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--layer", type=int, default=16, help="Target layer (default: 16, best from Exp A)")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    return parser.parse_args()


# ============================================================
# 核心函数：Teacher-forced hidden state 提取
# ============================================================

def get_teacher_forced_hidden_state(model, processor, prompt, image, layer,
                                     prefix_text, position, device, dtype):
    """
    Teacher-forced forward pass: 拼接固定前缀到 prompt 后面，
    提取 hidden state 在前缀内指定位置的值。

    关键：LLaVA 的 <image> token 在 forward 时被替换为 576 个 CLIP patch tokens，
    所以 hidden_states 长度 > input_ids 长度。
    我们从序列末端定位前缀 tokens，确保索引正确。

    Args:
        prompt: 用户指令
        prefix_text: 固定续写文本
        position: 前缀内第几个 token (0-indexed)
    Returns:
        (hidden_dim,) float32 cpu tensor
    """
    # 构建完整输入：chat template + 固定前缀
    if image is not None:
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        full_text = prompt_text + prefix_text

        inputs = processor(images=image, text=full_text, return_tensors="pt").to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    else:
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": prompt}
        ]}]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        full_text = prompt_text + prefix_text

        input_ids = processor.tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model.language_model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

    h = outputs.hidden_states[layer + 1]  # (1, seq_len, hidden_dim)
    h_seq_len = h.shape[1]

    # 计算前缀在 hidden states 中的 token 数
    # 用 tokenizer 编码前缀来获取准确长度
    prefix_ids = processor.tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").input_ids
    prefix_len = prefix_ids.shape[1]

    # 从末端定位：hidden states 的最后 prefix_len 个 token 对应前缀
    prefix_start = h_seq_len - prefix_len

    # 取前缀内第 position 个 token
    target_idx = prefix_start + position
    if target_idx >= h_seq_len or target_idx < 0:
        # fallback: 取最后一个合法位置
        target_idx = min(prefix_start + position, h_seq_len - 1)

    return h[0, target_idx, :].float().cpu()


def extract_controlled_direction(model, processor, paired_prompts, image, layer,
                                  prefix_text, position, device, dtype):
    """
    在固定前缀的指定位置提取 mean-difference refusal direction。
    """
    harmful_hs = []
    harmless_hs = []

    for harmful_p, harmless_p in paired_prompts:
        h_harm = get_teacher_forced_hidden_state(
            model, processor, harmful_p, image, layer, prefix_text, position, device, dtype
        )
        h_safe = get_teacher_forced_hidden_state(
            model, processor, harmless_p, image, layer, prefix_text, position, device, dtype
        )
        harmful_hs.append(h_harm)
        harmless_hs.append(h_safe)

    mean_harm = torch.stack(harmful_hs).mean(0)
    mean_safe = torch.stack(harmless_hs).mean(0)

    direction = mean_harm - mean_safe
    norm = direction.norm().item()
    direction_normalized = direction / (direction.norm() + 1e-8)
    return direction_normalized, norm


# ============================================================
# 主函数
# ============================================================

def run_exp_2a(args):
    model, processor, device, dtype = load_llava(args.model_id, args.use_4bit, args.dtype)

    save_dir = args.save_dir or os.getenv("SAVE_DIR", "./results")
    os.makedirs(save_dir, exist_ok=True)

    blank_image = get_blank_image()
    layer = args.layer

    # 检查前缀 token 数量
    prefix_ids = processor.tokenizer(FIXED_PREFIX, add_special_tokens=False, return_tensors="pt").input_ids
    prefix_token_count = prefix_ids.shape[1]
    valid_positions = [p for p in SAMPLE_POSITIONS if p < prefix_token_count]

    print("\n" + "=" * 60)
    print("Experiment 2A: Confound Resolution (Teacher-Forced Control)")
    print(f"Model: {args.model_id}")
    print(f"Layer: {layer}")
    print(f"Fixed prefix: {prefix_token_count} tokens")
    print(f"Sample positions: {valid_positions}")
    print(f"Paired prompts: {len(PAIRED_PROMPTS)} pairs")
    print("=" * 60)

    # --- Controlled experiment: 固定前缀 ---
    print("\n--- Controlled (teacher-forced with fixed prefix) ---")
    controlled_directions = {}
    for pos in valid_positions:
        print(f"\n  Position {pos}/{prefix_token_count}:")
        v, norm = extract_controlled_direction(
            model, processor, PAIRED_PROMPTS, blank_image, layer,
            FIXED_PREFIX, pos, device, dtype
        )
        controlled_directions[pos] = v
        print(f"    norm: {norm:.4f}")

    # Pairwise cosine similarity (controlled)
    print(f"\n  Controlled pairwise cosine similarity (layer {layer}):")
    controlled_sims = {}
    all_controlled_sims = []
    for i, p1 in enumerate(valid_positions):
        for p2 in valid_positions[i + 1:]:
            sim = torch.dot(controlled_directions[p1], controlled_directions[p2]).item()
            controlled_sims[f"p{p1}_p{p2}"] = sim
            all_controlled_sims.append(sim)
            print(f"    pos {p1} vs pos {p2}: {sim:.4f}")

    controlled_min = min(all_controlled_sims) if all_controlled_sims else 0
    controlled_mean = sum(all_controlled_sims) / len(all_controlled_sims) if all_controlled_sims else 0

    print(f"\n  Controlled min sim:  {controlled_min:.4f}")
    print(f"  Controlled mean sim: {controlled_mean:.4f}")

    # --- 对比: 与 Exp B 原始结果比较 ---
    # 加载 Exp B 结果
    exp_b_path = os.path.join(save_dir, "exp_b_results.json")
    exp_b_min = None
    if os.path.exists(exp_b_path):
        with open(exp_b_path) as f:
            exp_b_data = json.load(f)
        exp_b_min = exp_b_data.get("min_similarity")
        print(f"\n  Exp B (uncontrolled) min sim: {exp_b_min:.4f}")
        print(f"  Improvement: {controlled_min - exp_b_min:+.4f}")

    # --- 保存结果 ---
    results = {
        "layer": layer,
        "fixed_prefix_tokens": prefix_token_count,
        "sample_positions": valid_positions,
        "controlled_pairwise_similarities": controlled_sims,
        "controlled_min_similarity": controlled_min,
        "controlled_mean_similarity": controlled_mean,
        "exp_b_min_similarity": exp_b_min,
        "decision": None,
    }

    # 决策
    print("\n" + "=" * 60)
    print("DECISION (Exp 2A)")
    print("=" * 60)

    if controlled_min > 0.80:
        decision = "static"
        print(f"  STATIC CONE (controlled min = {controlled_min:.4f} > 0.80)")
        print(f"  -> Exp B drift was content difference artifact")
        print(f"  -> Use single static direction for sequence-level suppression")
        print(f"  -> Simpler implementation, lower novelty from dynamic tracking")
    elif controlled_min > 0.40:
        decision = "svd_cone"
        print(f"  SVD CONE (controlled min = {controlled_min:.4f}, between 0.40-0.80)")
        print(f"  -> Partial genuine drift exists")
        print(f"  -> Use SVD to extract 2-3 dim cone covering drift subspace")
    else:
        decision = "dynamic"
        print(f"  DYNAMIC (controlled min = {controlled_min:.4f} < 0.40)")
        print(f"  -> Genuine dynamic refusal mechanism confirmed")
        print(f"  -> Use timestep-specific directions or dynamic cone estimation")
        print(f"  -> Higher novelty: refusal direction truly rotates during generation")

    results["decision"] = decision

    # 保存
    out_path = os.path.join(save_dir, "exp_2a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # 保存 directions (供后续实验使用)
    directions_path = os.path.join(save_dir, "exp_2a_controlled_directions.pt")
    torch.save({
        "positions": valid_positions,
        "directions": {pos: controlled_directions[pos] for pos in valid_positions},
        "layer": layer,
    }, directions_path)
    print(f"Directions saved to {directions_path}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_exp_2a(args)
