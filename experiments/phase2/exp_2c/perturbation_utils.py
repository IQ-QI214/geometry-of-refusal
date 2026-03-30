"""
PGD 视觉扰动优化的 loss 函数和工具。
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict


def sequence_level_refusal_loss(hidden_states_list, refusal_direction, target_layers,
                                 prefix_len, total_len):
    """
    Sequence-level refusal suppression loss (Component 1)。
    最小化生成位置处 hidden state 在 refusal direction 上的投影。

    L_suppress = (1/T) Σ_t Σ_l (h_t^l · r)^2

    Args:
        hidden_states_list: list of (1, seq_len, hidden_dim) tensors, 索引 0=layer 0 output
        refusal_direction: (hidden_dim,) normalized tensor
        target_layers: list of layer indices
        prefix_len: prompt 部分的 token 数（在 hidden state 空间中）
        total_len: 总 sequence 长度
    """
    loss = torch.tensor(0.0, device=refusal_direction.device)
    n_layers = 0

    for layer_idx in target_layers:
        if layer_idx >= len(hidden_states_list):
            continue
        h = hidden_states_list[layer_idx]  # (1, seq_len, hidden_dim)
        # 只取生成部分
        h_gen = h[0, prefix_len:total_len, :]  # (T, hidden_dim)
        if h_gen.shape[0] == 0:
            continue

        # 投影到 refusal direction
        proj = h_gen @ refusal_direction  # (T,)
        loss = loss + (proj ** 2).mean()
        n_layers += 1

    if n_layers > 0:
        loss = loss / n_layers
    return loss


def harmful_alignment_loss(hidden_states_list, harmful_direction, best_layer,
                            prefix_len, total_len):
    """
    Harmful direction alignment loss (Component 2)。
    鼓励 hidden state 在 harmful direction 上有正投影。

    L_harmful = -(1/T) Σ_t (h_t · d_harmful)

    Args:
        harmful_direction: (hidden_dim,) normalized tensor pointing toward harmful content
        best_layer: 单层索引
    """
    if best_layer >= len(hidden_states_list):
        return torch.tensor(0.0, device=harmful_direction.device)

    h = hidden_states_list[best_layer]  # (1, seq_len, hidden_dim)
    h_gen = h[0, prefix_len:total_len, :]  # (T, hidden_dim)
    if h_gen.shape[0] == 0:
        return torch.tensor(0.0, device=harmful_direction.device)

    alignment = h_gen @ harmful_direction  # (T,)
    return -alignment.mean()


def cone_suppression_loss(hidden_states_list, cone_basis, target_layers,
                           prefix_len, total_len):
    """
    Cone-based suppression loss: 最小化 hidden state 在整个 cone subspace 上的投影。

    L = (1/T) Σ_t Σ_l ||proj_{cone}(h_t^l)||^2

    Args:
        cone_basis: (hidden_dim, cone_dim) tensor, 每列是一个 basis vector
    """
    loss = torch.tensor(0.0, device=cone_basis.device)
    n_layers = 0

    for layer_idx in target_layers:
        if layer_idx >= len(hidden_states_list):
            continue
        h = hidden_states_list[layer_idx]
        h_gen = h[0, prefix_len:total_len, :]  # (T, hidden_dim)
        if h_gen.shape[0] == 0:
            continue

        # 投影到 cone subspace: (T, hidden_dim) @ (hidden_dim, cone_dim) -> (T, cone_dim)
        proj = h_gen @ cone_basis
        loss = loss + (proj ** 2).sum(dim=-1).mean()
        n_layers += 1

    if n_layers > 0:
        loss = loss / n_layers
    return loss


def pgd_step(delta, grad, alpha, epsilon, base_pixel_values):
    """
    PGD 更新步: signed gradient descent + projection。

    Args:
        delta: 当前扰动 (float32)
        grad: delta 的梯度
        alpha: 步长
        epsilon: L∞ 约束
        base_pixel_values: 基础图像像素值
    """
    with torch.no_grad():
        # Signed gradient descent
        delta.data = delta.data - alpha * grad.sign()
        # Project to epsilon ball
        delta.data = delta.data.clamp(-epsilon, epsilon)
        # Ensure perturbed image stays in valid range
        delta.data = (base_pixel_values + delta.data).clamp(0, 1) - base_pixel_values
    return delta


def get_image_normalization_params(processor):
    """获取 processor 的图像归一化参数。"""
    image_mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1)
    return image_mean, image_std


def compute_prefix_len_in_hidden_space(processor, prompt_text, has_image=True):
    """
    计算 prompt 部分在 hidden state 空间中的 token 数。
    LLaVA 将 1 个 <image> token 扩展为 576 个 patch tokens。
    """
    prompt_ids = processor.tokenizer(prompt_text, return_tensors="pt").input_ids
    prefix_len = prompt_ids.shape[1]
    if has_image:
        # <image> token 被替换为 576 个 patch tokens，净增 575
        prefix_len += 575
    return prefix_len
