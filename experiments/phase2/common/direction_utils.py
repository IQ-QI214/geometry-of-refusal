"""
Refusal direction 加载与 cone basis 构建工具。
"""

import os
import torch
from typing import Dict, Optional


def load_exp_a_directions(results_dir: str) -> Dict:
    """
    加载 Exp A 保存的 refusal directions。
    返回 dict: {v_text, v_mm, layer}。
    """
    path = os.path.join(results_dir, "exp_a_directions.pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data


def load_exp_2a_directions(results_dir: str) -> Dict:
    """
    加载 Exp 2A 保存的 controlled position-specific directions。
    返回 dict: {positions: list, directions: {pos: tensor}, layer: int}。
    """
    path = os.path.join(results_dir, "exp_2a_controlled_directions.pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data


def build_cone_basis(directions_dict: Dict, cone_dim: int = 2) -> torch.Tensor:
    """
    从多个 position-specific directions 构建 cone basis (via SVD)。

    Args:
        directions_dict: {position: (hidden_dim,) tensor}
        cone_dim: 保留的 basis 维度

    Returns:
        (hidden_dim, cone_dim) tensor, 每列是一个 basis vector
    """
    vecs = torch.stack(list(directions_dict.values()))  # (K, hidden_dim)
    # 中心化
    vecs = vecs - vecs.mean(dim=0, keepdim=True)
    # SVD
    U, S, Vh = torch.linalg.svd(vecs, full_matrices=False)
    # 取前 cone_dim 个 right singular vectors
    basis = Vh[:cone_dim, :].T  # (hidden_dim, cone_dim)
    return basis


def get_refusal_direction(results_dir: str, mode: str = "mm") -> tuple:
    """
    便捷函数：获取 refusal direction 和对应 layer。

    Args:
        mode: "mm" (多模态，推荐) 或 "text" (纯文本)

    Returns:
        (direction, layer): direction 是 (hidden_dim,) normalized tensor
    """
    data = load_exp_a_directions(results_dir)
    key = "v_mm" if mode == "mm" else "v_text"
    direction = data[key]
    # 确保归一化
    direction = direction / (direction.norm() + 1e-8)
    return direction, data["layer"]
