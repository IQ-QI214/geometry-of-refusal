"""
LLaVA 上的 refusal direction ablation hook 工具。
适配自 refusal_direction/pipeline/utils/hook_utils.py，
针对 HuggingFace LlavaForConditionalGeneration 的 LLM backbone。
"""

import torch
import contextlib
import functools
from typing import List, Optional


def get_ablation_pre_hook(direction):
    """
    返回一个 forward pre-hook：从 layer input 中移除 direction 方向的投影。
    direction: (hidden_dim,) normalized tensor。
    """
    def hook_fn(module, args):
        if isinstance(args, tuple):
            activation = args[0]
        else:
            activation = args

        d = direction.to(activation.device).to(activation.dtype)
        # proj = (activation · d) * d
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_activation = activation - proj

        if isinstance(args, tuple):
            return (new_activation,) + args[1:]
        return new_activation
    return hook_fn


def get_ablation_output_hook(direction):
    """
    返回一个 forward hook：从 module output 中移除 direction 方向的投影。
    用于 self_attn 和 mlp 的 output。
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        d = direction.to(activation.device).to(activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_activation = activation - proj

        if isinstance(output, tuple):
            return (new_activation,) + output[1:]
        return new_activation
    return hook_fn


@contextlib.contextmanager
def ablation_context(model, direction, target_layers=None, ablate_attn=True, ablate_mlp=True):
    """
    Context manager: 在 LLaVA 的 LLM backbone 指定层上注册 ablation hooks。
    退出时自动移除所有 hooks。

    Args:
        model: LlavaForConditionalGeneration 实例
        direction: (hidden_dim,) normalized refusal direction
        target_layers: 层索引列表，None 表示全部层
        ablate_attn: 是否在 attention output 上 ablate
        ablate_mlp: 是否在 MLP output 上 ablate
    """
    llm_layers = model.language_model.model.layers
    n_layers = len(llm_layers)

    if target_layers is None:
        target_layers = list(range(n_layers))

    handles = []
    try:
        for layer_idx in target_layers:
            layer = llm_layers[layer_idx]
            # Pre-hook on layer input
            h = layer.register_forward_pre_hook(get_ablation_pre_hook(direction))
            handles.append(h)
            # Output hooks on attention and MLP
            if ablate_attn:
                h = layer.self_attn.register_forward_hook(get_ablation_output_hook(direction))
                handles.append(h)
            if ablate_mlp:
                h = layer.mlp.register_forward_hook(get_ablation_output_hook(direction))
                handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()
