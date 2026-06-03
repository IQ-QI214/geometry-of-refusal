"""Model loader utilities for MIBD experiments.

Qwen3-VL requires the qwen3-vl conda env (torch 2.9, transformers 5.5.4, qwen-vl-utils).
InternVL3 requires the rdo conda env (torch 2.5.1, timm).
"""
from __future__ import annotations


def load_qwen3vl(model_path: str, device: str = "cuda:0"):
    """Load Qwen3-VL-8B-Instruct. Requires qwen3-vl env."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    import torch

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def load_internvl3(model_path: str, device: str = "cuda:1"):
    """Load InternVL3-8B. Requires rdo env (timm)."""
    import sys
    import torch
    from transformers import AutoTokenizer

    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from modeling_internvl_chat import InternVLChatModel

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer
