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
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor


def load_internvl3(model_path: str, device: str = "cuda:1"):
    """Load InternVL3-8B via AutoModel. Requires rdo env (timm)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def load_gemma3(model_path: str, device: str = "cuda:0"):
    """Load Gemma3-4B-IT multimodal model. Requires qwen3-vl env."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor
