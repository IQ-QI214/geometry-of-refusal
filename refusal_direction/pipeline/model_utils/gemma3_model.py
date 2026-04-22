"""Adapter for Gemma-3-4B-it in text-only mode.

Loads the same checkpoint via Gemma3ForCausalLM if available, otherwise
extracts language_model submodule from Gemma3ForConditionalGeneration.
"""
import functools
import torch
from typing import List
from torch import Tensor
from jaxtyping import Float
from transformers import AutoTokenizer

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

_GEMMA3_EOI_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"


def tokenize_instructions_gemma3_text(
    tokenizer, instructions: List[str], outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
):
    prompts = []
    for instruction in instructions:
        messages = [{"role": "user", "content": instruction}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    if outputs is not None:
        prompts = [p + (o or "") for p, o in zip(prompts, outputs)]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


class Gemma3Model(ModelBase):
    """Text-only Gemma-3-4B. Tries Gemma3ForCausalLM first;
    falls back to extracting language_model from multimodal checkpoint."""

    def _load_model(self, model_path, dtype=torch.bfloat16):
        try:
            from transformers import Gemma3ForCausalLM
            model = Gemma3ForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, local_files_only=True
            ).eval().to("cuda:0")
        except (ImportError, OSError) as e:
            print(f"Gemma3ForCausalLM not available ({type(e).__name__}: {e}), falling back to extracting language_model from multimodal checkpoint")
            from transformers import Gemma3ForConditionalGeneration
            mm = Gemma3ForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, local_files_only=True
            ).eval().to("cuda:0")
            model = mm.language_model
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma3_text, tokenizer=self.tokenizer)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(_GEMMA3_EOI_SUFFIX, add_special_tokens=False)

    def _get_refusal_toks(self):
        return self.tokenizer.encode("I", add_special_tokens=False)[:1]

    def _get_model_block_modules(self):
        # Gemma3ForCausalLM: model.model.layers
        # Extracted language_model submodule: model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        return self.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([b.self_attn for b in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([b.mlp for b in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        def orthogonalize_fn(model):
            backbone = model.model if hasattr(model, "model") else model
            backbone.embed_tokens.weight.data = get_orthogonalized_matrix(
                backbone.embed_tokens.weight.data, direction
            )
            for block in backbone.layers:
                block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                    block.self_attn.o_proj.weight.data.T, direction
                ).T
                block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                    block.mlp.down_proj.weight.data.T, direction
                ).T
        return orthogonalize_fn

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        def act_add_fn(model):
            backbone = model.model if hasattr(model, "model") else model
            dtype = backbone.layers[layer - 1].mlp.down_proj.weight.dtype
            device = backbone.layers[layer - 1].mlp.down_proj.weight.device
            bias = (coeff * direction).to(dtype=dtype, device=device)
            backbone.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)
        return act_add_fn
