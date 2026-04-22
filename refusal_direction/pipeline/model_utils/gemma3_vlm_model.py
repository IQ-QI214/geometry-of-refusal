"""Adapter for Gemma-3-4B-it multimodal model.

Supports three image_modes like qwen_vlm_model.py:
  - 'text':  no image content block, no pixel_values
  - 'blank': 336x336 white image
  - 'noise': 336x336 uniform random image
"""
import functools
import torch
import numpy as np
from typing import List, Literal
from torch import Tensor
from jaxtyping import Float
from PIL import Image
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

_GEMMA3_BLANK = Image.new("RGB", (336, 336), (255, 255, 255))
_GEMMA3_EOI_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"


def _make_noise_image(seed: int = 42) -> Image.Image:
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, size=(336, 336, 3), dtype=np.uint8))


def tokenize_instructions_gemma3_vlm(
    processor: AutoProcessor,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
    image_mode: Literal["text", "blank", "noise"] = "blank",
    noise_seed: int = 42,
):
    if image_mode == "text":
        prompts = []
        for instruction in instructions:
            messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
        if outputs is not None:
            prompts = [p + (o or "") for p, o in zip(prompts, outputs)]
        return processor(text=prompts, padding=True, truncation=False, return_tensors="pt")

    img = _GEMMA3_BLANK if image_mode == "blank" else _make_noise_image(noise_seed)
    prompts = []
    for instruction in instructions:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": instruction},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    if outputs is not None:
        prompts = [p + (o or "") for p, o in zip(prompts, outputs)]
    images = [img] * len(prompts)
    return processor(text=prompts, images=images, padding=True, truncation=False, return_tensors="pt")


class Gemma3VLMModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True
        ).eval().to("cuda:0")
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self._processor = processor
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma3_vlm, processor=self._processor)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(_GEMMA3_EOI_SUFFIX, add_special_tokens=False)

    def _get_refusal_toks(self):
        # First token of common refusal patterns; populated fully via arditi_templates at runtime
        return self.tokenizer.encode("I", add_special_tokens=False)[:1]

    def _get_model_block_modules(self):
        # Gemma3ForConditionalGeneration: language_model IS Gemma3TextModel; backbone at .language_model.layers
        return self.model.language_model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        def orthogonalize_fn(model):
            lm = model.language_model
            lm.embed_tokens.weight.data = get_orthogonalized_matrix(
                lm.embed_tokens.weight.data, direction
            )
            for block in lm.layers:
                block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                    block.self_attn.o_proj.weight.data.T, direction
                ).T
                block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                    block.mlp.down_proj.weight.data.T, direction
                ).T
        return orthogonalize_fn

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        def act_add_fn(model):
            lm = model.language_model
            dtype = lm.layers[layer - 1].mlp.down_proj.weight.dtype
            device = lm.layers[layer - 1].mlp.down_proj.weight.device
            bias = (coeff * direction).to(dtype=dtype, device=device)
            lm.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)
        return act_add_fn

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[],
                             batch_size=8, max_new_tokens=64, temperature=0):
        """Override to conditionally pass pixel_values (absent for image_mode='text')."""
        from transformers import GenerationConfig
        from pipeline.utils.hook_utils import add_hooks

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x["instruction"] for x in dataset]
        categories = [x["category"] for x in dataset]

        if self.max_batch_size is None:
            self.max_batch_size = 4

        for i in range(0, len(dataset), self.max_batch_size):
            batch = instructions[i:i + self.max_batch_size]
            tokenized = self.tokenize_instructions_fn(instructions=batch)
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                gen_kwargs = {
                    "input_ids": tokenized.input_ids.to(self.model.device),
                    "attention_mask": tokenized.attention_mask.to(self.model.device),
                    "generation_config": generation_config,
                }
                if hasattr(tokenized, "pixel_values") and tokenized.pixel_values is not None:
                    gen_kwargs["pixel_values"] = tokenized.pixel_values.to(
                        device=self.model.device, dtype=self.model.dtype
                    )
                gen_toks = self.model.generate(**gen_kwargs)
                gen_toks = gen_toks[:, tokenized.input_ids.shape[-1]:]
                for idx, g in enumerate(gen_toks):
                    completions.append({
                        "category": categories[i + idx],
                        "prompt": instructions[i + idx],
                        "response": self.tokenizer.decode(g, skip_special_tokens=True).strip(),
                    })
        return completions
