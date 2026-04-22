import torch
import functools

import numpy as np
from torch import Tensor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Literal
try:
    from jaxtyping import Float
except ImportError:
    class _JaxStub:
        def __class_getitem__(cls, item): return cls
    Float = _JaxStub
from PIL import Image

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

QWEN_VLM_REFUSAL_TOKS = [40, 2121]  # ['I', 'As'] — shared tokenizer with text Qwen

_BLANK_IMAGE = Image.new("RGB", (336, 336), (255, 255, 255))

_QWEN_VLM_EOI_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def _make_noise_image(seed: int = 42) -> Image.Image:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(336, 336, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def tokenize_instructions_qwen_vlm(
    processor: AutoProcessor,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
    image_mode: Literal["text", "blank", "noise"] = "blank",
    noise_seed: int = 42,
):
    """Tokenize instructions for Qwen2.5-VL with configurable image mode.

    image_mode:
      'text'  — no image content block, no pixel_values (V-text condition)
      'blank' — 336×336 white image (original P0 / V-blank condition)
      'noise' — 336×336 uniform-random image (V-noise condition)
    """
    if image_mode == "text":
        prompts = []
        for instruction in instructions:
            messages = [{"role": "user", "content": instruction}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
        if outputs is not None:
            prompts = [p + o if o is not None else p for p, o in zip(prompts, outputs)]
        return processor(
            text=prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

    img = _BLANK_IMAGE if image_mode == "blank" else _make_noise_image(noise_seed)
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
        prompts = [p + o if o is not None else p for p, o in zip(prompts, outputs)]

    images = [img] * len(prompts)
    return processor(
        text=prompts,
        images=images,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


class QwenVLMModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
        ).eval()
        # Qwen-VL: manual .to() instead of device_map (avoids accelerate issues)
        model = model.to("cuda:0")
        model.requires_grad_(False)
        # Note: model.config.num_hidden_layers and hidden_size are already at top level
        # for Qwen2_5_VLConfig (verified: 28 layers, hidden_size=3584)
        return model

    def _load_tokenizer(self, model_path):
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._processor = processor
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen_vlm,
            processor=self._processor,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(_QWEN_VLM_EOI_SUFFIX, add_special_tokens=False)

    def _get_refusal_toks(self):
        return QWEN_VLM_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # Qwen2_5_VLForConditionalGeneration: model.model.language_model.layers
        return self.model.model.language_model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([
            block.self_attn for block in self.model_block_modules
        ])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([
            block.mlp for block in self.model_block_modules
        ])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        def orthogonalize_fn(model):
            backbone = model.model.language_model
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
            backbone = model.model.language_model
            dtype = backbone.layers[layer - 1].mlp.down_proj.weight.dtype
            device = backbone.layers[layer - 1].mlp.down_proj.weight.device
            bias = (coeff * direction).to(dtype=dtype, device=device)
            backbone.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)
        return act_add_fn

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[],
                             batch_size=8, max_new_tokens=64, temperature=0):
        """Override to pass pixel_values and image_grid_thw to generate."""
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
            batch_instructions = instructions[i:i + self.max_batch_size]
            tokenized = self.tokenize_instructions_fn(instructions=batch_instructions)

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks,
                          module_forward_hooks=fwd_hooks):
                gen_kwargs = {
                    "input_ids": tokenized.input_ids.to(self.model.device),
                    "attention_mask": tokenized.attention_mask.to(self.model.device),
                    "generation_config": generation_config,
                }
                if hasattr(tokenized, "pixel_values") and tokenized.pixel_values is not None:
                    gen_kwargs["pixel_values"] = tokenized.pixel_values.to(
                        device=self.model.device, dtype=self.model.dtype
                    )
                if hasattr(tokenized, "image_grid_thw") and tokenized.image_grid_thw is not None:
                    gen_kwargs["image_grid_thw"] = tokenized.image_grid_thw.to(self.model.device)

                generation_toks = self.model.generate(**gen_kwargs)
                generation_toks = generation_toks[:, tokenized.input_ids.shape[-1]:]

                for idx, gen in enumerate(generation_toks):
                    completions.append({
                        "category": categories[i + idx],
                        "prompt": instructions[i + idx],
                        "response": self.tokenizer.decode(gen, skip_special_tokens=True).strip(),
                    })

        return completions
