import torch
import functools
import os

from torch import Tensor
from transformers import LlavaForConditionalGeneration, AutoProcessor
from typing import List
from jaxtyping import Float
from PIL import Image

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# LLaVA-1.5 uses LLaMA-2 tokenizer
LLAVA_REFUSAL_TOKS = [306]  # token "I" in LLaMA-2 tokenizer

_LLAVA_EOI_SUFFIX = "\nASSISTANT:"

# Blank image: 336x336 white
_BLANK_IMAGE = Image.new("RGB", (336, 336), (255, 255, 255))


def tokenize_instructions_llava_vlm(
    processor: AutoProcessor,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
):
    """Tokenize instructions with blank image for LLaVA VLM."""
    conversations = []
    for instruction in instructions:
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}]
        conversations.append(conv)

    prompts = [
        processor.apply_chat_template(conv, add_generation_prompt=True)
        for conv in conversations
    ]

    if outputs is not None:
        prompts = [p + o for p, o in zip(prompts, outputs)]

    # All prompts share the same blank image
    images = [_BLANK_IMAGE] * len(prompts)

    result = processor(
        images=images,
        text=prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return result


class LlavaVLMModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
            local_files_only=True,
        ).eval()
        model.requires_grad_(False)
        # Patch top-level config for downstream pipeline compatibility
        model.config.num_hidden_layers = model.config.text_config.num_hidden_layers
        model.config.hidden_size = model.config.text_config.hidden_size
        return model

    def _load_tokenizer(self, model_path):
        processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
            local_files_only=True,
        )
        # Store full processor for image processing
        self._processor = processor
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_llava_vlm,
            processor=self._processor,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(_LLAVA_EOI_SUFFIX, add_special_tokens=False)

    def _get_refusal_toks(self):
        return LLAVA_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # LLaVA: language_model.model.layers
        return self.model.language_model.model.layers

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
            lm = model.language_model.model
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
            lm = model.language_model.model
            dtype = lm.layers[layer - 1].mlp.down_proj.weight.dtype
            device = lm.layers[layer - 1].mlp.down_proj.weight.device
            bias = (coeff * direction).to(dtype=dtype, device=device)
            lm.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)
        return act_add_fn

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[],
                             batch_size=8, max_new_tokens=64, temperature=0):
        """Override to pass pixel_values and image_sizes to generate."""
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
            self.max_batch_size = 4  # Conservative for VLM

        for i in range(0, len(dataset), self.max_batch_size):
            batch_instructions = instructions[i:i + self.max_batch_size]
            tokenized = self.tokenize_instructions_fn(instructions=batch_instructions)

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks,
                          module_forward_hooks=fwd_hooks):
                # Build kwargs for generate, passing all processor outputs
                gen_kwargs = {
                    "input_ids": tokenized.input_ids.to(self.model.device),
                    "attention_mask": tokenized.attention_mask.to(self.model.device),
                    "generation_config": generation_config,
                }
                if hasattr(tokenized, "pixel_values") and tokenized.pixel_values is not None:
                    gen_kwargs["pixel_values"] = tokenized.pixel_values.to(
                        device=self.model.device, dtype=self.model.dtype
                    )
                if hasattr(tokenized, "image_sizes") and tokenized.image_sizes is not None:
                    gen_kwargs["image_sizes"] = tokenized.image_sizes

                generation_toks = self.model.generate(**gen_kwargs)
                generation_toks = generation_toks[:, tokenized.input_ids.shape[-1]:]

                for idx, gen in enumerate(generation_toks):
                    completions.append({
                        "category": categories[i + idx],
                        "prompt": instructions[i + idx],
                        "response": self.tokenizer.decode(gen, skip_special_tokens=True).strip(),
                    })

        return completions
