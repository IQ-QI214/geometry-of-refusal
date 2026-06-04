"""VLM behavior generators for Phase 2A GPU runs.

InternVL3Generator  — rdo conda env, device=cuda:0 (or cuda:1)
Qwen3VLGenerator    — qwen3-vl conda env, device=cuda:0

Image loading: if image_path is None, missing, or a .txt placeholder, falls
back to a blank RGB image so every sample produces a forward pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from experiments.mibd.data.image_utils import blank_image
from experiments.mibd_routing.data.schema import PairedRoutingSample


def _load_image(image_path: str | None) -> Image.Image:
    """Return PIL image; blank if path is None, non-existent, or a .txt placeholder."""
    if image_path is None:
        return blank_image()
    p = Path(image_path)
    if p.suffix == ".txt" or not p.exists():
        return blank_image()
    return Image.open(p).convert("RGB")


# ---------------------------------------------------------------------------
# InternVL3Generator
# ---------------------------------------------------------------------------

class InternVL3Generator:
    """Text generation from InternVL3-8B for Phase 2A samples.

    Mirrors the inputs_embeds path used in InternVL3Adapter so that both
    behavior generation and hidden-state extraction share the same forward.
    """

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    INPUT_SIZE = 448

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._img_context_token_id: int | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _img_ctx_id(self) -> int:
        if self._img_context_token_id is None:
            self._img_context_token_id = self.tokenizer.convert_tokens_to_ids(
                self.IMG_CONTEXT_TOKEN
            )
        return self._img_context_token_id

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        tf = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(
                (self.INPUT_SIZE, self.INPUT_SIZE),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return tf(image).unsqueeze(0).to(self.device).to(torch.bfloat16)

    def _model_path(self) -> str:
        return getattr(self.model.config, "_name_or_path", "")

    def _conv_prompt(self, text: str, with_image: bool) -> str:
        import sys
        mp = self._model_path()
        if mp and mp not in sys.path:
            sys.path.insert(0, mp)
        from conversation import get_conv_template  # type: ignore[import]

        tmpl = get_conv_template(self.model.template)
        tmpl.system_message = self.model.system_message
        if with_image:
            n = self.model.num_image_token
            img_tok = "<img>" + self.IMG_CONTEXT_TOKEN * n + "</img>"
            tmpl.append_message(tmpl.roles[0], f"{img_tok}\n{text}")
        else:
            tmpl.append_message(tmpl.roles[0], text)
        tmpl.append_message(tmpl.roles[1], None)
        return tmpl.get_prompt()

    def _build_embeds(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        embed_fn = self.model.language_model.get_input_embeddings()
        embeds = embed_fn(input_ids)
        vit_embeds = self.model.extract_feature(pixel_values)
        ctx_id = self._img_ctx_id()
        B, N, C = embeds.shape
        flat_e = embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        mask = flat_ids == ctx_id
        flat_e[mask] = vit_embeds.reshape(-1, C).to(flat_e.device)
        return flat_e.reshape(B, N, C)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, sample: PairedRoutingSample) -> str:
        image = _load_image(sample.image_path)
        prompt = self._conv_prompt(sample.question, with_image=True)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        pixel_values = self._preprocess(image)
        inputs_embeds = self._build_embeds(input_ids, pixel_values)

        # generate() with inputs_embeds returns only generated token ids
        output_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Qwen3VLGenerator
# ---------------------------------------------------------------------------

class Qwen3VLGenerator:
    """Text generation from Qwen3-VL-8B-Instruct for Phase 2A samples."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, sample: PairedRoutingSample) -> str:
        from qwen_vl_utils import process_vision_info  # type: ignore[import]

        image = _load_image(sample.image_path)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": sample.question},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated = output_ids[:, input_len:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
