"""MIBD model adapters for hidden state extraction.

Qwen3VLAdapter  — Qwen3-VL-8B-Instruct (qwen3-vl env, 36 LLM layers)
InternVL3Adapter — InternVL3-8B       (rdo env,     28 LLM layers)
Gemma3Adapter   — Gemma3-4B-IT        (qwen3-vl env, 34 LLM layers)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.data.image_utils import blank_image, noise_image


class MIBDModelAdapter(ABC):

    @property
    @abstractmethod
    def num_llm_layers(self) -> int: ...

    @abstractmethod
    def prepare_inputs(
        self, sample: MIBDSample, image: Image.Image | None
    ) -> dict[str, Any]: ...

    @abstractmethod
    def extract_hidden(
        self,
        inputs: dict[str, Any],
        layers: tuple[int, ...],
        token_positions: tuple[int, ...],
    ) -> dict[tuple[int, int], np.ndarray]: ...

    def build_image_for_condition(
        self, sample: MIBDSample, seed: int = 0
    ) -> Image.Image | None:
        vc = sample.visual_condition
        if vc == "V-text":
            return None
        if vc == "V-blank":
            return blank_image()
        if vc == "V-noise":
            return noise_image(seed=seed)
        if vc in ("V-real", "FigStep"):
            if sample.image_path is None:
                return blank_image()
            return Image.open(sample.image_path).convert("RGB")
        return None


class Qwen3VLAdapter(MIBDModelAdapter):
    """
    Qwen3-VL-8B-Instruct hidden state extraction adapter.
    LLM layers: model.model.language_model.layers (36 layers for 8B variant).
    Uses forward hooks to capture layer outputs without extra memory overhead.
    """

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    @property
    def num_llm_layers(self) -> int:
        return len(self.model.model.language_model.layers)

    def prepare_inputs(
        self, sample: MIBDSample, image: Image.Image | None
    ) -> dict[str, Any]:
        if image is None:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": sample.text}
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.processor(text=[text], return_tensors="pt")
        else:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample.text},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt"
            )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_hidden(
        self,
        inputs: dict[str, Any],
        layers: tuple[int, ...],
        token_positions: tuple[int, ...],
    ) -> dict[tuple[int, int], np.ndarray]:
        layer_set = set(layers)
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for idx, layer in enumerate(self.model.model.language_model.layers):
            if idx not in layer_set:
                continue

            def make_hook(i):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[i] = h.detach().float().cpu()
                return hook

            hooks.append(layer.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            self.model(**inputs, output_hidden_states=False)

        for h in hooks:
            h.remove()

        result: dict[tuple[int, int], np.ndarray] = {}
        for layer_idx, hidden in captured.items():
            seq_len = hidden.shape[1]
            for pos in token_positions:
                abs_pos = seq_len + pos if pos < 0 else pos
                if 0 <= abs_pos < seq_len:
                    result[(layer_idx, pos)] = hidden[0, abs_pos, :].numpy()
        return result


class Gemma3Adapter(MIBDModelAdapter):
    """
    Gemma3-4B-IT hidden state extraction adapter.
    LLM layers: model.language_model.model.layers (34 layers for 4B variant).
    Uses forward hooks to capture layer outputs.
    """

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    @property
    def num_llm_layers(self) -> int:
        return len(self.model.model.language_model.layers)

    def prepare_inputs(
        self, sample: MIBDSample, image: Image.Image | None
    ) -> dict[str, Any]:
        if image is None:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": sample.text},
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": sample.text},
            ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if image is None:
            inputs = self.processor(text=text, return_tensors="pt")
        else:
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_hidden(
        self,
        inputs: dict[str, Any],
        layers: tuple[int, ...],
        token_positions: tuple[int, ...],
    ) -> dict[tuple[int, int], np.ndarray]:
        layer_set = set(layers)
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for idx, layer in enumerate(self.model.model.language_model.layers):
            if idx not in layer_set:
                continue

            def make_hook(i):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[i] = h.detach().float().cpu()
                return hook

            hooks.append(layer.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            self.model(**inputs, output_hidden_states=False)

        for h in hooks:
            h.remove()

        result: dict[tuple[int, int], np.ndarray] = {}
        for layer_idx, hidden in captured.items():
            seq_len = hidden.shape[1]
            for pos in token_positions:
                abs_pos = seq_len + pos if pos < 0 else pos
                if 0 <= abs_pos < seq_len:
                    result[(layer_idx, pos)] = hidden[0, abs_pos, :].numpy()
        return result


class InternVL3Adapter(MIBDModelAdapter):
    """
    InternVL3-8B hidden state extraction adapter.
    LLM layers: model.language_model.model.layers (28 layers).
    V-text: direct language_model(input_ids).
    MM: extract_feature → inject IMG_CONTEXT token embeddings → language_model(inputs_embeds).
    """

    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    INPUT_SIZE = 448

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._img_context_token_id: int | None = None

    @property
    def num_llm_layers(self) -> int:
        return len(self.model.language_model.model.layers)

    def _get_img_context_token_id(self) -> int:
        if self._img_context_token_id is None:
            self._img_context_token_id = self.tokenizer.convert_tokens_to_ids(
                self.IMG_CONTEXT_TOKEN
            )
        return self._img_context_token_id

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(
                (self.INPUT_SIZE, self.INPUT_SIZE),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform(image).unsqueeze(0).to(self.device).to(torch.bfloat16)

    def _get_model_path(self) -> str:
        return getattr(self.model.config, "_name_or_path", "")

    def _build_text_prompt(self, text: str) -> str:
        import sys
        model_path = self._get_model_path()
        if model_path and model_path not in sys.path:
            sys.path.insert(0, model_path)
        from conversation import get_conv_template
        tmpl = get_conv_template(self.model.template)
        tmpl.system_message = self.model.system_message
        tmpl.append_message(tmpl.roles[0], text)
        tmpl.append_message(tmpl.roles[1], None)
        return tmpl.get_prompt()

    def _build_mm_prompt(self, text: str) -> str:
        import sys
        model_path = self._get_model_path()
        if model_path and model_path not in sys.path:
            sys.path.insert(0, model_path)
        from conversation import get_conv_template
        num_image_token = self.model.num_image_token
        img_tokens = "<img>" + self.IMG_CONTEXT_TOKEN * num_image_token + "</img>"
        tmpl = get_conv_template(self.model.template)
        tmpl.system_message = self.model.system_message
        tmpl.append_message(tmpl.roles[0], f"{img_tokens}\n{text}")
        tmpl.append_message(tmpl.roles[1], None)
        return tmpl.get_prompt()

    def prepare_inputs(
        self, sample: MIBDSample, image: Image.Image | None
    ) -> dict[str, Any]:
        if image is None:
            prompt = self._build_text_prompt(sample.text)
            input_ids = self.tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(self.device)
            return {"input_ids": input_ids, "_pixel_values": None}
        else:
            prompt = self._build_mm_prompt(sample.text)
            input_ids = self.tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(self.device)
            pixel_values = self._preprocess_image(image)
            return {"input_ids": input_ids, "_pixel_values": pixel_values}

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
    ) -> torch.Tensor:
        embed_fn = self.model.language_model.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids)
        if pixel_values is None:
            return inputs_embeds
        vit_embeds = self.model.extract_feature(pixel_values)
        ctx_id = self._get_img_context_token_id()
        B, N, C = inputs_embeds.shape
        flat_embeds = inputs_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        mask = flat_ids == ctx_id
        flat_embeds[mask] = vit_embeds.reshape(-1, C).to(flat_embeds.device)
        return flat_embeds.reshape(B, N, C)

    def extract_hidden(
        self,
        inputs: dict[str, Any],
        layers: tuple[int, ...],
        token_positions: tuple[int, ...],
    ) -> dict[tuple[int, int], np.ndarray]:
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("_pixel_values")
        inputs_embeds = self._build_inputs_embeds(input_ids, pixel_values)

        layer_set = set(layers)
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for idx, layer in enumerate(self.model.language_model.model.layers):
            if idx not in layer_set:
                continue

            def make_hook(i):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    captured[i] = h.detach().float().cpu()
                return hook

            hooks.append(layer.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            self.model.language_model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=False,
            )

        for h in hooks:
            h.remove()

        result: dict[tuple[int, int], np.ndarray] = {}
        for layer_idx, hidden in captured.items():
            seq_len = hidden.shape[1]
            for pos in token_positions:
                abs_pos = seq_len + pos if pos < 0 else pos
                if 0 <= abs_pos < seq_len:
                    result[(layer_idx, pos)] = hidden[0, abs_pos, :].numpy()
        return result
