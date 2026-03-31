"""
Phase 3 模型适配层。

每个 adapter 封装模型架构差异，提供统一接口:
  - prepare_mm_inputs(prompt, image)   → dict, 多模态输入
  - prepare_text_inputs(prompt)        → dict, 纯文本输入 (直接走 LLM backbone)
  - get_llm_layers()                   → nn.ModuleList, LLM backbone 的 transformer layers
  - get_visual_token_count(inputs)     → int, 该次 forward 的 visual token 数量
  - get_last_text_token_pos(h_seq_len, visual_token_count) → int, last text token 在序列中的位置

所有 prepare_*_inputs 返回已移到正确 device/dtype 的 dict，
调用方只需直接 **unpack 传给 model.forward()。
"""

import torch
from abc import ABC, abstractmethod
from PIL import Image
from typing import Dict, Any, Optional


class ModelAdapter(ABC):
    """所有 adapter 的基类。"""

    def __init__(self, model, processor, device: str, model_name: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.model_name = model_name

    @abstractmethod
    def prepare_mm_inputs(self, prompt: str, image: Image.Image) -> Dict[str, Any]:
        """准备多模态 (image + text) 输入，返回可直接传给 model() 的 dict。"""
        pass

    @abstractmethod
    def prepare_text_inputs(self, prompt: str) -> Dict[str, Any]:
        """
        准备纯文本输入。
        返回可直接传给对应 text-backbone forward() 的 dict。
        注意：不同模型的 text backbone 调用方式不同:
          - LLaVA/InternVL2/InstructBLIP: model.language_model(**inputs)
          - Qwen2.5-VL: model(**inputs) 不含 pixel_values
        """
        pass

    @abstractmethod
    def get_llm_layers(self):
        """返回 LLM backbone 的 nn.ModuleList (transformer decoder layers)。"""
        pass

    @abstractmethod
    def get_visual_token_count(self, inputs: Optional[Dict] = None) -> int:
        """
        返回该次 forward 的 visual token 数量。
        dynamic 模型需传入 inputs dict；固定数量的模型可以忽略 inputs。
        """
        pass

    def get_last_text_token_pos(self, h_seq_len: int, visual_token_count: int) -> int:
        """
        返回 hidden_states 序列中最后一个 text token 的位置索引。
        对于 mm 模式: visual tokens 通常排在前面，text tokens 在后面。
        """
        # 一般规律: 最后一个 token = 最后一个 text token
        return h_seq_len - 1

    def forward_text_backbone(self, text_inputs: Dict) -> Any:
        """
        调用 text backbone 做 forward。子类可重写。
        默认: model.language_model(**text_inputs, output_hidden_states=True)
        """
        return self.model.language_model(
            **text_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward_mm(self, mm_inputs: Dict) -> Any:
        """调用完整 VLM 做 mm forward。"""
        return self.model(
            **mm_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # ── Generation 方法 (Exp 3C 使用) ──────────────────────────────────────
    def _get_pad_token_id(self) -> int:
        """获取 pad_token_id，子类可重写。"""
        if hasattr(self.processor, 'tokenizer'):
            return self.processor.tokenizer.eos_token_id
        return self.processor.eos_token_id

    def _decode_new_tokens(self, gen_ids: torch.Tensor, input_len: int) -> str:
        """解码新生成的 tokens（去掉 prompt 部分）。"""
        new_ids = gen_ids[0][input_len:]
        if hasattr(self.processor, 'tokenizer'):
            return self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        return self.processor.decode(new_ids, skip_special_tokens=True)

    def generate_mm(self, prompt: str, image: Image.Image,
                    max_new_tokens: int = 200) -> str:
        """MM 模式生成文本。返回 decoded response（不含 prompt）。子类可重写。"""
        mm_inputs = self.prepare_mm_inputs(prompt, image)
        input_len = mm_inputs["input_ids"].shape[1]
        pad_token_id = self._get_pad_token_id()
        with torch.no_grad():
            gen_ids = self.model.generate(
                **mm_inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad_token_id,
            )
        return self._decode_new_tokens(gen_ids, input_len)

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Text-only 模式生成。返回 decoded response。子类可重写。"""
        text_inputs = self.prepare_text_inputs(prompt)
        input_len = text_inputs["input_ids"].shape[1]
        pad_token_id = self._get_pad_token_id()
        with torch.no_grad():
            gen_ids = self.model.language_model.generate(
                **text_inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad_token_id,
            )
        return self._decode_new_tokens(gen_ids, input_len)


# ============================================================
# LLaVA-1.5-7B Adapter
# ============================================================
class LLaVAAdapter(ModelAdapter):
    """
    LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf) 适配。
    - Visual encoder: CLIP ViT-L/14, 336px
    - Visual tokens: 576 (固定)
    - LLM backbone: LLaMA-2-7B
    - text-only: model.language_model(input_ids=...)
    - mm: model(input_ids=..., pixel_values=..., image_sizes=...)
    """

    VISUAL_TOKEN_COUNT = 576

    def prepare_mm_inputs(self, prompt: str, image: Image.Image) -> Dict:
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.device)
        # 确保 pixel_values 用 bfloat16
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    def prepare_text_inputs(self, prompt: str) -> Dict:
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]}]
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        input_ids = self.processor.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        return {"input_ids": input_ids}

    def get_llm_layers(self):
        return self.model.language_model.model.layers

    def get_visual_token_count(self, inputs=None) -> int:
        return self.VISUAL_TOKEN_COUNT

    def forward_text_backbone(self, text_inputs: Dict) -> Any:
        # LLaVA: text backbone 是 model.language_model
        return self.model.language_model(
            **text_inputs,
            output_hidden_states=True,
            return_dict=True,
        )


# ============================================================
# Qwen2.5-VL-7B-Instruct Adapter
# ============================================================
class Qwen2VLAdapter(ModelAdapter):
    """
    Qwen2.5-VL-7B-Instruct 适配。
    - Visual encoder: Qwen2.5 Custom ViT (native resolution)
    - Visual tokens: 动态 (取决于图像尺寸，从 image_grid_thw 计算)
    - LLM backbone: Qwen2.5-7B (28 层, hidden_dim=3584)
    - text-only: model(**inputs) 不含 pixel_values (Qwen2.5-VL 不分离 backbone)
    - mm: model(**inputs) 含 pixel_values
    - layers: model.model.layers (非 model.language_model)
    """

    def prepare_mm_inputs(self, prompt: str, image: Image.Image) -> Dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        # Qwen2.5-VL 使用 qwen_vl_utils 或直接 processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    def prepare_text_inputs(self, prompt: str) -> Dict:
        # Qwen2.5-VL 没有分离的 language_model，text-only 通过不传 pixel_values 实现
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        return inputs

    def get_llm_layers(self):
        # Qwen2_5_VLForConditionalGeneration: model.model.language_model.layers
        if (hasattr(self.model, "model") and
                hasattr(self.model.model, "language_model") and
                hasattr(self.model.model.language_model, "layers")):
            return self.model.model.language_model.layers
        # AutoModel 加载的 Qwen2_5_VLModel: model.language_model.layers
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return self.model.language_model.layers
        # 其他 fallback
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if (hasattr(self.model, "language_model") and
                hasattr(self.model.language_model, "model")):
            return self.model.language_model.model.layers
        raise AttributeError(
            f"Cannot find LLM layers in Qwen2VL model. "
            f"Top-level attrs: {[k for k in self.model.__dict__ if not k.startswith('_')]}"
        )

    def get_visual_token_count(self, inputs=None) -> int:
        if inputs is None:
            return 196  # 336x336 默认估计
        # 从 image_grid_thw 计算: T * H * W
        if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
            thw = inputs.image_grid_thw[0]
            return int(thw[0].item() * thw[1].item() * thw[2].item())
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            thw = inputs["image_grid_thw"][0]
            return int(thw[0].item() * thw[1].item() * thw[2].item())
        return 196  # fallback

    def forward_text_backbone(self, text_inputs: Dict) -> Any:
        # Qwen2.5-VL: text-only 直接调用 model() 不含 pixel_values
        return self.model(
            **text_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward_mm(self, mm_inputs: Dict) -> Any:
        return self.model(
            **mm_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Qwen2.5-VL text-only: 直接调用 model.generate() 不含 pixel_values。"""
        text_inputs = self.prepare_text_inputs(prompt)
        input_len = text_inputs["input_ids"].shape[1]
        pad_token_id = self._get_pad_token_id()
        with torch.no_grad():
            gen_ids = self.model.generate(
                **text_inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad_token_id,
            )
        return self._decode_new_tokens(gen_ids, input_len)


# ============================================================
# InternVL2-8B Adapter
# ============================================================
class InternVL2Adapter(ModelAdapter):
    """
    InternVL2-8B (OpenGVLab/InternVL2-8B) 适配。
    - Visual encoder: InternViT-300M (非 CLIP, 专为 VLM 训练)
    - Visual tokens: 256 per tile (448px 输入, pixel_shuffle 0.5)
    - LLM backbone: InternLM2-8B (32 层)
    - layers: model.language_model.model.layers
    - 需要 trust_remote_code=True

    关键设计: InternVL2 的 forward() 强制需要 image_flags，不方便直接使用。
    参照其 generate() 方法的逻辑:
      MM:   extract_feature(pixel_values) → 替换 input_embeds 中的占位符 → language_model(inputs_embeds=...)
      Text: language_model(input_ids=...)
    两种模式都直接调用 language_model，hook 目标层一致。
    """

    VISUAL_TOKEN_COUNT = 256  # 448px 单 tile, ViT 1024 patches → pixel_shuffle → 256

    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    def __init__(self, model, processor, device, model_name):
        super().__init__(model, processor, device, model_name)
        # InternVL2 的 conversation.py 在模型目录下，需要加到 sys.path
        import sys
        from .model_configs import MODEL_CONFIGS
        model_path = MODEL_CONFIGS[model_name]["model_path"]
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

    def _ensure_img_context_token_id(self):
        """确保 img_context_token_id 已设置（首次调用时初始化）。"""
        if self.model.img_context_token_id is None:
            token_id = self.processor.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
            self.model.img_context_token_id = token_id

    def _build_mm_prompt(self, prompt: str) -> str:
        """按 InternVL2 的 chat() 逻辑构造 prompt: <image> → <img><IMG_CONTEXT>*N</img>"""
        from conversation import get_conv_template
        self._ensure_img_context_token_id()

        question = f'<image>\n{prompt}'
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        # 替换 <image> 为 N 个 IMG_CONTEXT tokens
        num_image_token = self.model.num_image_token  # 通常 256
        image_tokens = (self.IMG_START_TOKEN
                        + self.IMG_CONTEXT_TOKEN * num_image_token * 1  # 1 tile
                        + self.IMG_END_TOKEN)
        query = query.replace('<image>', image_tokens, 1)
        return query

    def _build_text_prompt(self, prompt: str) -> str:
        """纯文本 prompt，使用 InternVL2 的 conversation template。"""
        from conversation import get_conv_template
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    def prepare_mm_inputs(self, prompt: str, image: Image.Image) -> Dict:
        pixel_values = self._preprocess_image(image)
        query = self._build_mm_prompt(prompt)
        input_ids = self.processor(query, return_tensors="pt").input_ids.to(self.device)
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

    def prepare_text_inputs(self, prompt: str) -> Dict:
        query = self._build_text_prompt(prompt)
        input_ids = self.processor(query, return_tensors="pt").input_ids.to(self.device)
        return {"input_ids": input_ids}

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """InternVL2 图像预处理: resize→normalize→tensor。"""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        INPUT_SIZE = 448

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values = transform(image).unsqueeze(0).to(self.device).to(torch.bfloat16)
        return pixel_values

    def get_llm_layers(self):
        return self.model.language_model.model.layers

    def get_visual_token_count(self, inputs=None) -> int:
        return self.VISUAL_TOKEN_COUNT

    def forward_text_backbone(self, text_inputs: Dict) -> Any:
        return self.model.language_model(
            **text_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward_mm(self, mm_inputs: Dict) -> Any:
        """
        参照 InternVL2 generate() 的逻辑: 手动将 visual features 注入 input_embeds,
        然后直接调用 language_model。绕过 forward() 对 image_flags 的依赖。
        """
        self._ensure_img_context_token_id()

        pixel_values = mm_inputs["pixel_values"]
        input_ids = mm_inputs["input_ids"]

        # 1. 提取 visual features
        vit_embeds = self.model.extract_feature(pixel_values)

        # 2. 获取 text embeddings
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # 3. 替换 IMG_CONTEXT 占位符为 visual features
        flat_ids = input_ids.reshape(B * N)
        selected = (flat_ids == self.model.img_context_token_id)
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)

        # 4. 直接调用 language_model（与 text-only 的 hook 目标层一致）
        return self.model.language_model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

    def _build_inputs_embeds(self, mm_inputs: Dict) -> torch.Tensor:
        """构造注入了 visual features 的 inputs_embeds（供 generate_mm 复用）。"""
        self._ensure_img_context_token_id()
        pixel_values = mm_inputs["pixel_values"]
        input_ids = mm_inputs["input_ids"]
        vit_embeds = self.model.extract_feature(pixel_values)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        flat_ids = input_ids.reshape(B * N)
        selected = (flat_ids == self.model.img_context_token_id)
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        return input_embeds.reshape(B, N, C)

    def generate_mm(self, prompt: str, image: Image.Image,
                    max_new_tokens: int = 200) -> str:
        """InternVL2 MM 生成: 手动注入 visual features → language_model.generate()。"""
        mm_inputs = self.prepare_mm_inputs(prompt, image)
        inputs_embeds = self._build_inputs_embeds(mm_inputs)
        input_len = inputs_embeds.shape[1]
        pad_token_id = self.processor.eos_token_id
        with torch.no_grad():
            gen_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_ids = gen_ids[0]  # generate with inputs_embeds 输出不含 prompt tokens
        return self.processor.decode(new_ids, skip_special_tokens=True)

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """InternVL2 text-only 生成: language_model.generate(input_ids=...)。"""
        text_inputs = self.prepare_text_inputs(prompt)
        input_len = text_inputs["input_ids"].shape[1]
        pad_token_id = self.processor.eos_token_id
        with torch.no_grad():
            gen_ids = self.model.language_model.generate(
                **text_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        new_ids = gen_ids[0][input_len:]
        return self.processor.decode(new_ids, skip_special_tokens=True)

    def _decode_new_tokens(self, gen_ids: torch.Tensor, input_len: int) -> str:
        new_ids = gen_ids[0][input_len:]
        return self.processor.decode(new_ids, skip_special_tokens=True)


# ============================================================
# InstructBLIP-Vicuna-7B Adapter
# ============================================================
class InstructBLIPAdapter(ModelAdapter):
    """
    InstructBLIP-Vicuna-7B (Salesforce/instructblip-vicuna-7b) 适配。
    - Visual encoder: BLIP-2 ViT-G/14
    - VL Connector: Q-Former (learnable 32 query tokens → 32 output tokens)
    - LLM backbone: Vicuna-7B (32 层)
    - text-only: model.language_model(input_ids=...)
    - mm: model.forward() with pixel_values + input_ids + qformer_input_ids
    - layers: model.language_model.model.layers
    """

    VISUAL_TOKEN_COUNT = 32  # Q-Former 输出固定 32 个 tokens

    def prepare_mm_inputs(self, prompt: str, image: Image.Image) -> Dict:
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    def prepare_text_inputs(self, prompt: str) -> Dict:
        # InstructBLIP text-only: 直接用 language_model tokenizer
        input_ids = self.processor.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        return {"input_ids": input_ids}

    def get_llm_layers(self):
        return self.model.language_model.model.layers

    def get_visual_token_count(self, inputs=None) -> int:
        return self.VISUAL_TOKEN_COUNT

    def forward_text_backbone(self, text_inputs: Dict) -> Any:
        return self.model.language_model(
            **text_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward_mm(self, mm_inputs: Dict) -> Any:
        return self.model(
            **mm_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    def generate_mm(self, prompt: str, image: Image.Image,
                    max_new_tokens: int = 200) -> str:
        """InstructBLIP MM 生成: model.generate() 含 pixel_values + qformer_input_ids。"""
        mm_inputs = self.prepare_mm_inputs(prompt, image)
        pad_token_id = self.processor.tokenizer.eos_token_id
        with torch.no_grad():
            gen_ids = self.model.generate(
                **mm_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        # InstructBLIP generate() 输出不含 prompt (language model 输出)
        return self.processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
def create_adapter(model_name: str, model, processor, device: str) -> ModelAdapter:
    """根据 model_name 创建对应的 adapter 实例。"""
    from .model_configs import MODEL_CONFIGS
    model_class = MODEL_CONFIGS[model_name]["model_class"]

    adapter_map = {
        "llava": LLaVAAdapter,
        "qwen2vl": Qwen2VLAdapter,
        "internvl": InternVL2Adapter,
        "instructblip": InstructBLIPAdapter,
    }
    if model_class not in adapter_map:
        raise ValueError(f"No adapter for model_class: {model_class}")

    return adapter_map[model_class](model, processor, device, model_name)
