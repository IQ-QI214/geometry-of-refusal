"""
Phase 3 模型配置与统一加载接口。

支持模型:
  llava_7b       - LLaVA-1.5-7B  (from HF hub cache)
  llava_13b      - LLaVA-1.5-13B  (from HF hub cache)
  qwen2vl_7b     - Qwen2.5-VL-7B-Instruct  (from local models/)
  qwen2vl_32b    - Qwen2.5-VL-32B-Instruct  (from public models/)
  internvl2_8b   - InternVL2-8B  (from local models/)
  instructblip_7b - InstructBLIP-Vicuna-7B  (from local models/)
"""

import os
import torch

# 本地模型目录
_LOCAL_MODELS_DIR = "/inspire/hdd/global_user/wenming-253108090054/models"
_HF_CACHE_DIR = os.getenv(
    "HUGGINGFACE_CACHE_DIR",
    "/inspire/hdd/global_user/wenming-253108090054/models/hub"
)

MODEL_CONFIGS = {
    "llava_7b": {
        # 从 HF hub 缓存加载，与 Phase 1/2 保持一致
        "model_path": "llava-hf/llava-1.5-7b-hf",
        "use_hub_cache": True,
        "model_class": "llava",
        "total_layers": 32,
        # probe_layers 相对深度约 25%/38%/50%/63%/88%
        "probe_layers": [8, 12, 16, 20, 28],
        "hidden_dim": 4096,
        "blank_image_size": (336, 336),
        "visual_token_count": 576,   # CLIP 336px → 576 patch tokens (固定)
    },
    "llava_13b": {
        "model_path": "llava-hf/llava-1.5-13b-hf",
        "use_hub_cache": True,
        "model_class": "llava",
        "total_layers": 40,
        # probe_layers 相对深度约 25%/38%/50%/63%/88%
        "probe_layers": [10, 15, 20, 25, 35],
        "hidden_dim": 5120,
        "blank_image_size": (336, 336),
        "visual_token_count": 576,
    },
    "qwen2vl_7b": {
        "model_path": os.path.join(_LOCAL_MODELS_DIR, "Qwen2.5-VL-7B-Instruct"),
        "use_hub_cache": False,
        "model_class": "qwen2vl",
        "total_layers": 28,
        # probe_layers 相对深度约 25%/39%/50%/64%/86%
        "probe_layers": [7, 11, 14, 18, 24],
        "hidden_dim": 3584,
        "blank_image_size": (336, 336),
        "visual_token_count": "dynamic",  # 由 image_grid_thw 决定，运行时计算
    },
    "qwen2vl_32b": {
        "model_path": "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-VL-32B-Instruct",
        "use_hub_cache": False,
        "model_class": "qwen2vl",
        "total_layers": 64,
        # probe_layers 相对深度约 25%/39%/50%/64%/86%
        "probe_layers": [16, 25, 32, 41, 55],
        "hidden_dim": 5120,
        "blank_image_size": (336, 336),
        "visual_token_count": "dynamic",
    },
    "internvl2_8b": {
        "model_path": os.path.join(_LOCAL_MODELS_DIR, "InternVL2-8B"),
        "use_hub_cache": False,
        "model_class": "internvl",
        "total_layers": 32,
        "probe_layers": [8, 12, 16, 20, 28],
        "hidden_dim": 4096,
        "blank_image_size": (448, 448),  # InternVL 标准输入尺寸
        "visual_token_count": 256,       # 单 tile = 256 tokens
        "trust_remote_code": True,
    },
    "instructblip_7b": {
        "model_path": os.path.join(_LOCAL_MODELS_DIR, "InstructBLIP-7B"),
        "use_hub_cache": False,
        "model_class": "instructblip",
        "total_layers": 32,
        "probe_layers": [8, 12, 16, 20, 28],
        "hidden_dim": 4096,
        "blank_image_size": (224, 224),  # InstructBLIP 标准输入尺寸
        "visual_token_count": 32,        # Q-Former 输出 32 个 query tokens
    },
}


def load_model_by_name(model_name: str, device: str):
    """
    统一加载接口。返回 (model, processor)。

    所有模型使用 bfloat16 + device_map=device + local_files_only=True。
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]
    model_class = cfg["model_class"]
    model_path = cfg["model_path"]
    trust_remote_code = cfg.get("trust_remote_code", False)

    print(f"[{model_name}] Loading from: {model_path}")
    print(f"[{model_name}] Model class: {model_class}, device: {device}")

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    if cfg["use_hub_cache"]:
        load_kwargs["cache_dir"] = _HF_CACHE_DIR
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    if model_class == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        processor_kwargs = {"local_files_only": True}
        if cfg["use_hub_cache"]:
            processor_kwargs["cache_dir"] = _HF_CACHE_DIR
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    elif model_class == "qwen2vl":
        # 使用 qwen3-vl 环境（transformers >= 4.52）加载
        # Qwen2_5_VLForConditionalGeneration 有 generate()，AutoModel 只有 backbone
        # 新版 transformers: torch_dtype → dtype，device_map 需要 accelerate 故手动 .to()
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        qwen_load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
        )
        if cfg["use_hub_cache"]:
            qwen_load_kwargs["cache_dir"] = _HF_CACHE_DIR
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **qwen_load_kwargs)
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )

    elif model_class == "internvl":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(model_path, **load_kwargs)
        processor = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )

    elif model_class == "instructblip":
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        processor = InstructBlipProcessor.from_pretrained(model_path, local_files_only=True)

    else:
        raise ValueError(f"Unknown model_class: {model_class}")

    model.eval()
    print(f"[{model_name}] Model loaded successfully.")
    return model, processor
