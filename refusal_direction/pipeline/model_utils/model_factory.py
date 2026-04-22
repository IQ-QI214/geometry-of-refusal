from pipeline.model_utils.model_base import ModelBase

# VLM model path indicators — checked before text-only fallbacks
# (Qwen2.5-VL path contains "qwen" so VLM must be matched first)
VLM_INDICATORS = {
    "llava": ["llava-1.5", "llava-hf"],
    "qwen_vlm": ["Qwen2.5-VL", "qwen2.5-vl"],
}

# Explicit model_name → class mapping (takes priority over path-based dispatch)
_MODEL_NAME_MAP = {
    "gemma-3-4b-it-vlm": ("pipeline.model_utils.gemma3_vlm_model", "Gemma3VLMModel"),
    "gemma-3-4b-it":     ("pipeline.model_utils.gemma3_model",     "Gemma3Model"),
}


def construct_model_base(model_path: str, model_name: str = None) -> ModelBase:
    # 1. Dispatch by explicit model_name when provided
    if model_name is not None:
        key = model_name.lower()
        if key in _MODEL_NAME_MAP:
            module_path, class_name = _MODEL_NAME_MAP[key]
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(model_path)
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Known names: {list(_MODEL_NAME_MAP.keys())}")

    path_lower = model_path.lower()

    # 2. Check VLM models first (more specific patterns)
    for indicator in VLM_INDICATORS["llava"]:
        if indicator.lower() in path_lower:
            from pipeline.model_utils.llava_vlm_model import LlavaVLMModel
            return LlavaVLMModel(model_path)

    for indicator in VLM_INDICATORS["qwen_vlm"]:
        if indicator.lower() in path_lower:
            from pipeline.model_utils.qwen_vlm_model import QwenVLMModel
            return QwenVLMModel(model_path)

    # 3. Fallback to text-only models
    if 'qwen' in path_lower:
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in path_lower:
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in path_lower:
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma-3' in path_lower:
        # Path-based dispatch always loads VLM class for gemma-3.
        # For text-only mode, use: construct_model_base(path, model_name="gemma-3-4b-it")
        from pipeline.model_utils.gemma3_vlm_model import Gemma3VLMModel
        return Gemma3VLMModel(model_path)
    elif 'gemma' in path_lower:
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)
    elif 'yi' in path_lower:
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
