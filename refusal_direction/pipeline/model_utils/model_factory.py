from pipeline.model_utils.model_base import ModelBase

# VLM model path indicators — checked before text-only fallbacks
# (Qwen2.5-VL path contains "qwen" so VLM must be matched first)
VLM_INDICATORS = {
    "llava": ["llava-1.5", "llava-hf"],
    "qwen_vlm": ["Qwen2.5-VL", "qwen2.5-vl"],
}

def construct_model_base(model_path: str) -> ModelBase:
    path_lower = model_path.lower()

    # Check VLM models first (more specific patterns)
    for indicator in VLM_INDICATORS["llava"]:
        if indicator.lower() in path_lower:
            from pipeline.model_utils.llava_vlm_model import LlavaVLMModel
            return LlavaVLMModel(model_path)

    for indicator in VLM_INDICATORS["qwen_vlm"]:
        if indicator.lower() in path_lower:
            from pipeline.model_utils.qwen_vlm_model import QwenVLMModel
            return QwenVLMModel(model_path)

    # Fallback to text-only models
    if 'qwen' in path_lower:
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in path_lower:
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in path_lower:
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in path_lower:
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)
    elif 'yi' in path_lower:
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
