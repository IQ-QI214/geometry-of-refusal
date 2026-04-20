"""Single source of truth for model and judge paths."""

MODEL_PATHS = {
    "qwen2.5_7b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct",
    "llama3.1_8b": "/inspire/hdd/global_user/wenming-253108090054/models/Llama-3.1-8B-Instruct",
}

JUDGE_PATHS = {
    "llamaguard3":          "/inspire/hdd/global_user/wenming-253108090054/models/llama-guard-3-8b",
    "strongreject_base":    "/inspire/hdd/global_user/wenming-253108090054/models/gemma-2b",
    "strongreject_adapter": "/inspire/hdd/global_user/wenming-253108090054/models/strongreject-15k-v1",
}
