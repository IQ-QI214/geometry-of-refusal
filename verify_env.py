#!/usr/bin/env python3
"""verify_env.py — CPU-side static check that the gemma-4-heretic probe env is ready.

No GPU required. Does NOT load model weights. Exits 0 on PASS, 1 on any FAIL.

Checks:
  1. Imports: transformers, peft, strong_reject, sentencepiece, accelerate, torch
  2. Gemma4ForConditionalGeneration class resolvable from transformers
  3. gemma-4-heretic config.json declares architecture "Gemma4ForConditionalGeneration"
  4. Tokenizer config file exists and has extra_special_tokens patch target
  5. Version summary printout
"""
import json
import sys
from pathlib import Path

MODEL_PATH = Path("/inspire/hdd/global_user/wenming-253108090054/models/gemma-4-heretic")

results = []

def check(name, fn):
    try:
        fn()
        results.append((True, name, ""))
        print(f"  PASS  {name}")
    except Exception as e:
        results.append((False, name, str(e)))
        print(f"  FAIL  {name}  -- {e}")


print("=" * 64)
print("gemma-4-heretic probe env — static verification")
print("=" * 64)

def check_imports():
    import torch
    import transformers
    import accelerate
    import peft
    import sentencepiece
    import safetensors
    import tokenizers
    import huggingface_hub

def check_strong_reject():
    from strong_reject.evaluate import cached_models, evaluate  # noqa: F401

def check_gemma4_class():
    from transformers import Gemma4ForConditionalGeneration  # noqa: F401

def check_auto_classes():
    from transformers import (
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoTokenizer,
    )  # noqa: F401

def check_model_config():
    cfg_path = MODEL_PATH / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(str(cfg_path))
    with open(cfg_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    if "Gemma4ForConditionalGeneration" not in archs:
        raise ValueError(f"unexpected architectures: {archs}")

def check_tokenizer_config_shape():
    tc_path = MODEL_PATH / "tokenizer_config.json"
    if not tc_path.is_file():
        raise FileNotFoundError(str(tc_path))
    with open(tc_path) as f:
        cfg = json.load(f)
    if "extra_special_tokens" not in cfg:
        return
    val = cfg["extra_special_tokens"]
    if not isinstance(val, (list, dict)):
        raise ValueError(f"extra_special_tokens is {type(val).__name__}, expected list or dict")

def print_versions():
    import torch, transformers, accelerate, peft
    print()
    print(f"  torch         : {torch.__version__}")
    print(f"  transformers  : {transformers.__version__}")
    print(f"  accelerate    : {accelerate.__version__}")
    print(f"  peft          : {peft.__version__}")
    try:
        import strong_reject
        print(f"  strong_reject : {getattr(strong_reject, '__version__', 'unknown')}")
    except Exception:
        pass

check("imports (torch, transformers, accelerate, peft, sentencepiece, hf_hub)", check_imports)
check("strong_reject.evaluate importable", check_strong_reject)
check("Gemma4ForConditionalGeneration in transformers", check_gemma4_class)
check("AutoModelForImageTextToText / AutoModelForCausalLM / AutoTokenizer", check_auto_classes)
check("gemma-4-heretic config.json architecture", check_model_config)
check("tokenizer_config.json extra_special_tokens shape", check_tokenizer_config_shape)

print_versions()

print()
n_fail = sum(1 for ok, _, _ in results if not ok)
if n_fail == 0:
    print(f"ALL {len(results)} CHECKS PASS.")
    sys.exit(0)
else:
    print(f"FAILURES: {n_fail} / {len(results)}")
    sys.exit(1)
