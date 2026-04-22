"""Verification point alpha: Gemma-3-4B VLM language_model weights equal standalone text decoder."""
import argparse
import hashlib
import sys

import torch


def sha_of_state_dict(sd):
    h = hashlib.sha256()
    for name in sorted(sd.keys()):
        h.update(name.encode())
        h.update(sd[name].cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    args = p.parse_args()

    from transformers import Gemma3ForConditionalGeneration

    # Load as multimodal and extract language_model submodule
    model_mm = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    )
    lm_from_mm = {
        k.replace("model.language_model.", "", 1): v
        for k, v in model_mm.state_dict().items()
        if k.startswith("model.language_model.")
    }
    print(f"Num model.language_model.* keys from multimodal checkpoint: {len(lm_from_mm)}")
    del model_mm

    # Try loading as standalone CausalLM
    try:
        from transformers import Gemma3ForCausalLM
        model_lm = Gemma3ForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        lm_standalone = model_lm.state_dict()
        del model_lm
    except Exception as e:
        print(f"Cannot load as Gemma3ForCausalLM: {e}", file=sys.stderr)
        print("ALPHA_RESULT: STANDALONE_LOAD_FAILED")
        print("Interpretation: No separate text-only interface; L == V-text by construction.")
        return

    keys_mm = set(lm_from_mm.keys())
    keys_sa = set(lm_standalone.keys())
    print(f"Num language_model.* keys from standalone checkpoint: {len(lm_standalone)}")

    # Normalize SA keys: Gemma3ForCausalLM wraps backbone under model.* and adds lm_head.
    # Strip the model. prefix and drop lm_head to make keys comparable to MM extraction.
    lm_sa_normalized = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in lm_standalone.items()
        if k != "lm_head.weight"
    }
    keys_sa_norm = set(lm_sa_normalized.keys())

    if keys_mm != keys_sa_norm:
        print(f"Key mismatch after normalization. MM-only: {keys_mm - keys_sa_norm} | SA-only: {keys_sa_norm - keys_mm}")
        print("ALPHA_RESULT: KEY_MISMATCH")
        return

    mm_hash = sha_of_state_dict(lm_from_mm)
    sa_hash = sha_of_state_dict(lm_sa_normalized)
    print(f"multimodal language_model.* hash: {mm_hash}")
    print(f"standalone CausalLM state hash:  {sa_hash}")
    if mm_hash == sa_hash:
        print("ALPHA_RESULT: PASS (weights equal; L ≡ V-text collapse valid)")
    else:
        print("ALPHA_RESULT: FAIL (weights differ; run L and V-text as separate conditions)")


if __name__ == "__main__":
    main()
