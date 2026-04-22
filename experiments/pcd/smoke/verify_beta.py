"""Verification point beta: Qwen2.5-VL forward accepts no pixel_values when image_mode='text'."""
import argparse
import sys

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

DEFAULT_MODEL_PATH = "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-VL-7B-Instruct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    args = p.parse_args()
    model_path = args.model_path

    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).eval().to("cuda:0")

    # --- Test 1: raw processor path (no image block) ---
    msg = [{"role": "user", "content": "How to bake a cake?"}]
    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], padding=True, truncation=False, return_tensors="pt")

    input_ids = inputs.input_ids.to("cuda:0")
    attention_mask = inputs.attention_mask.to("cuda:0")
    has_pv = hasattr(inputs, "pixel_values") and inputs.pixel_values is not None
    print(f"input_ids shape: {input_ids.shape}")
    print(f"pixel_values present (raw processor): {has_pv}")

    with torch.no_grad():
        try:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
            gen = out[:, input_ids.shape[-1]:]
            text = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
            print(f"Generated via raw processor ({len(text)} chars): {text[:200]}")
            raw_ok = len(text.strip()) > 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Raw processor path failed: {type(e).__name__}: {e}", file=sys.stderr)
            raw_ok = False

    # --- Test 2: adapter path via tokenize_instructions_qwen_vlm(image_mode="text") ---
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
    try:
        from pipeline.model_utils.qwen_vlm_model import tokenize_instructions_qwen_vlm
        tok = tokenize_instructions_qwen_vlm(
            processor=processor,
            instructions=["How to bake a cake?"],
            image_mode="text",
        )
        adapter_has_pv = hasattr(tok, "pixel_values") and tok.pixel_values is not None
        print(f"pixel_values present (adapter image_mode='text'): {adapter_has_pv}")
        adapter_ok = not adapter_has_pv
        if adapter_has_pv:
            print("WARNING: adapter returned pixel_values for image_mode='text'")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Adapter path failed: {type(e).__name__}: {e}", file=sys.stderr)
        adapter_ok = False

    if raw_ok and adapter_ok:
        print("BETA_RESULT: PASS")
    elif not raw_ok:
        print("BETA_RESULT: FAIL_EMPTY" if True else "BETA_RESULT: FAIL_EXCEPTION")
    else:
        print("BETA_RESULT: FAIL_ADAPTER")


if __name__ == "__main__":
    main()
