"""Smoke test: load Gemma-3 adapters and verify tokenization (no GPU needed for tokenization checks)."""
import argparse
import sys
import torch


def smoke_vlm_tokenization(path):
    """Test tokenization paths without running model.generate (no GPU needed)."""
    from transformers import AutoProcessor
    from pipeline.model_utils.gemma3_vlm_model import tokenize_instructions_gemma3_vlm
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)

    # text mode: no pixel_values
    tok_text = tokenize_instructions_gemma3_vlm(
        processor=processor, instructions=["How to bake a cake?"], image_mode="text"
    )
    assert "input_ids" in tok_text, "text mode: missing input_ids"
    has_pv = "pixel_values" in tok_text and tok_text.pixel_values is not None
    assert not has_pv, f"text mode: unexpected pixel_values present"

    # blank mode: pixel_values present
    tok_blank = tokenize_instructions_gemma3_vlm(
        processor=processor, instructions=["How to bake a cake?"], image_mode="blank"
    )
    assert "pixel_values" in tok_blank and tok_blank.pixel_values is not None, \
        "blank mode: pixel_values missing"

    # noise mode: pixel_values present and different from blank
    tok_noise = tokenize_instructions_gemma3_vlm(
        processor=processor, instructions=["How to bake a cake?"], image_mode="noise"
    )
    assert "pixel_values" in tok_noise and tok_noise.pixel_values is not None, \
        "noise mode: pixel_values missing"
    # noise and blank should produce different pixel values
    assert not torch.allclose(tok_blank.pixel_values, tok_noise.pixel_values), \
        "noise and blank pixel_values are identical (expected different)"

    print("VLM_TOKENIZATION_SMOKE: PASS")


def smoke_text_tokenization(path):
    """Test text adapter tokenization (CPU only)."""
    from transformers import AutoTokenizer
    from pipeline.model_utils.gemma3_model import tokenize_instructions_gemma3_text
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok = tokenize_instructions_gemma3_text(
        tokenizer=tokenizer, instructions=["How to bake a cake?"]
    )
    assert "input_ids" in tok, "text adapter: missing input_ids"
    assert not (hasattr(tok, "pixel_values") and tok.pixel_values is not None), \
        "text adapter: unexpected pixel_values"
    print("TEXT_TOKENIZATION_SMOKE: PASS")


def smoke_vlm_generate(path):
    """Full model load + generate (GPU required)."""
    from pipeline.model_utils.gemma3_vlm_model import Gemma3VLMModel
    m = Gemma3VLMModel(path)
    resp = m.generate_completions(
        [{"instruction": "How to bake a cake?", "category": "smoke"}],
        max_new_tokens=16,
    )
    assert len(resp) == 1 and len(resp[0]["response"]) > 0, "VLM generate: empty response"
    print(f"VLM_GENERATE_SMOKE: PASS — response: {resp[0]['response'][:80]}")
    print(f"  num layers: {len(m.model_block_modules)}")


def smoke_text_generate(path):
    """Full model load + generate (GPU required)."""
    from pipeline.model_utils.gemma3_model import Gemma3Model
    m = Gemma3Model(path)
    resp = m.generate_completions(
        [{"instruction": "How to bake a cake?", "category": "smoke"}],
        max_new_tokens=16,
    )
    assert len(resp) == 1 and len(resp[0]["response"]) > 0, "Text generate: empty response"
    print(f"TEXT_GENERATE_SMOKE: PASS — response: {resp[0]['response'][:80]}")
    print(f"  num layers: {len(m.model_block_modules)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--mode", choices=["tokenization", "generate", "all"], default="tokenization",
                   help="tokenization = CPU-only checks; generate = GPU required; all = both")
    args = p.parse_args()

    if args.mode in ("tokenization", "all"):
        smoke_vlm_tokenization(args.model_path)
        smoke_text_tokenization(args.model_path)
    if args.mode in ("generate", "all"):
        smoke_vlm_generate(args.model_path)
        smoke_text_generate(args.model_path)
