"""
Exp A3: Refusal Direction Norm as SC Predictor

For each prompt under baseline_mm, manually generate token-by-token,
recording the NW-layer refusal direction projection norm at each step.
Then analyze: does norm predict self-correction (AUROC)?

Usage:
  python exp_a3_norm_prediction.py --model llava_7b --device cuda:0
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase2" / "common"))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "category_a"))
# phase3 inserted last → becomes sys.path[0] → 'common' resolves to phase3/common
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase3"))

from common.model_configs import MODEL_CONFIGS, load_model_by_name
from common.model_adapters import create_adapter
from eval_utils import evaluate_response
# Import data_utils directly from category_a/common (avoids 'common' namespace conflict)
import importlib.util as _ilu
_du_spec = _ilu.spec_from_file_location(
    "cat_a_data_utils",
    _PROJ_ROOT / "experiments" / "category_a" / "common" / "data_utils.py")
_du_mod = _ilu.module_from_spec(_du_spec)
_du_spec.loader.exec_module(_du_mod)
load_saladbench_test = _du_mod.load_saladbench_test


# ── Norm recorder hook ──────────────────────────────────────────────────────

class NormRecorderHook:
    """Record |h · v̂| at each generation step."""

    def __init__(self, direction: torch.Tensor):
        self.direction = direction / direction.norm()
        self.norms: List[float] = []

    def __call__(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        last_h = hidden[:, -1, :]  # last token position
        d = self.direction.to(last_h.device, last_h.dtype)
        proj_norm = (last_h @ d).abs().item()
        self.norms.append(proj_norm)
        return output

    def reset(self):
        self.norms = []


# ── Direction loading ────────────────────────────────────────────────────────

def load_directions(model_name: str) -> dict:
    directions_path = (_PROJ_ROOT / "results" / "phase3" / model_name
                       / "exp_3a_directions.pt")
    if not directions_path.exists():
        raise FileNotFoundError(f"Directions not found: {directions_path}. Run Exp 3A first.")
    data = torch.load(directions_path, map_location="cpu")
    nw_layer = data["narrow_waist_layer"]
    v_mm = data["directions"][nw_layer]["v_mm"]
    return {"narrow_waist_layer": nw_layer, "v_mm": v_mm}


# ── Manual token-by-token generation ─────────────────────────────────────────

def generate_with_norm_recording(
    model, adapter, prompt: str, image: Image.Image,
    direction: torch.Tensor, nw_layer: int,
    max_new_tokens: int = 200,
) -> Dict:
    """
    Generate token-by-token with KV-cache, recording norm at each step.
    Works for LLaVA (language_model.generate fallback) and Qwen2.5-VL.
    """
    mm_inputs = adapter.prepare_mm_inputs(prompt, image)
    llm_layers = adapter.get_llm_layers()

    # Register recorder on NW layer output
    recorder = NormRecorderHook(direction)
    handle = llm_layers[nw_layer].register_forward_hook(recorder)

    try:
        # Prefill
        with torch.no_grad():
            outputs = adapter.forward_mm(mm_inputs)

        past_kv = outputs.past_key_values
        recorder.reset()  # discard prefill norms, only track generation

        # First generated token
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = [next_token_id.item()]

        # Determine EOS token
        if hasattr(adapter.processor, 'tokenizer'):
            eos_id = adapter.processor.tokenizer.eos_token_id
        elif hasattr(adapter.processor, 'eos_token_id'):
            eos_id = adapter.processor.eos_token_id
        else:
            eos_id = 2  # fallback

        # Autoregressive loop
        for t in range(max_new_tokens - 1):
            with torch.no_grad():
                try:
                    # LLaVA: use language_model directly
                    outputs = adapter.model.language_model(
                        input_ids=next_token_id,
                        past_key_values=past_kv,
                    )
                except Exception:
                    # Qwen2.5-VL: use model directly
                    outputs = adapter.model(
                        input_ids=next_token_id,
                        past_key_values=past_kv,
                    )

            past_kv = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token_id.item())

            if next_token_id.item() == eos_id:
                break

    finally:
        handle.remove()

    # Decode
    if hasattr(adapter.processor, 'tokenizer'):
        text = adapter.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        text = adapter.processor.decode(generated_ids, skip_special_tokens=True)

    return {
        "generated_text": text,
        "generated_ids": generated_ids,
        "norms": recorder.norms,
        "n_tokens": len(generated_ids),
    }


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_norm_prediction(results: List[Dict]) -> Dict:
    """Compute AUROC and temporal causality metrics."""
    from sklearn.metrics import roc_auc_score

    features_max = []
    features_mean = []
    labels = []
    sc_spike_precedes = []

    for r in results:
        norms = r["norms"]
        if len(norms) == 0:
            continue

        is_sc = r["eval"]["self_correction_found"]
        labels.append(1 if is_sc else 0)
        features_max.append(max(norms))
        features_mean.append(float(np.mean(norms)))

        # Temporal causality for SC sequences
        if is_sc and r["eval"]["self_correction_pos_chars"] > 0:
            total_chars = len(r["generated_text"])
            total_tokens = len(norms)
            if total_chars > 0 and total_tokens > 1:
                sc_token_approx = int(
                    r["eval"]["self_correction_pos_chars"] / total_chars * total_tokens
                )
                if sc_token_approx > 1:
                    spike_pos = int(np.argmax(norms[:sc_token_approx]))
                    sc_spike_precedes.append(spike_pos < sc_token_approx)

    analysis = {"n_sequences": len(labels)}

    if len(set(labels)) < 2:
        analysis["auroc_max_norm"] = None
        analysis["auroc_mean_norm"] = None
        analysis["note"] = "Cannot compute AUROC: only one class present"
    else:
        analysis["auroc_max_norm"] = float(roc_auc_score(labels, features_max))
        analysis["auroc_mean_norm"] = float(roc_auc_score(labels, features_mean))

    analysis["n_sc"] = sum(labels)
    analysis["n_non_sc"] = len(labels) - sum(labels)

    if sc_spike_precedes:
        analysis["spike_precedes_sc_rate"] = float(
            sum(sc_spike_precedes) / len(sc_spike_precedes))
        analysis["n_sc_analyzed"] = len(sc_spike_precedes)
    else:
        analysis["spike_precedes_sc_rate"] = None
        analysis["n_sc_analyzed"] = 0

    return analysis


# ── Main ─────────────────────────────────────────────────────────────────────

def run_a3(model_name: str, device: str, max_new_tokens: int = 200, n_prompts_limit: int = 0):
    cfg = MODEL_CONFIGS[model_name]
    prompts_data = load_saladbench_test()
    prompts = [item["instruction"] for item in prompts_data]
    if n_prompts_limit > 0:
        prompts = prompts[:n_prompts_limit]

    save_dir = _PROJ_ROOT / "results" / "category_a" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "a3_norm_prediction.json"

    print(f"\n{'='*60}")
    print(f"Exp A3: Norm Prediction — {model_name}")
    print(f"Prompts: {len(prompts)}, max_new_tokens={max_new_tokens}")
    print(f"{'='*60}")

    model, processor = load_model_by_name(model_name, device)
    adapter = create_adapter(model_name, model, processor, device)
    blank_image = Image.new("RGB", cfg["blank_image_size"], color=(128, 128, 128))

    dir_data = load_directions(model_name)
    nw_layer = dir_data["narrow_waist_layer"]
    v_mm = dir_data["v_mm"]

    results = []
    for i, prompt in enumerate(prompts):
        torch.cuda.empty_cache()
        gen_result = generate_with_norm_recording(
            model, adapter, prompt, blank_image, v_mm, nw_layer, max_new_tokens
        )
        eval_result = evaluate_response(gen_result["generated_text"])

        entry = {
            "prompt_idx": i,
            "prompt": prompt,
            "generated_text": gen_result["generated_text"][:500],
            "norms": gen_result["norms"],
            "n_tokens": gen_result["n_tokens"],
            "eval": eval_result,
        }
        results.append(entry)

        status = ("SC" if eval_result["self_correction_found"]
                  else ("HARMFUL" if eval_result["full_harmful_completion"] else "REFUSED"))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(prompts)}] {prompt[:40]}... → {status} "
                  f"(max_norm={max(gen_result['norms']) if gen_result['norms'] else 0:.3f})")

        # Save periodically
        if (i + 1) % 100 == 0:
            _save_intermediate(output_path, model_name, nw_layer, results)

    # Final analysis
    analysis = analyze_norm_prediction(results)
    print(f"\n{'='*60}")
    print(f"A3 Analysis — {model_name}")
    print(f"  AUROC (max_norm): {analysis.get('auroc_max_norm', 'N/A')}")
    print(f"  AUROC (mean_norm): {analysis.get('auroc_mean_norm', 'N/A')}")
    print(f"  Spike precedes SC: {analysis.get('spike_precedes_sc_rate', 'N/A')}")

    output = {
        "model": model_name,
        "narrow_waist_layer": nw_layer,
        "n_prompts": len(prompts),
        "analysis": analysis,
        "sequences": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[saved] {output_path}")


def _save_intermediate(path, model_name, nw_layer, results):
    output = {"model": model_name, "narrow_waist_layer": nw_layer,
              "n_prompts_so_far": len(results), "sequences": results}
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Exp A3: Norm Prediction")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--n_prompts", type=int, default=0,
                        help="Limit number of prompts (0 = use all). Use 10 for smoke test.")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_a3(args.model, args.device, args.max_new_tokens, args.n_prompts)


if __name__ == "__main__":
    main()
