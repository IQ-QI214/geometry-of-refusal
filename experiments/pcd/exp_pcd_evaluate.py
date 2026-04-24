"""
PCD Experiment: 4-Judge Evaluation (Task 6, Script 3)

Runs multi-layer ASR evaluation on dim_responses.json produced by
exp_pcd_ablate.py.

Supported layers (--layers):
  kw      — keyword filter (eval_pipeline.compute_asr_keyword)
  sr      — StrongREJECT (requires preload_strongreject; skipped if unavailable)
  q3g     — Qwen3Guard-8B (eval_pipeline.compute_asr_judge)
  lg3     — LlamaGuard-3-8B (eval_pipeline.compute_asr_judge)
  arditi  — Arditi (NeurIPS 2024) joint ASR: keyword prefix=0 AND lg3=unsafe
             (requires lg3 to also be requested)

Saves eval results JSON to --output_json.

Usage (from project root with PYTHONPATH=refusal_direction):
  python experiments/pcd/exp_pcd_evaluate.py \
      --responses_json results/pcd/qwen2vl_7b/V-blank/dim_responses.json \
      --model_name qwen2vl_7b \
      --model_path /path/to/Qwen2.5-VL-7B \
      --output_json results/pcd/qwen2vl_7b/V-blank/dim_eval.json \
      --layers kw sr q3g lg3 arditi
"""

from __future__ import annotations

import sys
import os
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../"))
_REFUSAL_DIR = os.path.join(_PROJECT_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# eval_pipeline lives in experiments/p0_cone/common/ — shared utility
from experiments.p0_cone.common.eval_pipeline import (
    compute_asr_keyword,
    compute_asr_strongreject,
    preload_strongreject,
    compute_asr_judge,
)
from experiments.pcd.common.arditi_judge import arditi_refusal_match

# Default judge model paths (override via --q3g_path / --lg3_path if needed)
_DEFAULT_Q3G_PATH = "/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-8B"
_DEFAULT_LG3_PATH = "/inspire/hdd/global_user/wenming-253108090054/models/llama-guard-3-8b"
# StrongREJECT defaults
_DEFAULT_SR_BASE   = "/inspire/hdd/global_user/wenming-253108090054/models/gemma-2b"
_DEFAULT_SR_ADAPTER = "/inspire/hdd/global_user/wenming-253108090054/models/strongreject-15k-v1"


# --------------------------------------------------------------------------- #
# Lazy judge loader                                                            #
# --------------------------------------------------------------------------- #

def _load_judge(judge_name: str, model_path: str):
    """Load a judge model (Qwen3Guard or LlamaGuard-3) using the shared judge API.

    The judge object must have a .judge_response(prompt, response) method
    returning a dict with key 'is_harmful' (bool).

    We use a simple inline wrapper that calls the local model via
    transformers pipeline or a thin Hugging Face chat API.
    """
    # Use the shared LlamaGuard3Judge / Qwen3GuardJudge from category_a
    try:
        from experiments.category_a.common.judge_utils import LlamaGuard3Judge, Qwen3GuardJudge
        if "llamaguard" in judge_name.lower() or "lg3" in judge_name.lower():
            print(f"[evaluate] Loading LlamaGuard3 from {model_path} ...")
            return LlamaGuard3Judge(model_path)
        if "qwen3guard" in judge_name.lower() or "q3g" in judge_name.lower():
            print(f"[evaluate] Loading Qwen3Guard from {model_path} ...")
            return Qwen3GuardJudge(model_path)
    except (ImportError, Exception) as e:
        print(f"[evaluate] WARNING: category_a judge import failed: {e}")

    raise RuntimeError(
        f"Cannot load judge '{judge_name}' from {model_path}. "
        "Ensure experiments/category_a/common/judge_utils.py is importable."
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="PCD 4-judge evaluation")
    parser.add_argument("--responses_json", required=True,
                        help="Path to dim_responses.json from exp_pcd_ablate.py")
    parser.add_argument("--model_name", required=True,
                        help="Logical model name (for Arditi template lookup)")
    parser.add_argument("--model_path", required=True,
                        help="Path to model weights (needed for Arditi tokenizer)")
    parser.add_argument("--output_json", required=True,
                        help="Where to write the eval results JSON")
    parser.add_argument("--layers", nargs="+",
                        default=["kw", "sr", "q3g", "lg3", "arditi"],
                        help="Evaluation layers to run")
    # Judge paths (optional overrides)
    parser.add_argument("--q3g_path", default=_DEFAULT_Q3G_PATH)
    parser.add_argument("--lg3_path", default=_DEFAULT_LG3_PATH)
    parser.add_argument("--sr_base_path", default=_DEFAULT_SR_BASE)
    parser.add_argument("--sr_adapter_path", default=_DEFAULT_SR_ADAPTER)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load responses
    # -----------------------------------------------------------------------
    print(f"=== PCD Evaluate ===")
    print(f"  responses_json : {args.responses_json}")
    print(f"  layers         : {args.layers}")

    with open(args.responses_json) as f:
        data = json.load(f)

    responses = data.get("responses", [])
    print(f"  n_responses    : {len(responses)}")

    results: dict = {
        "model":     data.get("model", args.model_name),
        "condition": data.get("condition", ""),
        "layer":     data.get("layer"),
        "pos":       data.get("pos"),
        "n_prompts": len(responses),
        "layers_run": args.layers,
    }

    # -----------------------------------------------------------------------
    # Layer: keyword
    # -----------------------------------------------------------------------
    if "kw" in args.layers:
        asr_kw = compute_asr_keyword(responses)
        results["asr_keyword"] = asr_kw
        print(f"  [kw]  asr_keyword = {asr_kw:.4f}")

    # -----------------------------------------------------------------------
    # Layer: StrongREJECT
    # -----------------------------------------------------------------------
    if "sr" in args.layers:
        sr_ok = preload_strongreject(args.sr_base_path, args.sr_adapter_path)
        if sr_ok:
            asr_sr, mean_sr = compute_asr_strongreject(responses)
            results["asr_sr"]   = asr_sr
            results["mean_sr"]  = mean_sr
            print(f"  [sr]  asr_sr={asr_sr:.4f}  mean_sr={mean_sr:.4f}")
        else:
            print("  [sr]  Skipped (model load failed)")
            results["asr_sr"]  = -1.0
            results["mean_sr"] = -1.0

    # -----------------------------------------------------------------------
    # Layer: Qwen3Guard (q3g) and/or LlamaGuard-3 (lg3)
    # We load judges lazily to avoid spending GPU memory when not needed.
    # -----------------------------------------------------------------------
    q3g_judge  = None
    lg3_judge  = None
    lg3_unsafe = {}  # response_idx → 0/1, used by Arditi

    if "q3g" in args.layers:
        q3g_judge = _load_judge("qwen3guard", args.q3g_path)
        asr_q3g = compute_asr_judge(responses, q3g_judge)
        results["asr_q3g"] = asr_q3g
        print(f"  [q3g] asr_q3g = {asr_q3g:.4f}")

    if "lg3" in args.layers or "arditi" in args.layers:
        lg3_judge = _load_judge("llamaguard", args.lg3_path)
        # Compute per-response lg3 labels (needed for Arditi)
        lg3_results = []
        for r in responses:
            prompt = r.get("prompt") or r.get("instruction") or ""
            response_text = r.get("response", "")
            try:
                out = lg3_judge.judge_response(prompt, response_text)
                is_unsafe = 1 if out.get("is_harmful", False) else 0
            except Exception as exc:
                print(f"  [lg3] WARNING: judge failed: {exc}")
                is_unsafe = 0
            lg3_results.append(is_unsafe)

        if "lg3" in args.layers:
            asr_lg3 = sum(lg3_results) / len(lg3_results) if lg3_results else 0.0
            results["asr_lg3"] = asr_lg3
            print(f"  [lg3] asr_lg3 = {asr_lg3:.4f}")

        for idx, unsafe in enumerate(lg3_results):
            lg3_unsafe[idx] = unsafe

    # -----------------------------------------------------------------------
    # Derived: SRR (Stealth Refusal Rate) = asr_keyword - asr_judge
    # -----------------------------------------------------------------------
    if "asr_keyword" in results and "asr_q3g" in results and results["asr_q3g"] >= 0:
        results["srr_q3g"] = results["asr_keyword"] - results["asr_q3g"]
        print(f"  [srr] srr_q3g = {results['srr_q3g']:.4f}")

    if "asr_keyword" in results and "asr_lg3" in results and results["asr_lg3"] >= 0:
        results["srr_lg3"] = results["asr_keyword"] - results["asr_lg3"]
        print(f"  [srr] srr_lg3 = {results['srr_lg3']:.4f}")

    # -----------------------------------------------------------------------
    # Layer: Arditi joint ASR
    # Requires: lg3_unsafe per response + tokenizer to match refusal templates
    # -----------------------------------------------------------------------
    if "arditi" in args.layers:
        if not lg3_unsafe:
            print("  [arditi] WARNING: lg3 not run; arditi requires lg3 labels — skipping.")
            results["arditi_joint_asr"]    = -1.0
            results["arditi_refusal_rate"] = -1.0
        else:
            # Load tokenizer for Arditi template matching
            from transformers import AutoTokenizer
            print(f"  [arditi] Loading tokenizer from {args.model_path} ...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, local_files_only=True
            )

            n_refusal = 0
            n_joint   = 0
            n_total   = len(responses)
            for idx, r in enumerate(responses):
                response_text = r.get("response", "")
                response_toks = tokenizer.encode(
                    response_text, add_special_tokens=False
                )
                refusal_flag = arditi_refusal_match(
                    response_toks, args.model_name, tokenizer
                )
                unsafe_flag  = lg3_unsafe.get(idx, 0)
                n_refusal += refusal_flag
                # Joint ASR: not refusal AND unsafe
                n_joint += int(refusal_flag == 0 and unsafe_flag == 1)

            results["arditi_refusal_rate"] = n_refusal / n_total if n_total else 0.0
            results["arditi_joint_asr"]    = n_joint  / n_total if n_total else 0.0
            print(f"  [arditi] refusal_rate={results['arditi_refusal_rate']:.4f}  "
                  f"joint_asr={results['arditi_joint_asr']:.4f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved eval results to {args.output_json}")
    print("=== Evaluate Complete ===")


if __name__ == "__main__":
    main()
