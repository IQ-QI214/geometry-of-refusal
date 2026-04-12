"""
Exp A1 Cross-Validation: Validate Stealth Refusal with LLaMA-Guard-3-8B.

For each ablation config, identifies "disputed" cases where:
  - keyword: initial_bypass=True (potential harmful bypass)
  - Qwen3Guard: is_harmful=False  (judge says safe / stealth refusal)

Runs LLaMA-Guard (Meta family, independent of Qwen) on ONLY these disputed cases.

Usage (qwen3-vl env, transformers>=4.51):
  python exp_a1_cross_validate.py --model qwen2vl_7b --device cuda:0
  python exp_a1_cross_validate.py --model llava_7b   --device cuda:2
"""

import os
import sys
import argparse
import json
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "category_a"))

from common.judge_utils import create_judge

ABLATION_CONFIGS = ["ablation_nw_vmm", "ablation_all_vmm", "ablation_nw_vtext"]
DATASET = "saladbench"


def build_disputed_subset(gen_data: dict, qwen3guard_cfg: dict):
    """
    Extract disputed cases: keyword=bypass AND Qwen3Guard=safe.

    Returns list of dicts:
      {"idx": int, "prompt": str, "response": str}
    """
    responses = gen_data["responses"]
    judge_details = qwen3guard_cfg["judge_details"]

    assert len(responses) == len(judge_details), (
        f"Length mismatch: responses={len(responses)}, judge_details={len(judge_details)}"
    )
    disputed = []
    for i, (resp, jd) in enumerate(zip(responses, judge_details)):
        if resp.get("initial_bypass", False) and not jd["is_harmful"]:
            disputed.append({
                "idx": i,
                "prompt": resp["prompt"],
                "response": resp["response"],
            })
    return disputed


def compute_concordance(disputed: list, llamaguard_results: list) -> dict:
    """
    Compute concordance between Qwen3Guard (all safe) and LLaMA-Guard results.

    concordance_rate = fraction where LLaMA-Guard also says Safe
    discord_rate     = fraction where LLaMA-Guard says Unsafe
    """
    n = len(disputed)
    if n == 0:
        return {
            "n_disputed": 0,
            "n_concordant": 0,
            "n_discordant": 0,
            "concordance_rate": None,
            "discord_rate": None,
        }

    n_concordant = sum(1 for r in llamaguard_results if not r["is_harmful"])
    n_discordant = n - n_concordant

    return {
        "n_disputed": n,
        "n_concordant": n_concordant,
        "n_discordant": n_discordant,
        "concordance_rate": n_concordant / n,
        "discord_rate": n_discordant / n,
    }


def run_crossval(model_name: str, device: str):
    results_dir = _PROJ_ROOT / "results" / "category_a" / model_name
    qg_path = results_dir / f"a1_judged_qwen3guard_{DATASET}.json"
    output_path = results_dir / f"a1_crossval_llamaguard_{DATASET}.json"

    if not qg_path.exists():
        raise FileNotFoundError(f"Qwen3Guard judged results not found: {qg_path}")

    with open(qg_path) as f:
        qg_data = json.load(f)

    print(f"\n{'='*60}")
    print(f"A1 Cross-Validation: {model_name} | judge=llamaguard3 | device={device}")
    print(f"{'='*60}")

    # Load LLaMA-Guard once, reuse across configs
    judge = create_judge("llamaguard3", device)

    config_results = {}

    for config_name in ABLATION_CONFIGS:
        gen_path = results_dir / f"a1_{config_name}_{DATASET}.json"
        if not gen_path.exists():
            print(f"[SKIP] {config_name}: generation file not found")
            continue

        with open(gen_path) as f:
            gen_data = json.load(f)

        qg_cfg = qg_data["configs"].get(config_name)
        if qg_cfg is None:
            print(f"[SKIP] {config_name}: not in Qwen3Guard results")
            continue

        disputed = build_disputed_subset(gen_data, qg_cfg)
        print(f"\n[{config_name}] Disputed subset: {len(disputed)} / {len(gen_data['responses'])}")

        if len(disputed) == 0:
            print("  No disputed cases — skipping LLaMA-Guard inference.")
            concordance = compute_concordance(disputed, [])
            config_results[config_name] = {
                "config": config_name,
                "n_total": len(gen_data["responses"]),
                **concordance,
                "cross_validated_stealth_rate": None,
                "adjusted_FHCR": qg_cfg["fhcr_judge"],
                "llamaguard_details": [],
            }
            continue

        # Run LLaMA-Guard on disputed subset only
        pairs = [{"prompt": d["prompt"], "response": d["response"]} for d in disputed]
        llamaguard_results = judge.judge_batch(pairs)

        concordance = compute_concordance(disputed, llamaguard_results)

        # cross_validated_stealth_rate: both judges agree it's safe, as % of ALL prompts
        n_total = len(gen_data["responses"])
        n_crossval_stealth = concordance["n_concordant"]
        cross_validated_stealth_rate = n_crossval_stealth / n_total

        # adjusted_FHCR: harmful = Qwen3Guard harmful + disputed cases where LlamaGuard says harmful
        n_harmful_qg = qg_cfg["n_harmful_judge"]
        adjusted_n_harmful = n_harmful_qg + concordance["n_discordant"]
        adjusted_FHCR = adjusted_n_harmful / n_total

        print(f"  concordance_rate={concordance['concordance_rate']:.3f}  "
              f"discord_rate={concordance['discord_rate']:.3f}  "
              f"cross_validated_stealth_rate={cross_validated_stealth_rate:.3f}  "
              f"adjusted_FHCR={adjusted_FHCR:.3f}")

        config_results[config_name] = {
            "config": config_name,
            "n_total": n_total,
            **concordance,
            "cross_validated_stealth_rate": cross_validated_stealth_rate,
            "adjusted_FHCR": adjusted_FHCR,
            "disputed_indices": [d["idx"] for d in disputed],
            "llamaguard_details": llamaguard_results,
        }

    output = {
        "model": model_name,
        "judge_primary": "qwen3guard",
        "judge_cross": "llamaguard3",
        "dataset": DATASET,
        "configs": config_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="A1 Judge Cross-Validation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen2vl_7b", "qwen2vl_32b", "llava_7b", "llava_13b"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_crossval(args.model, args.device)


if __name__ == "__main__":
    main()
