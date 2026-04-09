"""
Exp A1 Judge: Post-hoc evaluation of A1 responses with judge models.

Reads a1_{config}_{dataset}.json files, runs judge model, outputs judged metrics.

Usage (in transformers>=4.51 env):
  python exp_a1_judge.py --model llava_7b --judge qwen3guard --device cuda:0
  python exp_a1_judge.py --model llava_7b --judge llamaguard3 --dataset harmbench
"""

import os
import sys
import argparse
import json
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "category_a"))

from common.judge_utils import create_judge

CONFIG_NAMES = [
    "baseline_text", "baseline_mm",
    "ablation_nw_vmm", "ablation_all_vmm", "ablation_nw_vtext",
]


def run_judge(model_name: str, judge_name: str, dataset_name: str,
              device: str):

    results_dir = _PROJ_ROOT / "results" / "category_a" / model_name
    output_path = results_dir / f"a1_judged_{judge_name}_{dataset_name}.json"

    print(f"\n{'='*60}")
    print(f"A1 Judge: {model_name} | judge={judge_name} | dataset={dataset_name}")
    print(f"{'='*60}")

    judge = create_judge(judge_name, device)
    all_judged = {}

    for config_name in CONFIG_NAMES:
        response_path = results_dir / f"a1_{config_name}_{dataset_name}.json"
        if not response_path.exists():
            print(f"[SKIP] {config_name}: {response_path} not found")
            continue

        with open(response_path) as f:
            data = json.load(f)

        responses = data.get("responses", data)  # handle both formats
        print(f"\n[{config_name}] Judging {len(responses)} responses...")

        pairs = [{"prompt": r["prompt"], "response": r["response"]} for r in responses]
        judge_results = judge.judge_batch(pairs)

        # Compute FHCR_judge
        n = len(judge_results)
        n_harmful = sum(1 for r in judge_results if r["is_harmful"])
        fhcr_judge = n_harmful / n if n > 0 else 0

        # Merge keyword metrics from original
        metrics_kw = data.get("metrics_kw", {})

        ibr = metrics_kw.get("initial_bypass_rate", -1)
        config_result = {
            "config": config_name,
            "n_prompts": n,
            "fhcr_judge": fhcr_judge,
            "n_harmful_judge": n_harmful,
            "metrics_kw": metrics_kw,
            "dsa_gap": ibr - fhcr_judge if ibr >= 0 else -1,
            "judge_details": judge_results,
        }
        all_judged[config_name] = config_result

        fhcr_kw = metrics_kw.get("full_harmful_completion_rate", -1)
        print(f"  IBR_kw={ibr:.3f}  FHCR_kw={fhcr_kw:.3f}  "
              f"FHCR_{judge_name}={fhcr_judge:.3f}  DSA_gap={ibr - fhcr_judge:.3f}")

    # Save
    output = {
        "model": model_name,
        "judge": judge_name,
        "dataset": dataset_name,
        "configs": all_judged,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="A1 Judge Evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--judge", type=str, required=True,
                        choices=["qwen3guard", "llamaguard3"])
    parser.add_argument("--dataset", type=str, default="saladbench",
                        choices=["saladbench", "harmbench"])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_judge(args.model, args.judge, args.dataset, args.device)


if __name__ == "__main__":
    main()
