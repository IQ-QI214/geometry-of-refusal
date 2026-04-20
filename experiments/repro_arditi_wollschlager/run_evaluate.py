"""
run_evaluate.py — Evaluate all completions from DIM, RDO, Cone.

Reads:
  results/repro_arditi_wollschlager/<model_alias>/completions/saladbench_{baseline,ablation}_completions.json
  results/repro_arditi_wollschlager/rdo/<model_alias>/<config>/completions/saladbench_{baseline,ablation}_completions.json

Writes: results/repro_arditi_wollschlager/evaluation.json (merge-safe)

Usage:
  # Keyword (CPU):
  python experiments/repro_arditi_wollschlager/run_evaluate.py --judge keyword --model all
  # LG3 (GPU):
  CUDA_VISIBLE_DEVICES=0 conda run -n rdo python experiments/repro_arditi_wollschlager/run_evaluate.py --judge llamaguard3 --model qwen2.5_7b --device cuda:0
  CUDA_VISIBLE_DEVICES=1 conda run -n rdo python experiments/repro_arditi_wollschlager/run_evaluate.py --judge llamaguard3 --model llama3.1_8b --device cuda:0
  # SR (GPU):
  CUDA_VISIBLE_DEVICES=2 conda run -n rdo python experiments/repro_arditi_wollschlager/run_evaluate.py --judge strongreject --model qwen2.5_7b --device cuda:0
  CUDA_VISIBLE_DEVICES=3 conda run -n rdo python experiments/repro_arditi_wollschlager/run_evaluate.py --judge strongreject --model llama3.1_8b --device cuda:0
"""
import sys, os, json, glob, argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from common.model_paths import MODEL_PATHS
from common.eval_judges import (
    compute_asr_keyword,
    compute_asr_llamaguard3, create_llamaguard3_judge,
    preload_strongreject, compute_asr_strongreject,
)
from common.stealth_analysis import compute_srr

SAVE_ROOT = "results/repro_arditi_wollschlager"
EVAL_JSON = os.path.join(SAVE_ROOT, "evaluation.json")


def find_completion_pairs(model_alias: str) -> dict:
    """Return dict: config_name → (baseline_path, ablation_path)"""
    configs = {}
    # DIM
    dim_base = os.path.join(SAVE_ROOT, model_alias, "completions")
    b = os.path.join(dim_base, "saladbench_baseline_completions.json")
    a = os.path.join(dim_base, "saladbench_ablation_completions.json")
    if os.path.exists(b) and os.path.exists(a):
        configs["dim_ablation"] = (b, a)
    # RDO/Cone
    rdo_base = os.path.join(SAVE_ROOT, "rdo", model_alias)
    for config_comp_dir in sorted(glob.glob(os.path.join(rdo_base, "*/completions"))):
        config_name = os.path.basename(os.path.dirname(config_comp_dir))
        b = os.path.join(config_comp_dir, "saladbench_baseline_completions.json")
        a = os.path.join(config_comp_dir, "saladbench_ablation_completions.json")
        if os.path.exists(b) and os.path.exists(a):
            configs[config_name] = (b, a)
    return configs


def evaluate_model(model_key: str, judge_name: str, judge=None) -> dict:
    model_alias = os.path.basename(MODEL_PATHS[model_key])
    pairs = find_completion_pairs(model_alias)
    if not pairs:
        print(f"  [WARN] No completion files found for {model_alias}. Run T7/T8/T9 first.")
        return {}
    results = {}
    for config, (baseline_path, ablation_path) in pairs.items():
        baseline_comps = json.load(open(baseline_path))
        ablation_comps = json.load(open(ablation_path))
        entry = {"n": len(ablation_comps)}
        if judge_name in ("keyword", "all"):
            entry["asr_kw_baseline"] = compute_asr_keyword(baseline_comps)
            entry["asr_kw"]          = compute_asr_keyword(ablation_comps)
        if judge_name in ("llamaguard3", "all") and judge is not None:
            entry["asr_lg3_baseline"] = compute_asr_llamaguard3(baseline_comps, judge)
            entry["asr_lg3"]          = compute_asr_llamaguard3(ablation_comps, judge)
            if "asr_kw" in entry:
                entry["srr"] = compute_srr(entry["asr_kw"], entry["asr_lg3"])
        if judge_name in ("strongreject", "all"):
            _, entry["mean_sr_baseline"] = compute_asr_strongreject(baseline_comps)
            entry["asr_sr"], entry["mean_sr"] = compute_asr_strongreject(ablation_comps)
        results[config] = entry
        print(f"  [{model_alias}] {config}: {entry}")
    return results


def merge_and_save(new_results: dict, model_key: str):
    existing = {}
    if os.path.exists(EVAL_JSON):
        existing = json.load(open(EVAL_JSON))
    if model_key not in existing:
        existing[model_key] = {}
    for config, entry in new_results.items():
        existing[model_key].setdefault(config, {}).update(entry)
    os.makedirs(os.path.dirname(EVAL_JSON), exist_ok=True)
    with open(EVAL_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Saved to {EVAL_JSON}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["keyword", "llamaguard3", "strongreject", "all"], required=True)
    parser.add_argument("--model", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_keys = list(MODEL_PATHS.keys()) if args.model == "all" else [args.model]
    judge = None
    if args.judge in ("llamaguard3", "all"):
        judge = create_llamaguard3_judge(args.device)
    if args.judge in ("strongreject", "all"):
        preload_strongreject(args.device)

    for model_key in model_keys:
        print(f"\n=== Evaluating {model_key} with {args.judge} ===")
        results = evaluate_model(model_key, args.judge, judge)
        if results:
            merge_and_save(results, model_key)

    print(f"\nDone. Results in {EVAL_JSON}")

if __name__ == "__main__":
    main()
