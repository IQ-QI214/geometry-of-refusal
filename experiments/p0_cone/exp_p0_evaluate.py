"""
exp_p0_evaluate.py — Main evaluation runner for P0 Cone Ablation.

Evaluates 12 response files across 4 layers:
  Layer 1: keyword
  Layer 2: StrongREJECT
  Layer 3: Qwen3Guard-8B
  Layer 4: LLaMA-Guard-3-8B

Usage:
  python exp_p0_evaluate.py [--layers keyword sr qwen3guard llamaguard] [--results_dir results/p0_cone] [--device cuda:0]
"""

import sys
import os

# ---------------------------------------------------------------------------
# sys.path setup — must come before any local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../refusal_direction"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../category_a"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
from pathlib import Path

from common.eval_pipeline import evaluate_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = ["llava_7b", "qwen2vl_7b"]
METHODS = ["dim", "rdo"]
KS = [1, 3, 5]

GUARD_MODELS = {
    "qwen3guard": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen3Guard-Gen-8B",
    "llamaguard": "/inspire/hdd/global_user/wenming-253108090054/models/llama-guard-3-8b",
}

SR_MODELS = {
    "base": "/inspire/hdd/global_user/wenming-253108090054/models/gemma-2b",
    "adapter": "/inspire/hdd/global_user/wenming-253108090054/models/strongreject-15k-v1",
}


# ---------------------------------------------------------------------------
# Judge loading (deferred — only when needed)
# ---------------------------------------------------------------------------

def load_judges(layers: list, device: str) -> dict:
    """Load guard models and SR model. Imports inside function to avoid
    pulling in transformers at module level.
    Uses importlib for judge_utils to avoid sys.path 'common' package collision."""
    judges = {}

    # Layer 2: StrongREJECT — preload gemma-2b + LoRA adapter into strong_reject cache
    if "sr" in layers:
        from common.eval_pipeline import preload_strongreject
        preload_strongreject(SR_MODELS["base"], SR_MODELS["adapter"], device=device)

    needs_guard = "qwen3guard" in layers or "llamaguard" in layers
    if not needs_guard:
        return judges

    # Load judge_utils directly by absolute path to avoid shadowing by p0_cone/common/
    import importlib.util
    judge_utils_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../category_a/common/judge_utils.py"
    )
    try:
        spec = importlib.util.spec_from_file_location("judge_utils", judge_utils_path)
        judge_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judge_utils)
        create_judge = judge_utils.create_judge
    except Exception as e:
        print(f"[evaluate] ERROR: could not load judge_utils from {judge_utils_path}: {e}")
        return judges

    if "qwen3guard" in layers:
        print(f"[evaluate] Loading Qwen3Guard on {device}...")
        try:
            judges["qwen3guard"] = create_judge("qwen3guard", device=device)
        except Exception as e:
            print(f"[evaluate] ERROR loading qwen3guard: {e}")

    if "llamaguard" in layers:
        print(f"[evaluate] Loading LlamaGuard3 on {device}...")
        try:
            judges["llamaguard"] = create_judge("llamaguard3", device=device)
        except Exception as e:
            print(f"[evaluate] ERROR loading llamaguard3: {e}")

    return judges


# ---------------------------------------------------------------------------
# Response file discovery
# ---------------------------------------------------------------------------

def find_response_files(results_dir: Path) -> dict:
    """Return a dict mapping (model, method, k) -> Path for all 12 expected files.
    Warns about missing files; missing entries are excluded from the dict.
    """
    found = {}
    for model in MODELS:
        for method in METHODS:
            for k in KS:
                rel = f"{model}/{method}/{method}_k{k}_responses.json"
                fpath = results_dir / rel
                if fpath.exists():
                    found[(model, method, k)] = fpath
                else:
                    print(f"[evaluate] WARNING: missing response file: {fpath}")
    return found


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _fmt(value, key: str) -> str:
    """Format a metric value for the markdown table."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value < 0:
            return "N/A"
        return f"{value:.3f}"
    return str(value)


def build_markdown_table(all_results: dict) -> str:
    header = (
        "| Model | Method | k | ASR_kw | ASR_sr | mean_sr | ASR_q3g | ASR_lg | SRR(q3g) | SRR(lg) |\n"
        "|-------|--------|---|--------|--------|---------|---------|--------|----------|---------|\n"
    )
    rows = []
    for model in sorted(all_results.keys()):
        model_data = all_results[model]
        for method in ["dim", "rdo"]:
            for k in [1, 3, 5]:
                key = f"{method}_k{k}"
                metrics = model_data.get(key, {})
                row = (
                    f"| {model} | {method} | {k} "
                    f"| {_fmt(metrics.get('asr_keyword'), 'asr_keyword')} "
                    f"| {_fmt(metrics.get('asr_sr'), 'asr_sr')} "
                    f"| {_fmt(metrics.get('mean_sr'), 'mean_sr')} "
                    f"| {_fmt(metrics.get('asr_qwen3guard'), 'asr_qwen3guard')} "
                    f"| {_fmt(metrics.get('asr_llamaguard'), 'asr_llamaguard')} "
                    f"| {_fmt(metrics.get('srr_qwen3guard'), 'srr_qwen3guard')} "
                    f"| {_fmt(metrics.get('srr_llamaguard'), 'srr_llamaguard')} |"
                )
                rows.append(row)
    return header + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P0 Cone Ablation — Evaluation Runner")
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=["keyword", "sr", "qwen3guard", "llamaguard"],
        default=["keyword", "sr", "qwen3guard", "llamaguard"],
        help="Evaluation layers to run.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/p0_cone",
        help="Path to results directory (relative to cwd or absolute).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for guard models.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        # Resolve relative to project root (two levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent
        results_dir = project_root / args.results_dir

    print(f"[evaluate] Results dir: {results_dir}")
    print(f"[evaluate] Layers: {args.layers}")
    print(f"[evaluate] Device: {args.device}")

    # 1. Load judges
    judges = load_judges(args.layers, args.device)

    # 2. Find response files
    response_files = find_response_files(results_dir)
    if not response_files:
        print("[evaluate] ERROR: No response files found. Exiting.")
        sys.exit(1)
    print(f"[evaluate] Found {len(response_files)}/12 response files.")

    # 3. Evaluate each file
    all_results: dict = {model: {} for model in MODELS}

    for (model, method, k), fpath in sorted(response_files.items()):
        print(f"\n[evaluate] Evaluating: {fpath.name}  ({model} / {method} / k={k})")
        with open(fpath, "r") as f:
            data = json.load(f)
        responses = data.get("responses", [])
        if not responses:
            print(f"[evaluate] WARNING: empty or missing 'responses' in {fpath.name}, skipping.")
            continue

        metrics = evaluate_file(responses, layers=args.layers, judges=judges)
        key = f"{method}_k{k}"
        all_results[model][key] = metrics

        # Print per-file results
        for metric_name, val in metrics.items():
            val_str = f"{val:.4f}" if isinstance(val, float) and val >= 0 else "N/A"
            print(f"  {metric_name}: {val_str}")

    # 4. Save eval_results.json
    # Merge with existing file to support two-phase runs (e.g. --layers keyword sr first,
    # then --layers qwen3guard llamaguard second) without losing prior results.
    out_json = results_dir / "eval_results.json"
    if out_json.exists():
        with open(out_json) as f:
            existing = json.load(f)
        for model, model_data in existing.items():
            for cond_key, old_metrics in model_data.items():
                if model in all_results and cond_key in all_results[model]:
                    # Merge: new metrics take precedence over old for same keys
                    all_results[model][cond_key] = {**old_metrics, **all_results[model][cond_key]}
                elif model in all_results:
                    all_results[model][cond_key] = old_metrics

    # Replace -1.0 sentinels (unavailable layer) with null in the JSON output
    def _clean(v):
        if isinstance(v, float) and v < 0:
            return None
        return v

    clean_results = {
        model: {k: {mk: _clean(mv) for mk, mv in metrics.items()} for k, metrics in model_data.items()}
        for model, model_data in all_results.items()
    }

    with open(out_json, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n[evaluate] Saved: {out_json}")

    # 5. Save eval_summary.md
    md_content = "# P0 Cone Ablation — Evaluation Summary\n\n"
    md_content += build_markdown_table(clean_results)
    md_content += "\n**Columns**: ASR_kw=keyword ASR, ASR_sr=StrongREJECT ASR, "
    md_content += "ASR_q3g=Qwen3Guard ASR, ASR_lg=LlamaGuard ASR, "
    md_content += "SRR=Stealth Refusal Rate (ASR_kw - ASR_judge). N/A=not computed or unavailable.\n"

    out_md = results_dir / "eval_summary.md"
    with open(out_md, "w") as f:
        f.write(md_content)
    print(f"[evaluate] Saved: {out_md}")

    print("\n[evaluate] Done.")


if __name__ == "__main__":
    main()
