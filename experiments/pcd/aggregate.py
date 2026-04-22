"""Aggregate per-condition PCD results into the 8x6 matrix and summary markdown.

Usage:
    python experiments/pcd/aggregate.py [--root results/pcd] [--out_json ...] [--out_md ...]
"""
import argparse
import json
import torch
from pathlib import Path


def cos(a, b):
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum())


def _safe_float(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and v < 0):
        return "-"
    try:
        return format(float(v), fmt)
    except Exception:
        return "-"


def main():
    p = argparse.ArgumentParser(description="Aggregate PCD results into 8x6 matrix")
    p.add_argument("--root",     default="results/pcd")
    p.add_argument("--out_json", default="results/pcd/pcd_8x6_matrix.json")
    p.add_argument("--out_md",   default="results/pcd/pcd_summary.md")
    args = p.parse_args()

    root = Path(args.root)
    matrix = {}

    # Load Qwen L (text-only) reference direction from repro if available
    qwen_L_dir = None
    gemma_L_dir = None
    for candidate in [
        "results/repro_arditi_wollschlager/Qwen2.5-7B-Instruct/direction.pt",
        "results/repro_arditi_wollschlager/qwen2.5-7b/direction.pt",
    ]:
        if Path(candidate).exists():
            qwen_L_dir = torch.load(candidate, map_location="cpu").float()
            break

    def add_row(family, condition, dir_path, sweep_json, eval_json, rdo_eval_json=None):
        row = {"family": family, "condition": condition}

        if sweep_json is not None and Path(sweep_json).exists():
            sj = json.loads(Path(sweep_json).read_text())
            row.update({
                "best_layer":    sj.get("layer"),
                "best_pos":      sj.get("pos"),
                "filter_passed": sj.get("filter_passed"),
            })

        if eval_json is not None and Path(eval_json).exists():
            ev = json.loads(Path(eval_json).read_text())
            for k in ("asr_keyword", "asr_sr", "asr_q3g", "asr_lg3",
                      "arditi_joint_asr", "arditi_refusal_rate"):
                if k in ev:
                    row[k] = ev[k]

        if rdo_eval_json is not None and Path(rdo_eval_json).exists():
            ev = json.loads(Path(rdo_eval_json).read_text())
            row["rdo_asr_kw"]  = ev.get("asr_keyword")
            row["rdo_asr_lg3"] = ev.get("asr_lg3")
            row["rdo_arditi"]  = ev.get("arditi_joint_asr")

        if dir_path is not None and Path(dir_path).exists():
            d = torch.load(dir_path, map_location="cpu").float()
            ref = qwen_L_dir if family == "qwen" else gemma_L_dir
            if ref is not None:
                row["cos_vs_L"] = cos(d, ref)

        matrix[f"{family}/{condition}"] = row

    # Qwen family (3 VLM conditions)
    for cond, subdir in [
        ("V-text",  "V-text"),
        ("V-blank", "V-blank-resweep"),
        ("V-noise", "V-noise"),
    ]:
        d = root / "qwen_family" / subdir
        add_row(
            "qwen", cond,
            dir_path      = str(d / "dim_k1.pt"),
            sweep_json    = str(d / "best_layer.json"),
            eval_json     = str(d / "dim_responses_eval.json"),
            rdo_eval_json = str(d / "rdo_k3" / "rdo_k3_responses_eval.json"),
        )

    # Gemma family (3 VLM conditions; alpha=PASS means V-text == L)
    for cond in ["V-text", "V-blank", "V-noise"]:
        d = root / "gemma_family" / cond
        add_row(
            "gemma", cond,
            dir_path      = str(d / "dim_k1.pt"),
            sweep_json    = str(d / "best_layer.json"),
            eval_json     = str(d / "dim_responses_eval.json"),
            rdo_eval_json = str(d / "rdo_k3" / "rdo_k3_responses_eval.json"),
        )

    # Gemma L (only present if alpha=FAIL and we ran L separately)
    gemma_L_result_dir = root / "gemma_family" / "L"
    if gemma_L_result_dir.exists():
        lp = gemma_L_result_dir / "dim_k1.pt"
        if lp.exists():
            gemma_L_dir = torch.load(str(lp), map_location="cpu").float()
        add_row(
            "gemma", "L",
            dir_path   = str(lp),
            sweep_json = str(gemma_L_result_dir / "best_layer.json"),
            eval_json  = str(gemma_L_result_dir / "dim_responses_eval.json"),
        )

    # Write JSON
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(matrix, indent=2))
    print(f"Wrote {args.out_json}  ({len(matrix)} rows)")

    # Write markdown summary table
    cols = [
        ("Family",    lambda r: r["family"]),
        ("Condition", lambda r: r["condition"]),
        ("Layer",     lambda r: str(r.get("best_layer", "-"))),
        ("Pos",       lambda r: str(r.get("best_pos", "-"))),
        ("ASR_kw",    lambda r: _safe_float(r.get("asr_keyword"))),
        ("ASR_SR",    lambda r: _safe_float(r.get("asr_sr"))),
        ("ASR_q3g",   lambda r: _safe_float(r.get("asr_q3g"))),
        ("ASR_LG3",   lambda r: _safe_float(r.get("asr_lg3"))),
        ("Arditi",    lambda r: _safe_float(r.get("arditi_joint_asr"))),
        ("cos_vs_L",  lambda r: _safe_float(r.get("cos_vs_L"))),
    ]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep    = "|" + "|".join(["---" if i < 2 else ":---:" for i in range(len(cols))]) + "|"
    rows_md = ["| " + " | ".join(fn(row) for _, fn in cols) + " |"
               for row in matrix.values()]

    md_lines = ["# PCD 8x6 Summary\n", header, sep] + rows_md
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
