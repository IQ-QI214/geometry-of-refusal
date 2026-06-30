"""Data-purity guard: detect the blank-placeholder AUC saturation confound.

Last round's offline oracle found AUC=1.0 at *every* layer including layer 0,
which is the signature of the ``generated_blank_placeholder`` confound: probes
separate "blank image vs content image" (a trivial low-level feature readable at
layer 0) rather than harmful semantics.

This guard inspects a ``probe_summary.json`` (or a ``hidden_states.npz`` via the
offline oracle) and decides whether the dataset is *suspiciously saturated*. Run
it after re-extracting hidden states from the format-matched neutral-carrier
dataset to confirm the fix actually broke the confound::

    python3 -m experiments.mibd_routing_v2.eval.check_saturation \
        --probe-summary results/mibd_routing_v2/sensor_probe/<run>/probe_summary.json

Exit code is non-zero when saturation is detected, so it can gate a pipeline.
CPU-only, numpy-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def evaluate_probe_summary(
    summary: dict,
    early_layer_threshold: int = 4,
    early_auc_ceiling: float = 0.98,
) -> dict:
    """Decide whether a probe summary shows the layer-0 saturation confound.

    Heuristic: if an *early* layer (<= ``early_layer_threshold``) already has
    single-direction AUC >= ``early_auc_ceiling`` for every condition, harmful
    semantics cannot plausibly be the only separable variable that early -- it
    is the blank-vs-content confound. We report per-condition early-layer AUCs
    and an overall ``saturated`` verdict.
    """
    conditions = summary.get("conditions", {})
    if not conditions:
        raise ValueError("probe summary has no 'conditions'")

    per_condition: dict[str, dict] = {}
    saturated_flags = []
    for cond, rows in conditions.items():
        early = [r for r in rows if r.get("layer", 1_000_000) <= early_layer_threshold]
        if not early:
            continue
        early_aucs = [float(r.get("single_direction_auc", 0.0)) for r in early]
        max_early = max(early_aucs)
        is_sat = max_early >= early_auc_ceiling
        saturated_flags.append(is_sat)
        per_condition[cond] = {
            "min_early_layer": min(r["layer"] for r in early),
            "max_early_auc": round(max_early, 6),
            "saturated": is_sat,
        }

    overall = bool(saturated_flags) and all(saturated_flags)
    return {
        "early_layer_threshold": early_layer_threshold,
        "early_auc_ceiling": early_auc_ceiling,
        "per_condition": per_condition,
        "saturated": overall,
        "verdict": (
            "SATURATED: early-layer AUC at ceiling -> likely blank-vs-content "
            "confound; re-extract with format-matched neutral carriers."
            if overall
            else "OK: early layers are not saturated; harmful semantics is "
            "plausibly the separable variable."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe-summary saturation guard (CPU).")
    parser.add_argument("--probe-summary", required=True, type=Path)
    parser.add_argument("--early-layer-threshold", type=int, default=4)
    parser.add_argument("--early-auc-ceiling", type=float, default=0.98)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(Path(args.probe_summary).read_text())
    report = evaluate_probe_summary(
        summary,
        early_layer_threshold=args.early_layer_threshold,
        early_auc_ceiling=args.early_auc_ceiling,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["saturated"] else 0


if __name__ == "__main__":
    sys.exit(main())
