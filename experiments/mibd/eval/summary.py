from __future__ import annotations
import json
from pathlib import Path

from experiments.mibd.probes.train import find_best_locus


def build_phase1_summary(
    model_id: str,
    signal_type: str,
    probe_results_by_condition: dict[str, dict[tuple[int, int], dict]],
    condition_cosines: dict[tuple[str, str], float],
    static_transfer_auc: dict[tuple[str, str], float],
) -> dict:
    """Build Phase 1 JSON summary per HANDOFF.md schema."""
    results = []
    for vc, pr in probe_results_by_condition.items():
        if not pr:
            continue
        best = find_best_locus(pr)
        layer, pos = best
        results.append({
            "visual_condition": vc,
            "layer": layer,
            "token_pos": pos,
            "auc": float(pr[best]["auc"]),
        })

    return {
        "model_id": model_id,
        "signal_type": signal_type,
        "results": results,
        "condition_cosines": {
            f"{a}|{b}": float(v)
            for (a, b), v in condition_cosines.items()
        },
        "static_transfer_auc": {
            f"{a}|{b}": float(v)
            for (a, b), v in static_transfer_auc.items()
        },
    }


def save_summary(summary: dict, output_dir: str, filename: str = "summary.json") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return path
