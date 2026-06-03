from __future__ import annotations

import numpy as np

from experiments.mibd.probes.direction import project_scores


def compute_score_margins(
    direction: np.ndarray,
    hidden_map: dict[tuple[int, int], dict[str, np.ndarray]],
    layer: int,
    pos: int,
) -> dict:
    """
    Project harmful/harmless hidden states onto direction and compute gap statistics.
    """
    lp_map = hidden_map.get((layer, pos))
    if lp_map is None:
        raise KeyError(f"(layer={layer}, pos={pos}) not found in hidden_map")

    harmful = lp_map.get("harmful")
    harmless = lp_map.get("harmless")
    if harmful is None or harmless is None:
        raise ValueError("hidden_map entry must have both 'harmful' and 'harmless' keys")

    harmful_scores = project_scores(harmful, direction)
    harmless_scores = project_scores(harmless, direction)

    q75_h, q25_h = np.percentile(harmful_scores, [75, 25])
    q75_l, q25_l = np.percentile(harmless_scores, [75, 25])

    return {
        "mean_gap": float(harmful_scores.mean() - harmless_scores.mean()),
        "median_gap": float(np.median(harmful_scores) - np.median(harmless_scores)),
        "iqr_harmful": float(q75_h - q25_h),
        "iqr_harmless": float(q75_l - q25_l),
        "n_harmful": int(len(harmful_scores)),
        "n_harmless": int(len(harmless_scores)),
    }


def condition_margin_table(
    condition_directions: dict[str, np.ndarray],
    all_hidden: dict[str, dict[tuple[int, int], dict[str, np.ndarray]]],
    layer: int,
    pos: int,
) -> dict[str, dict]:
    """Compute margins for each visual condition using its own direction."""
    result = {}
    for vc, direction in condition_directions.items():
        hidden_map = all_hidden.get(vc)
        if hidden_map is None:
            continue
        try:
            result[vc] = compute_score_margins(direction, hidden_map, layer, pos)
        except (KeyError, ValueError):
            continue
    return result
