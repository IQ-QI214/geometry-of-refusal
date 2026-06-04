from __future__ import annotations

import numpy as np

from experiments.mibd.probes.direction import project_scores


def compute_score_margins(
    direction: np.ndarray,
    hidden_map: dict[tuple[int, int], dict[str, np.ndarray]],
    layer: int,
    pos: int,
    pos_label: str = "harmful",
    neg_label: str = "harmless",
) -> dict:
    """Project pos/neg hidden states onto direction and compute gap statistics."""
    lp_map = hidden_map.get((layer, pos))
    if lp_map is None:
        raise KeyError(f"(layer={layer}, pos={pos}) not found in hidden_map")

    pos_arr = lp_map.get(pos_label)
    neg_arr = lp_map.get(neg_label)
    if pos_arr is None or neg_arr is None:
        raise ValueError(f"hidden_map entry must have both '{pos_label}' and '{neg_label}' keys")

    pos_scores = project_scores(pos_arr, direction)
    neg_scores = project_scores(neg_arr, direction)

    q75_h, q25_h = np.percentile(pos_scores, [75, 25])
    q75_l, q25_l = np.percentile(neg_scores, [75, 25])

    return {
        "mean_gap": float(pos_scores.mean() - neg_scores.mean()),
        "median_gap": float(np.median(pos_scores) - np.median(neg_scores)),
        "iqr_harmful": float(q75_h - q25_h),
        "iqr_harmless": float(q75_l - q25_l),
        "n_harmful": int(len(pos_scores)),
        "n_harmless": int(len(neg_scores)),
    }


def condition_margin_table(
    condition_directions: dict[str, np.ndarray],
    all_hidden: dict[str, dict[tuple[int, int], dict[str, np.ndarray]]],
    layer: int,
    pos: int,
    pos_label: str = "harmful",
    neg_label: str = "harmless",
) -> dict[str, dict]:
    """Compute margins for each visual condition using its own direction."""
    result = {}
    for vc, direction in condition_directions.items():
        hidden_map = all_hidden.get(vc)
        if hidden_map is None:
            continue
        try:
            result[vc] = compute_score_margins(
                direction, hidden_map, layer, pos, pos_label, neg_label
            )
        except (KeyError, ValueError):
            continue
    return result
