from __future__ import annotations

import numpy as np

from experiments.mibd.probes.direction import (
    mean_difference_direction,
    project_scores,
    cosine_similarity,
)
from experiments.mibd.probes.metrics import binary_auc


def train_probes_for_condition(
    hidden_map: dict[tuple[int, int], dict[str, np.ndarray]],
) -> dict[tuple[int, int], dict]:
    """Train mean-difference probes for all (layer, pos) in one visual condition."""
    results = {}
    for (layer, pos), label_map in hidden_map.items():
        harmful = label_map.get("harmful")
        harmless = label_map.get("harmless")
        if harmful is None or harmless is None:
            continue
        if len(harmful) < 2 or len(harmless) < 2:
            continue
        direction = mean_difference_direction(harmful, harmless)
        all_hidden = np.vstack([harmful, harmless])
        labels = np.array([1] * len(harmful) + [0] * len(harmless))
        scores = project_scores(all_hidden, direction)
        auc = binary_auc(labels, scores)
        results[(layer, pos)] = {"direction": direction, "auc": auc}
    return results


def find_best_locus(
    probe_results: dict[tuple[int, int], dict],
) -> tuple[int, int]:
    """Return (layer, pos) with highest AUC."""
    return max(probe_results, key=lambda k: probe_results[k]["auc"])


def compute_condition_cosines(
    condition_directions: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Cosine similarity between all pairs of condition directions."""
    conditions = sorted(condition_directions.keys())
    cosines = {}
    for i, ca in enumerate(conditions):
        for cb in conditions[i + 1:]:
            cosines[(ca, cb)] = cosine_similarity(
                condition_directions[ca], condition_directions[cb]
            )
    return cosines


def compute_static_transfer_aucs(
    source_direction: np.ndarray,
    source_condition: str,
    target_hidden_maps: dict[str, dict[tuple[int, int], dict[str, np.ndarray]]],
    layer: int,
    pos: int,
) -> dict[tuple[str, str], float]:
    """Apply source_condition direction to target conditions at (layer, pos)."""
    aucs = {}
    for target_cond, hidden_map in target_hidden_maps.items():
        if target_cond == source_condition:
            continue
        lp_map = hidden_map.get((layer, pos))
        if lp_map is None:
            continue
        harmful = lp_map.get("harmful")
        harmless = lp_map.get("harmless")
        if harmful is None or harmless is None:
            continue
        all_hidden = np.vstack([harmful, harmless])
        labels = np.array([1] * len(harmful) + [0] * len(harmless))
        scores = project_scores(all_hidden, source_direction)
        aucs[(source_condition, target_cond)] = binary_auc(labels, scores)
    return aucs
