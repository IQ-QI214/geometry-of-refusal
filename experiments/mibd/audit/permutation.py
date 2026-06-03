from __future__ import annotations

import numpy as np

from experiments.mibd.probes.direction import mean_difference_direction, project_scores
from experiments.mibd.probes.metrics import binary_auc


def permutation_auc(
    hidden_map: dict[tuple[int, int], dict[str, np.ndarray]],
    layer: int,
    pos: int,
    n_permutations: int = 100,
    seed: int = 42,
    test_frac: float = 0.2,
) -> float:
    """
    Correct permutation test with train/test separation:
    1. Fix a held-out test split (real labels kept intact).
    2. Each permutation: shuffle ONLY train labels → train probe → evaluate on test with real labels.
    Expected ~0.5 when there is no real signal.

    Previous bug: evaluated against same shuffled labels used for training (no separation),
    which naturally yields high AUC regardless of signal.
    """
    lp_map = hidden_map.get((layer, pos))
    if lp_map is None:
        raise KeyError(f"(layer={layer}, pos={pos}) not found in hidden_map")

    harmful = lp_map.get("harmful")
    harmless = lp_map.get("harmless")
    if harmful is None or harmless is None:
        raise ValueError("hidden_map entry must have both 'harmful' and 'harmless' keys")

    all_hidden = np.vstack([harmful, harmless])
    labels = np.array([1] * len(harmful) + [0] * len(harmless))

    rng = np.random.default_rng(seed)

    # Fixed train/test split (same across all permutations)
    idx = rng.permutation(len(labels))
    n_test = max(1, round(len(labels) * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    test_hidden = all_hidden[test_idx]
    test_labels = labels[test_idx]   # real labels — never shuffled
    train_hidden = all_hidden[train_idx]
    train_labels = labels[train_idx]

    if len(np.unique(test_labels)) < 2:
        # Test set lacks one class; can't compute AUC
        return float("nan")

    aucs = []
    for _ in range(n_permutations):
        shuffled_train = rng.permutation(train_labels)  # shuffle only train labels
        perm_pos = train_hidden[shuffled_train == 1]
        perm_neg = train_hidden[shuffled_train == 0]
        if len(perm_pos) < 2 or len(perm_neg) < 2:
            continue
        direction = mean_difference_direction(perm_pos, perm_neg)
        scores = project_scores(test_hidden, direction)
        aucs.append(binary_auc(test_labels, scores))  # evaluate against REAL test labels

    if not aucs:
        return float("nan")
    return float(np.mean(aucs))
