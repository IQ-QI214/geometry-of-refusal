"""Small dependency-light probe metrics."""

from __future__ import annotations

import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length")
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("AUC requires at least one positive and one negative sample")

    wins = 0.0
    total = float(len(positives) * len(negatives))
    for pos_score in positives:
        wins += float(np.sum(pos_score > negatives))
        wins += 0.5 * float(np.sum(pos_score == negatives))
    return wins / total

