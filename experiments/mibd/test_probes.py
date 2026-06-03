import numpy as np

from experiments.mibd.probes.direction import (
    cosine_similarity,
    mean_difference_direction,
    project_scores,
)
from experiments.mibd.probes.metrics import binary_auc


def test_mean_difference_direction_points_from_harmless_to_harmful():
    harmful = np.array([[2.0, 0.0], [4.0, 0.0]])
    harmless = np.array([[0.0, 0.0], [0.0, 2.0]])

    direction = mean_difference_direction(harmful, harmless)
    scores = project_scores(np.vstack([harmful, harmless]), direction)

    assert direction.shape == (2,)
    assert scores[:2].mean() > scores[2:].mean()
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_binary_auc_handles_separable_scores():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    assert binary_auc(labels, scores) == 1.0


def test_cosine_similarity_normalizes_inputs():
    assert cosine_similarity(np.array([2.0, 0.0]), np.array([3.0, 0.0])) == 1.0

