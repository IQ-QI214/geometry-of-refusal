import numpy as np
import pytest

from experiments.mibd.probes.train import (
    train_probes_for_condition,
    compute_condition_cosines,
    compute_static_transfer_aucs,
)


def _make_hidden_map(n: int = 20, dim: int = 32):
    rng = np.random.default_rng(0)
    result = {}
    for layer in (0, 1):
        for pos in (-1, -5):
            harmful = rng.standard_normal((n, dim)) + 2.0
            harmless = rng.standard_normal((n, dim))
            result[(layer, pos)] = {"harmful": harmful, "harmless": harmless}
    return result


def test_train_probes_returns_auc_above_chance():
    hidden_map = _make_hidden_map()
    probe_results = train_probes_for_condition(hidden_map)
    for (layer, pos), info in probe_results.items():
        assert "auc" in info
        assert "direction" in info
        assert 0.0 <= info["auc"] <= 1.0
        assert info["auc"] > 0.5


def test_compute_condition_cosines_symmetric_keys():
    cond_directions = {
        "V-text": np.array([1.0, 0.0, 0.0]),
        "V-blank": np.array([0.9, 0.1, 0.0]),
        "V-noise": np.array([0.85, 0.15, 0.0]),
    }
    cosines = compute_condition_cosines(cond_directions)
    assert ("V-text", "V-blank") in cosines or ("V-blank", "V-text") in cosines
    for val in cosines.values():
        assert -1.0 <= val <= 1.0


def test_compute_static_transfer_aucs():
    rng = np.random.default_rng(1)
    dim = 16
    direction = np.ones(dim) / np.sqrt(dim)
    target_hidden = {
        "V-blank": {
            (0, -1): {
                "harmful": rng.standard_normal((10, dim)) + 2.0,
                "harmless": rng.standard_normal((10, dim)),
            }
        }
    }
    aucs = compute_static_transfer_aucs(
        source_direction=direction,
        source_condition="V-text",
        target_hidden_maps=target_hidden,
        layer=0,
        pos=-1,
    )
    assert ("V-text", "V-blank") in aucs
    assert aucs[("V-text", "V-blank")] > 0.5
