"""Per-layer linear risk probes and cross-carrier transfer (numpy, CPU-only).

Turns extracted hidden states into a *bank* of single-direction probes (one per
transformer layer) that the router consumes as ``per_layer_probe_scores`` for
``aggregate_risk_score``. The interesting science here is the *cross-carrier*
behaviour: a probe fit on one visual carrier (e.g. FigStep) is evaluated on
another (e.g. V-real). If risk evidence is re-encoded across layers per carrier
(the CaRoB premise, arXiv:2603.11114 / 2606.24952), a fixed single-layer probe
should transfer *worse* across carriers than within a carrier -- the empirical
signal that motivates carrier-conditioned routing.

Reuses the well-tested ``mean_difference_direction`` / ``project_scores`` /
``binary_auc`` primitives so results stay consistent with the rest of the repo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.mibd.probes.direction import mean_difference_direction, project_scores
from experiments.mibd.probes.metrics import binary_auc
from experiments.mibd_routing_v2.eval.load_hidden_states import (
    available_layers,
    build_carrier_feature_matrix,
)


@dataclass(frozen=True)
class LayerProbe:
    """A single-direction linear risk probe for one layer."""

    layer: int
    direction: np.ndarray  # (d,) unit vector, harmful - harmless
    bias: float  # midpoint of class-mean projections (decision offset)


def fit_layer_probe(harmful: np.ndarray, harmless: np.ndarray, layer: int) -> LayerProbe:
    """Fit a diff-of-means probe; ``bias`` is the midpoint threshold."""
    direction = mean_difference_direction(harmful, harmless)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pos_mean = float(project_scores(harmful, direction).mean())
        neg_mean = float(project_scores(harmless, direction).mean())
    bias = 0.5 * (pos_mean + neg_mean)
    return LayerProbe(layer=layer, direction=direction, bias=bias)


def score_samples(probe: LayerProbe, hidden: np.ndarray) -> np.ndarray:
    """Signed risk score: projection onto the probe direction minus bias.

    Positive => harmful side. Preserves orientation, unlike projection norms.

    The matmul is wrapped in ``np.errstate`` because macOS's Accelerate BLAS
    backend emits spurious divide/overflow warnings for large finite matrices;
    results match a warning-free einsum to ~1e-14.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return project_scores(hidden, probe.direction) - probe.bias


def fit_probe_bank(
    states: dict[str, np.ndarray],
    carrier: str,
    position: int = -1,
) -> dict[int, LayerProbe]:
    """Fit one probe per available layer for ``carrier``."""
    bank: dict[int, LayerProbe] = {}
    for layer in available_layers(states, carrier):
        feats, labels = build_carrier_feature_matrix(states, carrier, layer, position)
        harmful = feats[labels == 1]
        harmless = feats[labels == 0]
        bank[layer] = fit_layer_probe(harmful, harmless, layer)
    return bank


def layer_auc(
    probe: LayerProbe,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """AUC of a probe's risk scores against binary labels."""
    scores = score_samples(probe, features)
    return binary_auc(labels, scores)


def cross_carrier_auc(
    states: dict[str, np.ndarray],
    train_carrier: str,
    test_carrier: str,
    position: int = -1,
) -> dict[int, float]:
    """Per-layer AUC of probes fit on ``train_carrier`` evaluated on ``test_carrier``.

    Only layers present in *both* carriers are scored. When
    ``train_carrier == test_carrier`` this is the within-carrier (train==test)
    upper bound, which should reproduce the saturated probe summary.
    """
    train_bank = fit_probe_bank(states, train_carrier, position)
    shared = sorted(
        set(train_bank) & set(available_layers(states, test_carrier))
    )
    if not shared:
        raise ValueError(
            f"no shared layers between {train_carrier!r} and {test_carrier!r}"
        )
    result: dict[int, float] = {}
    for layer in shared:
        feats, labels = build_carrier_feature_matrix(
            states, test_carrier, layer, position
        )
        result[layer] = layer_auc(train_bank[layer], feats, labels)
    return result


def best_layer(per_layer_auc: dict[int, float]) -> int:
    """Layer with the highest AUC (ties broken by smallest layer index)."""
    if not per_layer_auc:
        raise ValueError("per_layer_auc is empty")
    return min(per_layer_auc, key=lambda layer: (-per_layer_auc[layer], layer))
