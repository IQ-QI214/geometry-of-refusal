"""Oracle dynamic sensor-to-gate bridge (H3 upper-bound probe).

The bridge is the CPU-side math that later GPU hooks will call after they
extract evidence vectors from sensor loci and identify a gate locus via
forward-only interventions. It does not run a model; it only aggregates
evidence and maps it into the gate hidden-state space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


Locus = tuple[int, int]


@dataclass(frozen=True)
class OracleBridgeConfig:
    """Configuration for applying dynamic evidence to a gate hidden state.

    Attributes:
        loci: Ordered sensor loci whose evidence should be aggregated.
        weights: Optional per-locus weights. Missing loci default to 1.0.
        normalize_weights: If true, divide weighted evidence by the sum of
            absolute weights. This keeps bridge strength comparable when top-k
            locus count changes.
        bridge_matrix: Optional linear map from evidence space into gate space.
            For 1D vectors it must have shape ``(gate_dim, evidence_dim)``.
        scale: Scalar multiplier for the mapped aggregate before adding it to
            ``gate_hidden``.
    """

    loci: list[Locus]
    weights: dict[Locus, float] = field(default_factory=dict)
    normalize_weights: bool = False
    bridge_matrix: np.ndarray | None = None
    scale: float = 1.0


def _validate_same_shape(name: str, value: np.ndarray, expected: np.ndarray) -> None:
    if value.shape != expected.shape:
        raise ValueError(f"{name} must have shape {expected.shape}, got {value.shape}")


def _weighted_evidence(
    gate: np.ndarray,
    evidence_by_locus: dict[Locus, np.ndarray],
    config: OracleBridgeConfig,
) -> np.ndarray:
    if not config.loci:
        raise ValueError("OracleBridgeConfig.loci must not be empty")

    aggregate = np.zeros_like(gate, dtype=float)
    weight_norm = 0.0
    for locus in config.loci:
        if locus not in evidence_by_locus:
            raise ValueError(f"Missing evidence for locus {locus}")
        evidence = np.asarray(evidence_by_locus[locus], dtype=float)
        _validate_same_shape("evidence", evidence, gate)
        weight = float(config.weights.get(locus, 1.0))
        aggregate += weight * evidence
        weight_norm += abs(weight)

    if config.normalize_weights and weight_norm > 0.0:
        aggregate = aggregate / weight_norm
    return aggregate


def _apply_bridge_matrix(
    aggregate: np.ndarray,
    gate_shape: tuple[int, ...],
    bridge_matrix: np.ndarray | None,
) -> np.ndarray:
    if bridge_matrix is None:
        return aggregate

    matrix = np.asarray(bridge_matrix, dtype=float)
    if aggregate.ndim == 1:
        expected = (gate_shape[0], aggregate.shape[0])
        if matrix.shape != expected:
            raise ValueError(f"bridge_matrix must have shape {expected}, got {matrix.shape}")
        return matrix @ aggregate

    if aggregate.ndim == 2:
        expected = (gate_shape[-1], aggregate.shape[-1])
        if matrix.shape != expected:
            raise ValueError(f"bridge_matrix must have shape {expected}, got {matrix.shape}")
        return aggregate @ matrix.T

    raise ValueError("gate_hidden/evidence must be 1D or 2D arrays")


def apply_oracle_bridge(
    gate_hidden: np.ndarray,
    evidence_by_locus: dict[Locus, np.ndarray],
    config: OracleBridgeConfig,
) -> np.ndarray:
    """Return ``gate_hidden + scale * mapped(weighted_sensor_evidence)``."""
    gate = np.asarray(gate_hidden, dtype=float)
    if gate.ndim not in (1, 2):
        raise ValueError("gate_hidden must be a 1D or 2D array")

    aggregate = _weighted_evidence(gate, evidence_by_locus, config)
    mapped = _apply_bridge_matrix(aggregate, gate.shape, config.bridge_matrix)
    _validate_same_shape("mapped evidence", mapped, gate)
    return gate + float(config.scale) * mapped


def summarize_bridge_effect(
    baseline_safe_policy_rate: float,
    bridged_safe_policy_rate: float,
    baseline_over_refusal_rate: float,
    bridged_over_refusal_rate: float,
    baseline_degeneration_rate: float,
    bridged_degeneration_rate: float,
) -> dict[str, float]:
    """Compute Go/No-Go report inputs in percentage points."""
    return {
        "safe_policy_gain_pp": round(
            100.0 * (bridged_safe_policy_rate - baseline_safe_policy_rate), 10
        ),
        "over_refusal_delta_pp": round(
            100.0 * (bridged_over_refusal_rate - baseline_over_refusal_rate), 10
        ),
        "degeneration_delta_pp": round(
            100.0 * (bridged_degeneration_rate - baseline_degeneration_rate), 10
        ),
    }
