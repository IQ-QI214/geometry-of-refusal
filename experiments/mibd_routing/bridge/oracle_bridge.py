"""Oracle dynamic sensor-to-gate bridge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class OracleBridgeConfig:
    loci: list[tuple[int, int]]
    weights: dict[tuple[int, int], float] = field(default_factory=dict)
    bridge_matrix: np.ndarray | None = None
    scale: float = 1.0


def apply_oracle_bridge(
    gate_hidden: np.ndarray,
    evidence_by_locus: dict[tuple[int, int], np.ndarray],
    config: OracleBridgeConfig,
) -> np.ndarray:
    gate = np.asarray(gate_hidden, dtype=float)
    if not config.loci:
        raise ValueError("OracleBridgeConfig.loci must not be empty")
    aggregate = np.zeros_like(gate)
    for locus in config.loci:
        if locus not in evidence_by_locus:
            raise ValueError(f"Missing evidence for locus {locus}")
        evidence = np.asarray(evidence_by_locus[locus], dtype=float)
        if evidence.shape != gate.shape:
            raise ValueError("evidence and gate_hidden must have the same shape")
        aggregate += float(config.weights.get(locus, 1.0)) * evidence
    bridge_matrix = config.bridge_matrix
    if bridge_matrix is not None:
        aggregate = aggregate @ np.asarray(bridge_matrix, dtype=float).T
    return gate + float(config.scale) * aggregate

