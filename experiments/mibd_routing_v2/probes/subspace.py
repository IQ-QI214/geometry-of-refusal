"""Multi-direction subspace readout (sensor upgrade).

Single-direction refusal probes can under-fit when risk evidence is spread
across several directions. Following *Refusal Beyond a Single Direction*
(arXiv:2606.13720), we extract a low-rank *risk subspace* and measure how much
the extra directions improve linear readability over the best single
direction.

Method (dependency-light, CPU-only):
* Iteratively extract difference-of-means directions; after each one, project
  the residual hidden states orthogonal to the directions found so far
  (a simple INLP-style deflation). This yields ``rank`` orthonormal directions.
* Score a sample as the signed sum of projections onto the extracted
  risk-oriented directions. This preserves the harmful-vs-harmless orientation;
  using only projection norms can destroy rank-1 separability.
* Compare subspace AUC against the best single-direction AUC.

numpy-only. No model, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.mibd.probes.metrics import binary_auc


@dataclass(frozen=True)
class SubspaceReadoutReport:
    rank: int
    single_direction_auc: float
    subspace_auc: float
    subspace_gain: float
    directions: np.ndarray  # shape (rank, hidden_dim), orthonormal rows

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "single_direction_auc": self.single_direction_auc,
            "subspace_auc": self.subspace_auc,
            "subspace_gain": self.subspace_gain,
        }


def _mean_diff(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    direction = harmful.mean(axis=0) - harmless.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return np.zeros_like(direction)
    return direction / norm


def extract_risk_subspace(
    risk_hidden: np.ndarray,
    safe_hidden: np.ndarray,
    rank: int = 3,
) -> np.ndarray:
    """Extract ``rank`` orthonormal risk directions via deflated diff-of-means.

    Returns an array of shape (k, hidden_dim) with k <= rank orthonormal rows
    (k can be smaller if the data is exhausted / degenerate).
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    risk = np.asarray(risk_hidden, dtype=float)
    safe = np.asarray(safe_hidden, dtype=float)
    if risk.ndim != 2 or safe.ndim != 2:
        raise ValueError("risk_hidden and safe_hidden must be 2D arrays")
    if risk.shape[1] != safe.shape[1]:
        raise ValueError("risk_hidden and safe_hidden must share hidden dim")

    risk_res = risk.copy()
    safe_res = safe.copy()
    directions: list[np.ndarray] = []
    for _ in range(rank):
        direction = _mean_diff(risk_res, safe_res)
        if np.linalg.norm(direction) <= 1e-12:
            break
        # Orthogonalize against directions found so far (numerical safety).
        for prev in directions:
            direction = direction - np.dot(direction, prev) * prev
        norm = np.linalg.norm(direction)
        if norm <= 1e-9:
            break
        direction = direction / norm
        directions.append(direction)
        # Deflate: remove this direction's component from the residuals.
        risk_res = risk_res - np.outer(risk_res @ direction, direction)
        safe_res = safe_res - np.outer(safe_res @ direction, direction)
    if not directions:
        raise ValueError("Failed to extract any risk direction (degenerate data)")
    return np.vstack(directions)


def _subspace_scores(hidden: np.ndarray, directions: np.ndarray) -> np.ndarray:
    projections = np.asarray(hidden, dtype=float) @ directions.T  # (n, k)
    return projections.sum(axis=1)


def evaluate_subspace_readout(
    labels: np.ndarray,
    risk_hidden: np.ndarray,
    safe_hidden: np.ndarray,
    pooled_hidden: np.ndarray,
    rank: int = 3,
) -> SubspaceReadoutReport:
    """Compare best-single-direction AUC vs multi-direction subspace AUC.

    Args:
        labels: binary labels (1 = risk) aligned with ``pooled_hidden``.
        risk_hidden / safe_hidden: class-conditional hidden states used to
            *extract* the subspace (typically the training split).
        pooled_hidden: hidden states to *score* (typically held-out), aligned
            with ``labels``.
        rank: target subspace rank.
    """
    labels = np.asarray(labels)
    directions = extract_risk_subspace(risk_hidden, safe_hidden, rank=rank)

    single_scores = np.asarray(pooled_hidden, dtype=float) @ directions[0]
    single_auc = binary_auc(labels, single_scores)

    subspace_scores = _subspace_scores(pooled_hidden, directions)
    subspace_auc = binary_auc(labels, subspace_scores)

    return SubspaceReadoutReport(
        rank=directions.shape[0],
        single_direction_auc=single_auc,
        subspace_auc=subspace_auc,
        subspace_gain=subspace_auc - single_auc,
        directions=directions,
    )
