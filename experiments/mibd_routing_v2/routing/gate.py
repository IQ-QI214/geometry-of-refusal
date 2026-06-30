"""Threshold gate operators (numpy, CPU-only).

Implements CaRoB part B (CaRoB代码实现交接蓝图 §四): the threshold-driven gate
that turns the routed risk score ``s(x)`` into an intervention strength
``g(x)``, plus a null-space projection (AlphaSteer 2506.07022 spirit) that
guarantees benign inputs see zero perturbation along the protected benign
basis.
"""

from __future__ import annotations

import numpy as np


def soft_gate(risk_score: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    """A2 soft gate: g(x) = sigmoid(alpha * (s - tau))."""
    s = np.asarray(risk_score, dtype=np.float64)
    z = alpha * (s - tau)
    # Numerically stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def hard_gate(risk_score: np.ndarray, tau: float) -> np.ndarray:
    """A1 hard gate: 1 if s > tau else 0 (strict inequality at the boundary)."""
    s = np.asarray(risk_score, dtype=np.float64)
    return (s > tau).astype(np.float64)


def nullspace_projection(delta: np.ndarray, benign_basis: np.ndarray) -> np.ndarray:
    """Project ``delta`` onto the null space of the benign basis.

    Args:
        delta: ``(n, d)`` intervention vectors.
        benign_basis: ``(k, d)`` benign directions assumed to be orthonormal
            rows (caller-provided; QR-orthonormalised upstream).

    Returns:
        ``delta`` with its components along each benign direction removed,
        i.e. ``proj = delta - delta @ B^T @ B`` (where ``B`` is the benign basis).
    """
    delta = np.asarray(delta, dtype=np.float64)
    B = np.asarray(benign_basis, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError(f"benign_basis must be 2D (k, d), got {B.shape}")
    if delta.shape[-1] != B.shape[-1]:
        raise ValueError(
            f"delta dim {delta.shape[-1]} != benign_basis dim {B.shape[-1]}"
        )
    return delta - delta @ B.T @ B


def apply_gated_delta(
    hidden: np.ndarray,
    delta: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    """h' = h + gate[:, None] * delta."""
    hidden = np.asarray(hidden, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    if hidden.shape != delta.shape:
        raise ValueError(f"hidden/delta shape mismatch: {hidden.shape} vs {delta.shape}")
    if gate.shape != (hidden.shape[0],):
        raise ValueError(f"gate must be ({hidden.shape[0]},), got {gate.shape}")
    return hidden + gate[:, None] * delta
