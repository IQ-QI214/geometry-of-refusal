"""Orthogonal projection baseline for removing a subspace."""

from __future__ import annotations

import numpy as np


def remove_subspace(hidden: np.ndarray, basis: np.ndarray) -> np.ndarray:
    h = np.asarray(hidden, dtype=float)
    b = np.asarray(basis, dtype=float)
    if b.ndim == 1:
        b = b[None, :]
    q, _ = np.linalg.qr(b.T)
    projection = h @ q @ q.T
    return h - projection

