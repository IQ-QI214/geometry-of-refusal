"""Fixed safety-direction steering baseline."""

from __future__ import annotations

import numpy as np


def apply_fixed_direction(hidden: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    h = np.asarray(hidden, dtype=float)
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("direction must be non-zero")
    return h + float(scale) * d / norm

