"""Direction extraction and projection utilities."""

from __future__ import annotations

import numpy as np


def mean_difference_direction(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    harmful = _as_2d(harmful, "harmful")
    harmless = _as_2d(harmless, "harmless")
    direction = harmful.mean(axis=0) - harmless.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise ValueError("Mean-difference direction has near-zero norm.")
    return direction / norm


def project_scores(hidden_states: np.ndarray, direction: np.ndarray) -> np.ndarray:
    hidden_states = _as_2d(hidden_states, "hidden_states")
    direction = np.asarray(direction, dtype=float)
    if hidden_states.shape[1] != direction.shape[0]:
        raise ValueError(
            f"Hidden dim {hidden_states.shape[1]} does not match direction dim {direction.shape[0]}"
        )
    return hidden_states @ direction


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        raise ValueError("Cannot compute cosine for near-zero vector.")
    return float(np.dot(a, b) / denom)


def _as_2d(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    return arr

