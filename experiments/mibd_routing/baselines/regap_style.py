"""ReGap-style global modality drift correction."""

from __future__ import annotations

import numpy as np


def compute_regap_correction(text_hidden: np.ndarray, multimodal_hidden: np.ndarray) -> np.ndarray:
    text = np.asarray(text_hidden, dtype=float)
    multimodal = np.asarray(multimodal_hidden, dtype=float)
    if text.shape != multimodal.shape:
        raise ValueError("text_hidden and multimodal_hidden must have the same shape")
    return np.mean(text - multimodal, axis=0)


def apply_regap_correction(multimodal_hidden: np.ndarray, correction: np.ndarray) -> np.ndarray:
    return np.asarray(multimodal_hidden, dtype=float) + np.asarray(correction, dtype=float)

