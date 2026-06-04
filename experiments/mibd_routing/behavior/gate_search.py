"""Behavioral gate scoring utilities."""

from __future__ import annotations

import numpy as np


def safe_policy_margin(safe_logprobs: np.ndarray, unsafe_logprobs: np.ndarray) -> np.ndarray:
    safe = np.asarray(safe_logprobs, dtype=float)
    unsafe = np.asarray(unsafe_logprobs, dtype=float)
    if safe.shape != unsafe.shape:
        raise ValueError("safe_logprobs and unsafe_logprobs must have the same shape")
    return safe - unsafe


def gate_effect(
    baseline_safe_logprobs: np.ndarray,
    baseline_unsafe_logprobs: np.ndarray,
    intervened_safe_logprobs: np.ndarray,
    intervened_unsafe_logprobs: np.ndarray,
) -> float:
    baseline = safe_policy_margin(baseline_safe_logprobs, baseline_unsafe_logprobs)
    intervened = safe_policy_margin(intervened_safe_logprobs, intervened_unsafe_logprobs)
    return float(np.mean(intervened - baseline))

