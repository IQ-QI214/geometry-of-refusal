"""Bridge losses for CaRoB (numpy, CPU-only).

Implements CaRoB part C/D (CaRoB代码实现交接蓝图 §五):

* ``lora_delta``: the low-rank bridge ``delta = scaling * (h @ A @ B)``.
* ``distillation_loss_c2``: pull visual gate-locus hidden states toward the
  "text-only would refuse" teacher (counterfactual teacher; clean signal).
* ``self_supervised_corr_loss_c3``: ``-corr(s(x), safety_margin)``; encourages
  the sensor score to track the gate's safety margin without explicit labels.
* ``utility_anchor_penalty``: a simple anchor that penalises non-zero delta on
  benign samples (gate close to 0), a proxy for over-refusal control.
"""

from __future__ import annotations

import numpy as np


def lora_delta(
    h: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    scaling: float,
) -> np.ndarray:
    """Low-rank bridge: ``delta = scaling * (h @ A @ B)``.

    ``A`` has shape ``(d, r)`` and ``B`` shape ``(r, d)``; the resulting delta
    has shape ``(n, d)`` and rank at most ``r``.
    """
    h = np.asarray(h, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A,B must be 2D; got {A.shape}, {B.shape}")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"rank mismatch: A.shape[1]={A.shape[1]} vs B.shape[0]={B.shape[0]}")
    if h.shape[-1] != A.shape[0]:
        raise ValueError(f"hidden dim mismatch: h.shape[-1]={h.shape[-1]} vs A.shape[0]={A.shape[0]}")
    return float(scaling) * (h @ A @ B)


def distillation_loss_c2(
    student_gate_hidden: np.ndarray,
    teacher_gate_hidden: np.ndarray,
) -> float:
    """MSE between student and teacher gate-locus hidden states."""
    s = np.asarray(student_gate_hidden, dtype=np.float64)
    t = np.asarray(teacher_gate_hidden, dtype=np.float64)
    if s.shape != t.shape:
        raise ValueError(f"shape mismatch: {s.shape} vs {t.shape}")
    return float(np.mean((s - t) ** 2))


def self_supervised_corr_loss_c3(
    risk_score: np.ndarray,
    safety_margin: np.ndarray,
) -> float:
    """``-corr(s(x), safety_margin)``; returns 0 when either input is constant.

    Returned values lie in ``[-1, +1]``: lower is better (positive correlation
    between sensor score and safety margin).
    """
    s = np.asarray(risk_score, dtype=np.float64).ravel()
    m = np.asarray(safety_margin, dtype=np.float64).ravel()
    if s.shape != m.shape:
        raise ValueError(f"shape mismatch: {s.shape} vs {m.shape}")
    s_std = s.std()
    m_std = m.std()
    if s_std < 1e-12 or m_std < 1e-12:
        return 0.0
    corr = float(np.corrcoef(s, m)[0, 1])
    return -corr


def utility_anchor_penalty(delta: np.ndarray, gate: np.ndarray) -> float:
    """Penalise non-zero delta on benign samples (gate close to 0).

    Defined as ``mean_i ((1 - gate_i) * ||delta_i||^2)``: samples that the gate
    classifies as benign should leave the hidden state alone. Risky samples
    (gate=1) contribute zero so the penalty cannot fight the safety objective.
    """
    delta = np.asarray(delta, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2D, got {delta.shape}")
    if gate.shape != (delta.shape[0],):
        raise ValueError(f"gate shape {gate.shape} mismatched with delta {delta.shape}")
    weight = 1.0 - np.clip(gate, 0.0, 1.0)
    sq_norm = (delta ** 2).sum(axis=-1)
    return float(np.mean(weight * sq_norm))
