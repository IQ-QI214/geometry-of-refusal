"""Routing gating primitives (numpy, CPU-only).

Implements the four anti-collapse / sampling primitives that the CaRoB router
depends on (CaRoB代码实现交接蓝图 §二):

* ``softmax`` with temperature.
* ``switch_load_balancing_loss`` from Switch Transformer (arXiv:2101.03961).
* ``router_z_loss`` from ST-MoE (arXiv:2202.08906).
* ``LossFreeBiasState`` / ``update_loss_free_bias``: Loss-Free dynamic bias
  balancing (arXiv:2408.15664), preferred over auxiliary balancing losses.
* ``gumbel_softmax_sample`` with optional straight-through hard mode
  (arXiv:1611.01144).
* ``anneal_temperature``: linear schedule used for training-soft / inference-hard.

All tensors are ``numpy.ndarray``; no torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional temperature."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def switch_load_balancing_loss(gate_probs: np.ndarray, expert_mask: np.ndarray) -> float:
    """Switch Transformer load balancing loss (arXiv:2101.03961).

    For ``L`` experts and ``n`` samples, the loss is

        loss = L * sum_i f_i * P_i

    where ``f_i`` is the fraction of samples routed to expert ``i`` (from
    ``expert_mask``) and ``P_i`` is the mean gate probability for expert ``i``.

    The original paper multiplies by an auxiliary coefficient alpha; we leave
    that to the caller and return the raw ``L * f.dot(P)`` value so the minimum
    for a perfectly balanced configuration is ``1.0``.
    """
    probs = np.asarray(gate_probs, dtype=np.float64)
    mask = np.asarray(expert_mask, dtype=np.float64)
    if probs.shape != mask.shape:
        raise ValueError(f"shape mismatch: {probs.shape} vs {mask.shape}")
    n, L = probs.shape
    f = mask.sum(axis=0) / max(n, 1)
    P = probs.mean(axis=0)
    return float(L * np.dot(f, P))


def router_z_loss(logits: np.ndarray) -> float:
    """ST-MoE router z-loss (arXiv:2202.08906).

    Penalises ``logsumexp(logits)`` growing above the uniform-logits baseline
    ``log(K)``, which keeps the softmax temperature-stable. Returns the mean
    over the batch. With all-zero logits the loss is exactly 0, so the
    minimum corresponds to the "uniform routing magnitude" state.
    """
    x = np.asarray(logits, dtype=np.float64)
    K = x.shape[-1]
    m = np.max(x, axis=-1, keepdims=True)
    lse = (m + np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))).squeeze(-1)
    return float(np.mean((lse - np.log(K)) ** 2))


@dataclass(frozen=True)
class LossFreeBiasState:
    """Per-expert bias added to router logits (arXiv:2408.15664)."""

    bias: np.ndarray  # shape (L,)


def update_loss_free_bias(
    state: LossFreeBiasState,
    expert_load: np.ndarray,
    target_load: np.ndarray,
    lr: float = 1e-3,
) -> LossFreeBiasState:
    """One step of Loss-Free dynamic bias balancing.

    Overloaded experts (load > target) get their bias decreased; underloaded
    experts get their bias increased. The update has no interaction with the
    main loss gradient, so it does not fight the safety objective.
    """
    load = np.asarray(expert_load, dtype=np.float64)
    target = np.asarray(target_load, dtype=np.float64)
    if load.shape != target.shape or load.shape != state.bias.shape:
        raise ValueError("expert_load, target_load, and bias must share shape")
    delta = lr * (target - load)
    return LossFreeBiasState(bias=state.bias + delta)


def gumbel_softmax_sample(
    logits: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
    hard: bool = False,
) -> np.ndarray:
    """Gumbel-Softmax sampling (arXiv:1611.01144).

    Args:
        logits: ``(..., K)`` real-valued logits.
        temperature: positive scalar; lower -> sharper.
        rng: ``numpy.random.Generator`` for reproducibility.
        hard: if True, return a one-hot row (straight-through style).
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    logits = np.asarray(logits, dtype=np.float64)
    u = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=logits.shape)
    gumbel_noise = -np.log(-np.log(u))
    y = softmax((logits + gumbel_noise) / temperature, axis=-1, temperature=1.0)
    if not hard:
        return y
    idx = np.argmax(y, axis=-1)
    one_hot = np.zeros_like(y)
    np.put_along_axis(one_hot, idx[..., None], 1.0, axis=-1)
    return one_hot


def anneal_temperature(
    step: int,
    t_start: float = 2.0,
    t_end: float = 0.5,
    total_steps: int = 1000,
) -> float:
    """Linear temperature schedule clamped to ``[t_end, t_start]``."""
    if total_steps <= 0:
        return float(t_end)
    progress = min(max(step / total_steps, 0.0), 1.0)
    return float(t_start + (t_end - t_start) * progress)
