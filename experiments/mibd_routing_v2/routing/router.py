"""Carrier-conditioned routing (numpy, CPU-only).

Implements CaRoB part A (CaRoB代码实现交接蓝图 §三): a 2-layer MLP gating with
carrier-type embedding and the decoupled DirMoE-style two heads
(arXiv:2602.09001): one for layer selection, one for trust weights. The router
returns both the soft routing distribution and the mode-specific ``selected``
(soft probabilities or one-hot top-1) plus per-layer ``trust`` weights, which
downstream components combine with per-layer probe scores into the risk score
``s(x) = sum_l p_l * probe_l(h_l(x))`` (arXiv:2603.11114 / 2606.24952).

This is the pure-numpy "forward + scoring" surface; the trainable wrapper
(autograd parameters, gradient flow) lives in the GPU layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.mibd_routing_v2.routing.gating import softmax


@dataclass(frozen=True)
class RouterConfig:
    hidden_dim: int
    num_layers: int
    num_carriers: int
    carrier_embed_dim: int = 8
    mlp_hidden: int = 32
    mode: str = "soft"  # "soft" (B1) | "top1" (B2)


@dataclass(frozen=True)
class RouterParams:
    w1: np.ndarray
    b1: np.ndarray
    w_layer: np.ndarray
    b_layer: np.ndarray
    w_trust: np.ndarray
    b_trust: np.ndarray
    carrier_embed: np.ndarray  # (num_carriers, carrier_embed_dim)


@dataclass(frozen=True)
class RoutingOutput:
    layer_probs: np.ndarray  # (n, L) soft distribution over layers
    selected: np.ndarray  # (n, L) soft = layer_probs, top1 = one-hot
    trust: np.ndarray  # (n, L)


def _check_shapes(
    features: np.ndarray,
    carrier_ids: np.ndarray,
    params: RouterParams,
    config: RouterConfig,
) -> None:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    n, d = features.shape
    if d != config.hidden_dim:
        raise ValueError(f"features hidden_dim={d} != config.hidden_dim={config.hidden_dim}")
    if carrier_ids.shape != (n,):
        raise ValueError(f"carrier_ids must have shape ({n},), got {carrier_ids.shape}")
    if carrier_ids.min() < 0 or carrier_ids.max() >= config.num_carriers:
        raise ValueError(
            f"carrier_ids out of [0, {config.num_carriers}): "
            f"min={int(carrier_ids.min())}, max={int(carrier_ids.max())}"
        )
    expected_in = config.hidden_dim + config.carrier_embed_dim
    if params.w1.shape != (expected_in, config.mlp_hidden):
        raise ValueError(f"w1 shape {params.w1.shape} != ({expected_in}, {config.mlp_hidden})")
    if params.w_layer.shape != (config.mlp_hidden, config.num_layers):
        raise ValueError("w_layer shape mismatch")
    if params.w_trust.shape != (config.mlp_hidden, config.num_layers):
        raise ValueError("w_trust shape mismatch")
    if params.carrier_embed.shape != (config.num_carriers, config.carrier_embed_dim):
        raise ValueError("carrier_embed shape mismatch")


def route(
    features: np.ndarray,
    carrier_ids: np.ndarray,
    params: RouterParams,
    config: RouterConfig,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> RoutingOutput:
    """Carrier-conditioned routing forward pass.

    Args:
        features: ``(n, hidden_dim)`` per-sample sensor features (e.g. mean
            pooled hidden states from a representative layer).
        carrier_ids: ``(n,)`` integer carrier-type ids.
        params, config: weights and shapes.
        temperature: softmax temperature (used for the layer head).
        rng: kept for API symmetry with the future Gumbel mode; unused in the
            deterministic forward.

    Returns:
        RoutingOutput with ``layer_probs`` (soft distribution), ``selected``
        (soft = ``layer_probs``; top1 = one-hot of argmax), and ``trust``.
    """
    del rng  # deterministic in this numpy forward
    features = np.asarray(features, dtype=np.float64)
    carrier_ids = np.asarray(carrier_ids, dtype=np.int64)
    _check_shapes(features, carrier_ids, params, config)

    carrier_vec = params.carrier_embed[carrier_ids]
    x = np.concatenate([features, carrier_vec], axis=-1)
    h = np.tanh(x @ params.w1 + params.b1)
    layer_logits = h @ params.w_layer + params.b_layer
    trust_logits = h @ params.w_trust + params.b_trust
    layer_probs = softmax(layer_logits, axis=-1, temperature=temperature)
    trust = softmax(trust_logits, axis=-1)

    if config.mode == "soft":
        selected = layer_probs
    elif config.mode == "top1":
        idx = np.argmax(layer_probs, axis=-1)
        selected = np.zeros_like(layer_probs)
        np.put_along_axis(selected, idx[:, None], 1.0, axis=-1)
    else:
        raise ValueError(f"unknown router mode: {config.mode}")

    return RoutingOutput(layer_probs=layer_probs, selected=selected, trust=trust)


def aggregate_risk_score(
    layer_probs: np.ndarray,
    per_layer_probe_scores: np.ndarray,
) -> np.ndarray:
    """s(x) = sum_l p_l * probe_l(h_l(x))."""
    p = np.asarray(layer_probs, dtype=np.float64)
    s = np.asarray(per_layer_probe_scores, dtype=np.float64)
    if p.shape != s.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {s.shape}")
    return (p * s).sum(axis=-1)
