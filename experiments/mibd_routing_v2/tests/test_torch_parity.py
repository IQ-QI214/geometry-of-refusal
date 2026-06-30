"""Parity tests: torch routing core must match the numpy reference.

These are skipped automatically when torch is unavailable (the CPU-only dev
box), and run on the GPU machine to guarantee the trainable torch port is a
faithful, numerically-aligned wrapper of the numpy core.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.mibd_routing_v2.routing import bridge_losses as np_bridge
from experiments.mibd_routing_v2.routing import gate as np_gate
from experiments.mibd_routing_v2.routing import gating as np_gating
from experiments.mibd_routing_v2.routing import router as np_router
from experiments.mibd_routing_v2.routing import torch_modules as tm


ATOL = 1e-5


def test_switch_loss_parity() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random((6, 4))
    probs = probs / probs.sum(axis=1, keepdims=True)
    mask = np.zeros((6, 4))
    for i in range(6):
        mask[i, i % 4] = 1.0
    np_val = np_gating.switch_load_balancing_loss(probs, mask)
    t_val = tm.switch_load_balancing_loss(
        torch.tensor(probs), torch.tensor(mask)
    ).item()
    assert t_val == pytest.approx(np_val, abs=ATOL)


def test_z_loss_parity() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(5, 7))
    np_val = np_gating.router_z_loss(logits)
    t_val = tm.router_z_loss(torch.tensor(logits)).item()
    assert t_val == pytest.approx(np_val, abs=ATOL)
    # zero logits -> 0 on both sides
    assert tm.router_z_loss(torch.zeros(3, 5)).item() == pytest.approx(0.0, abs=ATOL)


def test_loss_free_bias_parity() -> None:
    bias = np.zeros(4)
    load = np.array([0.7, 0.1, 0.1, 0.1])
    target = np.full(4, 0.25)
    np_state = np_gating.update_loss_free_bias(
        np_gating.LossFreeBiasState(bias=bias), load, target, lr=0.5
    )
    t_bias = tm.update_loss_free_bias(
        torch.tensor(bias), torch.tensor(load), torch.tensor(target), lr=0.5
    ).numpy()
    np.testing.assert_allclose(t_bias, np_state.bias, atol=ATOL)


def test_anneal_parity() -> None:
    for step in (0, 250, 500, 1000):
        assert tm.anneal_temperature(step, 2.0, 0.5, 1000) == pytest.approx(
            np_gating.anneal_temperature(step, 2.0, 0.5, 1000), abs=ATOL
        )


def test_router_forward_parity() -> None:
    """Torch router with weights copied from numpy params must match route()."""
    hidden_dim, L, C = 6, 4, 3
    cfg_np = np_router.RouterConfig(
        hidden_dim=hidden_dim, num_layers=L, num_carriers=C,
        carrier_embed_dim=4, mlp_hidden=8, mode="soft",
    )
    rng = np.random.default_rng(0)
    in_dim = hidden_dim + cfg_np.carrier_embed_dim
    params = np_router.RouterParams(
        w1=rng.normal(size=(in_dim, cfg_np.mlp_hidden)),
        b1=rng.normal(size=cfg_np.mlp_hidden),
        w_layer=rng.normal(size=(cfg_np.mlp_hidden, L)),
        b_layer=rng.normal(size=L),
        w_trust=rng.normal(size=(cfg_np.mlp_hidden, L)),
        b_trust=rng.normal(size=L),
        carrier_embed=rng.normal(size=(C, cfg_np.carrier_embed_dim)),
    )
    feats = rng.normal(size=(5, hidden_dim))
    carrier_ids = np.array([0, 1, 2, 0, 1])
    np_out = np_router.route(feats, carrier_ids, params, cfg_np)

    cfg_t = tm.TorchRouterConfig(
        hidden_dim=hidden_dim, num_layers=L, num_carriers=C,
        carrier_embed_dim=4, mlp_hidden=8, mode="soft",
    )
    router = tm.CarrierRouter(cfg_t).double()
    with torch.no_grad():
        router.fc1.weight.copy_(torch.tensor(params.w1.T))
        router.fc1.bias.copy_(torch.tensor(params.b1))
        router.layer_head.weight.copy_(torch.tensor(params.w_layer.T))
        router.layer_head.bias.copy_(torch.tensor(params.b_layer))
        router.trust_head.weight.copy_(torch.tensor(params.w_trust.T))
        router.trust_head.bias.copy_(torch.tensor(params.b_trust))
        router.carrier_embed.weight.copy_(torch.tensor(params.carrier_embed))

    t_out = router(torch.tensor(feats), torch.tensor(carrier_ids, dtype=torch.long))
    np.testing.assert_allclose(
        t_out["layer_probs"].detach().numpy(), np_out.layer_probs, atol=ATOL
    )
    np.testing.assert_allclose(t_out["trust"].detach().numpy(), np_out.trust, atol=ATOL)


def test_aggregate_risk_score_parity() -> None:
    rng = np.random.default_rng(2)
    probs = rng.random((4, 5))
    probs = probs / probs.sum(axis=1, keepdims=True)
    probe = rng.normal(size=(4, 5))
    np_val = np_router.aggregate_risk_score(probs, probe)
    t_val = tm.aggregate_risk_score(torch.tensor(probs), torch.tensor(probe)).numpy()
    np.testing.assert_allclose(t_val, np_val, atol=ATOL)


def test_gate_parity() -> None:
    s = np.array([-1.0, 0.0, 0.5, 2.0])
    gate = tm.ThresholdGate(tau=0.5, alpha=4.0)
    np.testing.assert_allclose(
        gate.soft(torch.tensor(s)).numpy(),
        np_gate.soft_gate(s, tau=0.5, alpha=4.0),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        gate.hard(torch.tensor(s)).numpy(),
        np_gate.hard_gate(s, tau=0.5),
        atol=ATOL,
    )


def test_nullspace_and_gated_delta_parity() -> None:
    rng = np.random.default_rng(3)
    d, k = 6, 2
    basis = np.linalg.qr(rng.normal(size=(d, k)))[0].T
    delta = rng.normal(size=(3, d))
    np_proj = np_gate.nullspace_projection(delta, basis)
    t_proj = tm.nullspace_projection(torch.tensor(delta), torch.tensor(basis)).numpy()
    np.testing.assert_allclose(t_proj, np_proj, atol=ATOL)

    h = rng.normal(size=(3, d))
    g = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        tm.apply_gated_delta(torch.tensor(h), torch.tensor(delta), torch.tensor(g)).numpy(),
        np_gate.apply_gated_delta(h, delta, g),
        atol=ATOL,
    )


def test_bridge_losses_parity() -> None:
    rng = np.random.default_rng(4)
    student = rng.normal(size=(4, 6))
    teacher = rng.normal(size=(4, 6))
    assert tm.distillation_loss_c2(
        torch.tensor(student), torch.tensor(teacher)
    ).item() == pytest.approx(np_bridge.distillation_loss_c2(student, teacher), abs=ATOL)

    s = rng.normal(size=10)
    margin = 2.0 * s + 0.1 * rng.normal(size=10)
    assert tm.self_supervised_corr_loss_c3(
        torch.tensor(s), torch.tensor(margin)
    ).item() == pytest.approx(np_bridge.self_supervised_corr_loss_c3(s, margin), abs=ATOL)

    delta = rng.normal(size=(4, 6))
    g = np.array([0.0, 0.0, 1.0, 0.5])
    assert tm.utility_anchor_penalty(
        torch.tensor(delta), torch.tensor(g)
    ).item() == pytest.approx(np_bridge.utility_anchor_penalty(delta, g), abs=ATOL)


def test_lowrank_bridge_identity_at_init() -> None:
    """B initialized to zero => bridge is identity at init (no perturbation)."""
    bridge = tm.LowRankBridge(hidden_dim=8, rank=2, alpha=4.0).double()
    h = torch.randn(3, 8, dtype=torch.float64)
    gate = torch.ones(3, dtype=torch.float64)
    out = bridge(h, gate)
    np.testing.assert_allclose(out.detach().numpy(), h.numpy(), atol=ATOL)


def test_router_is_trainable() -> None:
    """A gradient step must change router parameters (autograd wired)."""
    cfg = tm.TorchRouterConfig(hidden_dim=6, num_layers=4, num_carriers=3, mlp_hidden=8)
    router = tm.CarrierRouter(cfg)
    opt = torch.optim.SGD(router.parameters(), lr=0.1)
    feats = torch.randn(5, 6)
    carrier_ids = torch.tensor([0, 1, 2, 0, 1])
    before = router.layer_head.weight.detach().clone()
    out = router(feats, carrier_ids)
    # encourage uniform routing via z-loss + balancing
    loss = tm.router_z_loss(out["layer_logits"])
    loss.backward()
    opt.step()
    after = router.layer_head.weight.detach()
    assert not torch.allclose(before, after)
