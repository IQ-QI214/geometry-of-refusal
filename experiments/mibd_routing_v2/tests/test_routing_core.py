"""CPU-only tests for CaRoB routing core (numpy).

Tests are organised in blueprint order (CaRoB代码实现交接蓝图_20260630.md §二-§五):
gating -> router -> gate -> bridge_losses. Each block mirrors the TDD points
listed in the blueprint. Pure numpy; no torch import, so this module can be
collected independently of ``test_v2_cpu.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.mibd_routing_v2.routing import (
    bridge_losses,
    gate,
    gating,
    router,
)


# ---------------------------------------------------------------------------
# Module 1: gating.py
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_rows_sum_to_one(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        probs = gating.softmax(logits, axis=-1)
        assert probs.shape == logits.shape
        np.testing.assert_allclose(probs.sum(axis=-1), np.ones(2), atol=1e-9)

    def test_higher_temperature_is_more_uniform(self) -> None:
        logits = np.array([[3.0, 0.0, -1.0]])
        low_t = gating.softmax(logits, temperature=0.5)
        high_t = gating.softmax(logits, temperature=4.0)

        def entropy(p: np.ndarray) -> float:
            return float(-(p * np.log(p + 1e-12)).sum(axis=-1)[0])

        assert entropy(high_t) > entropy(low_t)

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            gating.softmax(np.array([[1.0, 2.0]]), temperature=0.0)


class TestSwitchLoadBalancingLoss:
    def test_balanced_load_minimises_loss(self) -> None:
        # 8 samples uniformly spread over L=4 layers
        n, L = 8, 4
        gate_probs = np.full((n, L), 1.0 / L)
        expert_mask = np.zeros((n, L))
        # Cycle the chosen expert evenly so f_i = 1/L for all i
        for i in range(n):
            expert_mask[i, i % L] = 1.0

        balanced = gating.switch_load_balancing_loss(gate_probs, expert_mask)

        # Stacked-on-one-layer baseline
        gate_probs_skew = np.zeros((n, L))
        gate_probs_skew[:, 0] = 1.0
        expert_mask_skew = np.zeros((n, L))
        expert_mask_skew[:, 0] = 1.0
        skewed = gating.switch_load_balancing_loss(gate_probs_skew, expert_mask_skew)

        assert balanced < skewed
        # Theoretical balanced minimum for the (alpha = 1, sum f*P scaled by N) form is 1.0
        assert balanced == pytest.approx(1.0, abs=1e-9)
        assert skewed == pytest.approx(L, abs=1e-9)


class TestRouterZLoss:
    def test_zero_logits_loss_is_zero(self) -> None:
        logits = np.zeros((4, 5))
        assert gating.router_z_loss(logits) == pytest.approx(0.0, abs=1e-12)

    def test_scaled_logits_increase_loss(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(4, 5))
        small = gating.router_z_loss(logits)
        large = gating.router_z_loss(10.0 * logits)
        assert large > small


class TestLossFreeBias:
    def test_bias_moves_against_overload(self) -> None:
        L = 4
        state = gating.LossFreeBiasState(bias=np.zeros(L))
        expert_load = np.array([0.7, 0.1, 0.1, 0.1])
        target_load = np.full(L, 1.0 / L)
        new_state = gating.update_loss_free_bias(state, expert_load, target_load, lr=1.0)
        # Overloaded layer 0 must have bias decreased; underloaded layers increased.
        assert new_state.bias[0] < 0
        assert new_state.bias[1] > 0

    def test_iteration_reduces_load_variance(self) -> None:
        L = 4
        # synthetic system: load drifts toward target as bias is added
        state = gating.LossFreeBiasState(bias=np.zeros(L))
        target = np.full(L, 1.0 / L)
        load_history = []
        load = np.array([0.7, 0.1, 0.1, 0.1])
        for _ in range(20):
            state = gating.update_loss_free_bias(state, load, target, lr=0.2)
            # Simulate load adjusting toward equilibrium proportional to bias
            base_logits = np.log(load + 1e-12)
            new_load = gating.softmax(base_logits + state.bias)
            load_history.append(float(load.var()))
            load = new_load
        # Variance should fall over time
        assert load_history[-1] < load_history[0]


class TestGumbelSoftmax:
    def test_deterministic_with_seeded_rng(self) -> None:
        logits = np.array([[1.0, 2.0, 0.5]])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a = gating.gumbel_softmax_sample(logits, temperature=1.0, rng=rng1)
        b = gating.gumbel_softmax_sample(logits, temperature=1.0, rng=rng2)
        np.testing.assert_allclose(a, b)

    def test_hard_sample_is_one_hot(self) -> None:
        logits = np.array([[1.0, 2.0, 0.5], [0.1, -1.0, 4.0]])
        rng = np.random.default_rng(0)
        hard = gating.gumbel_softmax_sample(logits, temperature=1.0, rng=rng, hard=True)
        np.testing.assert_allclose(hard.sum(axis=-1), np.ones(2))
        # Each row has exactly one entry equal to 1.0
        assert ((hard == 1.0).sum(axis=-1) == 1).all()

    def test_low_temperature_sharper(self) -> None:
        logits = np.array([[1.0, 0.0, -1.0]])
        rng_lo = np.random.default_rng(7)
        rng_hi = np.random.default_rng(7)
        low_t = gating.gumbel_softmax_sample(logits, temperature=0.1, rng=rng_lo)
        high_t = gating.gumbel_softmax_sample(logits, temperature=5.0, rng=rng_hi)
        assert low_t.max() > high_t.max()


class TestAnnealTemperature:
    def test_endpoints(self) -> None:
        assert gating.anneal_temperature(0, t_start=2.0, t_end=0.5, total_steps=1000) == pytest.approx(
            2.0
        )
        assert gating.anneal_temperature(
            1000, t_start=2.0, t_end=0.5, total_steps=1000
        ) == pytest.approx(0.5)

    def test_monotonic_decrease(self) -> None:
        steps = [0, 100, 500, 900, 1000]
        vals = [gating.anneal_temperature(s, 2.0, 0.5, 1000) for s in steps]
        for a, b in zip(vals, vals[1:]):
            assert a >= b

    def test_clamped_beyond_total_steps(self) -> None:
        assert gating.anneal_temperature(2000, 2.0, 0.5, 1000) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Module 2: router.py
# ---------------------------------------------------------------------------


def _make_router(hidden_dim: int = 6, L: int = 4, C: int = 3, seed: int = 0) -> tuple[
    router.RouterConfig, router.RouterParams
]:
    rng = np.random.default_rng(seed)
    cfg = router.RouterConfig(
        hidden_dim=hidden_dim,
        num_layers=L,
        num_carriers=C,
        carrier_embed_dim=4,
        mlp_hidden=8,
        mode="soft",
    )
    in_dim = hidden_dim + cfg.carrier_embed_dim
    params = router.RouterParams(
        w1=rng.normal(scale=0.5, size=(in_dim, cfg.mlp_hidden)),
        b1=np.zeros(cfg.mlp_hidden),
        w_layer=rng.normal(scale=0.5, size=(cfg.mlp_hidden, L)),
        b_layer=np.zeros(L),
        w_trust=rng.normal(scale=0.5, size=(cfg.mlp_hidden, L)),
        b_trust=np.zeros(L),
        carrier_embed=rng.normal(scale=0.5, size=(C, cfg.carrier_embed_dim)),
    )
    return cfg, params


class TestRouter:
    def test_shape_and_probs_sum_to_one(self) -> None:
        cfg, params = _make_router()
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(5, cfg.hidden_dim))
        carrier_ids = np.array([0, 1, 2, 0, 1])
        out = router.route(feats, carrier_ids, params, cfg)
        assert out.layer_probs.shape == (5, cfg.num_layers)
        np.testing.assert_allclose(out.layer_probs.sum(axis=-1), np.ones(5), atol=1e-9)
        assert out.selected.shape == (5, cfg.num_layers)
        assert out.trust.shape == (5, cfg.num_layers)

    def test_top1_is_one_hot(self) -> None:
        cfg0, params = _make_router()
        cfg = router.RouterConfig(
            hidden_dim=cfg0.hidden_dim,
            num_layers=cfg0.num_layers,
            num_carriers=cfg0.num_carriers,
            carrier_embed_dim=cfg0.carrier_embed_dim,
            mlp_hidden=cfg0.mlp_hidden,
            mode="top1",
        )
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(4, cfg.hidden_dim))
        carrier_ids = np.array([0, 1, 2, 0])
        out = router.route(feats, carrier_ids, params, cfg)
        np.testing.assert_allclose(out.selected.sum(axis=-1), np.ones(4))
        assert ((out.selected == 1.0).sum(axis=-1) == 1).all()

    def test_carrier_conditioned(self) -> None:
        cfg, params = _make_router()
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(3, cfg.hidden_dim))
        out_a = router.route(feats, np.array([0, 0, 0]), params, cfg)
        out_b = router.route(feats, np.array([1, 1, 1]), params, cfg)
        # The same features under different carriers must yield different routing.
        assert not np.allclose(out_a.layer_probs, out_b.layer_probs)

    def test_aggregate_risk_score_pure_layer(self) -> None:
        # If routing is fully concentrated on layer k, aggregated risk == probe[k]
        n, L = 4, 5
        probe = np.array(
            [
                [0.1, 0.2, 0.7, 0.3, 0.4],
                [0.0, 0.5, 0.1, 0.2, 0.9],
                [0.4, 0.4, 0.4, 0.4, 0.4],
                [-1.0, 0.0, 1.0, 2.0, -2.0],
            ]
        )
        for k in range(L):
            one_hot = np.zeros((n, L))
            one_hot[:, k] = 1.0
            agg = router.aggregate_risk_score(one_hot, probe)
            np.testing.assert_allclose(agg, probe[:, k])

    def test_determinism(self) -> None:
        cfg, params = _make_router()
        rng = np.random.default_rng(2024)
        feats = rng.normal(size=(4, cfg.hidden_dim))
        carrier_ids = np.array([0, 1, 2, 1])
        a = router.route(feats, carrier_ids, params, cfg)
        b = router.route(feats, carrier_ids, params, cfg)
        np.testing.assert_allclose(a.layer_probs, b.layer_probs)
        np.testing.assert_allclose(a.selected, b.selected)
        np.testing.assert_allclose(a.trust, b.trust)

    def test_dim_mismatch_raises(self) -> None:
        cfg, params = _make_router(hidden_dim=6)
        feats = np.zeros((2, 5))  # wrong hidden_dim
        with pytest.raises(ValueError):
            router.route(feats, np.array([0, 1]), params, cfg)

    def test_carrier_id_out_of_range(self) -> None:
        cfg, params = _make_router()
        feats = np.zeros((1, cfg.hidden_dim))
        with pytest.raises(ValueError):
            router.route(feats, np.array([cfg.num_carriers]), params, cfg)


# ---------------------------------------------------------------------------
# Module 3: gate.py
# ---------------------------------------------------------------------------


class TestSoftGate:
    def test_at_threshold_is_half(self) -> None:
        s = np.array([1.0, 2.0, 3.0])
        g = gate.soft_gate(s, tau=2.0, alpha=4.0)
        assert g[1] == pytest.approx(0.5, abs=1e-9)

    def test_alpha_steepness(self) -> None:
        s = np.array([2.05])
        g_shallow = gate.soft_gate(s, tau=2.0, alpha=1.0)
        g_steep = gate.soft_gate(s, tau=2.0, alpha=50.0)
        assert g_steep[0] > g_shallow[0]

    def test_extremes(self) -> None:
        g = gate.soft_gate(np.array([-100.0, 100.0]), tau=0.0, alpha=1.0)
        assert g[0] == pytest.approx(0.0, abs=1e-6)
        assert g[1] == pytest.approx(1.0, abs=1e-6)


class TestHardGate:
    def test_threshold(self) -> None:
        s = np.array([0.4, 0.5, 0.6])
        g = gate.hard_gate(s, tau=0.5)
        assert list(g) == [0.0, 0.0, 1.0]


class TestNullspaceProjection:
    def test_orthogonal_to_basis(self) -> None:
        rng = np.random.default_rng(0)
        d, k, n = 8, 3, 5
        basis = np.linalg.qr(rng.normal(size=(d, k)))[0].T  # (k, d), orthonormal rows
        delta = rng.normal(size=(n, d))
        projected = gate.nullspace_projection(delta, basis)
        for b in basis:
            inner = projected @ b
            np.testing.assert_allclose(inner, np.zeros(n), atol=1e-9)

    def test_idempotent(self) -> None:
        rng = np.random.default_rng(0)
        d, k = 6, 2
        basis = np.linalg.qr(rng.normal(size=(d, k)))[0].T
        delta = rng.normal(size=(3, d))
        once = gate.nullspace_projection(delta, basis)
        twice = gate.nullspace_projection(once, basis)
        np.testing.assert_allclose(once, twice, atol=1e-9)


class TestApplyGatedDelta:
    def test_zero_gate_is_identity(self) -> None:
        rng = np.random.default_rng(0)
        h = rng.normal(size=(4, 5))
        delta = rng.normal(size=(4, 5))
        g = np.zeros(4)
        out = gate.apply_gated_delta(h, delta, g)
        np.testing.assert_allclose(out, h)

    def test_full_gate(self) -> None:
        rng = np.random.default_rng(0)
        h = rng.normal(size=(2, 3))
        delta = rng.normal(size=(2, 3))
        g = np.ones(2)
        out = gate.apply_gated_delta(h, delta, g)
        np.testing.assert_allclose(out, h + delta)


# ---------------------------------------------------------------------------
# Module 4: bridge_losses.py
# ---------------------------------------------------------------------------


class TestLoraDelta:
    def test_shape_and_zero_scaling(self) -> None:
        rng = np.random.default_rng(0)
        d, r, n = 8, 2, 5
        A = rng.normal(size=(d, r))
        B = rng.normal(size=(r, d))
        h = rng.normal(size=(n, d))
        out_zero = bridge_losses.lora_delta(h, A, B, scaling=0.0)
        np.testing.assert_allclose(out_zero, np.zeros_like(out_zero))
        out = bridge_losses.lora_delta(h, A, B, scaling=2.0)
        assert out.shape == (n, d)
        # Rank at most r
        assert np.linalg.matrix_rank(out) <= r


class TestDistillationLossC2:
    def test_zero_when_identical(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 6))
        assert bridge_losses.distillation_loss_c2(x, x) == pytest.approx(0.0, abs=1e-12)

    def test_monotone_in_difference(self) -> None:
        rng = np.random.default_rng(0)
        teacher = rng.normal(size=(4, 6))
        small = teacher + 0.1 * rng.normal(size=teacher.shape)
        large = teacher + 2.0 * rng.normal(size=teacher.shape)
        assert bridge_losses.distillation_loss_c2(small, teacher) < bridge_losses.distillation_loss_c2(
            large, teacher
        )


class TestSelfSupervisedCorrLossC3:
    def test_perfect_positive_corr_min(self) -> None:
        s = np.array([0.1, 0.2, 0.3, 0.5, 0.9])
        margin = 2.0 * s + 1.0
        assert bridge_losses.self_supervised_corr_loss_c3(s, margin) == pytest.approx(-1.0, abs=1e-9)

    def test_perfect_negative_corr_max(self) -> None:
        s = np.array([0.1, 0.2, 0.3, 0.5, 0.9])
        margin = -s
        assert bridge_losses.self_supervised_corr_loss_c3(s, margin) == pytest.approx(1.0, abs=1e-9)

    def test_constant_input_safe(self) -> None:
        s = np.zeros(5)
        margin = np.ones(5)
        loss = bridge_losses.self_supervised_corr_loss_c3(s, margin)
        assert math.isfinite(loss)
        assert loss == pytest.approx(0.0, abs=1e-9)


class TestUtilityAnchorPenalty:
    def test_zero_when_no_delta_on_benign(self) -> None:
        delta = np.zeros((3, 4))
        g = np.zeros(3)
        assert bridge_losses.utility_anchor_penalty(delta, g) == pytest.approx(0.0, abs=1e-12)

    def test_increases_with_benign_delta(self) -> None:
        rng = np.random.default_rng(0)
        delta_small = 0.1 * rng.normal(size=(4, 6))
        delta_large = 2.0 * rng.normal(size=(4, 6))
        g = np.zeros(4)  # benign
        small = bridge_losses.utility_anchor_penalty(delta_small, g)
        large = bridge_losses.utility_anchor_penalty(delta_large, g)
        assert large > small

    def test_only_penalises_benign_samples(self) -> None:
        rng = np.random.default_rng(0)
        delta = rng.normal(size=(4, 6))
        # if every sample is risky (gate=1), penalty should be small / zero
        risky_only = bridge_losses.utility_anchor_penalty(delta, np.ones(4))
        benign_only = bridge_losses.utility_anchor_penalty(delta, np.zeros(4))
        assert benign_only > risky_only
