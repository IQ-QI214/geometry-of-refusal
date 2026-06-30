"""CPU-only tests for the offline evaluation path (loader + probe bank + oracle).

These exercise the modules against *synthetic* npz-shaped data so they run with
numpy alone (no torch, no real model files). A separate manual run validates
them on the real extracted hidden states.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.mibd_routing_v2.eval import load_hidden_states as lh
from experiments.mibd_routing_v2.eval import run_offline_oracle as roo
from experiments.mibd_routing_v2.eval import check_saturation as cs
from experiments.mibd_routing_v2.routing import probe_bank as pb


def _make_states(
    carriers=("FigStep", "V-real"),
    layers=(0, 4, 8),
    d=16,
    n_per_class=20,
    separable=True,
    seed=0,
):
    """Build a synthetic ``states`` dict matching the real npz key schema.

    When ``separable`` is True, harmful/harmless are linearly separable within
    each (carrier, layer) along a *carrier-specific* direction, so within-carrier
    AUC is ~1.0 but cross-carrier transfer degrades.
    """
    rng = np.random.default_rng(seed)
    states: dict[str, np.ndarray] = {}
    # one distinct separating axis per carrier
    carrier_axis = {c: i % d for i, c in enumerate(carriers)}
    for c in carriers:
        for layer in layers:
            harmless = rng.normal(scale=0.3, size=(n_per_class, d))
            harmful = rng.normal(scale=0.3, size=(n_per_class, d))
            if separable:
                axis = carrier_axis[c]
                harmful[:, axis] += 5.0  # push harmful along carrier-specific axis
            states[f"{c}__layer{layer}__pos-1__harmful"] = harmful.astype(np.float32)
            states[f"{c}__layer{layer}__pos-1__harmless"] = harmless.astype(np.float32)
    return states


# ---------------------------------------------------------------------------
# Module B: load_hidden_states
# ---------------------------------------------------------------------------


class TestLoader:
    def test_available_carriers_and_layers(self) -> None:
        states = _make_states()
        assert lh.available_carriers(states) == ["FigStep", "V-real"]
        assert lh.available_layers(states, "FigStep") == [0, 4, 8]

    def test_missing_carrier_raises(self) -> None:
        states = _make_states()
        with pytest.raises(ValueError):
            lh.available_layers(states, "Nope")

    def test_build_feature_matrix_labels(self) -> None:
        states = _make_states(n_per_class=10)
        feats, labels = lh.build_carrier_feature_matrix(states, "FigStep", 0)
        assert feats.shape == (20, 16)
        assert labels.sum() == 10  # 10 harmful
        assert set(labels.tolist()) == {0, 1}

    def test_missing_key_raises(self) -> None:
        states = _make_states()
        with pytest.raises(ValueError):
            lh.build_carrier_feature_matrix(states, "FigStep", 999)

    def test_split_is_stratified_and_leakproof(self) -> None:
        states = _make_states(n_per_class=20)
        feats, labels = lh.build_carrier_feature_matrix(states, "FigStep", 0)
        split = lh.split_train_test(feats, labels, frac_train=0.5, seed=1)
        # both classes present on both sides
        assert set(split.train_labels.tolist()) == {0, 1}
        assert set(split.test_labels.tolist()) == {0, 1}
        # no leakage: train+test sizes add up, no overlap by reconstruction
        assert (
            split.train_features.shape[0] + split.test_features.shape[0]
            == feats.shape[0]
        )

    def test_split_deterministic(self) -> None:
        states = _make_states()
        feats, labels = lh.build_carrier_feature_matrix(states, "FigStep", 0)
        a = lh.split_train_test(feats, labels, seed=7)
        b = lh.split_train_test(feats, labels, seed=7)
        np.testing.assert_array_equal(a.train_labels, b.train_labels)
        np.testing.assert_array_equal(a.train_features, b.train_features)

    def test_invalid_frac_raises(self) -> None:
        states = _make_states()
        feats, labels = lh.build_carrier_feature_matrix(states, "FigStep", 0)
        with pytest.raises(ValueError):
            lh.split_train_test(feats, labels, frac_train=1.5)


# ---------------------------------------------------------------------------
# Module A: probe_bank
# ---------------------------------------------------------------------------


class TestProbeBank:
    def test_same_carrier_auc_high(self) -> None:
        states = _make_states(separable=True)
        aucs = pb.cross_carrier_auc(states, "FigStep", "FigStep")
        assert all(a == pytest.approx(1.0, abs=1e-9) for a in aucs.values())

    def test_cross_carrier_degrades(self) -> None:
        # carrier-specific separating axis => cross-carrier transfer should drop
        states = _make_states(separable=True)
        same = pb.cross_carrier_auc(states, "FigStep", "FigStep")
        cross = pb.cross_carrier_auc(states, "FigStep", "V-real")
        assert np.mean(list(cross.values())) < np.mean(list(same.values()))

    def test_score_orientation(self) -> None:
        states = _make_states()
        feats, labels = lh.build_carrier_feature_matrix(states, "FigStep", 0)
        harmful = feats[labels == 1]
        harmless = feats[labels == 0]
        probe = pb.fit_layer_probe(harmful, harmless, layer=0)
        # harmful mean score should exceed harmless mean score
        assert pb.score_samples(probe, harmful).mean() > pb.score_samples(
            probe, harmless
        ).mean()

    def test_best_layer_picks_max(self) -> None:
        per_layer = {0: 0.7, 4: 0.95, 8: 0.8}
        assert pb.best_layer(per_layer) == 4

    def test_best_layer_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            pb.best_layer({})


# ---------------------------------------------------------------------------
# Module C: run_offline_oracle
# ---------------------------------------------------------------------------


class TestOfflineOracle:
    def test_report_schema(self) -> None:
        states = _make_states()
        report = roo.build_offline_report(states, model_name="synthetic")
        assert set(report) == {"summary", "o1_o2", "transfer_matrix", "gate_sweeps"}
        summary = report["summary"]
        assert summary["model"] == "synthetic"
        assert summary["carriers"] == ["FigStep", "V-real"]
        # JSON serializable
        json.dumps(report)

    def test_oracle_not_worse_than_fixed(self) -> None:
        states = _make_states()
        report = roo.build_offline_report(states, model_name="synthetic")
        for tr in report["o1_o2"]:
            for te in report["o1_o2"][tr]:
                cell = report["o1_o2"][tr][te]
                assert cell["o2_oracle_auc"] >= cell["o1_fixed_auc"] - 1e-9

    def test_cross_carrier_drop_nonnegative_when_carrier_specific(self) -> None:
        states = _make_states(separable=True)
        report = roo.build_offline_report(states, model_name="synthetic")
        drop = report["summary"]["cross_carrier_transfer_drop"]
        # within >= cross on average for carrier-specific separation
        assert drop >= -1e-9

    def test_gate_sweep_monotone_endpoints(self) -> None:
        states = _make_states()
        report = roo.build_offline_report(states, model_name="synthetic")
        for carrier, sweep in report["gate_sweeps"].items():
            # lowest tau => everything activates; highest tau => nothing
            assert sweep[0]["harmful_activation_rate"] >= sweep[-1]["harmful_activation_rate"]
            assert sweep[0]["benign_leak_rate"] >= sweep[-1]["benign_leak_rate"]


# ---------------------------------------------------------------------------
# Saturation guard: check_saturation
# ---------------------------------------------------------------------------


class TestSaturationGuard:
    def test_flags_layer0_saturation(self) -> None:
        # every layer AUC=1.0 incl layer 0 => blank-vs-content confound
        summary = {
            "conditions": {
                "FigStep": [
                    {"layer": 0, "single_direction_auc": 1.0},
                    {"layer": 8, "single_direction_auc": 1.0},
                ],
                "V-real": [
                    {"layer": 0, "single_direction_auc": 1.0},
                    {"layer": 8, "single_direction_auc": 1.0},
                ],
            }
        }
        report = cs.evaluate_probe_summary(summary)
        assert report["saturated"] is True

    def test_passes_when_early_layers_weak(self) -> None:
        # early layers weak, separability emerges later => semantics, not confound
        summary = {
            "conditions": {
                "FigStep": [
                    {"layer": 0, "single_direction_auc": 0.55},
                    {"layer": 16, "single_direction_auc": 0.93},
                ],
            }
        }
        report = cs.evaluate_probe_summary(summary)
        assert report["saturated"] is False

    def test_empty_conditions_raises(self) -> None:
        with pytest.raises(ValueError):
            cs.evaluate_probe_summary({"conditions": {}})
