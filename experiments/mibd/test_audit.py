"""CPU-only tests for Phase 1.5 audit modules. No GPU or model required."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.audit.splits import (
    held_out_split,
    group_split_by_paired_id,
    cross_category_split,
    available_categories,
)
from experiments.mibd.audit.permutation import permutation_auc
from experiments.mibd.audit.margins import compute_score_margins, condition_margin_table
from experiments.mibd.eval.phase1p5_report import AuditResult, build_phase1p5_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(
    id: str,
    label: str = "harmful",
    category: str = "cat_a",
    paired_id: str | None = None,
    visual_condition: str = "V-text",
) -> MIBDSample:
    return MIBDSample(
        id=id,
        text="hello",
        image_path=None,
        label=label,
        category=category,
        source="test",
        paired_id=paired_id,
        visual_condition=visual_condition,
    )


def _make_hidden_map(
    n_harmful: int = 20,
    n_harmless: int = 20,
    dim: int = 8,
    layer: int = 0,
    pos: int = 0,
    seed: int = 0,
    separable: bool = True,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    if separable:
        harmful = rng.standard_normal((n_harmful, dim)) + 2.0
        harmless = rng.standard_normal((n_harmless, dim))
    else:
        harmful = rng.standard_normal((n_harmful, dim))
        harmless = rng.standard_normal((n_harmless, dim))
    return {(layer, pos): {"harmful": harmful, "harmless": harmless}}


# ---------------------------------------------------------------------------
# splits tests
# ---------------------------------------------------------------------------

def test_held_out_split_stratified():
    samples = (
        [_make_sample(f"h{i}", label="harmful") for i in range(40)]
        + [_make_sample(f"l{i}", label="harmless") for i in range(40)]
    )
    train, test = held_out_split(samples, test_frac=0.2, seed=0)
    assert len(train) + len(test) == len(samples)
    # Both splits should have both labels
    train_labels = {s.label for s in train}
    test_labels = {s.label for s in test}
    assert "harmful" in train_labels and "harmless" in train_labels
    assert "harmful" in test_labels and "harmless" in test_labels
    # Roughly 20% in test
    assert 10 <= len(test) <= 20


def test_group_split_no_pair_leakage():
    # 20 pairs: a_i.paired_id = b_i.id and b_i.paired_id = a_i.id (real schema convention)
    samples = []
    for i in range(20):
        samples.append(_make_sample(f"a_{i}", label="harmful",  paired_id=f"b_{i}"))
        samples.append(_make_sample(f"b_{i}", label="harmless", paired_id=f"a_{i}"))

    train, test = group_split_by_paired_id(samples, test_frac=0.2, seed=42)
    assert len(train) + len(test) == len(samples)

    # No pair group should be split across train and test.
    # Build a map: sample_id -> which split it's in.
    train_ids = {s.id for s in train}
    test_ids = {s.id for s in test}
    for s in samples:
        if s.paired_id is not None:
            # Both members of the pair must be on the same side.
            partner_in_train = s.paired_id in train_ids
            self_in_train = s.id in train_ids
            assert partner_in_train == self_in_train, (
                f"Pair leaked: {s.id} and {s.paired_id} on different sides"
            )


def test_group_split_none_paired_ids():
    # All samples have paired_id=None — should still split without error
    samples = [_make_sample(f"s{i}", paired_id=None) for i in range(20)]
    train, test = group_split_by_paired_id(samples, test_frac=0.2, seed=0)
    assert len(train) + len(test) == len(samples)
    assert len(test) >= 1


def test_cross_category_split_correct_partition():
    samples = (
        [_make_sample(f"a{i}", category="cat_a") for i in range(10)]
        + [_make_sample(f"b{i}", category="cat_b") for i in range(10)]
        + [_make_sample(f"c{i}", category="cat_c") for i in range(10)]
    )
    train, test = cross_category_split(samples, test_category="cat_b")
    assert all(s.category == "cat_b" for s in test)
    assert all(s.category != "cat_b" for s in train)
    assert len(train) == 20
    assert len(test) == 10


def test_available_categories():
    samples = (
        [_make_sample(f"a{i}", category="z_cat") for i in range(3)]
        + [_make_sample(f"b{i}", category="a_cat") for i in range(3)]
    )
    cats = available_categories(samples)
    assert cats == ["a_cat", "z_cat"]


# ---------------------------------------------------------------------------
# permutation test
# ---------------------------------------------------------------------------

def test_permutation_auc_near_half():
    """With shuffled labels, permutation AUC should average close to 0.5."""
    # Use clearly separable data so original AUC is near 1.0.
    # After permutation the labels are random, so AUC should collapse to ~0.5.
    from experiments.mibd.probes.direction import mean_difference_direction, project_scores
    from experiments.mibd.probes.metrics import binary_auc

    hidden_map = _make_hidden_map(n_harmful=30, n_harmless=30, dim=16, separable=True, seed=1)
    lp = hidden_map[(0, 0)]
    # Verify the data is actually separable — original AUC should be near 1.0
    direction = mean_difference_direction(lp["harmful"], lp["harmless"])
    all_hidden = np.vstack([lp["harmful"], lp["harmless"]])
    labels = np.array([1] * 30 + [0] * 30)
    original_auc = binary_auc(labels, project_scores(all_hidden, direction))
    assert original_auc >= 0.9, f"Test data not separable enough: original AUC={original_auc:.4f}"

    perm_auc = permutation_auc(hidden_map, layer=0, pos=0, n_permutations=200, seed=42)
    # Permutation AUC must be well below the original — detects the bug where
    # original labels were used instead of shuffled labels.
    assert perm_auc < 0.8, (
        f"Permutation AUC too high ({perm_auc:.4f}) — labels may not be shuffled correctly"
    )
    # Also check it is roughly centered around 0.5
    assert 0.30 <= perm_auc <= 0.75, f"Expected ~0.5, got {perm_auc:.4f}"


def test_permutation_auc_random_data():
    """With non-separable data the permutation AUC should also be near 0.5."""
    hidden_map = _make_hidden_map(n_harmful=20, n_harmless=20, dim=8, separable=False, seed=2)
    perm_auc = permutation_auc(hidden_map, layer=0, pos=0, n_permutations=100, seed=0)
    assert 0.25 <= perm_auc <= 0.75, f"Expected ~0.5, got {perm_auc:.4f}"


# ---------------------------------------------------------------------------
# margin tests
# ---------------------------------------------------------------------------

def test_compute_score_margins_basic():
    hidden_map = _make_hidden_map(n_harmful=20, n_harmless=20, dim=8, separable=True, seed=3)
    direction = np.ones(8) / np.sqrt(8)
    margins = compute_score_margins(direction, hidden_map, layer=0, pos=0)
    assert set(margins.keys()) == {
        "mean_gap", "median_gap", "iqr_harmful", "iqr_harmless", "n_harmful", "n_harmless"
    }
    assert margins["n_harmful"] == 20
    assert margins["n_harmless"] == 20
    # With separable data (harmful shifted +2 in all dims), mean_gap should be positive
    assert margins["mean_gap"] > 0


def test_condition_margin_table():
    hmap = _make_hidden_map(n_harmful=10, n_harmless=10, dim=4, separable=True, seed=5)
    direction = np.ones(4) / 2.0
    table = condition_margin_table(
        condition_directions={"V-text": direction, "V-blank": direction},
        all_hidden={"V-text": hmap, "V-blank": hmap},
        layer=0,
        pos=0,
    )
    assert "V-text" in table
    assert "V-blank" in table
    assert table["V-text"]["n_harmful"] == 10


# ---------------------------------------------------------------------------
# report smoke test
# ---------------------------------------------------------------------------

def test_build_phase1p5_report_runs():
    ar = AuditResult(
        model_id="test_model",
        signal_type="harmfulness",
        visual_condition="V-text",
        train_auc=0.99,
        held_out_auc=0.95,
        group_split_auc=0.92,
        permutation_auc=0.51,
        cross_category_aucs={"cat_a": 0.88, "cat_b": 0.91},
        margins={
            "mean_gap": 1.5,
            "median_gap": 1.4,
            "iqr_harmful": 0.8,
            "iqr_harmless": 0.7,
            "n_harmful": 50,
            "n_harmless": 50,
        },
        static_transfer_margin_drop={"V-blank": 0.3, "V-noise": 0.4},
    )
    report = build_phase1p5_report([ar], model_id="test_model", signal_type="harmfulness")
    assert isinstance(report, str)
    assert "Phase 1.5" in report
    assert "V-text" in report
    assert "Permutation" in report
    assert "Cross-Category" in report
    assert "Margin" in report


def test_build_phase1p5_report_no_paired_ids():
    ar = AuditResult(
        model_id="m",
        signal_type="refusal",
        visual_condition="V-blank",
        train_auc=0.80,
        held_out_auc=0.75,
        group_split_auc=-1.0,   # sentinel for N/A
        permutation_auc=0.50,
        cross_category_aucs={},
        margins={},
        static_transfer_margin_drop={},
    )
    report = build_phase1p5_report([ar], model_id="m", signal_type="refusal")
    assert "N/A" in report


# ---------------------------------------------------------------------------
# NEW: nested permutation tests
# ---------------------------------------------------------------------------

def _make_multi_locus_hidden_map(
    n_harmful: int = 20,
    n_harmless: int = 20,
    dim: int = 8,
    n_layers: int = 3,
    n_pos: int = 2,
    seed: int = 0,
    separable: bool = True,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    hmap = {}
    for layer in range(n_layers):
        for pos in range(n_pos):
            if separable:
                harmful = rng.standard_normal((n_harmful, dim)) + 2.0
                harmless = rng.standard_normal((n_harmless, dim))
            else:
                harmful = rng.standard_normal((n_harmful, dim))
                harmless = rng.standard_normal((n_harmless, dim))
            hmap[(layer, pos)] = {"harmful": harmful, "harmless": harmless}
    return hmap


def test_nested_permutation_returns_stats_dict():
    """nested permutation_auc 返回包含 mean/std/p95/n_valid 的 dict"""
    hmap = _make_multi_locus_hidden_map(n_harmful=20, n_harmless=20, dim=8, seed=10)
    result = permutation_auc(hmap, n_permutations=20, seed=0)
    assert isinstance(result, dict)
    for key in ("mean", "std", "min", "max", "p95", "n_valid"):
        assert key in result, f"missing key: {key}"
    assert result["n_valid"] > 0


def test_nested_permutation_mean_near_half_separable():
    """完全可分数据下，nested permutation mean 接近 0.5（< 0.65）"""
    hmap = _make_multi_locus_hidden_map(
        n_harmful=30, n_harmless=30, dim=16, n_layers=2, n_pos=2,
        seed=7, separable=True,
    )
    result = permutation_auc(hmap, n_permutations=100, seed=42)
    assert isinstance(result, dict)
    assert result["n_valid"] > 0
    assert result["mean"] < 0.65, (
        f"Nested permutation mean too high ({result['mean']:.4f}) — locus selection leakage"
    )


def test_array_held_out_auc_train_selected_returns_dict():
    """_array_held_out_auc_train_selected 返回包含 held_out_auc/best_layer/best_pos 的 dict"""
    from experiments.mibd.audit.held_out import array_held_out_auc_train_selected

    hmap = _make_multi_locus_hidden_map(n_harmful=30, n_harmless=30, dim=8, seed=20)
    result = array_held_out_auc_train_selected(
        hmap, pos_label="harmful", neg_label="harmless", seed=0
    )
    assert isinstance(result, dict)
    for key in ("held_out_auc", "train_auc", "best_layer", "best_pos",
                "train_pos_n", "train_neg_n", "test_pos_n", "test_neg_n"):
        assert key in result, f"missing key: {key}"
    assert result["held_out_auc"] != -1.0
    assert result["best_layer"] >= 0
    assert result["best_pos"] >= 0


def test_held_out_auc_train_selected_less_than_full_data_on_small_data():
    """在样本量极小时，train-only locus 选出的 AUC 应 <= full-data locus AUC"""
    from experiments.mibd.audit.held_out import array_held_out_auc, array_held_out_auc_train_selected
    from experiments.mibd.probes.train import find_best_locus, train_probes_for_condition

    # Very small dataset to maximize selection leakage effect.
    hmap = _make_multi_locus_hidden_map(
        n_harmful=8, n_harmless=8, dim=4, n_layers=4, n_pos=3, seed=99, separable=True
    )

    # full-data locus selection
    probe_results = train_probes_for_condition(
        {k: {"harmful": v["harmful"], "harmless": v["harmless"]} for k, v in hmap.items()}
    )
    full_layer, full_pos = find_best_locus(probe_results)
    full_ho_auc = array_held_out_auc(
        hmap, full_layer, full_pos,
        pos_label="harmful", neg_label="harmless", seed=0
    )

    ts_result = array_held_out_auc_train_selected(
        hmap, pos_label="harmful", neg_label="harmless", seed=0
    )
    ts_auc = ts_result["held_out_auc"]

    # Both should be valid (not -1.0 sentinel).
    assert full_ho_auc != -1.0, "full-data held-out AUC should be computable"
    assert ts_auc != -1.0, "train-selected held-out AUC should be computable"
    # Train-only selection should not exceed full-data selection (no leakage advantage).
    assert ts_auc <= full_ho_auc + 0.05, (
        f"train-selected AUC ({ts_auc:.4f}) exceeds full-data AUC ({full_ho_auc:.4f}) "
        "by more than tolerance — unexpected"
    )


def test_phase1p5_report_no_negative_one():
    """报告中不出现 -1.0000 字符串"""
    ar = AuditResult(
        model_id="m",
        signal_type="harmfulness",
        visual_condition="V-text",
        train_auc=0.95,
        held_out_auc=0.90,
        group_split_auc=-1.0,
        permutation_auc=0.51,
        cross_category_aucs={"cat_a": 0.88},
        margins={
            "mean_gap": 1.2,
            "median_gap": 1.1,
            "iqr_harmful": 0.5,
            "iqr_harmless": 0.4,
            "n_harmful": 40,
            "n_harmless": 40,
        },
        static_transfer_margin_drop={},
        train_selected_locus=(2, 1),
        full_data_locus=(3, 0),
        held_out_auc_train_selected=0.88,
        permutation_stats={"mean": 0.51, "std": 0.04, "min": 0.43, "max": 0.59, "p95": 0.58, "n_valid": 50},
    )
    report = build_phase1p5_report([ar], model_id="m", signal_type="harmfulness")
    assert "-1.0000" not in report, f"report contains -1.0000:\n{report}"
