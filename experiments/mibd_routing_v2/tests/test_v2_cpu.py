"""CPU-only tests for the mibd_routing_v2 iteration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.mibd_routing_v2.data.matched_safe_images import (
    load_category_safe_pool,
    select_matched_safe_image,
)
from experiments.mibd_routing_v2.probes.dissociation import (
    compute_condition_dissociation,
    compute_dissociation,
    detection_direction_from_hidden,
)
from experiments.mibd_routing_v2.probes.subspace import (
    evaluate_subspace_readout,
    extract_risk_subspace,
)


# --------------------------------------------------------------------------- #
# dissociation
# --------------------------------------------------------------------------- #
def test_orthogonal_directions_are_dissociated() -> None:
    det = np.array([1.0, 0.0, 0.0])
    ctrl = np.array([0.0, 1.0, 0.0])
    score = compute_dissociation(det, ctrl)
    assert abs(score.cosine) < 1e-9
    assert abs(score.angle_degrees - 90.0) < 1e-6
    assert score.is_dissociated is True


def test_aligned_directions_not_dissociated() -> None:
    det = np.array([1.0, 1.0, 0.0])
    ctrl = np.array([2.0, 2.0, 0.0])
    score = compute_dissociation(det, ctrl)
    assert score.cosine > 0.99
    assert score.angle_degrees < 1.0
    assert score.is_dissociated is False


def test_antiparallel_treated_as_same_axis() -> None:
    det = np.array([1.0, 0.0])
    ctrl = np.array([-1.0, 0.0])
    score = compute_dissociation(det, ctrl)
    # opposite signs span the same axis -> angle 0, not 180
    assert score.angle_degrees < 1e-6


def test_detection_direction_from_hidden_is_unit() -> None:
    rng = np.random.default_rng(0)
    risk = rng.normal(loc=1.0, size=(20, 8))
    safe = rng.normal(loc=-1.0, size=(20, 8))
    direction = detection_direction_from_hidden(risk, safe)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-9


def test_condition_dissociation_only_shared_keys() -> None:
    det = {"FigStep": np.array([1.0, 0.0]), "V-real": np.array([1.0, 0.0])}
    ctrl = {"FigStep": np.array([0.0, 1.0]), "V-blank": np.array([1.0, 0.0])}
    table = compute_condition_dissociation(det, ctrl)
    assert set(table) == {"FigStep"}
    assert table["FigStep"].is_dissociated is True


# --------------------------------------------------------------------------- #
# subspace
# --------------------------------------------------------------------------- #
def test_extract_subspace_orthonormal() -> None:
    rng = np.random.default_rng(1)
    risk = rng.normal(loc=1.0, size=(40, 16))
    safe = rng.normal(loc=-1.0, size=(40, 16))
    directions = extract_risk_subspace(risk, safe, rank=3)
    assert directions.shape[0] <= 3
    gram = directions @ directions.T
    assert np.allclose(gram, np.eye(directions.shape[0]), atol=1e-6)


def test_subspace_single_direction_signal_is_well_behaved() -> None:
    # When the discriminative signal lies along a SINGLE mean-difference axis
    # under isotropic noise, the first diff-of-means direction is already
    # near-optimal; extra orthogonal directions only add noise. The subspace
    # AUC should therefore stay close to (not collapse below) the single one.
    rng = np.random.default_rng(2)
    n = 60
    dim = 12
    base = rng.normal(size=(n, dim))
    labels = np.array([1, 0] * (n // 2))
    signal = np.zeros((n, dim))
    signal[labels == 1, 0] = 2.0
    signal[labels == 1, 1] = 2.0
    hidden = base + signal
    report = evaluate_subspace_readout(
        labels=labels,
        risk_hidden=hidden[labels == 1],
        safe_hidden=hidden[labels == 0],
        pooled_hidden=hidden,
        rank=3,
    )
    assert 0.0 <= report.subspace_auc <= 1.0
    assert report.subspace_auc >= report.single_direction_auc - 0.05


def test_subspace_extraction_is_deterministic_and_bounded() -> None:
    # diff-of-means deflation is a first-moment method: we guarantee
    # determinism, orthonormality and bounded AUC, but NOT a synthetic
    # improvement (that depends on real residual mean structure).
    rng = np.random.default_rng(3)
    dim = 10
    per = 40
    risk = rng.normal(loc=0.0, scale=1.0, size=(per, dim))
    risk[:, 0] += 2.0
    risk[:, 3] += 1.0
    safe = rng.normal(loc=0.0, scale=1.0, size=(per, dim))
    pooled = np.vstack([risk, safe])
    labels = np.array([1] * per + [0] * per)

    report_a = evaluate_subspace_readout(
        labels=labels, risk_hidden=risk, safe_hidden=safe,
        pooled_hidden=pooled, rank=3,
    )
    report_b = evaluate_subspace_readout(
        labels=labels, risk_hidden=risk, safe_hidden=safe,
        pooled_hidden=pooled, rank=3,
    )
    assert 0.0 <= report_a.subspace_auc <= 1.0
    assert 0.0 <= report_a.single_direction_auc <= 1.0
    # deterministic given identical inputs
    assert report_a.subspace_auc == report_b.subspace_auc
    assert np.allclose(report_a.directions, report_b.directions)


# --------------------------------------------------------------------------- #
# matched safe images
# --------------------------------------------------------------------------- #
def test_category_matched_selection(tmp_path: Path) -> None:
    (tmp_path / "weapons").mkdir()
    (tmp_path / "cyber").mkdir()
    (tmp_path / "weapons" / "a.png").write_bytes(b"x")
    (tmp_path / "weapons" / "b.png").write_bytes(b"x")
    (tmp_path / "cyber" / "c.png").write_bytes(b"x")
    (tmp_path / "global.png").write_bytes(b"x")

    pool = load_category_safe_pool(tmp_path)
    assert pool.by_category.keys() == {"weapons", "cyber"}
    assert len(pool.global_pool) == 1

    path, mode = select_matched_safe_image(pool, "weapons", index=0)
    assert mode == "category_matched"
    assert path.endswith("a.png")

    path, mode = select_matched_safe_image(pool, "self_harm", index=0)
    assert mode == "global_fallback"
    assert path.endswith("global.png")


def test_empty_pool_returns_none() -> None:
    pool = load_category_safe_pool(None)
    assert pool.is_empty()
    path, mode = select_matched_safe_image(pool, "weapons", index=0)
    assert path is None
    assert mode == "none"
