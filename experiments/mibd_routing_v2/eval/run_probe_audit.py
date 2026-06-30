"""Leakage-aware probe audit for extracted hidden states (CPU-only).

This module exists because the original ``probe_summary`` and offline oracle
fit a mean-difference probe and evaluate it on the same hidden-state rows. In a
high-dimensional setting that metric is useful as a smoke check, but it is not a
valid Go/No-Go gate. This audit reports held-out and permutation baselines so
early-layer AUC saturation can be interpreted as evidence rather than an
artifact of same-set evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mibd.probes.direction import mean_difference_direction, project_scores
from experiments.mibd.probes.metrics import binary_auc
from experiments.mibd_routing_v2.eval.load_hidden_states import (
    available_carriers,
    available_layers,
    build_carrier_feature_matrix,
    load_npz,
    split_train_test,
)


def _fit_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return mean_difference_direction(features[labels == 1], features[labels == 0])


def _auc_for_direction(direction: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    return float(binary_auc(labels, project_scores(features, direction)))


def _round(value: float) -> float:
    return round(float(value), 6)


def evaluate_within_carrier_layer(
    features: np.ndarray,
    labels: np.ndarray,
    frac_train: float,
    seed: int,
    permutation_seed: int,
) -> dict:
    """Return same-set, held-out, and permutation held-out AUC for one layer."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    same_direction = _fit_direction(features, labels)
    same_set_auc = _auc_for_direction(same_direction, features, labels)

    split = split_train_test(features, labels, frac_train=frac_train, seed=seed)
    heldout_direction = _fit_direction(split.train_features, split.train_labels)
    heldout_auc = _auc_for_direction(
        heldout_direction,
        split.test_features,
        split.test_labels,
    )

    rng = np.random.default_rng(permutation_seed)
    permuted = labels.copy()
    rng.shuffle(permuted)
    perm_split = split_train_test(features, permuted, frac_train=frac_train, seed=seed)
    perm_direction = _fit_direction(perm_split.train_features, perm_split.train_labels)
    permutation_heldout_auc = _auc_for_direction(
        perm_direction,
        perm_split.test_features,
        perm_split.test_labels,
    )

    return {
        "n_samples": int(features.shape[0]),
        "hidden_dim": int(features.shape[1]),
        "same_set_auc": _round(same_set_auc),
        "heldout_auc": _round(heldout_auc),
        "permutation_heldout_auc": _round(permutation_heldout_auc),
    }


def build_probe_audit_report(
    states: dict[str, np.ndarray],
    model_name: str,
    frac_train: float = 0.5,
    seed: int = 20260630,
    permutation_seed: int = 20260701,
    position: int = -1,
) -> dict:
    """Build a leakage-aware audit report for extracted hidden states."""
    carriers = available_carriers(states)
    within: dict[str, dict[str, dict]] = {}
    cross: dict[str, dict[str, dict[str, float]]] = {}

    for carrier in carriers:
        within[carrier] = {}
        for layer in available_layers(states, carrier):
            features, labels = build_carrier_feature_matrix(states, carrier, layer, position)
            within[carrier][str(layer)] = evaluate_within_carrier_layer(
                features,
                labels,
                frac_train=frac_train,
                seed=seed + layer,
                permutation_seed=permutation_seed + layer,
            )

    for train_carrier in carriers:
        cross[train_carrier] = {}
        for test_carrier in carriers:
            if train_carrier == test_carrier:
                continue
            shared = sorted(
                set(available_layers(states, train_carrier))
                & set(available_layers(states, test_carrier))
            )
            cross[train_carrier][test_carrier] = {}
            for layer in shared:
                train_features, train_labels = build_carrier_feature_matrix(
                    states, train_carrier, layer, position
                )
                test_features, test_labels = build_carrier_feature_matrix(
                    states, test_carrier, layer, position
                )
                direction = _fit_direction(train_features, train_labels)
                cross_auc = _auc_for_direction(direction, test_features, test_labels)
                cross[train_carrier][test_carrier][str(layer)] = _round(cross_auc)

    within_values = [
        row["heldout_auc"]
        for carrier_rows in within.values()
        for row in carrier_rows.values()
    ]
    perm_values = [
        row["permutation_heldout_auc"]
        for carrier_rows in within.values()
        for row in carrier_rows.values()
    ]
    cross_values = [
        auc
        for train_rows in cross.values()
        for test_rows in train_rows.values()
        for auc in test_rows.values()
    ]
    mean_within = float(np.mean(within_values)) if within_values else float("nan")
    mean_cross = float(np.mean(cross_values)) if cross_values else float("nan")

    return {
        "summary": {
            "model": model_name,
            "carriers": carriers,
            "frac_train": frac_train,
            "seed": seed,
            "permutation_seed": permutation_seed,
            "mean_within_heldout_auc": _round(mean_within),
            "mean_cross_carrier_auc": _round(mean_cross),
            "heldout_cross_carrier_transfer_drop": _round(mean_within - mean_cross),
            "mean_permutation_heldout_auc": _round(float(np.mean(perm_values)))
            if perm_values
            else None,
        },
        "within_carrier": within,
        "cross_carrier": cross,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware probe audit (CPU).")
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--frac-train", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--permutation-seed", type=int, default=20260701)
    parser.add_argument("--position", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    states = load_npz(str(args.npz))
    report = build_probe_audit_report(
        states,
        model_name=args.model or args.npz.parent.name,
        frac_train=args.frac_train,
        seed=args.seed,
        permutation_seed=args.permutation_seed,
        position=args.position,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[probe-audit] wrote {args.out}")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
