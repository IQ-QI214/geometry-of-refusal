"""Offline round-0 oracle calibration for CaRoB (CPU-only, numpy).

This is the local, GPU-free counterpart of the experiment matrix's round 0
(防御方法升级方案_CaRoB_MoE融合_20260630.md §七). It does *not* run a model or
generate text; it operates on already-extracted hidden states and exercises the
full numpy routing path -- probe bank -> per-layer scores ->
``router.aggregate_risk_score`` -> ``gate`` -- to produce:

* O1: fixed single-layer baseline AUC.
* O2: oracle layer-selection AUC (upper bound of "pick the right layer").
* Cross-carrier transfer matrix (the empirical motivation signal: does a probe
  fit on one carrier degrade on another?).
* A gate threshold sweep reporting harmful activation vs benign leakage.

Run::

    python3 -m experiments.mibd_routing_v2.eval.run_offline_oracle \
        --npz results/mibd_routing_v2/sensor_probe/qwen3_vl_8b_loci10/hidden_states.npz \
        --out results/mibd_routing_v2/offline_oracle/qwen3_vl_8b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mibd.probes.metrics import binary_auc
from experiments.mibd_routing_v2.eval.load_hidden_states import (
    available_carriers,
    available_layers,
    build_carrier_feature_matrix,
    load_npz,
)
from experiments.mibd_routing_v2.routing.gate import hard_gate
from experiments.mibd_routing_v2.routing.probe_bank import (
    best_layer,
    fit_probe_bank,
    layer_auc,
    score_samples,
)
from experiments.mibd_routing_v2.routing.router import aggregate_risk_score


def per_layer_score_matrix(
    bank: dict,
    states: dict[str, np.ndarray],
    test_carrier: str,
    shared_layers: list[int],
    position: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(n, L)`` per-layer risk scores and ``(n,)`` labels for a carrier.

    Column ``j`` holds ``bank[shared_layers[j]]`` scores on the test carrier's
    layer-``shared_layers[j]`` features. All layers must share the sample order,
    which holds because ``build_carrier_feature_matrix`` is deterministic.
    """
    columns = []
    labels_ref = None
    for layer in shared_layers:
        feats, labels = build_carrier_feature_matrix(states, test_carrier, layer, position)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError("label order mismatch across layers")
        columns.append(score_samples(bank[layer], feats))
    score_matrix = np.stack(columns, axis=1)
    return score_matrix, labels_ref


def _one_hot_layer_probs(n: int, n_layers: int, col: int) -> np.ndarray:
    probs = np.zeros((n, n_layers))
    probs[:, col] = 1.0
    return probs


def gate_threshold_sweep(
    risk_scores: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 21,
) -> list[dict]:
    """Sweep hard-gate thresholds; report harmful activation vs benign leak."""
    s = np.asarray(risk_scores, dtype=np.float64)
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        hi = lo + 1.0
    taus = np.linspace(lo, hi, num_thresholds)
    harmful = labels == 1
    benign = labels == 0
    sweep = []
    for tau in taus:
        g = hard_gate(s, tau=float(tau))
        harmful_rate = float(g[harmful].mean()) if harmful.any() else 0.0
        benign_leak = float(g[benign].mean()) if benign.any() else 0.0
        sweep.append(
            {
                "tau": round(float(tau), 6),
                "harmful_activation_rate": round(harmful_rate, 6),
                "benign_leak_rate": round(benign_leak, 6),
            }
        )
    return sweep


def build_offline_report(
    states: dict[str, np.ndarray],
    model_name: str,
    fixed_layer: int | None = None,
    position: int = -1,
) -> dict:
    """Assemble the round-0 offline report dict (JSON-serializable)."""
    carriers = available_carriers(states)
    if len(carriers) < 1:
        raise ValueError("no carriers found in states")

    # Cross-carrier transfer matrix (train_carrier -> test_carrier), per layer
    transfer_matrix: dict[str, dict[str, dict[str, float]]] = {}
    o1_o2: dict[str, dict[str, dict[str, float]]] = {}
    gate_sweeps: dict[str, list[dict]] = {}

    for train_carrier in carriers:
        bank = fit_probe_bank(states, train_carrier, position)
        transfer_matrix[train_carrier] = {}
        o1_o2[train_carrier] = {}
        for test_carrier in carriers:
            shared = sorted(set(bank) & set(available_layers(states, test_carrier)))
            score_matrix, labels = per_layer_score_matrix(
                bank, states, test_carrier, shared, position
            )
            n, n_layers = score_matrix.shape

            per_layer_int = {
                layer: float(
                    layer_auc(
                        bank[layer],
                        *build_carrier_feature_matrix(states, test_carrier, layer, position),
                    )
                )
                for layer in shared
            }
            per_layer = {str(layer): round(auc, 6) for layer, auc in per_layer_int.items()}
            transfer_matrix[train_carrier][test_carrier] = per_layer

            # O1: fixed single layer (default: median shared layer)
            fixed = fixed_layer if fixed_layer in shared else shared[len(shared) // 2]
            fixed_col = shared.index(fixed)
            o1_probs = _one_hot_layer_probs(n, n_layers, fixed_col)
            o1_scores = aggregate_risk_score(o1_probs, score_matrix)
            o1_auc = binary_auc(labels, o1_scores)

            # O2: oracle layer (best AUC on this test carrier)
            oracle_layer = best_layer(per_layer_int)
            oracle_col = shared.index(oracle_layer)
            o2_probs = _one_hot_layer_probs(n, n_layers, oracle_col)
            o2_scores = aggregate_risk_score(o2_probs, score_matrix)
            o2_auc = binary_auc(labels, o2_scores)

            o1_o2[train_carrier][test_carrier] = {
                "fixed_layer": int(fixed),
                "o1_fixed_auc": round(float(o1_auc), 6),
                "oracle_layer": int(oracle_layer),
                "o2_oracle_auc": round(float(o2_auc), 6),
                "oracle_gain": round(float(o2_auc - o1_auc), 6),
            }

            # Gate sweep only for the within-carrier diagonal, oracle layer
            if train_carrier == test_carrier:
                gate_sweeps[train_carrier] = gate_threshold_sweep(o2_scores, labels)

    # Headline: average within-carrier vs cross-carrier oracle AUC
    within, cross = [], []
    for tr in carriers:
        for te in carriers:
            auc = o1_o2[tr][te]["o2_oracle_auc"]
            (within if tr == te else cross).append(auc)
    summary = {
        "model": model_name,
        "carriers": carriers,
        "mean_within_carrier_oracle_auc": round(float(np.mean(within)), 6) if within else None,
        "mean_cross_carrier_oracle_auc": round(float(np.mean(cross)), 6) if cross else None,
        "cross_carrier_transfer_drop": (
            round(float(np.mean(within) - np.mean(cross)), 6)
            if within and cross
            else None
        ),
    }
    return {
        "summary": summary,
        "o1_o2": o1_o2,
        "transfer_matrix": transfer_matrix,
        "gate_sweeps": gate_sweeps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline round-0 oracle calibration (CPU).")
    parser.add_argument("--npz", required=True, help="path to hidden_states.npz")
    parser.add_argument("--out", required=True, help="path to write the JSON report")
    parser.add_argument("--model", default=None, help="model name label (defaults to npz parent dir)")
    parser.add_argument("--fixed-layer", type=int, default=None, help="layer for O1 baseline")
    parser.add_argument("--position", type=int, default=-1, help="token position key")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    states = load_npz(args.npz)
    model = args.model or Path(args.npz).parent.name
    report = build_offline_report(
        states, model_name=model, fixed_layer=args.fixed_layer, position=args.position
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[offline-oracle] wrote {out_path}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
