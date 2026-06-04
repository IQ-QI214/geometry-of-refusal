from __future__ import annotations

import numpy as np

from experiments.mibd.probes.direction import mean_difference_direction, project_scores
from experiments.mibd.probes.metrics import binary_auc
from experiments.mibd.probes.train import find_best_locus, train_probes_for_condition


def array_held_out_auc(
    vc_hidden: dict[tuple[int, int], dict[str, np.ndarray]],
    layer: int,
    pos: int,
    pos_label: str,
    neg_label: str,
    seed: int,
    test_frac: float = 0.2,
) -> float:
    """Held-out AUC at a fixed (layer, pos) locus from pre-extracted arrays.

    Splits pos/neg arrays into train/test, trains probe on train, evaluates on test.
    The locus is fixed (caller-supplied) — selection leakage is the caller's concern.
    Returns -1.0 (N/A sentinel) if the split is too small.
    """
    lp = vc_hidden.get((layer, pos))
    if lp is None:
        return -1.0
    pos_arr = lp.get(pos_label)
    neg_arr = lp.get(neg_label)
    if pos_arr is None or neg_arr is None:
        return -1.0

    rng = np.random.default_rng(seed)
    n_test_pos = min(max(1, round(len(pos_arr) * test_frac)), len(pos_arr) - 1)
    n_test_neg = min(max(1, round(len(neg_arr) * test_frac)), len(neg_arr) - 1)

    pos_idx = rng.permutation(len(pos_arr))
    neg_idx = rng.permutation(len(neg_arr))

    train_pos = pos_arr[pos_idx[n_test_pos:]]
    train_neg = neg_arr[neg_idx[n_test_neg:]]
    test_pos  = pos_arr[pos_idx[:n_test_pos]]
    test_neg  = neg_arr[neg_idx[:n_test_neg]]

    if len(train_pos) < 2 or len(train_neg) < 2 or len(test_pos) < 1 or len(test_neg) < 1:
        return -1.0

    direction = mean_difference_direction(train_pos, train_neg)
    test_all = np.vstack([test_pos, test_neg])
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))
    return float(binary_auc(test_labels, project_scores(test_all, direction)))


def array_held_out_auc_train_selected(
    vc_hidden: dict[tuple[int, int], dict[str, np.ndarray]],
    pos_label: str,
    neg_label: str,
    seed: int,
    test_frac: float = 0.2,
) -> dict:
    """Train-only locus selection held-out AUC.

    1. Stratified train/test split per locus (pos and neg split separately).
    2. Scan all (layer, pos) on train-only hidden states to select best locus.
    3. Evaluate on test with the train-selected locus direction.

    Returns dict with held_out_auc, train_auc, best_layer, best_pos,
    train_pos_n, train_neg_n, test_pos_n, test_neg_n.
    Returns held_out_auc=-1.0 if split is not computable.
    """
    _fail: dict = {
        "held_out_auc": -1.0,
        "train_auc": -1.0,
        "best_layer": -1,
        "best_pos": -1,
        "train_pos_n": 0,
        "train_neg_n": 0,
        "test_pos_n": 0,
        "test_neg_n": 0,
    }
    if not vc_hidden:
        return _fail

    rng = np.random.default_rng(seed)

    # Derive consistent split indices from the first valid locus.
    first_key = next(iter(vc_hidden))
    first_lp = vc_hidden[first_key]
    pos_arr_ref = first_lp.get(pos_label)
    neg_arr_ref = first_lp.get(neg_label)
    if pos_arr_ref is None or neg_arr_ref is None:
        return _fail

    n_pos = len(pos_arr_ref)
    n_neg = len(neg_arr_ref)
    n_test_pos = min(max(1, round(n_pos * test_frac)), n_pos - 1)
    n_test_neg = min(max(1, round(n_neg * test_frac)), n_neg - 1)

    if n_pos - n_test_pos < 2 or n_neg - n_test_neg < 2:
        return _fail

    pos_idx = rng.permutation(n_pos)
    neg_idx = rng.permutation(n_neg)
    train_pos_global = pos_idx[n_test_pos:]
    test_pos_global = pos_idx[:n_test_pos]
    train_neg_global = neg_idx[n_test_neg:]
    test_neg_global = neg_idx[:n_test_neg]

    train_hidden_map: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    test_hidden_map: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    for key, lp in vc_hidden.items():
        p = lp.get(pos_label)
        n = lp.get(neg_label)
        if p is None or n is None:
            continue
        tp_idx = train_pos_global[train_pos_global < len(p)]
        tn_idx = train_neg_global[train_neg_global < len(n)]
        ep_idx = test_pos_global[test_pos_global < len(p)]
        en_idx = test_neg_global[test_neg_global < len(n)]
        if len(tp_idx) < 2 or len(tn_idx) < 2 or len(ep_idx) < 1 or len(en_idx) < 1:
            continue
        train_hidden_map[key] = {pos_label: p[tp_idx], neg_label: n[tn_idx]}
        test_hidden_map[key] = {pos_label: p[ep_idx], neg_label: n[en_idx]}

    if not train_hidden_map:
        return _fail

    probe_results = train_probes_for_condition(train_hidden_map, pos_label=pos_label, neg_label=neg_label)
    if not probe_results:
        return _fail

    best_key = find_best_locus(probe_results)
    best_layer, best_pos = best_key
    direction = probe_results[best_key]["direction"]
    train_auc = float(probe_results[best_key]["auc"])

    test_lp = test_hidden_map.get(best_key)
    if test_lp is None:
        return _fail

    test_pos_arr = test_lp["harmful"]
    test_neg_arr = test_lp["harmless"]
    test_all = np.vstack([test_pos_arr, test_neg_arr])
    test_labels = np.array([1] * len(test_pos_arr) + [0] * len(test_neg_arr))
    held_out_auc = float(binary_auc(test_labels, project_scores(test_all, direction)))

    train_lp = train_hidden_map[best_key]
    return {
        "held_out_auc": held_out_auc,
        "train_auc": train_auc,
        "best_layer": best_layer,
        "best_pos": best_pos,
        "train_pos_n": len(train_lp["harmful"]),
        "train_neg_n": len(train_lp["harmless"]),
        "test_pos_n": len(test_pos_arr),
        "test_neg_n": len(test_neg_arr),
    }
