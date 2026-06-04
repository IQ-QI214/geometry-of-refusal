from __future__ import annotations

import numpy as np

from experiments.mibd.probes.direction import mean_difference_direction, project_scores
from experiments.mibd.probes.metrics import binary_auc
from experiments.mibd.probes.train import find_best_locus, train_probes_for_condition

_UNSET = object()


def permutation_auc(
    hidden_map: dict[tuple[int, int], dict[str, np.ndarray]],
    layer: object = _UNSET,
    pos: object = _UNSET,
    n_permutations: int = 100,
    seed: int = 42,
    test_frac: float = 0.2,
    pos_label: str = "harmful",
    neg_label: str = "harmless",
) -> float | dict:
    """Permutation test with train/test separation.

    Called with layer/pos: legacy single-locus mode, returns float (mean AUC).
    Called without layer/pos: nested permutation — each permutation re-scans all
    loci on shuffled train labels to select best locus, then evaluates on real test
    labels. Returns a stats dict.

    Nested permutation stats dict keys:
        mean, std, min, max, p95, n_valid
    """
    if layer is not _UNSET and pos is not _UNSET:
        return _single_locus_permutation(
            hidden_map,
            layer=int(layer),
            pos=int(pos),
            n_permutations=n_permutations,
            seed=seed,
            test_frac=test_frac,
            pos_label=pos_label,
            neg_label=neg_label,
        )
    return _nested_permutation(
        hidden_map,
        n_permutations=n_permutations,
        seed=seed,
        test_frac=test_frac,
        pos_label=pos_label,
        neg_label=neg_label,
    )


def _single_locus_permutation(
    hidden_map: dict[tuple[int, int], dict[str, np.ndarray]],
    layer: int,
    pos: int,
    n_permutations: int,
    seed: int,
    test_frac: float,
    pos_label: str,
    neg_label: str,
) -> float:
    lp_map = hidden_map.get((layer, pos))
    if lp_map is None:
        raise KeyError(f"(layer={layer}, pos={pos}) not found in hidden_map")

    pos_arr = lp_map.get(pos_label)
    neg_arr = lp_map.get(neg_label)
    if pos_arr is None or neg_arr is None:
        raise ValueError(f"hidden_map entry must have both '{pos_label}' and '{neg_label}' keys")

    all_hidden = np.vstack([pos_arr, neg_arr])
    labels = np.array([1] * len(pos_arr) + [0] * len(neg_arr))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    n_test = max(1, round(len(labels) * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    test_hidden = all_hidden[test_idx]
    test_labels = labels[test_idx]
    train_hidden = all_hidden[train_idx]
    train_labels = labels[train_idx]

    if len(np.unique(test_labels)) < 2:
        return float("nan")

    aucs = []
    for _ in range(n_permutations):
        shuffled_train = rng.permutation(train_labels)
        perm_pos = train_hidden[shuffled_train == 1]
        perm_neg = train_hidden[shuffled_train == 0]
        if len(perm_pos) < 2 or len(perm_neg) < 2:
            continue
        direction = mean_difference_direction(perm_pos, perm_neg)
        scores = project_scores(test_hidden, direction)
        aucs.append(binary_auc(test_labels, scores))

    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _nested_permutation(
    hidden_map: dict[tuple[int, int], dict[str, np.ndarray]],
    n_permutations: int,
    seed: int,
    test_frac: float,
    pos_label: str,
    neg_label: str,
) -> dict:
    """Nested permutation: each iteration re-selects locus on shuffled train data."""
    _nan_result = {
        "mean": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "p95": float("nan"),
        "n_valid": 0,
    }

    all_keys = list(hidden_map.keys())
    if not all_keys:
        return _nan_result

    first_lp = hidden_map[all_keys[0]]
    pos_ref = first_lp.get(pos_label)
    neg_ref = first_lp.get(neg_label)
    if pos_ref is None or neg_ref is None:
        return _nan_result

    n_pos = len(pos_ref)
    n_neg = len(neg_ref)

    rng = np.random.default_rng(seed)

    n_test_pos = max(1, round(n_pos * test_frac))
    n_test_neg = max(1, round(n_neg * test_frac))
    if n_pos - n_test_pos < 2 or n_neg - n_test_neg < 2:
        return _nan_result

    pos_idx = rng.permutation(n_pos)
    neg_idx = rng.permutation(n_neg)
    test_pos_idx = pos_idx[:n_test_pos]
    train_pos_idx = pos_idx[n_test_pos:]
    test_neg_idx = neg_idx[:n_test_neg]
    train_neg_idx = neg_idx[n_test_neg:]

    # Build fixed test hidden map (real labels).
    test_hidden_map: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for key, lp in hidden_map.items():
        h = lp.get(pos_label)
        hn = lp.get(neg_label)
        if h is None or hn is None:
            continue
        tp_idx = test_pos_idx[test_pos_idx < len(h)]
        tn_idx = test_neg_idx[test_neg_idx < len(hn)]
        if len(tp_idx) == 0 or len(tn_idx) == 0:
            continue
        test_hidden_map[key] = {
            pos_label: h[tp_idx],
            neg_label: hn[tn_idx],
        }

    has_both = any(
        len(v[pos_label]) >= 1 and len(v[neg_label]) >= 1
        for v in test_hidden_map.values()
    )
    if not has_both:
        return _nan_result

    aucs: list[float] = []

    for _ in range(n_permutations):
        n_train_pos = len(train_pos_idx)
        n_train_neg = len(train_neg_idx)
        n_train = n_train_pos + n_train_neg
        shuffled = rng.permutation(n_train)
        shuf_pos_local = shuffled[:n_train_pos]
        shuf_neg_local = shuffled[n_train_pos:]

        perm_train_map: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for key, lp in hidden_map.items():
            h = lp.get(pos_label)
            hn = lp.get(neg_label)
            if h is None or hn is None:
                continue
            tp_idx = train_pos_idx[train_pos_idx < len(h)]
            tn_idx = train_neg_idx[train_neg_idx < len(hn)]
            if len(tp_idx) < 1 or len(tn_idx) < 1:
                continue
            train_arr = np.vstack([h[tp_idx], hn[tn_idx]])
            n_actual = len(train_arr)
            valid_pos = shuf_pos_local[shuf_pos_local < n_actual]
            valid_neg = shuf_neg_local[shuf_neg_local < n_actual]
            if len(valid_pos) < 2 or len(valid_neg) < 2:
                continue
            perm_train_map[key] = {
                pos_label: train_arr[valid_pos],
                neg_label: train_arr[valid_neg],
            }

        if not perm_train_map:
            continue

        try:
            probe_results = train_probes_for_condition(
                perm_train_map, pos_label=pos_label, neg_label=neg_label
            )
            if not probe_results:
                continue
            best_key = find_best_locus(probe_results)
            direction = probe_results[best_key]["direction"]
        except (ValueError, KeyError):
            continue

        test_lp = test_hidden_map.get(best_key)
        if test_lp is None:
            continue
        test_h = test_lp[pos_label]
        test_hn = test_lp[neg_label]
        if len(test_h) < 1 or len(test_hn) < 1:
            continue
        test_all = np.vstack([test_h, test_hn])
        test_labels = np.array([1] * len(test_h) + [0] * len(test_hn))
        try:
            auc = binary_auc(test_labels, project_scores(test_all, direction))
            aucs.append(auc)
        except Exception:
            continue

    if not aucs:
        return _nan_result

    arr = np.array(aucs)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "n_valid": len(aucs),
    }
