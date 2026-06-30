"""Load extracted VLM hidden states for offline (CPU) routing analysis.

The sensor-probe extraction stage (GPU) saves per-(carrier, layer, label)
hidden-state matrices as a single ``.npz`` per model under
``results/mibd_routing_v2/sensor_probe/<model>/hidden_states.npz``. Keys follow
the pattern ``{carrier}__layer{L}__pos{P}__{label}`` (e.g.
``FigStep__layer12__pos-1__harmful``).

This module is the CPU-side reader that turns those arrays into feature/label
matrices the numpy routing core can consume, with a leakage-free stratified
train/test split. numpy-only; no torch, no model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_HARMFUL = "harmful"
_HARMLESS = "harmless"


@dataclass(frozen=True)
class CarrierSplit:
    """A stratified train/test split for one (carrier, layer) feature matrix."""

    train_features: np.ndarray  # (n_train, d)
    train_labels: np.ndarray  # (n_train,) 1=harmful, 0=harmless
    test_features: np.ndarray  # (n_test, d)
    test_labels: np.ndarray  # (n_test,)


def load_npz(path: str) -> dict[str, np.ndarray]:
    """Load the hidden-state archive as a plain dict of arrays.

    The ``manifest_json`` scalar entry, if present, is dropped so that callers
    only see numeric feature matrices.
    """
    archive = np.load(path, allow_pickle=True)
    return {key: archive[key] for key in archive.files if key != "manifest_json"}


def _key(carrier: str, layer: int, label: str, position: int = -1) -> str:
    return f"{carrier}__layer{layer}__pos{position}__{label}"


def available_carriers(states: dict[str, np.ndarray]) -> list[str]:
    return sorted({key.split("__", 1)[0] for key in states})


def available_layers(states: dict[str, np.ndarray], carrier: str) -> list[int]:
    """Layers present for ``carrier`` (sorted ascending)."""
    prefix = f"{carrier}__layer"
    layers = set()
    for key in states:
        if key.startswith(prefix):
            layer_token = key[len(prefix):].split("__", 1)[0]
            layers.add(int(layer_token))
    if not layers:
        raise ValueError(f"carrier {carrier!r} not found in states")
    return sorted(layers)


def build_carrier_feature_matrix(
    states: dict[str, np.ndarray],
    carrier: str,
    layer: int,
    position: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack harmful/harmless into ``(features, labels)``.

    Labels: 1 for harmful, 0 for harmless. Order is harmful rows then harmless
    rows (callers that need shuffling should use ``split_train_test``).
    """
    harmful_key = _key(carrier, layer, _HARMFUL, position)
    harmless_key = _key(carrier, layer, _HARMLESS, position)
    if harmful_key not in states or harmless_key not in states:
        raise ValueError(
            f"missing keys for carrier={carrier} layer={layer} pos={position}"
        )
    harmful = np.asarray(states[harmful_key], dtype=np.float64)
    harmless = np.asarray(states[harmless_key], dtype=np.float64)
    features = np.concatenate([harmful, harmless], axis=0)
    labels = np.concatenate(
        [np.ones(harmful.shape[0]), np.zeros(harmless.shape[0])]
    ).astype(np.int64)
    return features, labels


def split_train_test(
    features: np.ndarray,
    labels: np.ndarray,
    frac_train: float = 0.5,
    seed: int = 0,
) -> CarrierSplit:
    """Leakage-free *stratified* split: each class split independently.

    ``frac_train`` of each class goes to train, the rest to test. The same
    ``seed`` yields the same split. At least one sample per class is kept on
    each side when the class has >= 2 samples.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels length mismatch")
    if not 0.0 < frac_train < 1.0:
        raise ValueError(f"frac_train must be in (0, 1), got {frac_train}")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in (0, 1):
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size == 0:
            raise ValueError(f"class {cls} has no samples")
        rng.shuffle(cls_idx)
        n_train = int(round(frac_train * cls_idx.size))
        if cls_idx.size >= 2:
            n_train = min(max(n_train, 1), cls_idx.size - 1)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    train_idx_arr = np.array(sorted(train_idx), dtype=np.int64)
    test_idx_arr = np.array(sorted(test_idx), dtype=np.int64)
    return CarrierSplit(
        train_features=features[train_idx_arr],
        train_labels=labels[train_idx_arr],
        test_features=features[test_idx_arr],
        test_labels=labels[test_idx_arr],
    )
