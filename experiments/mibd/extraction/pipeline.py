from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from experiments.mibd.data.schema import MIBDSample

if TYPE_CHECKING:  # torch-backed adapters are only needed for type hints
    from experiments.mibd.models.adapters import MIBDModelAdapter


def run_extraction(
    adapter: MIBDModelAdapter,
    samples: Sequence[MIBDSample],
    layers: tuple[int, ...],
    token_positions: tuple[int, ...],
    seed: int = 0,
) -> dict[str, dict[tuple[int, int], dict[str, np.ndarray]]]:
    """
    Extract hidden states for all samples across visual conditions.
    Returns: {visual_condition: {(layer, pos): {"harmful": array, "harmless": array}}}
    """
    storage: dict[str, dict[tuple[int, int], dict[str, list[np.ndarray]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total = len(samples)
    log_interval = max(1, total // 20)  # print ~20 progress updates

    for i, sample in enumerate(samples):
        if i % log_interval == 0 or i == total - 1:
            print(f"[extraction] {i+1}/{total}  vc={sample.visual_condition}  label={sample.label}", flush=True)
        image = adapter.build_image_for_condition(sample, seed=seed + i)
        inputs = adapter.prepare_inputs(sample, image)
        hidden_map = adapter.extract_hidden(inputs, layers, token_positions)

        for (layer_idx, pos), vec in hidden_map.items():
            storage[sample.visual_condition][(layer_idx, pos)][sample.label].append(vec)

    print(f"[extraction] done — {total} samples processed", flush=True)
    return _consolidate_storage(storage)


def run_extraction_with_metadata(
    adapter: MIBDModelAdapter,
    samples: Sequence[MIBDSample],
    layers: tuple[int, ...],
    token_positions: tuple[int, ...],
    seed: int = 0,
    row_meta_fn: Callable[[MIBDSample, int], dict] | None = None,
) -> tuple[
    dict[str, dict[tuple[int, int], dict[str, np.ndarray]]],
    dict[str, dict[tuple[int, int], dict[str, list[dict]]]],
]:
    """Like :func:`run_extraction` but also returns per-row metadata.

    The v3 audit could not do pair-level splits because the npz stored no row
    identity. This variant records, for every ``(visual_condition, (layer, pos),
    label)`` group, the metadata of each stacked row **in the exact stacking
    order**, so a later audit can recover which dataset row each hidden-state
    vector came from.

    ``row_meta_fn(sample, global_index) -> dict`` customises what is recorded.
    The default records id / paired_id / label / visual_condition / category /
    source / image_path plus the global dataset row index.

    Returns ``(hidden, row_metadata)`` where ``row_metadata`` mirrors the hidden
    structure but holds a list of metadata dicts instead of a stacked array.
    """
    if row_meta_fn is None:
        def row_meta_fn(sample: MIBDSample, index: int) -> dict:  # noqa: E306
            return {
                "row_index": index,
                "sample_id": sample.id,
                "paired_id": sample.paired_id,
                "label": sample.label,
                "visual_condition": sample.visual_condition,
                "category": sample.category,
                "source": sample.source,
                "image_path": sample.image_path,
            }

    storage: dict[str, dict[tuple[int, int], dict[str, list[np.ndarray]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta_storage: dict[str, dict[tuple[int, int], dict[str, list[dict]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total = len(samples)
    log_interval = max(1, total // 20)

    for i, sample in enumerate(samples):
        if i % log_interval == 0 or i == total - 1:
            print(f"[extraction] {i+1}/{total}  vc={sample.visual_condition}  label={sample.label}", flush=True)
        image = adapter.build_image_for_condition(sample, seed=seed + i)
        inputs = adapter.prepare_inputs(sample, image)
        hidden_map = adapter.extract_hidden(inputs, layers, token_positions)

        meta = row_meta_fn(sample, i)
        for (layer_idx, pos), vec in hidden_map.items():
            storage[sample.visual_condition][(layer_idx, pos)][sample.label].append(vec)
            meta_storage[sample.visual_condition][(layer_idx, pos)][sample.label].append(meta)

    print(f"[extraction] done — {total} samples processed", flush=True)
    return _consolidate_storage(storage), _plain_meta(meta_storage)


def _plain_meta(meta_storage: dict) -> dict:
    """Convert nested defaultdicts to plain dicts (JSON-friendly, order preserved)."""
    result: dict[str, dict[tuple[int, int], dict[str, list[dict]]]] = {}
    for vc, lp_map in meta_storage.items():
        result[vc] = {}
        for lp, label_map in lp_map.items():
            result[vc][lp] = {label: list(rows) for label, rows in label_map.items()}
    return result



def _consolidate_storage(
    storage: dict,
) -> dict[str, dict[tuple[int, int], dict[str, np.ndarray]]]:
    result: dict[str, dict[tuple[int, int], dict[str, np.ndarray]]] = {}
    for vc, lp_map in storage.items():
        result[vc] = {}
        for (l, p), label_map in lp_map.items():
            result[vc][(l, p)] = {
                label: np.stack(vecs, axis=0)
                for label, vecs in label_map.items()
                if len(vecs) > 0
            }
    return result
