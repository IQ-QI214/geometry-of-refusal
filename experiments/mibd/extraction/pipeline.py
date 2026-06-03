from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from experiments.mibd.data.schema import MIBDSample
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

    for i, sample in enumerate(samples):
        image = adapter.build_image_for_condition(sample, seed=seed + i)
        inputs = adapter.prepare_inputs(sample, image)
        hidden_map = adapter.extract_hidden(inputs, layers, token_positions)

        for (layer_idx, pos), vec in hidden_map.items():
            storage[sample.visual_condition][(layer_idx, pos)][sample.label].append(vec)

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
