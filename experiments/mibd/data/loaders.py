from __future__ import annotations
import json
import random
import uuid
from pathlib import Path
from typing import Sequence

from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.config import SUPPORTED_VISUAL_CONDITIONS


def load_harmbench_phase1(
    data_dir: str,
    visual_conditions: Sequence[str],
    max_samples: int = 512,
    seed: int = 42,
    split: str = "test",
) -> list[MIBDSample]:
    """Load HarmBench text samples and expand across visual conditions."""
    for vc in visual_conditions:
        if vc not in SUPPORTED_VISUAL_CONDITIONS:
            raise ValueError(f"Unsupported visual condition: {vc}")

    data_path = Path(data_dir)
    harmful_raw = json.loads((data_path / f"harmful_{split}.json").read_text())
    harmless_raw = json.loads((data_path / f"harmless_{split}.json").read_text())

    rng = random.Random(seed)
    n_per_label = max_samples // (2 * len(visual_conditions))
    n_per_label = max(1, n_per_label)

    harmful_sel = rng.sample(harmful_raw, min(n_per_label, len(harmful_raw)))
    harmless_sel = rng.sample(harmless_raw, min(n_per_label, len(harmless_raw)))

    samples: list[MIBDSample] = []
    for vc in visual_conditions:
        for item in harmful_sel:
            samples.append(MIBDSample.from_dict({
                "id": str(uuid.uuid4()),
                "text": item["instruction"],
                "image_path": None,
                "label": "harmful",
                "category": str(item.get("category") or "unknown"),
                "source": "harmbench",
                "paired_id": None,
                "visual_condition": vc,
            }))
        for item in harmless_sel:
            samples.append(MIBDSample.from_dict({
                "id": str(uuid.uuid4()),
                "text": item["instruction"],
                "image_path": None,
                "label": "harmless",
                "category": str(item.get("category") or "general"),
                "source": "alpaca",
                "paired_id": None,
                "visual_condition": vc,
            }))
    return samples
