from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.config import SUPPORTED_VISUAL_CONDITIONS


def _stable_id(text: str, visual_condition: str, label: str, source: str) -> str:
    """Deterministic sample ID so refusal-label JSONs match across runs.

    visual_condition is intentionally excluded: the same text across V-text /
    V-blank / V-noise / V-real / FigStep must share the same ID so that refusal
    labels generated on V-text samples can be applied to all conditions.
    """
    key = f"{source}|{label}|{text}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _make_text_sample(
    item: dict,
    label: str,
    visual_condition: str,
    source: str,
    default_category: str,
    image_path: str | None = None,
) -> MIBDSample:
    text = item["instruction"]
    return MIBDSample.from_dict({
        "id": _stable_id(text, visual_condition, label, source),
        "text": text,
        "image_path": image_path,
        "label": label,
        "category": str(item.get("category") or default_category),
        "source": source,
        "paired_id": None,
        "visual_condition": visual_condition,
    })


def _collect_real_image_paths(mmsafety_dir: str | Path | None, seed: int = 42) -> list[str]:
    """Collect figstep image paths from mm-safetybench for use as V-real images."""
    if mmsafety_dir is None:
        return []
    base = Path(mmsafety_dir)
    paths: list[str] = []
    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        figstep_dir = cat_dir / "images_figstep"
        if not figstep_dir.exists():
            continue
        for img_path in sorted(figstep_dir.glob("*.png")):
            paths.append(str(img_path))
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths


def load_harmbench_phase1(
    data_dir: str,
    visual_conditions: Sequence[str],
    max_samples: int = 512,
    seed: int = 42,
    split: str = "test",
    mmsafety_dir: str | Path | None = None,
) -> list[MIBDSample]:
    """Load HarmBench text samples and expand across visual conditions.

    For V-real condition: assigns real images from mmsafety_dir if provided,
    otherwise falls back to blank (same as V-blank — not ideal for experiments).
    """
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

    # Pre-collect real image pool for V-real condition
    real_image_pool = _collect_real_image_paths(mmsafety_dir, seed=seed) if mmsafety_dir else []
    real_rng = random.Random(seed + 1)

    samples: list[MIBDSample] = []
    for vc in visual_conditions:
        for item in harmful_sel:
            img = None
            if vc == "V-real" and real_image_pool:
                img = real_rng.choice(real_image_pool)
            samples.append(_make_text_sample(item, "harmful", vc, "harmbench", "unknown", image_path=img))
        for item in harmless_sel:
            img = None
            if vc == "V-real" and real_image_pool:
                img = real_rng.choice(real_image_pool)
            samples.append(_make_text_sample(item, "harmless", vc, "alpaca", "general", image_path=img))
    return samples


def load_mmsafety_figstep(
    mmsafety_dir: str,
    max_samples: int = 512,
    seed: int = 42,
) -> list[MIBDSample]:
    """Load MM-SafetyBench FigStep images as harmful MIBDSamples."""
    base = Path(mmsafety_dir)
    all_items: list[dict] = []
    for cat_dir in sorted(base.iterdir()):
        if not cat_dir.is_dir():
            continue
        data_file = cat_dir / "data.json"
        figstep_dir = cat_dir / "images_figstep"
        if not data_file.exists() or not figstep_dir.exists():
            continue
        items = json.loads(data_file.read_text())
        for item in items:
            img_path = figstep_dir / f"{item['id']}.png"
            if img_path.exists():
                all_items.append({
                    "text": item.get("qr_prompt", item.get("original_prompt", "")),
                    "image_path": str(img_path),
                    "category": cat_dir.name,
                })

    rng = random.Random(seed)
    rng.shuffle(all_items)
    selected = all_items[:max_samples]

    return [
        MIBDSample.from_dict({
            "id": _stable_id(it["text"], "FigStep", "harmful", "mm-safetybench"),
            "text": it["text"],
            "image_path": it["image_path"],
            "label": "harmful",
            "category": it["category"],
            "source": "mm-safetybench",
            "paired_id": None,
            "visual_condition": "FigStep",
        })
        for it in selected
    ]
