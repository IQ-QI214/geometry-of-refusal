"""Build v2 Phase 2A paired datasets with category-matched safe images."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from experiments.mibd_routing.data.convert_phase2a import (
    AMBIGUOUS_QUESTIONS,
    collect_mmsafety_items,
    _visual_condition_for_carrier,
)
from experiments.mibd_routing.data.schema import (
    DatasetCard,
    PairedRoutingSample,
    RiskLabel,
)
from experiments.mibd_routing_v2.data.matched_safe_images import (
    load_category_safe_pool,
    select_matched_safe_image,
)


def build_phase2a_paired_dataset_v2(
    mmsafety_dir: str | Path,
    output_dir: str | Path,
    num_pairs: int = 200,
    seed: int = 20260604,
    carriers: Sequence[str] = ("figstep", "typographic"),
    safe_image_dir: str | Path | None = None,
) -> tuple[list[PairedRoutingSample], DatasetCard]:
    """Build the v2 matched-benign Phase 2A dataset.

    v2 deliberately reuses the stable Phase 2A schema and MM-SafetyBench item
    collector from ``mibd_routing`` while keeping the upgraded matched-safe
    selection and output identity under ``mibd_routing_v2``.
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")

    rng = random.Random(seed)
    items = collect_mmsafety_items(mmsafety_dir=mmsafety_dir, carriers=carriers)
    if not items:
        raise ValueError("No MM-SafetyBench carrier items found")
    rng.shuffle(items)
    selected = items[: min(num_pairs, len(items))]
    safe_pool = load_category_safe_pool(safe_image_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_safe_dir = output_path / "safe_controls"

    records: list[PairedRoutingSample] = []
    for index, item in enumerate(selected):
        paired_id = f"phase2a-v2-{index:04d}"
        question = AMBIGUOUS_QUESTIONS[index % len(AMBIGUOUS_QUESTIONS)]
        safe_image, safe_mode = select_matched_safe_image(
            safe_pool,
            category=item.category,
            index=index,
        )
        if safe_image is None:
            generated_safe_dir.mkdir(parents=True, exist_ok=True)
            safe_path = generated_safe_dir / f"safe_blank_{index:04d}.txt"
            if not safe_path.exists():
                safe_path.write_text("blank safe control placeholder\n", encoding="utf-8")
            safe_image = str(safe_path)
            safe_mode = "generated_blank_placeholder"

        common = {
            "paired_id": paired_id,
            "question": question,
            "carrier_type": item.carrier_type,
            "risk_category": item.category,
            "expected_safe_behavior": "safe_policy",
            "expected_benign_behavior": "benign_helpful",
            "visual_condition": _visual_condition_for_carrier(item.carrier_type),
            "source": "mm-safetybench-phase2a-v2",
        }
        metadata = {
            "safe_image_mode": safe_mode,
            "risk_item_id": item.item_id,
            "original_prompt": item.original_prompt,
            "risk_text": item.risk_text,
            **item.metadata,
        }
        records.append(
            PairedRoutingSample(
                sample_id=f"{paired_id}-safe",
                image_path=safe_image,
                counterpart_image_path=item.image_path,
                risk_label=RiskLabel.SAFE,
                metadata=metadata,
                **common,
            )
        )
        records.append(
            PairedRoutingSample(
                sample_id=f"{paired_id}-risk",
                image_path=item.image_path,
                counterpart_image_path=safe_image,
                risk_label=RiskLabel.RISK,
                metadata=metadata,
                **common,
            )
        )

    card = DatasetCard(
        name="mibd_phase2a_matched_v2",
        num_paired_ids=len(selected),
        num_samples=len(records),
        carrier_counts=dict(Counter(sample.carrier_type.value for sample in records if sample.is_risk)),
        risk_category_counts=dict(Counter(sample.risk_category for sample in records if sample.is_risk)),
        dataset_hash=_dataset_hash(records),
        seed=seed,
    )
    _write_dataset(output_path, records, card, len(items), safe_image_dir)
    return records, card


def _dataset_hash(records: Iterable[PairedRoutingSample]) -> str:
    payload = "\n".join(sample.to_json() for sample in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _write_dataset(
    output_path: Path,
    records: list[PairedRoutingSample],
    card: DatasetCard,
    available_risk_items: int,
    safe_image_dir: str | Path | None,
) -> None:
    (output_path / "paired_dataset.jsonl").write_text(
        "\n".join(sample.to_json() for sample in records) + "\n",
        encoding="utf-8",
    )
    (output_path / "dataset_card.json").write_text(
        json.dumps(card.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    safe_modes = Counter(
        sample.metadata.get("safe_image_mode", "unknown")
        for sample in records
        if sample.risk_label == RiskLabel.SAFE
    )
    report = "\n".join(
        [
            "# MIBD Phase 2A Matched V2 Dataset Build Report",
            "",
            f"- paired IDs: {card.num_paired_ids}",
            f"- samples: {card.num_samples}",
            f"- available risk carrier items: {available_risk_items}",
            f"- dataset hash: {card.dataset_hash}",
            f"- safe image dir: {safe_image_dir or 'generated_blank_placeholder'}",
            f"- safe image modes: {dict(safe_modes)}",
            f"- carrier counts: {card.carrier_counts}",
            f"- risk category counts: {card.risk_category_counts}",
            "",
        ]
    )
    (output_path / "build_report.md").write_text(report, encoding="utf-8")
