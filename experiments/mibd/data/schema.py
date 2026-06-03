"""Canonical sample schema for MIBD experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from experiments.mibd.config import SUPPORTED_VISUAL_CONDITIONS


SUPPORTED_LABELS = {"harmful", "harmless"}


@dataclass(frozen=True)
class MIBDSample:
    id: str
    text: str
    image_path: str | None
    label: str
    category: str
    source: str
    paired_id: str | None
    visual_condition: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MIBDSample":
        missing = {
            key
            for key in (
                "id",
                "text",
                "image_path",
                "label",
                "category",
                "source",
                "paired_id",
                "visual_condition",
            )
            if key not in data
        }
        if missing:
            raise ValueError(f"Missing required sample fields: {sorted(missing)}")
        if data["label"] not in SUPPORTED_LABELS:
            raise ValueError(f"Unsupported label {data['label']!r}")
        if data["visual_condition"] not in SUPPORTED_VISUAL_CONDITIONS:
            raise ValueError(f"Unsupported visual condition {data['visual_condition']!r}")
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            image_path=None if data["image_path"] is None else str(data["image_path"]),
            label=str(data["label"]),
            category=str(data["category"]),
            source=str(data["source"]),
            paired_id=None if data["paired_id"] is None else str(data["paired_id"]),
            visual_condition=str(data["visual_condition"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

