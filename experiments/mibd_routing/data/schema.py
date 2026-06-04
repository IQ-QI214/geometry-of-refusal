"""Schema for Phase 2 paired routing diagnostic samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any

from experiments.mibd.data.schema import MIBDSample


class CarrierType(str, Enum):
    NATURAL_RISK = "natural_risk"
    FIGSTEP = "figstep"
    BLANK = "blank"
    NOISE = "noise"
    TYPOGRAPHIC = "typographic"


class RiskLabel(str, Enum):
    SAFE = "safe"
    RISK = "risk"


class BehaviorLabel(str, Enum):
    SAFE_POLICY = "safe_policy"
    UNSAFE_COMPLIANCE = "unsafe_compliance"
    BENIGN_HELPFUL = "benign_helpful"
    OVER_REFUSAL = "over_refusal"
    DEGENERATION = "degeneration"


@dataclass(frozen=True)
class PairedRoutingSample:
    sample_id: str
    paired_id: str
    question: str
    image_path: str | None
    counterpart_image_path: str | None
    risk_label: RiskLabel
    carrier_type: CarrierType
    risk_category: str
    expected_safe_behavior: str
    expected_benign_behavior: str
    visual_condition: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_risk(self) -> bool:
        return self.risk_label == RiskLabel.RISK

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PairedRoutingSample":
        required = {
            "sample_id",
            "paired_id",
            "question",
            "image_path",
            "counterpart_image_path",
            "risk_label",
            "carrier_type",
            "risk_category",
            "expected_safe_behavior",
            "expected_benign_behavior",
            "visual_condition",
            "source",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"Missing paired routing fields: {sorted(missing)}")
        try:
            risk_label = RiskLabel(data["risk_label"])
            carrier_type = CarrierType(data["carrier_type"])
        except ValueError as exc:
            raise ValueError(f"Unsupported paired routing enum value: {exc}") from exc
        return cls(
            sample_id=str(data["sample_id"]),
            paired_id=str(data["paired_id"]),
            question=str(data["question"]),
            image_path=None if data["image_path"] is None else str(data["image_path"]),
            counterpart_image_path=(
                None
                if data["counterpart_image_path"] is None
                else str(data["counterpart_image_path"])
            ),
            risk_label=risk_label,
            carrier_type=carrier_type,
            risk_category=str(data["risk_category"]),
            expected_safe_behavior=str(data["expected_safe_behavior"]),
            expected_benign_behavior=str(data["expected_benign_behavior"]),
            visual_condition=str(data["visual_condition"]),
            source=str(data["source"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["risk_label"] = self.risk_label.value
        data["carrier_type"] = self.carrier_type.value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def to_mibd_sample(self) -> MIBDSample:
        return MIBDSample(
            id=self.sample_id,
            text=self.question,
            image_path=self.image_path,
            label="harmful" if self.is_risk else "harmless",
            category=self.risk_category,
            source=self.source,
            paired_id=self.paired_id,
            visual_condition=self.visual_condition,
        )


@dataclass(frozen=True)
class DatasetCard:
    name: str
    num_paired_ids: int
    num_samples: int
    carrier_counts: dict[str, int]
    risk_category_counts: dict[str, int]
    dataset_hash: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

