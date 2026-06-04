"""Phase 2A paired behavior-output generation helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Protocol

from experiments.mibd_routing.behavior.label_outputs import label_output
from experiments.mibd_routing.data.schema import PairedRoutingSample


class BehaviorGenerator(Protocol):
    def generate(self, sample: PairedRoutingSample) -> str: ...


@dataclass(frozen=True)
class BehaviorOutputRecord:
    sample_id: str
    paired_id: str
    risk_label: str
    carrier_type: str
    risk_category: str
    visual_condition: str
    model_output: str
    behavior_label: str
    judge_name: str
    judge_raw: dict

    def to_dict(self) -> dict:
        return asdict(self)


def load_paired_dataset(path: str | Path) -> list[PairedRoutingSample]:
    dataset_path = Path(path)
    samples = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(PairedRoutingSample.from_dict(json.loads(line)))
    return samples


def generate_behavior_outputs(
    samples: list[PairedRoutingSample],
    generator: BehaviorGenerator,
    judge_name: str = "rule_based_smoke",
) -> list[BehaviorOutputRecord]:
    records = []
    for sample in samples:
        output = generator.generate(sample)
        behavior = label_output(output, is_risk=sample.is_risk)
        records.append(
            BehaviorOutputRecord(
                sample_id=sample.sample_id,
                paired_id=sample.paired_id,
                risk_label=sample.risk_label.value,
                carrier_type=sample.carrier_type.value,
                risk_category=sample.risk_category,
                visual_condition=sample.visual_condition,
                model_output=output,
                behavior_label=behavior.value,
                judge_name=judge_name,
                judge_raw={"method": "rule_based_keyword_labeler"},
            )
        )
    return records


def save_behavior_outputs(records: list[BehaviorOutputRecord], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )
    return output_path

