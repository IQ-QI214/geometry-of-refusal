"""Configuration loading for MIBD experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


SUPPORTED_VISUAL_CONDITIONS = {
    "V-text",
    "V-blank",
    "V-noise",
    "V-real",
    "FigStep",
}


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    dataset: str
    visual_conditions: tuple[str, ...]
    layers: tuple[int, ...]
    token_positions: tuple[int, ...]
    seed: int
    max_samples: int
    batch_size: int
    output_dir: Path

    def experiment_matrix(self) -> list[tuple[str, int, int]]:
        return [
            (condition, layer, pos)
            for condition in self.visual_conditions
            for layer in self.layers
            for pos in self.token_positions
        ]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}

    visual_conditions = tuple(raw.get("visual_conditions", ()))
    _validate_visual_conditions(visual_conditions)

    return ExperimentConfig(
        model_id=str(raw["model_id"]),
        dataset=str(raw["dataset"]),
        visual_conditions=visual_conditions,
        layers=_parse_grid(raw["layer_grid"]),
        token_positions=tuple(int(v) for v in raw["token_grid"]),
        seed=int(raw["seed"]),
        max_samples=int(raw["max_samples"]),
        batch_size=int(raw["batch_size"]),
        output_dir=Path(raw["output_dir"]),
    )


def _validate_visual_conditions(conditions: Iterable[str]) -> None:
    for condition in conditions:
        if condition not in SUPPORTED_VISUAL_CONDITIONS:
            supported = ", ".join(sorted(SUPPORTED_VISUAL_CONDITIONS))
            raise ValueError(
                f"Unsupported visual condition {condition!r}. Supported: {supported}"
            )


def _parse_grid(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        start = int(value["start"])
        stop = int(value["stop"])
        step = int(value.get("step", 1))
        return tuple(range(start, stop, step))
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    raise TypeError("Grid must be a list or {start, stop, step} mapping.")

