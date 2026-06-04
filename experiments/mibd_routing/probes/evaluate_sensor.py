"""Sensor readout and relocation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.mibd.probes.direction import cosine_similarity
from experiments.mibd.probes.metrics import binary_auc


@dataclass(frozen=True)
class MultiLocusReadoutReport:
    best_locus: tuple[int, int]
    best_locus_auc: float
    multi_locus_auc: float
    multi_locus_gain: float

    def to_dict(self) -> dict[str, object]:
        return {
            "best_locus": list(self.best_locus),
            "best_locus_auc": self.best_locus_auc,
            "multi_locus_auc": self.multi_locus_auc,
            "multi_locus_gain": self.multi_locus_gain,
        }


@dataclass(frozen=True)
class RelocationScore:
    cosine_relocation: float
    layer_relocation: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cosine_relocation": self.cosine_relocation,
            "layer_relocation": self.layer_relocation,
        }


def evaluate_multi_locus_readout(
    labels: np.ndarray,
    locus_scores: dict[tuple[int, int], np.ndarray],
) -> MultiLocusReadoutReport:
    if not locus_scores:
        raise ValueError("locus_scores must not be empty")
    labels = np.asarray(labels)
    aucs = {
        locus: binary_auc(labels, np.asarray(scores, dtype=float))
        for locus, scores in locus_scores.items()
    }
    margins = {
        locus: _class_margin(labels, np.asarray(scores, dtype=float))
        for locus, scores in locus_scores.items()
    }
    best_locus = max(aucs, key=lambda locus: (aucs[locus], margins[locus], locus))
    stacked = np.vstack([np.asarray(scores, dtype=float) for scores in locus_scores.values()])
    multi_scores = np.mean(stacked, axis=0)
    multi_auc = binary_auc(labels, multi_scores)
    best_auc = aucs[best_locus]
    return MultiLocusReadoutReport(
        best_locus=best_locus,
        best_locus_auc=best_auc,
        multi_locus_auc=multi_auc,
        multi_locus_gain=multi_auc - best_auc,
    )


def compute_relocation_scores(
    standard_direction: np.ndarray,
    condition_directions: dict[str, np.ndarray],
    standard_layer: int,
    condition_layers: dict[str, int],
) -> dict[str, RelocationScore]:
    scores = {}
    for condition, direction in condition_directions.items():
        scores[condition] = RelocationScore(
            cosine_relocation=1.0 - cosine_similarity(standard_direction, direction),
            layer_relocation=abs(int(condition_layers[condition]) - int(standard_layer)),
        )
    return scores


def _class_margin(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    return float(np.mean(positives) - np.mean(negatives))
