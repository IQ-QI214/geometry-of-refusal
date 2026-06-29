"""Detection-vs-control dissociation metrics (H2 core).

This module quantifies the sensor-gate dissociation by measuring the angle
between two directions in hidden-state space:

* ``detection_direction``: the direction along which the risk evidence is most
  linearly readable (a sensor-side object, typically a difference-of-means
  vector between risk and safe hidden states).
* ``control_direction``: the direction whose intervention most strongly flips
  the model's safe-policy behavior (a gate-side object, obtained from
  forward-only causal patching elsewhere in the pipeline).

A large angle (cosine near zero) means "the model can read the risk but the
readout direction is not the one that controls refusal" -- i.e. the routing
failure. This mirrors the LLM-only finding in *Perfect Detection, Failed
Control* (arXiv:2606.24952) and lifts it to the VLM / visual-carrier setting.

CPU-only, numpy-only. No model, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from experiments.mibd.probes.direction import cosine_similarity, mean_difference_direction


@dataclass(frozen=True)
class DissociationScore:
    """Angle-based dissociation between a detection and a control direction."""

    cosine: float
    angle_degrees: float
    is_dissociated: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "cosine": self.cosine,
            "angle_degrees": self.angle_degrees,
            "is_dissociated": self.is_dissociated,
        }


def detection_direction_from_hidden(
    risk_hidden: np.ndarray,
    safe_hidden: np.ndarray,
) -> np.ndarray:
    """Sensor-side detection direction = difference-of-means(risk, safe).

    Reuses the existing, well-tested ``mean_difference_direction`` so the v2
    iteration stays consistent with the rest of the codebase. Returns a unit
    vector.
    """
    return mean_difference_direction(risk_hidden, safe_hidden)


def compute_dissociation(
    detection_direction: np.ndarray,
    control_direction: np.ndarray,
    dissociation_angle_threshold: float = 30.0,
) -> DissociationScore:
    """Angle between detection and control directions.

    Args:
        detection_direction: sensor-side direction (any nonzero vector).
        control_direction: gate-side direction (any nonzero vector).
        dissociation_angle_threshold: degrees above which we call the pair
            dissociated. 30 deg (cosine ~0.87) is a conservative default; the
            LLM-only literature reports angles up to ~83 deg.

    The cosine is taken in absolute value because a direction and its negation
    span the same axis; only the angle between *axes* is meaningful here.
    """
    raw_cos = cosine_similarity(detection_direction, control_direction)
    cos = abs(raw_cos)
    cos = max(-1.0, min(1.0, cos))
    angle = math.degrees(math.acos(cos))
    return DissociationScore(
        cosine=cos,
        angle_degrees=angle,
        is_dissociated=angle >= dissociation_angle_threshold,
    )


def compute_condition_dissociation(
    detection_directions: dict[str, np.ndarray],
    control_directions: dict[str, np.ndarray],
    dissociation_angle_threshold: float = 30.0,
) -> dict[str, DissociationScore]:
    """Per-visual-condition dissociation table.

    Only conditions present in *both* mappings are scored; conditions missing a
    counterpart are skipped (the caller can diff the key sets if needed).
    """
    if not detection_directions:
        raise ValueError("detection_directions must not be empty")
    if not control_directions:
        raise ValueError("control_directions must not be empty")
    shared = sorted(set(detection_directions) & set(control_directions))
    if not shared:
        raise ValueError("No shared conditions between detection and control directions")
    return {
        condition: compute_dissociation(
            detection_directions[condition],
            control_directions[condition],
            dissociation_angle_threshold=dissociation_angle_threshold,
        )
        for condition in shared
    }
