"""Go/No-Go report helpers for Phase 1 mislocalization experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocusResult:
    visual_condition: str
    layer: int
    token_pos: int
    auc: float


@dataclass(frozen=True)
class Phase1ResultSet:
    model_id: str
    signal_type: str
    results: list[LocusResult]
    condition_cosines: dict[tuple[str, str], float]
    static_transfer_auc: dict[tuple[str, str], float]


@dataclass(frozen=True)
class GoNoGoReport:
    decision: str
    go: bool
    locus_shift: bool
    blank_noise_equivalent: bool
    static_transfer_drop: bool
    markdown: str


def build_go_no_go_report(
    result_set: Phase1ResultSet,
    cosine_shift_threshold: float = 0.85,
    blank_noise_threshold: float = 0.85,
    static_auc_drop_threshold: float = 0.10,
) -> GoNoGoReport:
    by_condition = {r.visual_condition: r for r in result_set.results}
    v_text = by_condition.get("V-text")
    if v_text is None:
        raise ValueError("Phase 1 report requires a V-text result")

    locus_shift = any(
        (r.layer, r.token_pos) != (v_text.layer, v_text.token_pos)
        for condition, r in by_condition.items()
        if condition != "V-text"
    )
    direction_shift = any(
        value < cosine_shift_threshold
        for (left, right), value in result_set.condition_cosines.items()
        if "V-text" in (left, right)
    )
    blank_noise_equivalent = (
        result_set.condition_cosines.get(("V-blank", "V-noise"))
        or result_set.condition_cosines.get(("V-noise", "V-blank"))
        or 0.0
    ) >= blank_noise_threshold
    static_transfer_drop = any(
        (v_text.auc - auc) >= static_auc_drop_threshold
        for (_, target), auc in result_set.static_transfer_auc.items()
        if target != "V-text"
    )
    go = (locus_shift or direction_shift) and blank_noise_equivalent and static_transfer_drop
    decision = "CONTINUE_MIBD" if go else "STOP_OR_PIVOT"
    markdown = _render_markdown(
        result_set,
        decision,
        locus_shift=locus_shift or direction_shift,
        blank_noise_equivalent=blank_noise_equivalent,
        static_transfer_drop=static_transfer_drop,
    )
    return GoNoGoReport(
        decision=decision,
        go=go,
        locus_shift=locus_shift or direction_shift,
        blank_noise_equivalent=blank_noise_equivalent,
        static_transfer_drop=static_transfer_drop,
        markdown=markdown,
    )


def _render_markdown(
    result_set: Phase1ResultSet,
    decision: str,
    locus_shift: bool,
    blank_noise_equivalent: bool,
    static_transfer_drop: bool,
) -> str:
    rows = [
        "| Condition | Layer | Token Pos | AUC |",
        "|---|:---:|:---:|:---:|",
    ]
    for result in result_set.results:
        rows.append(
            f"| {result.visual_condition} | {result.layer} | {result.token_pos} | {result.auc:.3f} |"
        )
    checks = [
        f"- Locus/direction shift: {'PASS' if locus_shift else 'FAIL'}",
        f"- Blank/noise equivalence: {'PASS' if blank_noise_equivalent else 'FAIL'}",
        f"- Static transfer drop: {'PASS' if static_transfer_drop else 'FAIL'}",
    ]
    return "\n".join(
        [
            f"# Phase 1 Go/No-Go Report: {result_set.model_id}",
            "",
            f"Signal: `{result_set.signal_type}`",
            f"Decision: **{decision}**",
            "",
            *rows,
            "",
            *checks,
            "",
        ]
    )

