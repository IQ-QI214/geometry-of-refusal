from __future__ import annotations

from dataclasses import dataclass

_NA = -1.0  # sentinel: metric not available (not a real score)


@dataclass(frozen=True)
class AuditResult:
    model_id: str
    signal_type: str
    visual_condition: str
    train_auc: float
    held_out_auc: float           # _NA if not computed for this condition
    group_split_auc: float        # _NA if no paired_ids
    permutation_auc: float
    cross_category_aucs: dict[str, float]   # {test_category: auc or _NA}
    margins: dict
    static_transfer_margin_drop: dict[str, float]


def _fmt(value: float, na_str: str = "N/A") -> str:
    if value == _NA or (value != value):  # nan check
        return na_str
    return f"{value:.4f}"


def build_phase1p5_report(
    audit_results: list[AuditResult],
    model_id: str,
    signal_type: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Phase 1.5 Probe Validity Audit: {model_id}")
    lines.append("")
    lines.append(f"Signal: `{signal_type}`")
    lines.append("")

    for ar in audit_results:
        lines.append(f"## Condition: {ar.visual_condition}")
        lines.append("")

        lines.append("### Split AUCs")
        lines.append("")
        lines.append("| Split | AUC |")
        lines.append("|---|:---:|")
        lines.append(f"| Train (full) | {_fmt(ar.train_auc)} |")
        lines.append(f"| Held-out (random 20%) | {_fmt(ar.held_out_auc)} |")
        if ar.group_split_auc != _NA:
            lines.append(f"| Group split (by paired_id) | {_fmt(ar.group_split_auc)} |")
        else:
            lines.append("| Group split (by paired_id) | N/A (no paired ids) |")
        lines.append(f"| Permutation (mean over 100) | {_fmt(ar.permutation_auc)} |")
        lines.append("")

        if ar.cross_category_aucs:
            lines.append("### Cross-Category AUCs")
            lines.append("")
            lines.append("| Held-out Category | AUC |")
            lines.append("|---|:---:|")
            for cat, auc in sorted(ar.cross_category_aucs.items()):
                lines.append(f"| {cat} | {_fmt(auc, 'N/A (single-class test set)')} |")
            lines.append("")

        m = ar.margins
        if m:
            lines.append("### Margin Statistics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---:|")
            lines.append(f"| Mean gap (harmful − harmless) | {m.get('mean_gap', float('nan')):.4f} |")
            lines.append(f"| Median gap | {m.get('median_gap', float('nan')):.4f} |")
            lines.append(f"| IQR harmful | {m.get('iqr_harmful', float('nan')):.4f} |")
            lines.append(f"| IQR harmless | {m.get('iqr_harmless', float('nan')):.4f} |")
            lines.append(f"| N harmful | {m.get('n_harmful', 0)} |")
            lines.append(f"| N harmless | {m.get('n_harmless', 0)} |")
            lines.append("")

        if ar.static_transfer_margin_drop:
            lines.append("### Static Transfer Margin Drop (V-text probe → other conditions)")
            lines.append("")
            lines.append("| Target Condition | Margin Drop |")
            lines.append("|---|---:|")
            for target, drop in sorted(ar.static_transfer_margin_drop.items()):
                lines.append(f"| {target} | {drop:.4f} |")
            lines.append("")

        verdict = _condition_verdict(ar)
        lines.append(f"**Audit verdict:** {verdict}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Overall Audit Conclusion")
    lines.append("")
    lines.append(_overall_conclusion(audit_results))
    lines.append("")

    return "\n".join(lines)


def _condition_verdict(ar: AuditResult) -> str:
    issues = []
    # Only flag held-out AUC if it was actually computed
    if ar.held_out_auc != _NA and ar.held_out_auc == ar.held_out_auc:
        if ar.held_out_auc < 0.70:
            issues.append(f"held-out AUC low ({ar.held_out_auc:.3f})")
        gap = ar.train_auc - ar.held_out_auc
        if gap > 0.15:
            issues.append(f"large train/held-out gap ({gap:.3f}) — possible overfit")
    if ar.permutation_auc == ar.permutation_auc and ar.permutation_auc != _NA:
        if ar.permutation_auc > 0.60:
            issues.append(f"permutation AUC high ({ar.permutation_auc:.3f}) — possible artifact")
    if not issues:
        return "PASS — probe appears valid"
    return "WARN — " + "; ".join(issues)


def _overall_conclusion(audit_results: list[AuditResult]) -> str:
    if not audit_results:
        return "No audit results available."
    n_warn = sum(1 for ar in audit_results if "WARN" in _condition_verdict(ar))
    n_total = len(audit_results)
    if n_warn == 0:
        return (
            f"All {n_total} condition(s) passed validity checks. "
            "Probes appear non-artifactual and generalize to held-out data."
        )
    return (
        f"{n_warn}/{n_total} condition(s) flagged with warnings. "
        "Review per-condition sections above before proceeding to Phase 2."
    )
