from __future__ import annotations

from dataclasses import dataclass, field

_NA = -1.0  # sentinel: metric not available (not a real score)


@dataclass(frozen=True)
class AuditResult:
    model_id: str
    signal_type: str
    visual_condition: str
    train_auc: float
    held_out_auc: float           # _NA if not computed for this condition
    group_split_auc: float        # _NA if no paired_ids
    permutation_auc: float        # legacy float field (mean); kept for backward compat
    cross_category_aucs: dict[str, float]   # {test_category: auc or _NA}
    margins: dict
    static_transfer_margin_drop: dict[str, float]
    # New fields (default to sentinel / empty so old construction still works)
    train_selected_locus: tuple[int, int] | None = None
    full_data_locus: tuple[int, int] | None = None
    held_out_auc_train_selected: float = _NA
    permutation_stats: dict = field(default_factory=dict)


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

        # Train-only locus selection held-out AUC
        if ar.held_out_auc_train_selected != _NA:
            locus_str = ""
            if ar.train_selected_locus is not None:
                bl, bp = ar.train_selected_locus
                locus_str = f" (locus: layer={bl} pos={bp})"
            lines.append(
                f"| Train-only held-out AUC | {_fmt(ar.held_out_auc_train_selected)}"
                f"{locus_str} |"
            )

        # Permutation row: use permutation_stats if available, else fall back to float
        perm_stats = ar.permutation_stats
        if perm_stats and perm_stats.get("n_valid", 0) > 0:
            mean = perm_stats.get("mean", float("nan"))
            std = perm_stats.get("std", float("nan"))
            p95 = perm_stats.get("p95", float("nan"))
            n_valid = perm_stats.get("n_valid", 0)
            perm_cell = f"{mean:.4f} ± {std:.4f} (p95={p95:.4f}, n={n_valid})"
        else:
            perm_cell = _fmt(ar.permutation_auc)
        lines.append(f"| Permutation (nested) | {perm_cell} |")
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
    if ar.held_out_auc != _NA and ar.held_out_auc == ar.held_out_auc:
        if ar.held_out_auc < 0.70:
            issues.append(f"held-out AUC low ({ar.held_out_auc:.3f})")
        gap = ar.train_auc - ar.held_out_auc
        if gap > 0.15:
            issues.append(f"large train/held-out gap ({gap:.3f}) — possible overfit")

    # Use permutation_stats["mean"] if available, else fall back to permutation_auc float
    perm_stats = ar.permutation_stats
    if perm_stats and perm_stats.get("n_valid", 0) > 0:
        perm_mean = perm_stats.get("mean", float("nan"))
    else:
        perm_mean = ar.permutation_auc

    if perm_mean == perm_mean and perm_mean != _NA:  # not nan, not sentinel
        if perm_mean > 0.60:
            issues.append(f"permutation AUC high ({perm_mean:.3f}) — possible artifact")

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
            "Probes pass the implemented validity checks and generalize to random held-out splits; "
            "group/category controls remain unavailable under the current dataset structure."
        )
    return (
        f"{n_warn}/{n_total} condition(s) flagged with warnings. "
        "Review per-condition sections above before proceeding to Phase 2."
    )
