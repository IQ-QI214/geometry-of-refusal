from experiments.mibd.eval.phase1_report import (
    LocusResult,
    Phase1ResultSet,
    build_go_no_go_report,
)


def test_phase1_report_marks_continue_when_shift_and_blank_noise_match():
    results = Phase1ResultSet(
        model_id="unit-vlm",
        signal_type="harmfulness",
        results=[
            LocusResult("V-text", 3, -5, 0.91),
            LocusResult("V-blank", 8, -1, 0.88),
            LocusResult("V-noise", 8, -1, 0.87),
        ],
        condition_cosines={
            ("V-text", "V-blank"): 0.52,
            ("V-text", "V-noise"): 0.55,
            ("V-blank", "V-noise"): 0.93,
        },
        static_transfer_auc={
            ("V-text", "V-blank"): 0.71,
            ("V-text", "V-noise"): 0.72,
        },
    )

    report = build_go_no_go_report(results)

    assert report.go is True
    assert "CONTINUE_MIBD" in report.decision
    assert report.blank_noise_equivalent is True
    assert report.static_transfer_drop is True

