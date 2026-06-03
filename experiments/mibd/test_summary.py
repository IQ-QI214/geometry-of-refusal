import json
import numpy as np
import pytest

from experiments.mibd.eval.summary import build_phase1_summary


def _make_probe_results(auc: float):
    return {(17, -5): {"direction": np.ones(32) / np.sqrt(32), "auc": auc}}


def test_build_phase1_summary_schema():
    probe_results_by_condition = {
        "V-text":  _make_probe_results(0.91),
        "V-blank": _make_probe_results(0.72),
        "V-noise": _make_probe_results(0.70),
    }
    condition_cosines = {("V-text", "V-blank"): 0.52, ("V-blank", "V-noise"): 0.93}
    static_transfer = {("V-text", "V-blank"): 0.71}

    summary = build_phase1_summary(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        signal_type="harmfulness",
        probe_results_by_condition=probe_results_by_condition,
        condition_cosines=condition_cosines,
        static_transfer_auc=static_transfer,
    )
    raw = json.loads(json.dumps(summary))
    assert raw["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert raw["signal_type"] == "harmfulness"
    assert len(raw["results"]) == 3
    for r in raw["results"]:
        assert "visual_condition" in r
        assert "layer" in r
        assert "token_pos" in r
        assert "auc" in r
    assert "V-text|V-blank" in raw["condition_cosines"]
    assert "V-text|V-blank" in raw["static_transfer_auc"]
