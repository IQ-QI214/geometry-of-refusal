import numpy as np
import pytest
from unittest.mock import MagicMock

from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.extraction.pipeline import run_extraction


def _make_sample(vc: str, label: str) -> MIBDSample:
    return MIBDSample.from_dict({
        "id": f"{vc}_{label}", "text": "test text", "image_path": None,
        "label": label, "category": "test", "source": "test",
        "paired_id": None, "visual_condition": vc,
    })


def test_run_extraction_returns_correct_keys():
    mock_adapter = MagicMock()
    mock_adapter.build_image_for_condition.return_value = None
    mock_adapter.prepare_inputs.return_value = {"input_ids": MagicMock()}
    mock_adapter.extract_hidden.return_value = {
        (0, -1): np.ones(16), (0, -5): np.ones(16),
        (1, -1): np.ones(16) * 2,
    }

    samples = [_make_sample("V-text", "harmful"), _make_sample("V-text", "harmless")]
    layers = (0, 1)
    token_positions = (-1, -5)

    result = run_extraction(mock_adapter, samples, layers, token_positions)

    assert "V-text" in result
    assert (0, -1) in result["V-text"]
    assert "harmful" in result["V-text"][(0, -1)]
    assert "harmless" in result["V-text"][(0, -1)]
    arr = result["V-text"][(0, -1)]["harmful"]
    assert arr.ndim == 2  # (n_samples, hidden_dim)


def test_run_extraction_skips_missing_hidden():
    mock_adapter = MagicMock()
    mock_adapter.build_image_for_condition.return_value = None
    mock_adapter.prepare_inputs.return_value = {"input_ids": MagicMock()}
    mock_adapter.extract_hidden.return_value = {(0, -1): np.ones(8)}

    samples = [_make_sample("V-blank", "harmful")]
    result = run_extraction(mock_adapter, samples, layers=(0, 1), token_positions=(-1, -5))

    assert (0, -1) in result["V-blank"]
    assert (1, -5) not in result.get("V-blank", {})
