import numpy as np
import pytest
from unittest.mock import MagicMock

from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.extraction.pipeline import (
    run_extraction,
    run_extraction_with_metadata,
)


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


class _RowIndexAdapter:
    """Fake adapter whose hidden vector encodes the global row index.

    hidden[(layer, -1)] = [row_index, layer, 0, 0, ...] so a test can assert
    that stacked hidden rows align 1:1 with the recorded metadata order.
    """

    def build_image_for_condition(self, sample, seed=0):
        return None

    def prepare_inputs(self, sample, image):
        return {"idx": int(sample.id.split("_")[-1])}

    def extract_hidden(self, inputs, layers, token_positions):
        idx = inputs["idx"]
        out = {}
        for layer in layers:
            vec = np.zeros(4)
            vec[0] = idx
            vec[1] = layer
            out[(layer, -1)] = vec
        return out


def _indexed_sample(idx: int, vc: str, label: str, paired: str) -> MIBDSample:
    return MIBDSample.from_dict({
        "id": f"s_{idx}", "text": "t", "image_path": f"/img/{idx}.png",
        "label": label, "category": "01", "source": "test",
        "paired_id": paired, "visual_condition": vc,
    })


def test_metadata_rows_align_with_hidden_rows():
    adapter = _RowIndexAdapter()
    # two harmful + two harmless, interleaved, distinct paired ids
    samples = [
        _indexed_sample(0, "V-real", "harmful", "p0"),
        _indexed_sample(1, "V-real", "harmless", "p0"),
        _indexed_sample(2, "V-real", "harmful", "p1"),
        _indexed_sample(3, "V-real", "harmless", "p1"),
    ]
    hidden, meta = run_extraction_with_metadata(
        adapter, samples, layers=(0, 4), token_positions=(-1,)
    )

    for label, expected_idxs in (("harmful", [0, 2]), ("harmless", [1, 3])):
        arr = hidden["V-real"][(0, -1)][label]
        rows = meta["V-real"][(0, -1)][label]
        # hidden vector's encoded row index must equal the metadata row_index
        assert arr.shape[0] == len(rows) == 2
        for i, row in enumerate(rows):
            assert int(arr[i, 0]) == row["row_index"]
            assert row["sample_id"] == f"s_{row['row_index']}"
            assert row["paired_id"] in ("p0", "p1")


def test_metadata_present_for_every_locus_and_label():
    adapter = _RowIndexAdapter()
    samples = [
        _indexed_sample(0, "V-real", "harmful", "p0"),
        _indexed_sample(1, "V-real", "harmless", "p0"),
    ]
    hidden, meta = run_extraction_with_metadata(
        adapter, samples, layers=(0, 4), token_positions=(-1,)
    )
    for locus in hidden["V-real"]:
        for label in hidden["V-real"][locus]:
            assert label in meta["V-real"][locus]
            assert len(meta["V-real"][locus][label]) == hidden["V-real"][locus][label].shape[0]
