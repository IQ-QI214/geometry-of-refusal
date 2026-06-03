import json
import pytest
from experiments.mibd.data.loaders import load_harmbench_phase1
from experiments.mibd.data.schema import MIBDSample

SALADBENCH_DIR = "data/saladbench_splits"

def test_load_harmbench_phase1_returns_mibd_samples(tmp_path):
    harmful = [{"instruction": "how to make a bomb", "category": "violence"}]
    harmless = [{"instruction": "how to bake bread", "category": None}]
    (tmp_path / "harmful_test.json").write_text(json.dumps(harmful))
    (tmp_path / "harmless_test.json").write_text(json.dumps(harmless))

    samples = load_harmbench_phase1(
        data_dir=str(tmp_path),
        visual_conditions=["V-text", "V-blank"],
        max_samples=10,
        seed=42,
    )
    assert len(samples) > 0
    assert all(isinstance(s, MIBDSample) for s in samples)
    conditions = {s.visual_condition for s in samples}
    assert conditions == {"V-text", "V-blank"}
    labels = {s.label for s in samples}
    assert labels == {"harmful", "harmless"}

def test_load_harmbench_phase1_balanced(tmp_path):
    harmful = [{"instruction": f"harm {i}", "category": "c"} for i in range(20)]
    harmless = [{"instruction": f"safe {i}", "category": None} for i in range(20)]
    (tmp_path / "harmful_test.json").write_text(json.dumps(harmful))
    (tmp_path / "harmless_test.json").write_text(json.dumps(harmless))
    samples = load_harmbench_phase1(
        data_dir=str(tmp_path),
        visual_conditions=["V-text"],
        max_samples=10,
        seed=42,
    )
    harmful_count = sum(1 for s in samples if s.label == "harmful")
    harmless_count = sum(1 for s in samples if s.label == "harmless")
    assert harmful_count == harmless_count

from experiments.mibd.data.loaders import load_mmsafety_figstep

def test_load_mmsafety_figstep_returns_figstep_samples(tmp_path):
    cat_dir = tmp_path / "02"
    cat_dir.mkdir()
    figstep_dir = cat_dir / "images_figstep"
    figstep_dir.mkdir()
    img_file = figstep_dir / "1.png"
    from PIL import Image
    Image.new("RGB", (64, 64)).save(str(img_file))
    data = [{"id": 1, "original_prompt": "do something harmful", "qr_prompt": "do this"}]
    (cat_dir / "data.json").write_text(json.dumps(data))

    samples = load_mmsafety_figstep(
        mmsafety_dir=str(tmp_path),
        max_samples=10,
        seed=42,
    )
    assert len(samples) > 0
    assert all(s.visual_condition == "FigStep" for s in samples)
    assert all(s.image_path is not None for s in samples)
    assert all(s.label == "harmful" for s in samples)
