from pathlib import Path

import pytest

from experiments.mibd.config import load_experiment_config


def test_load_phase1_config_expands_layer_and_token_grids(tmp_path):
    config_path = tmp_path / "phase1.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_id: Qwen/Qwen3-VL-8B-Instruct",
                "dataset: harmbench_smoke",
                "visual_conditions: [V-text, V-blank]",
                "layer_grid: {start: 0, stop: 4, step: 2}",
                "token_grid: [-5, -1]",
                "seed: 7",
                "max_samples: 16",
                "batch_size: 2",
                "output_dir: results/mibd/phase1_smoke",
            ]
        )
    )

    cfg = load_experiment_config(config_path)

    assert cfg.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.visual_conditions == ("V-text", "V-blank")
    assert cfg.layers == (0, 2)
    assert cfg.token_positions == (-5, -1)
    assert cfg.experiment_matrix() == [
        ("V-text", 0, -5),
        ("V-text", 0, -1),
        ("V-text", 2, -5),
        ("V-text", 2, -1),
        ("V-blank", 0, -5),
        ("V-blank", 0, -1),
        ("V-blank", 2, -5),
        ("V-blank", 2, -1),
    ]


def test_config_rejects_unknown_visual_condition(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_id: model",
                "dataset: data",
                "visual_conditions: [V-text, V-bad]",
                "layer_grid: [0]",
                "token_grid: [-1]",
                "seed: 0",
                "max_samples: 1",
                "batch_size: 1",
                "output_dir: out",
            ]
        )
    )

    with pytest.raises(ValueError, match="Unsupported visual condition"):
        load_experiment_config(config_path)

