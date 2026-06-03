import json
import subprocess
import sys
from pathlib import Path


def test_inspect_config_cli_prints_experiment_count(tmp_path):
    config_path = tmp_path / "phase1.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_id: unit-model",
                "dataset: unit-data",
                "visual_conditions: [V-text, V-blank]",
                "layer_grid: [0, 1]",
                "token_grid: [-1]",
                "seed: 0",
                "max_samples: 4",
                "batch_size: 2",
                "output_dir: out",
            ]
        )
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mibd.inspect_config",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "unit-model" in proc.stdout
    assert "experiment_count: 4" in proc.stdout


def test_phase1_report_cli_writes_markdown(tmp_path):
    input_path = tmp_path / "summary.json"
    output_path = tmp_path / "report.md"
    input_path.write_text(
        json.dumps(
            {
                "model_id": "unit-vlm",
                "signal_type": "harmfulness",
                "results": [
                    {"visual_condition": "V-text", "layer": 1, "token_pos": -5, "auc": 0.91},
                    {"visual_condition": "V-blank", "layer": 3, "token_pos": -1, "auc": 0.86},
                    {"visual_condition": "V-noise", "layer": 3, "token_pos": -1, "auc": 0.85},
                ],
                "condition_cosines": {
                    "V-text|V-blank": 0.5,
                    "V-text|V-noise": 0.55,
                    "V-blank|V-noise": 0.91,
                },
                "static_transfer_auc": {
                    "V-text|V-blank": 0.71,
                    "V-text|V-noise": 0.72,
                },
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mibd.make_phase1_report",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CONTINUE_MIBD" in output_path.read_text()

