"""Inspect an MIBD YAML config and print its experiment matrix size."""

from __future__ import annotations

import argparse

from experiments.mibd.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an MIBD experiment config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    matrix = cfg.experiment_matrix()
    print(f"model_id: {cfg.model_id}")
    print(f"dataset: {cfg.dataset}")
    print(f"visual_conditions: {','.join(cfg.visual_conditions)}")
    print(f"layers: {len(cfg.layers)}")
    print(f"token_positions: {len(cfg.token_positions)}")
    print(f"experiment_count: {len(matrix)}")
    print(f"output_dir: {cfg.output_dir}")


if __name__ == "__main__":
    main()

