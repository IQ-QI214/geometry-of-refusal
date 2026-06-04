"""CLI for preparing Phase 2 routing scaffold artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mibd_routing.data.build_paired_dataset import build_pilot_paired_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mibd_routing/paired_dataset/pilot"),
        help="Directory for paired_dataset.jsonl and dataset_card.json.",
    )
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, card = build_pilot_paired_dataset(
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        seed=args.seed,
    )
    print(f"Wrote paired dataset: {args.output_dir / 'paired_dataset.jsonl'}")
    print(f"Wrote dataset card: {args.output_dir / 'dataset_card.json'}")
    print(f"Dataset hash: {card.dataset_hash}")


if __name__ == "__main__":
    main()

