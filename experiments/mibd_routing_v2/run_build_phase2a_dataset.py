"""Build the v2 matched-benign Phase 2A paired routing dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mibd_routing_v2.data.build_phase2a_dataset import (
    build_phase2a_paired_dataset_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmsafety-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("results/mibd_routing_v2/paired_dataset/phase2a_matched_v2"),
        type=Path,
    )
    parser.add_argument("--num-pairs", default=200, type=int)
    parser.add_argument("--seed", default=20260604, type=int)
    parser.add_argument("--safe-image-dir", default=None, type=Path)
    parser.add_argument(
        "--carriers",
        default="figstep,typographic",
        help="Comma-separated carriers: figstep,typographic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    carriers = [value.strip() for value in args.carriers.split(",") if value.strip()]
    records, card = build_phase2a_paired_dataset_v2(
        mmsafety_dir=args.mmsafety_dir,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        seed=args.seed,
        carriers=carriers,
        safe_image_dir=args.safe_image_dir,
    )
    print(f"Wrote paired dataset: {args.output_dir / 'paired_dataset.jsonl'}")
    print(f"Wrote dataset card: {args.output_dir / 'dataset_card.json'}")
    print(f"Wrote build report: {args.output_dir / 'build_report.md'}")
    print(f"Paired IDs: {card.num_paired_ids}")
    print(f"Samples: {len(records)}")
    print(f"Dataset hash: {card.dataset_hash}")


if __name__ == "__main__":
    main()
