"""Run Phase 2A behavior generation on a paired routing dataset.

This runner supports a CPU smoke mode. Real VLM generation should be wired in
on the GPU environment after model-specific generation is confirmed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mibd_routing.behavior.generate_outputs import (
    generate_behavior_outputs,
    load_paired_dataset,
    save_behavior_outputs,
)
from experiments.mibd_routing.data.schema import PairedRoutingSample


class SmokeGenerator:
    def generate(self, sample: PairedRoutingSample) -> str:
        if sample.is_risk:
            return "First gather the materials, then follow these steps."
        return "The image shows everyday objects and a safe next action."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["smoke"],
        default="smoke",
        help="Only smoke mode is implemented in this CPU scaffold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_paired_dataset(args.dataset)
    records = generate_behavior_outputs(samples, SmokeGenerator())
    output_path = save_behavior_outputs(records, args.output)
    print(f"Loaded samples: {len(samples)}")
    print(f"Wrote behavior outputs: {output_path}")


if __name__ == "__main__":
    main()

