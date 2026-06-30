"""Generate the C1 neutral safe-image pool (format-matched, neutral content).

This produces a category-indexed pool of benign carrier images that share the
visual FORMAT of MM-SafetyBench risk carriers (white canvas, rendered text,
typographic / figstep layouts) while their CONTENT is completely neutral. Point
the v2 dataset builder at the resulting directory via ``--safe-image-dir`` to
eliminate the ``generated_blank_placeholder`` confound.

Example
-------
    python -m experiments.mibd_routing_v2.run_build_neutral_safe_pool \
      --output-dir data/mibd_routing_v2/benign_safe_images \
      --per-category 5

CPU-only. No model, no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mibd_routing_v2.data.neutral_carrier_renderer import (
    NeutralPoolSpec,
    NeutralRenderConfig,
    generate_neutral_safe_pool,
)

# MM-SafetyBench ships 13 numbered category directories (01..13).
DEFAULT_CATEGORIES = tuple(f"{i:02d}" for i in range(1, 14))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=Path("data/mibd_routing_v2/benign_safe_images"),
        type=Path,
    )
    parser.add_argument("--per-category", default=5, type=int)
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated category subdirectory names (default 01..13).",
    )
    parser.add_argument(
        "--carriers",
        default="typographic,figstep",
        help="Comma-separated carrier formats to render: typographic,figstep.",
    )
    parser.add_argument("--image-size", default=336, type=int)
    parser.add_argument("--font-size", default=22, type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    carriers = [c.strip() for c in args.carriers.split(",") if c.strip()]
    config = NeutralRenderConfig(
        size=(args.image_size, args.image_size),
        font_size=args.font_size,
    )
    spec = NeutralPoolSpec(
        categories=categories,
        per_category=args.per_category,
        carriers=carriers,
        config=config,
    )
    written = generate_neutral_safe_pool(args.output_dir, spec)
    total = sum(len(v) for v in written.values())
    print(f"Wrote neutral safe pool to: {args.output_dir}")
    print(f"Categories: {len(written)}  |  Images per category: {args.per_category}")
    print(f"Total images: {total}")
    print(f"Carriers: {carriers}")


if __name__ == "__main__":
    main()
