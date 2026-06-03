"""Build a Phase 1 Go/No-Go markdown report from JSON summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.mibd.eval.phase1_report import (
    LocusResult,
    Phase1ResultSet,
    build_go_no_go_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an MIBD Phase 1 Go/No-Go report.")
    parser.add_argument("--input", required=True, help="JSON summary path")
    parser.add_argument("--output", required=True, help="Markdown report path")
    args = parser.parse_args()

    result_set = _load_result_set(Path(args.input))
    report = build_go_no_go_report(result_set)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.markdown)
    print(f"decision: {report.decision}")
    print(f"wrote: {output_path}")


def _load_result_set(path: Path) -> Phase1ResultSet:
    raw = json.loads(path.read_text())
    return Phase1ResultSet(
        model_id=str(raw["model_id"]),
        signal_type=str(raw["signal_type"]),
        results=[
            LocusResult(
                visual_condition=str(row["visual_condition"]),
                layer=int(row["layer"]),
                token_pos=int(row["token_pos"]),
                auc=float(row["auc"]),
            )
            for row in raw["results"]
        ],
        condition_cosines=_parse_pair_map(raw.get("condition_cosines", {})),
        static_transfer_auc=_parse_pair_map(raw.get("static_transfer_auc", {})),
    )


def _parse_pair_map(raw: dict[str, Any]) -> dict[tuple[str, str], float]:
    parsed = {}
    for key, value in raw.items():
        left, right = key.split("|", 1)
        parsed[(left, right)] = float(value)
    return parsed


if __name__ == "__main__":
    main()

