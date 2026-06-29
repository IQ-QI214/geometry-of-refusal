"""V2 Phase 2A GPU behavior generation runner.

This is a v2-scoped wrapper around the existing VLM generation utilities. It
keeps default paths under `results/mibd_routing_v2/` and supports `--max-samples`
for quick offline GPU smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from experiments.mibd_routing.behavior.generate_outputs import (
    BehaviorOutputRecord,
    load_paired_dataset,
)
from experiments.mibd_routing.behavior.label_outputs import label_output
from experiments.mibd_routing.data.schema import PairedRoutingSample
from experiments.mibd_routing.run_phase2a_vlm_behavior import (
    MODEL_CHOICES,
    load_generator,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args(argv)


def _load_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["sample_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _process_sample(
    sample: PairedRoutingSample,
    generator,
    judge_name: str,
) -> BehaviorOutputRecord:
    model_output = generator.generate(sample)
    behavior = label_output(model_output, is_risk=sample.is_risk)
    return BehaviorOutputRecord(
        sample_id=sample.sample_id,
        paired_id=sample.paired_id,
        risk_label=sample.risk_label.value,
        carrier_type=sample.carrier_type.value,
        risk_category=sample.risk_category,
        visual_condition=sample.visual_condition,
        model_output=model_output,
        behavior_label=behavior.value,
        judge_name=judge_name,
        judge_raw={"method": "rule_based_keyword_labeler"},
    )


def run(
    model_name: str,
    dataset_path: Path,
    output_path: Path,
    device: str,
    resume: bool,
    max_samples: int | None,
    log_every: int,
) -> None:
    samples = load_paired_dataset(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    print(f"Loaded {len(samples)} v2 samples from {dataset_path}", flush=True)

    done_ids = _load_done_ids(output_path) if resume else set()
    pending = [sample for sample in samples if sample.sample_id not in done_ids]
    print(f"Pending samples: {len(pending)}", flush=True)
    if not pending:
        return

    print(f"Loading {model_name} on {device}", flush=True)
    generator, judge_name = load_generator(model_name, device)
    print("Model loaded.", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    errors: list[dict[str, str]] = []
    with output_path.open(mode, encoding="utf-8") as fout:
        for index, sample in enumerate(pending, start=1):
            start = time.time()
            try:
                record = _process_sample(sample, generator, judge_name)
                fout.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
                fout.write("\n")
                fout.flush()
            except Exception as exc:  # noqa: BLE001
                errors.append({"sample_id": sample.sample_id, "error": str(exc)})
                print(f"ERROR {sample.sample_id}: {exc}", flush=True)
                continue
            if index % log_every == 0 or index == len(pending):
                print(
                    f"[{index}/{len(pending)}] {sample.sample_id} "
                    f"{record.behavior_label} {time.time() - start:.1f}s",
                    flush=True,
                )

    if errors:
        err_path = output_path.parent / "errors.jsonl"
        err_path.write_text(
            "\n".join(json.dumps(error, ensure_ascii=False) for error in errors) + "\n",
            encoding="utf-8",
        )
        print(f"Errors: {len(errors)}; see {err_path}", flush=True)


def main() -> None:
    args = parse_args()
    run(
        model_name=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        device=args.device,
        resume=args.resume,
        max_samples=args.max_samples,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
