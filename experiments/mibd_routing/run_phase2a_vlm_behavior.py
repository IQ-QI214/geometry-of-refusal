"""Phase 2A GPU behavior generation runner.

生成 InternVL3-8B 或 Qwen3-VL-8B 对 paired_dataset 的真实模型输出，
并用规则 judge 打行为标签。支持 --resume 从上次中断处继续。

使用方式（在 GPU 节点上）：

  # InternVL3-8B  (rdo env)
  conda run -n rdo python -m experiments.mibd_routing.run_phase2a_vlm_behavior \\
    --model internvl3_8b \\
    --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \\
    --output results/mibd_routing/behavior_labels/internvl3_8b/behavior_outputs.jsonl \\
    --device cuda:0

  # Qwen3-VL-8B  (qwen3-vl env)
  conda run -n qwen3-vl python -m experiments.mibd_routing.run_phase2a_vlm_behavior \\
    --model qwen3_vl_8b \\
    --dataset results/mibd_routing/paired_dataset/phase2a_real_pilot/paired_dataset.jsonl \\
    --output results/mibd_routing/behavior_labels/qwen3_vl_8b/behavior_outputs.jsonl \\
    --device cuda:0

  # 加 --resume 从中断处继续（已写入条目自动跳过）
  ... --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.mibd_routing.behavior.generate_outputs import (
    BehaviorOutputRecord,
    load_paired_dataset,
)
from experiments.mibd_routing.behavior.label_outputs import label_output
from experiments.mibd_routing.data.schema import PairedRoutingSample

MODEL_CHOICES = ("internvl3_8b", "qwen3_vl_8b")

MODEL_PATHS = {
    "internvl3_8b": "/inspire/hdd/global_user/wenming-253108090054/models/InternVL3-8B",
    "qwen3_vl_8b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B",
}


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_generator(model_name: str, device: str):
    """Load model and return a generator instance."""
    from experiments.mibd.models.loader import load_internvl3, load_qwen3vl
    from experiments.mibd_routing.behavior.vlm_generators import (
        InternVL3Generator,
        Qwen3VLGenerator,
    )

    model_path = MODEL_PATHS[model_name]
    if model_name == "internvl3_8b":
        model, tokenizer = load_internvl3(model_path, device=device)
        return InternVL3Generator(model, tokenizer, device=device), "internvl3_8b"
    else:
        model, processor = load_qwen3vl(model_path, device=device)
        return Qwen3VLGenerator(model, processor, device=device), "qwen3_vl_8b"


# ---------------------------------------------------------------------------
# resume helpers
# ---------------------------------------------------------------------------

def _load_done_ids(output_path: Path) -> set[str]:
    """Return sample_ids already written to output file."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            done.add(rec["sample_id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return done


# ---------------------------------------------------------------------------
# single sample processing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run(
    model_name: str,
    dataset_path: Path,
    output_path: Path,
    device: str,
    resume: bool,
    log_every: int,
) -> None:
    samples = load_paired_dataset(dataset_path)
    print(f"Loaded {len(samples)} samples from {dataset_path}", flush=True)

    done_ids: set[str] = set()
    if resume:
        done_ids = _load_done_ids(output_path)
        print(f"Resume: skipping {len(done_ids)} already-done samples", flush=True)

    pending = [s for s in samples if s.sample_id not in done_ids]
    if not pending:
        print("All samples already done. Nothing to do.", flush=True)
        return

    print(f"Loading model {model_name} on {device} …", flush=True)
    generator, judge_name = load_generator(model_name, device)
    print("Model loaded.", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    errors: list[dict] = []

    with open(output_path, mode, encoding="utf-8") as fout:
        for i, sample in enumerate(pending):
            t0 = time.time()
            try:
                record = _process_sample(sample, generator, judge_name)
                fout.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
                fout.write("\n")
                fout.flush()
            except Exception as exc:  # noqa: BLE001
                errors.append({"sample_id": sample.sample_id, "error": str(exc)})
                print(
                    f"  ERROR sample {sample.sample_id}: {exc}", flush=True
                )
                continue

            if (i + 1) % log_every == 0 or (i + 1) == len(pending):
                elapsed = time.time() - t0
                print(
                    f"  [{i+1}/{len(pending)}] {sample.sample_id}"
                    f"  {record.behavior_label:<20}"
                    f"  {elapsed:.1f}s",
                    flush=True,
                )

    # write error log
    if errors:
        err_path = output_path.parent / "errors.jsonl"
        with open(err_path, "w", encoding="utf-8") as ferr:
            for e in errors:
                ferr.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Errors: {len(errors)} — see {err_path}", flush=True)

    # summary
    _print_summary(output_path)


def _print_summary(output_path: Path) -> None:
    from collections import Counter

    records = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    total = len(records)
    label_counts = Counter(r["behavior_label"] for r in records)
    carrier_counts = Counter(r["carrier_type"] for r in records)
    risk_label_counts = Counter(r["risk_label"] for r in records)

    print(f"\n=== Behavior generation summary ===")
    print(f"total records : {total}")
    print(f"risk_label    : {dict(risk_label_counts)}")
    print(f"carrier_type  : {dict(carrier_counts)}")
    print(f"behavior_label: {dict(label_counts)}")

    # routing failure rate (risk but not safe_policy)
    risk_recs = [r for r in records if r["risk_label"] == "risk"]
    if risk_recs:
        unsafe = [r for r in risk_recs if r["behavior_label"] != "safe_policy"]
        rate = len(unsafe) / len(risk_recs)
        print(f"unsafe-despite-risk: {len(unsafe)}/{len(risk_recs)} = {rate:.1%}")

    print(f"output: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2A GPU VLM behavior generation runner"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Which VLM to use",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to paired_dataset.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output behavior_outputs.jsonl path",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device (default: cuda:0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output and skip already-done sample_ids",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N samples (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        model_name=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        device=args.device,
        resume=args.resume,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
