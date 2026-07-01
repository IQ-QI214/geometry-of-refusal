"""V2 Phase 2B GPU hidden extraction plus CPU probe summary.

Run this on the offline GPU machine after building a v2 paired dataset. It
extracts hidden states for safe/risk paired samples, writes a compressed NPZ,
and computes first-pass single-direction and v2 subspace AUCs per locus.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.mibd.extraction.pipeline import run_extraction_with_metadata
from experiments.mibd.models.adapters import InternVL3Adapter, Qwen3VLAdapter
from experiments.mibd.models.loader import load_internvl3, load_qwen3vl
from experiments.mibd.probes.train import train_probes_for_condition
from experiments.mibd_routing.behavior.generate_outputs import load_paired_dataset
from experiments.mibd_routing_v2.probes.subspace import evaluate_subspace_readout

MODEL_CHOICES = ("internvl3_8b", "qwen3_vl_8b")
MODEL_PATHS = {
    "internvl3_8b": "/inspire/hdd/global_user/wenming-253108090054/models/InternVL3-8B",
    "qwen3_vl_8b": "/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--positions", default="-1", help="Comma-separated token positions")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--subspace-rank", type=int, default=3)
    return parser.parse_args(argv)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _load_adapter(model_name: str, device: str):
    model_path = MODEL_PATHS[model_name]
    if model_name == "internvl3_8b":
        model, tokenizer = load_internvl3(model_path, device=device)
        return InternVL3Adapter(model=model, tokenizer=tokenizer, device=device)
    model, processor = load_qwen3vl(model_path, device=device)
    return Qwen3VLAdapter(model=model, processor=processor, device=device)


def _save_hidden_npz(output_path: Path, hidden: dict, row_metadata: dict | None = None) -> None:
    arrays: dict[str, np.ndarray] = {}
    manifest = []
    row_meta_out: dict[str, list[dict]] = {}
    for condition, locus_map in hidden.items():
        for (layer, pos), label_map in locus_map.items():
            for label, values in label_map.items():
                key = f"{condition}__layer{layer}__pos{pos}__{label}"
                arrays[key] = values
                manifest.append(
                    {
                        "key": key,
                        "visual_condition": condition,
                        "layer": layer,
                        "position": pos,
                        "label": label,
                        "shape": list(values.shape),
                    }
                )
                # Attach per-row metadata for this key, in stacking order, so a
                # later audit can recover sample_id / paired_id / row_index and
                # do pair-level splits (v3 could not: no row identity was saved).
                if row_metadata is not None:
                    rows = (
                        row_metadata.get(condition, {})
                        .get((layer, pos), {})
                        .get(label, [])
                    )
                    row_meta_out[key] = rows
    arrays["manifest_json"] = np.array(json.dumps(manifest, ensure_ascii=False))
    if row_metadata is not None:
        arrays["row_metadata_json"] = np.array(json.dumps(row_meta_out, ensure_ascii=False))
    np.savez_compressed(output_path, **arrays)


def _build_probe_summary(hidden: dict, subspace_rank: int) -> dict[str, object]:
    summary: dict[str, object] = {"conditions": {}}
    for condition, locus_map in hidden.items():
        condition_summary = []
        probes = train_probes_for_condition(
            locus_map,
            pos_label="harmful",
            neg_label="harmless",
        )
        for (layer, pos), result in sorted(probes.items()):
            label_map = locus_map[(layer, pos)]
            risk = label_map["harmful"]
            safe = label_map["harmless"]
            pooled = np.vstack([risk, safe])
            labels = np.array([1] * len(risk) + [0] * len(safe))
            subspace = evaluate_subspace_readout(
                labels=labels,
                risk_hidden=risk,
                safe_hidden=safe,
                pooled_hidden=pooled,
                rank=subspace_rank,
            )
            condition_summary.append(
                {
                    "layer": layer,
                    "position": pos,
                    "single_direction_auc": float(result["auc"]),
                    "subspace_auc": float(subspace.subspace_auc),
                    "subspace_gain": float(subspace.subspace_gain),
                    "rank": int(subspace.rank),
                }
            )
        summary["conditions"][condition] = condition_summary
    return summary


def run(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    device: str,
    layers: tuple[int, ...],
    positions: tuple[int, ...],
    max_samples: int | None,
    seed: int,
    subspace_rank: int,
) -> None:
    samples = load_paired_dataset(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    mibd_samples = [sample.to_mibd_sample() for sample in samples]

    print(
        f"Loaded {len(mibd_samples)} samples; model={model_name}; "
        f"layers={layers}; positions={positions}",
        flush=True,
    )
    adapter = _load_adapter(model_name, device)
    print(f"Model loaded; num_llm_layers={adapter.num_llm_layers}", flush=True)

    hidden, row_metadata = run_extraction_with_metadata(
        adapter=adapter,
        samples=mibd_samples,
        layers=layers,
        token_positions=positions,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_hidden_npz(output_dir / "hidden_states.npz", hidden, row_metadata=row_metadata)
    summary = _build_probe_summary(hidden, subspace_rank=subspace_rank)
    (output_dir / "probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'hidden_states.npz'}", flush=True)
    print(f"Wrote {output_dir / 'probe_summary.json'}", flush=True)


def main() -> None:
    args = parse_args()
    run(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        device=args.device,
        layers=_parse_int_tuple(args.layers),
        positions=_parse_int_tuple(args.positions),
        max_samples=args.max_samples,
        seed=args.seed,
        subspace_rank=args.subspace_rank,
    )


if __name__ == "__main__":
    main()
