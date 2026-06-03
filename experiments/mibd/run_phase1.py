# experiments/mibd/run_phase1.py
"""
Phase 1 experiment entry point.

Usage:
  # Qwen3-VL (qwen3-vl env, GPU 0)
  conda run -n qwen3-vl python -m experiments.mibd.run_phase1 \
    --model qwen3vl --gpu 0 \
    --config experiments/mibd/configs/phase1_probe.yaml \
    --data-dir data/saladbench_splits \
    --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench

  # InternVL3 (rdo env, GPU 1)
  conda run -n rdo python -m experiments.mibd.run_phase1 \
    --model internvl3 --gpu 1 \
    --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
    --data-dir data/saladbench_splits \
    --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench
"""
from __future__ import annotations

import argparse
import os
import sys
import functools

# Force line-buffered stdout so every print() appears immediately in tee/tmux
print = functools.partial(print, flush=True)

from experiments.mibd.config import load_experiment_config
from experiments.mibd.data.loaders import load_harmbench_phase1, load_mmsafety_figstep
from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.eval.phase1_report import build_go_no_go_report, Phase1ResultSet, LocusResult
from experiments.mibd.eval.summary import build_phase1_summary, save_summary
from experiments.mibd.extraction.pipeline import run_extraction
from experiments.mibd.probes.train import (
    train_probes_for_condition,
    compute_condition_cosines,
    compute_static_transfer_aucs,
    find_best_locus,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3vl", "internvl3"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data/saladbench_splits")
    parser.add_argument(
        "--mmsafety-dir",
        default="/inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    cfg = load_experiment_config(args.config)
    print(f"[run_phase1] model={args.model} gpu={args.gpu} "
          f"conditions={cfg.visual_conditions} layers={len(cfg.layers)} "
          f"max_samples={cfg.max_samples}")

    print(f"[run_phase1] loading model from {cfg.model_id} ...")
    if args.model == "qwen3vl":
        from experiments.mibd.models.loader import load_qwen3vl
        from experiments.mibd.models.adapters import Qwen3VLAdapter
        model, processor = load_qwen3vl(cfg.model_id, device=device)
        adapter = Qwen3VLAdapter(model=model, processor=processor, device=device)
    else:
        from experiments.mibd.models.loader import load_internvl3
        from experiments.mibd.models.adapters import InternVL3Adapter
        model, tokenizer = load_internvl3(cfg.model_id, device=device)
        adapter = InternVL3Adapter(model=model, tokenizer=tokenizer, device=device)
    print(f"[run_phase1] model loaded, num_llm_layers={adapter.num_llm_layers}")

    print("[run_phase1] loading datasets ...")
    text_conditions = [vc for vc in cfg.visual_conditions if vc != "FigStep"]
    text_samples = load_harmbench_phase1(
        data_dir=args.data_dir,
        visual_conditions=text_conditions,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )
    figstep_samples = []
    if "FigStep" in cfg.visual_conditions:
        figstep_samples = load_mmsafety_figstep(
            mmsafety_dir=args.mmsafety_dir,
            max_samples=cfg.max_samples // len(cfg.visual_conditions),
            seed=cfg.seed,
        )
        # FigStep has no harmless pairs; reuse V-text harmless as FigStep harmless
        vtext_harmless = [s for s in text_samples if s.label == "harmless" and s.visual_condition == "V-text"]
        figstep_harmless = [
            MIBDSample.from_dict({
                "id": s.id + "_figstep",
                "text": s.text,
                "image_path": None,
                "label": "harmless",
                "category": s.category,
                "source": s.source,
                "paired_id": None,
                "visual_condition": "FigStep",
            })
            for s in vtext_harmless
        ]
        figstep_samples = figstep_samples + figstep_harmless
    all_samples = text_samples + figstep_samples
    print(f"[run_phase1] loaded {len(all_samples)} samples")
    print(f"[run_phase1] starting hidden state extraction "
          f"({len(cfg.layers)} layers × {len(cfg.token_positions)} positions × {len(all_samples)} samples) ...")
    all_hidden = run_extraction(
        adapter=adapter,
        samples=all_samples,
        layers=cfg.layers,
        token_positions=cfg.token_positions,
        seed=cfg.seed,
    )

    probe_results_by_condition = {
        vc: train_probes_for_condition(all_hidden[vc])
        for vc in all_hidden
    }

    vtext_probes = probe_results_by_condition.get("V-text", {})
    if vtext_probes:
        best_layer, best_pos = find_best_locus(vtext_probes)
        condition_directions = {
            vc: probe_results_by_condition[vc][(best_layer, best_pos)]["direction"]
            for vc in probe_results_by_condition
            if (best_layer, best_pos) in probe_results_by_condition[vc]
        }
        condition_cosines = compute_condition_cosines(condition_directions)
        static_transfer = compute_static_transfer_aucs(
            source_direction=vtext_probes[(best_layer, best_pos)]["direction"],
            source_condition="V-text",
            target_hidden_maps=all_hidden,
            layer=best_layer,
            pos=best_pos,
        )
    else:
        condition_cosines = {}
        static_transfer = {}

    summary = build_phase1_summary(
        model_id=cfg.model_id,
        signal_type="harmfulness",
        probe_results_by_condition=probe_results_by_condition,
        condition_cosines=condition_cosines,
        static_transfer_auc=static_transfer,
    )
    summary_path = save_summary(summary, output_dir=str(cfg.output_dir))
    print(f"[run_phase1] summary saved to {summary_path}")

    result_set = Phase1ResultSet(
        model_id=summary["model_id"],
        signal_type=summary["signal_type"],
        results=[
            LocusResult(
                visual_condition=r["visual_condition"],
                layer=r["layer"],
                token_pos=r["token_pos"],
                auc=r["auc"],
            )
            for r in summary["results"]
        ],
        condition_cosines={
            tuple(k.split("|", 1)): v
            for k, v in summary["condition_cosines"].items()
        },
        static_transfer_auc={
            tuple(k.split("|", 1)): v
            for k, v in summary["static_transfer_auc"].items()
        },
    )
    report = build_go_no_go_report(result_set)
    report_path = cfg.output_dir / "phase1_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.markdown)
    print(f"[run_phase1] decision={report.decision}")
    print(f"[run_phase1] report saved to {report_path}")


if __name__ == "__main__":
    main()
