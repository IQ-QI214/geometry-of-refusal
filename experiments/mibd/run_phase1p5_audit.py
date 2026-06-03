# experiments/mibd/run_phase1p5_audit.py
"""
Phase 1.5 Probe Validity Audit entry point.

Usage:
  # InternVL3 harmfulness audit (rdo env, GPU 0)
  conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
    --model internvl3 --gpu 0 \
    --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
    --signal-type harmfulness \
    --data-dir data/saladbench_splits \
    --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench

  # InternVL3 refusal audit (needs --refusal-labels)
  conda run -n rdo python -m experiments.mibd.run_phase1p5_audit \
    --model internvl3 --gpu 0 \
    --config experiments/mibd/configs/phase1_probe_internvl3.yaml \
    --signal-type refusal \
    --refusal-labels results/mibd/phase1_probe/internvl3_8b/refusal_labels.json \
    --data-dir data/saladbench_splits \
    --mmsafety-dir /inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

print = functools.partial(print, flush=True)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

from experiments.mibd.audit.margins import compute_score_margins, condition_margin_table
from experiments.mibd.audit.permutation import permutation_auc
from experiments.mibd.audit.splits import (
    available_categories,
    cross_category_split,
    group_split_by_paired_id,
    held_out_split,
)
from experiments.mibd.config import load_experiment_config
from experiments.mibd.data.loaders import load_harmbench_phase1, load_mmsafety_figstep
from experiments.mibd.data.schema import MIBDSample
from experiments.mibd.eval.phase1p5_report import AuditResult, build_phase1p5_report
from experiments.mibd.extraction.pipeline import run_extraction
from experiments.mibd.probes.direction import mean_difference_direction, project_scores
from experiments.mibd.probes.metrics import binary_auc
from experiments.mibd.probes.train import (
    compute_static_transfer_aucs,
    find_best_locus,
    train_probes_for_condition,
)

# Type alias for hidden state maps returned by run_extraction
HiddenMap = dict[str, dict[tuple[int, int], dict[str, np.ndarray]]]


def _remap_labels(
    samples: list[MIBDSample],
    refusal_labels: dict[str, str],
) -> list[MIBDSample]:
    """Replace sample.label with refusal/compliance from refusal_labels JSON.

    Samples whose id is not in the map are dropped with a warning so the probe
    always has clean positive/negative sets.
    """
    remapped = []
    missing = 0
    for s in samples:
        behaviour = refusal_labels.get(s.id)
        if behaviour is None:
            missing += 1
            continue
        remapped.append(
            MIBDSample(
                id=s.id,
                text=s.text,
                image_path=s.image_path,
                label=behaviour,   # "refusal" or "compliance"
                category=s.category,
                source=s.source,
                paired_id=s.paired_id,
                visual_condition=s.visual_condition,
            )
        )
    if missing:
        print(f"[run_phase1p5_audit] WARNING: {missing} samples had no refusal label and were dropped")
    return remapped


def _probe_auc_on_split(
    adapter,
    train_samples: list[MIBDSample],
    test_samples: list[MIBDSample],
    layers: tuple[int, ...],
    token_positions: tuple[int, ...],
    seed: int,
    vc: str,
    layer: int,
    pos: int,
    pos_label: str,
    neg_label: str,
) -> float:
    """Extract hidden states for train/test splits separately, train probe, return AUC.

    Only extracts for the given visual condition to keep GPU time low.
    Returns -1.0 (N/A sentinel) if either split lacks both classes or is too small.
    """
    vc_train = [s for s in train_samples if s.visual_condition == vc]
    vc_test = [s for s in test_samples if s.visual_condition == vc]

    # Early label checks before any GPU extraction
    train_labels = {s.label for s in vc_train}
    test_labels = {s.label for s in vc_test}
    if pos_label not in train_labels or neg_label not in train_labels:
        return -1.0
    if pos_label not in test_labels or neg_label not in test_labels:
        return -1.0

    train_pos_n = sum(1 for s in vc_train if s.label == pos_label)
    train_neg_n = sum(1 for s in vc_train if s.label == neg_label)
    if train_pos_n < 2 or train_neg_n < 2:
        return -1.0

    train_hidden = run_extraction(adapter, vc_train, layers, token_positions, seed=seed)
    test_hidden = run_extraction(adapter, vc_test, layers, token_positions, seed=seed)

    train_lp = train_hidden.get(vc, {}).get((layer, pos), {})
    test_lp = test_hidden.get(vc, {}).get((layer, pos), {})

    h_train = train_lp.get(pos_label)
    l_train = train_lp.get(neg_label)
    h_test = test_lp.get(pos_label)
    l_test = test_lp.get(neg_label)

    if h_train is None or l_train is None or h_test is None or l_test is None:
        return -1.0
    if len(h_train) < 2 or len(l_train) < 2 or len(h_test) < 1 or len(l_test) < 1:
        return -1.0

    direction = mean_difference_direction(h_train, l_train)
    test_all = np.vstack([h_test, l_test])
    test_labels = np.array([1] * len(h_test) + [0] * len(l_test))
    return float(binary_auc(test_labels, project_scores(test_all, direction)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3vl", "internvl3", "gemma3"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", required=True)
    parser.add_argument("--signal-type", default="harmfulness", choices=["harmfulness", "refusal"])
    parser.add_argument("--refusal-labels", default=None,
                        help="Path to JSON {sample_id: 'refusal'|'compliance'} (required for --signal-type refusal)")
    parser.add_argument("--data-dir", default="data/saladbench_splits")
    parser.add_argument(
        "--mmsafety-dir",
        default="/inspire/hdd/global_user/wenming-253108090054/czk/MML/dataset/mm-safebench",
    )
    parser.add_argument("--log-file", default=None,
                        help="Path to save a copy of all stdout output (tee to file)")
    args = parser.parse_args()

    if args.signal_type == "refusal" and args.refusal_labels is None:
        parser.error("--refusal-labels is required when --signal-type=refusal")

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = log_path.open("w", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        sys.stderr = _Tee(sys.__stderr__, _log_fh)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    cfg = load_experiment_config(args.config)
    print(f"[run_phase1p5_audit] model={args.model} gpu={args.gpu} signal={args.signal_type} "
          f"conditions={cfg.visual_conditions} layers={len(cfg.layers)}")

    print(f"[run_phase1p5_audit] loading model from {cfg.model_id} ...")
    if args.model == "qwen3vl":
        from experiments.mibd.models.adapters import Qwen3VLAdapter
        from experiments.mibd.models.loader import load_qwen3vl
        model, processor = load_qwen3vl(cfg.model_id, device=device)
        adapter = Qwen3VLAdapter(model=model, processor=processor, device=device)
    elif args.model == "gemma3":
        from experiments.mibd.models.loader import load_gemma3
        from experiments.mibd.models.adapters import Gemma3Adapter
        model, processor = load_gemma3(cfg.model_id, device=device)
        adapter = Gemma3Adapter(model=model, processor=processor, device=device)
    else:
        from experiments.mibd.models.adapters import InternVL3Adapter
        from experiments.mibd.models.loader import load_internvl3
        model, tokenizer = load_internvl3(cfg.model_id, device=device)
        adapter = InternVL3Adapter(model=model, tokenizer=tokenizer, device=device)
    print(f"[run_phase1p5_audit] model loaded, num_llm_layers={adapter.num_llm_layers}")

    print("[run_phase1p5_audit] loading datasets ...")
    text_conditions = [vc for vc in cfg.visual_conditions if vc != "FigStep"]
    text_samples = load_harmbench_phase1(
        data_dir=args.data_dir,
        visual_conditions=text_conditions,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )
    figstep_samples: list[MIBDSample] = []
    if "FigStep" in cfg.visual_conditions:
        figstep_harmful = load_mmsafety_figstep(
            mmsafety_dir=args.mmsafety_dir,
            max_samples=cfg.max_samples // len(cfg.visual_conditions),
            seed=cfg.seed,
        )
        import random as _random
        _rng = _random.Random(cfg.seed)
        harmful_image_paths = [s.image_path for s in figstep_harmful if s.image_path]
        vtext_harmless = [s for s in text_samples if s.label == "harmless" and s.visual_condition == "V-text"]
        figstep_harmless = [
            MIBDSample.from_dict({
                "id": s.id + "_figstep",
                "text": s.text,
                "image_path": _rng.choice(harmful_image_paths) if harmful_image_paths else None,
                "label": "harmless",
                "category": s.category,
                "source": s.source,
                "paired_id": None,
                "visual_condition": "FigStep",
            })
            for s in vtext_harmless
        ]
        figstep_samples = figstep_harmful + figstep_harmless

    all_samples = text_samples + figstep_samples
    print(f"[run_phase1p5_audit] loaded {len(all_samples)} samples")

    # Remap labels for refusal signal
    pos_label = "harmful"
    neg_label = "harmless"
    if args.signal_type == "refusal":
        with open(args.refusal_labels) as f:
            refusal_labels = json.loads(f.read())
        all_samples = _remap_labels(all_samples, refusal_labels)
        pos_label = "refusal"
        neg_label = "compliance"
        print(f"[run_phase1p5_audit] after refusal remap: {len(all_samples)} samples")

    # --- Full extraction on all samples (for train AUC, permutation, margins) ---
    print("[run_phase1p5_audit] extracting hidden states (full dataset) ...")
    all_hidden: HiddenMap = run_extraction(
        adapter=adapter,
        samples=all_samples,
        layers=cfg.layers,
        token_positions=cfg.token_positions,
        seed=cfg.seed,
    )

    # Train full probes per condition to find best locus
    probe_results_by_condition = {
        vc: train_probes_for_condition(all_hidden[vc])
        for vc in all_hidden
    }

    vtext_probes = probe_results_by_condition.get("V-text", {})
    if not vtext_probes:
        print("[run_phase1p5_audit] ERROR: no V-text probes trained, cannot continue")
        return

    best_layer, best_pos = find_best_locus(vtext_probes)
    print(f"[run_phase1p5_audit] best locus: layer={best_layer} pos={best_pos} "
          f"auc={vtext_probes[(best_layer, best_pos)]['auc']:.4f}")

    # V-text probe direction for static transfer
    vtext_direction = vtext_probes[(best_layer, best_pos)]["direction"]

    # Static transfer AUCs (V-text probe applied to other conditions)
    static_transfer = compute_static_transfer_aucs(
        source_direction=vtext_direction,
        source_condition="V-text",
        target_hidden_maps=all_hidden,
        layer=best_layer,
        pos=best_pos,
    )

    # All condition directions at best locus
    condition_directions = {
        vc: probe_results_by_condition[vc][(best_layer, best_pos)]["direction"]
        for vc in probe_results_by_condition
        if (best_layer, best_pos) in probe_results_by_condition[vc]
    }

    # Margin for V-text source (for transfer drop calculation)
    vtext_hidden_map = all_hidden.get("V-text", {})
    vtext_margin_dict = compute_score_margins(
        vtext_direction, vtext_hidden_map, best_layer, best_pos
    )
    vtext_mean_gap = vtext_margin_dict["mean_gap"]

    print("[run_phase1p5_audit] running permutation test on V-text best locus ...")
    perm_auc = permutation_auc(
        vtext_hidden_map, layer=best_layer, pos=best_pos, n_permutations=100, seed=cfg.seed
    )
    print(f"[run_phase1p5_audit] permutation AUC: {perm_auc:.4f}")

    # Per-condition margin table
    all_margins = condition_margin_table(condition_directions, all_hidden, best_layer, best_pos)

    # --- Held-out split: proper sample-level stratified split ---
    print("[run_phase1p5_audit] computing held-out split AUC (V-text) ...")
    vtext_samples = [s for s in all_samples if s.visual_condition == "V-text"]
    ho_train_samples, ho_test_samples = held_out_split(vtext_samples, test_frac=0.2, seed=cfg.seed)
    held_out_auc = _probe_auc_on_split(
        adapter=adapter,
        train_samples=ho_train_samples,
        test_samples=ho_test_samples,
        layers=cfg.layers,
        token_positions=cfg.token_positions,
        seed=cfg.seed,
        vc="V-text",
        layer=best_layer,
        pos=best_pos,
        pos_label=pos_label,
        neg_label=neg_label,
    )
    print(f"[run_phase1p5_audit] held-out AUC: {held_out_auc:.4f}")

    # --- Group split: split by paired_id so both pair members stay together ---
    has_pairs = any(s.paired_id is not None for s in vtext_samples)
    if has_pairs:
        print("[run_phase1p5_audit] computing group split AUC (V-text, paired-id isolated) ...")
        g_train_samples, g_test_samples = group_split_by_paired_id(
            vtext_samples, test_frac=0.2, seed=cfg.seed
        )
        group_split_auc = _probe_auc_on_split(
            adapter=adapter,
            train_samples=g_train_samples,
            test_samples=g_test_samples,
            layers=cfg.layers,
            token_positions=cfg.token_positions,
            seed=cfg.seed,
            vc="V-text",
            layer=best_layer,
            pos=best_pos,
            pos_label=pos_label,
            neg_label=neg_label,
        )
        print(f"[run_phase1p5_audit] group split AUC: {group_split_auc:.4f}")
    else:
        group_split_auc = -1.0
        print("[run_phase1p5_audit] no paired IDs found — group split AUC skipped")

    # --- Cross-category split: leave-one-category-out over V-text samples ---
    cats = available_categories(vtext_samples)
    cross_cat_aucs: dict[str, float] = {}
    print(f"[run_phase1p5_audit] computing cross-category AUCs for {len(cats)} categories ...")
    for test_cat in cats:
        cc_train_samples, cc_test_samples = cross_category_split(vtext_samples, test_category=test_cat)
        if not cc_train_samples or not cc_test_samples:
            print(f"[run_phase1p5_audit]   category={test_cat} skipped (empty split)")
            continue
        auc = _probe_auc_on_split(
            adapter=adapter,
            train_samples=cc_train_samples,
            test_samples=cc_test_samples,
            layers=cfg.layers,
            token_positions=cfg.token_positions,
            seed=cfg.seed,
            vc="V-text",
            layer=best_layer,
            pos=best_pos,
            pos_label=pos_label,
            neg_label=neg_label,
        )
        cross_cat_aucs[test_cat] = auc
        auc_str = "N/A (single-class test set)" if auc == -1.0 else f"{auc:.4f}"
        print(f"[run_phase1p5_audit]   category={test_cat} cross-cat AUC={auc_str}")

    # --- Build AuditResult per visual condition ---
    audit_results: list[AuditResult] = []
    for vc, pr in probe_results_by_condition.items():
        if not pr:
            continue
        train_auc = pr.get((best_layer, best_pos), {}).get("auc", -1.0)

        # held_out, group_split, cross_cat are computed on V-text; other conditions reuse
        if vc == "V-text":
            vc_held_out_auc = held_out_auc
            vc_group_split_auc = group_split_auc
            vc_cross_cat_aucs = cross_cat_aucs
        else:
            vc_held_out_auc = -1.0
            vc_group_split_auc = -1.0
            vc_cross_cat_aucs = {}

        # Per-condition permutation AUC (reuse V-text result, compute for others)
        if vc == "V-text":
            vc_perm_auc = perm_auc
        else:
            vc_hidden = all_hidden.get(vc, {})
            try:
                vc_perm_auc = permutation_auc(
                    vc_hidden, layer=best_layer, pos=best_pos,
                    n_permutations=50, seed=cfg.seed
                )
            except (KeyError, ValueError):
                vc_perm_auc = float("nan")

        # Static transfer margin drop (V-text only)
        transfer_margin_drop: dict[str, float] = {}
        if vc == "V-text":
            for target_vc, target_hidden in all_hidden.items():
                if target_vc == "V-text":
                    continue
                try:
                    target_margins = compute_score_margins(
                        vtext_direction, target_hidden, best_layer, best_pos
                    )
                    drop = vtext_mean_gap - target_margins["mean_gap"]
                    transfer_margin_drop[target_vc] = drop
                except (KeyError, ValueError):
                    pass

        margins = all_margins.get(vc, {})
        audit_results.append(AuditResult(
            model_id=cfg.model_id,
            signal_type=args.signal_type,
            visual_condition=vc,
            train_auc=float(train_auc),
            held_out_auc=float(vc_held_out_auc),
            group_split_auc=float(vc_group_split_auc),
            permutation_auc=float(vc_perm_auc),
            cross_category_aucs=vc_cross_cat_aucs,
            margins=margins,
            static_transfer_margin_drop=transfer_margin_drop,
        ))

    report_md = build_phase1p5_report(
        audit_results=audit_results,
        model_id=cfg.model_id,
        signal_type=args.signal_type,
    )
    report_path = cfg.output_dir / "phase1p5_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md)
    print(f"[run_phase1p5_audit] report saved to {report_path}")


if __name__ == "__main__":
    main()
