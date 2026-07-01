"""CPU-only data audit for v2 paired routing datasets.

Motivation
----------
The v3 run showed that fixing the blank-placeholder confound was *necessary but
not sufficient*: held-out cross-carrier transfer did not drop, because the safe
controls still differed from the risk carriers in ways a probe can exploit
(canvas size 336 vs 500, pixel std 27 vs 73, per-sample carrier mismatch on
90/200 pairs, and short fixed neutral text vs variable risk text). Those are
*visual/source distribution confounds*, not harmful semantics.

This audit inspects a paired dataset ``paired_dataset.jsonl`` on CPU and reports
the confounds that must be clean **before** any GPU re-extraction:

* image size distribution (safe vs risk)
* pixel mean/std distribution after a common resize (safe vs risk)
* per-sample carrier match rate (safe carrier vs risk carrier)
* text length distribution (safe vs risk)
* category balance
* pair integrity (every paired_id has exactly one safe + one risk, matched
  category/carrier)

It emits a JSON report and a Go/No-Go verdict. Exit code is non-zero when the
dataset is judged confounded, so it can gate a pipeline before GPU extraction.

PIL is optional: when unavailable, image-pixel checks are skipped and flagged as
``skipped_no_pil`` rather than failing the whole audit (so the tool still runs
in a minimal env). numpy-only otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on env
    _PIL_AVAILABLE = False


def load_paired_records(jsonl_path: str | Path) -> list[dict]:
    """Load a paired_dataset.jsonl into a list of dicts."""
    path = Path(jsonl_path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def _carrier_of(record: dict) -> str:
    """Carrier of the image actually attached to this record.

    The safe record's ``image_path`` points at the neutral safe image; its
    filename encodes the neutral carrier (``neutral_<carrier>_..``). The risk
    record's ``image_path`` points at the MM-SafetyBench carrier image, whose
    directory encodes the carrier (``images_wr`` -> typographic,
    ``images_figstep`` -> figstep).
    """
    path = (record.get("image_path") or "").lower()
    if "neutral_figstep" in path or "images_figstep" in path or "figstep" in path:
        return "figstep"
    if "neutral_typographic" in path or "images_wr" in path or "typographic" in path:
        return "typographic"
    return "unknown"


def _text_length(record: dict) -> int:
    md = record.get("metadata", {})
    text = md.get("risk_text") or md.get("replaced_prompt") or record.get("question") or ""
    return len(text)


def audit_pair_integrity(records: list[dict]) -> dict:
    """Every paired_id must have exactly one safe and one risk of same category."""
    by_pair: dict[str, list[dict]] = {}
    for r in records:
        by_pair.setdefault(r["paired_id"], []).append(r)

    problems = []
    safe_count = risk_count = 0
    for pid, rows in by_pair.items():
        labels = sorted(r["risk_label"] for r in rows)
        if labels != ["risk", "safe"]:
            problems.append({"paired_id": pid, "issue": f"labels={labels}"})
            continue
        safe_count += 1
        risk_count += 1
        cats = {r["risk_category"] for r in rows}
        if len(cats) != 1:
            problems.append({"paired_id": pid, "issue": f"category_mismatch={cats}"})
    return {
        "pairs": len(by_pair),
        "safe": safe_count,
        "risk": risk_count,
        "pair_problems": len(problems),
        "problem_examples": problems[:10],
    }


def audit_carrier_match(records: list[dict]) -> dict:
    """Per-sample: does the safe image carrier match its paired risk carrier?"""
    by_pair: dict[str, dict[str, dict]] = {}
    for r in records:
        by_pair.setdefault(r["paired_id"], {})[r["risk_label"]] = r

    matched = mismatched = 0
    mismatch_examples = []
    for pid, rows in by_pair.items():
        if "safe" not in rows or "risk" not in rows:
            continue
        safe_carrier = _carrier_of(rows["safe"])
        risk_carrier = _carrier_of(rows["risk"])
        if safe_carrier == risk_carrier and safe_carrier != "unknown":
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {"paired_id": pid, "safe": safe_carrier, "risk": risk_carrier}
                )
    total = matched + mismatched
    return {
        "matched": matched,
        "mismatched": mismatched,
        "match_rate": round(matched / total, 6) if total else 0.0,
        "mismatch_examples": mismatch_examples,
    }


def audit_text_length(records: list[dict]) -> dict:
    safe_lens = [_text_length(r) for r in records if r["risk_label"] == "safe"]
    risk_lens = [_text_length(r) for r in records if r["risk_label"] == "risk"]

    def _stats(xs: list[int]) -> dict:
        arr = np.asarray(xs, dtype=np.float64) if xs else np.zeros(0)
        return {
            "n": len(xs),
            "mean": round(float(arr.mean()), 3) if xs else 0.0,
            "std": round(float(arr.std()), 3) if xs else 0.0,
            "min": int(arr.min()) if xs else 0,
            "max": int(arr.max()) if xs else 0,
        }

    return {"safe": _stats(safe_lens), "risk": _stats(risk_lens)}


def _image_stats(path: str, resize_to: int) -> tuple[tuple[int, int] | None, float | None]:
    """Return (original_size, pixel_std_after_resize) or (None, None) on failure."""
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return None, None
    try:
        img = Image.open(p).convert("RGB")
    except Exception:  # pragma: no cover - corrupt/missing image
        return None, None
    size = img.size  # (w, h)
    resized = img.resize((resize_to, resize_to))
    arr = np.asarray(resized, dtype=np.float64)
    return size, float(arr.std())


def audit_images(records: list[dict], resize_to: int = 336, max_images: int | None = None) -> dict:
    if not _PIL_AVAILABLE:
        return {"status": "skipped_no_pil"}

    safe_sizes: list[tuple[int, int]] = []
    risk_sizes: list[tuple[int, int]] = []
    safe_stds: list[float] = []
    risk_stds: list[float] = []

    for label, sizes, stds in (
        ("safe", safe_sizes, safe_stds),
        ("risk", risk_sizes, risk_stds),
    ):
        subset = [r for r in records if r["risk_label"] == label]
        if max_images is not None:
            subset = subset[:max_images]
        for r in subset:
            size, std = _image_stats(r.get("image_path", ""), resize_to)
            if size is not None:
                sizes.append(size)
            if std is not None:
                stds.append(std)

    def _size_summary(sizes: list[tuple[int, int]]) -> dict:
        if not sizes:
            return {"n": 0, "unique_sizes": []}
        uniq = sorted({f"{w}x{h}" for (w, h) in sizes})
        return {"n": len(sizes), "unique_sizes": uniq}

    def _std_summary(stds: list[float]) -> dict:
        if not stds:
            return {"n": 0, "mean_std": None}
        return {"n": len(stds), "mean_std": round(float(np.mean(stds)), 3)}

    return {
        "status": "ok",
        "resize_to": resize_to,
        "safe_sizes": _size_summary(safe_sizes),
        "risk_sizes": _size_summary(risk_sizes),
        "safe_pixel_std": _std_summary(safe_stds),
        "risk_pixel_std": _std_summary(risk_stds),
    }


def audit_category_balance(records: list[dict]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for r in records:
        cat = r["risk_category"]
        counts.setdefault(cat, {"safe": 0, "risk": 0})
        counts[cat][r["risk_label"]] += 1
    balanced = all(v["safe"] == v["risk"] for v in counts.values())
    return {"balanced": balanced, "per_category": counts}


def build_data_audit_report(
    records: list[dict],
    dataset_name: str,
    resize_to: int = 336,
    max_images: int | None = None,
    carrier_match_min: float = 0.99,
    pixel_std_ratio_max: float = 1.5,
) -> dict:
    """Assemble the full data-audit report with a Go/No-Go verdict."""
    integrity = audit_pair_integrity(records)
    carrier = audit_carrier_match(records)
    text = audit_text_length(records)
    images = audit_images(records, resize_to=resize_to, max_images=max_images)
    category = audit_category_balance(records)

    problems: list[str] = []
    if integrity["pair_problems"] > 0:
        problems.append(f"pair_integrity: {integrity['pair_problems']} problems")
    if carrier["match_rate"] < carrier_match_min:
        problems.append(
            f"carrier_match_rate={carrier['match_rate']} < {carrier_match_min}"
        )
    if not category["balanced"]:
        problems.append("category imbalance between safe/risk")

    if images.get("status") == "ok":
        safe_sizes = set(images["safe_sizes"]["unique_sizes"])
        risk_sizes = set(images["risk_sizes"]["unique_sizes"])
        if safe_sizes and risk_sizes and safe_sizes != risk_sizes:
            problems.append(
                f"image size mismatch: safe={sorted(safe_sizes)} risk={sorted(risk_sizes)}"
            )
        s = images["safe_pixel_std"]["mean_std"]
        r = images["risk_pixel_std"]["mean_std"]
        if s and r and s > 0:
            ratio = max(r / s, s / r)
            if ratio > pixel_std_ratio_max:
                problems.append(
                    f"pixel_std ratio {round(ratio, 3)} > {pixel_std_ratio_max} "
                    f"(safe={s}, risk={r})"
                )

    clean = len(problems) == 0
    return {
        "summary": {
            "dataset": dataset_name,
            "clean": clean,
            "problems": problems,
            "verdict": (
                "CLEAN: no gross visual/source confound detected; safe to extract."
                if clean
                else "CONFOUNDED: fix the listed problems before GPU extraction."
            ),
        },
        "pair_integrity": integrity,
        "carrier_match": carrier,
        "text_length": text,
        "images": images,
        "category_balance": category,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU data audit for paired routing datasets.")
    parser.add_argument("--dataset", required=True, type=Path, help="paired_dataset.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="output JSON report path")
    parser.add_argument("--name", default=None, help="dataset name label")
    parser.add_argument("--resize-to", type=int, default=336)
    parser.add_argument("--max-images", type=int, default=None, help="cap images scanned per label")
    parser.add_argument("--carrier-match-min", type=float, default=0.99)
    parser.add_argument("--pixel-std-ratio-max", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_paired_records(args.dataset)
    report = build_data_audit_report(
        records,
        dataset_name=args.name or args.dataset.parent.name,
        resize_to=args.resize_to,
        max_images=args.max_images,
        carrier_match_min=args.carrier_match_min,
        pixel_std_ratio_max=args.pixel_std_ratio_max,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[data-audit] wrote {args.out}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0 if report["summary"]["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
