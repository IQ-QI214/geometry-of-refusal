#!/usr/bin/env python3
"""compute_alignment.py — Cross-modal cosine alignment matrix.

계산 대상 (각 모델마다):
  c1 = cos(r_LLM, r_V-text)       VL 대齐训练 편이
  c2 = cos(r_V-text, r_V-blank)    이미지 토큰 추가 편이
  c3 = cos(r_LLM, r_V-blank)       두 인수 총 편이

"级联 예측 c1*c2 vs 실측 c3" 오차 검증.
결과 저장: results/ara_opd_vlm_0427/cross_modal_geometry_diag/cross_modal_alignment.json

Run from project root:
  python3 experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/compute_alignment.py
"""
import json
import math
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "ara_opd_vlm_0427" / "cross_modal_geometry_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a / a.norm()) @ (b / b.norm()))


def load_direction(mean_diffs_pt: Path, best_layer_json: Path) -> torch.Tensor:
    """mean_diffs.pt에서 best_layer 기준으로 방향 벡터 추출.

    mean_diffs shape: (n_eoi_toks, n_layers, d_model)
    pos는 음수 인덱스 (예: -5)이고, n_eoi_toks로 변환.
    """
    best = json.loads(best_layer_json.read_text())
    layer = best["layer"]
    pos = best.get("pos", -5)
    mean_diffs = torch.load(mean_diffs_pt, map_location="cpu")
    n_eoi_toks = mean_diffs.shape[0]
    # pos는 음수 (-5 ~ -1), pos_idx = pos + n_eoi_toks
    pos_idx = pos + n_eoi_toks
    return mean_diffs[pos_idx, layer]


def analyze_model(name: str, llm_dir: torch.Tensor,
                  vtext_dir: torch.Tensor, vblank_dir: torch.Tensor) -> dict:
    c1 = cosine(llm_dir, vtext_dir)
    c2 = cosine(vtext_dir, vblank_dir)
    c3 = cosine(llm_dir, vblank_dir)
    cascade_pred = c1 * c2
    angle_sum_pred = math.cos(math.acos(max(-1.0, min(1.0, c1))) +
                               math.acos(max(-1.0, min(1.0, c2))))
    return {
        "model": name,
        "c1_llm_vs_vtext": round(c1, 4),
        "c2_vtext_vs_vblank": round(c2, 4),
        "c3_llm_vs_vblank": round(c3, 4),
        "cascade_prediction_c1xc2": round(cascade_pred, 4),
        "angle_sum_prediction": round(angle_sum_pred, 4),
        "cascade_error_c3_minus_pred": round(c3 - cascade_pred, 4),
        "interpretation": (
            "cascade" if abs(c3 - cascade_pred) < 0.08
            else "cascade_with_amplification" if c3 < cascade_pred
            else "partial_cancellation"
        ),
    }


def main():
    results = []

    # ---- Qwen2.5-VL (PCD 기존 데이터) ----
    pcd_qwen = RESULTS / "pcd" / "qwen_family"
    repro_qwen = RESULTS / "repro_arditi_wollschlager" / "dim" / "Qwen2.5-7B-Instruct"
    if (repro_qwen / "direction.pt").exists() and (pcd_qwen / "V-text" / "mean_diffs.pt").exists():
        llm_dir = torch.load(repro_qwen / "direction.pt", map_location="cpu").float()
        vtext_dir = load_direction(
            pcd_qwen / "V-text" / "mean_diffs.pt",
            pcd_qwen / "V-text" / "best_layer.json",
        )
        # V-blank-resweep 우선, 없으면 V-blank 사용 (mean_diffs.pt와 best_layer.json 모두 같은 디렉토리에서)
        if (pcd_qwen / "V-blank-resweep" / "mean_diffs.pt").exists():
            vblank_md = pcd_qwen / "V-blank-resweep" / "mean_diffs.pt"
            vblank_best = pcd_qwen / "V-blank-resweep" / "best_layer.json"
        else:
            vblank_md = pcd_qwen / "V-blank" / "mean_diffs.pt"
            vblank_best = pcd_qwen / "V-blank" / "best_layer.json"
        vblank_dir = load_direction(vblank_md, vblank_best)
        results.append(analyze_model("Qwen2.5-VL-7B", llm_dir, vtext_dir, vblank_dir))
    else:
        results.append({"model": "Qwen2.5-VL-7B", "status": "PCD data not found"})

    # ---- Qwen3-VL (Task 5 새 데이터) ----
    diag = RESULTS / "ara_opd_vlm_0427" / "cross_modal_geometry_diag" / "qwen3vl"
    if (diag / "V-text" / "sweep" / "mean_diffs.pt").exists():
        vtext3_dir = load_direction(
            diag / "V-text" / "sweep" / "mean_diffs.pt",
            diag / "V-text" / "sweep" / "best_layer.json",
        )
        vblank3_dir = load_direction(
            diag / "V-blank" / "sweep" / "mean_diffs.pt",
            diag / "V-blank" / "sweep" / "best_layer.json",
        )
        c2 = cosine(vtext3_dir, vblank3_dir)
        results.append({
            "model": "Qwen3-VL-8B",
            "c1_llm_vs_vtext": "N/A (no separate LLM checkpoint)",
            "c2_vtext_vs_vblank": round(c2, 4),
            "c3_llm_vs_vblank": "N/A",
            "note": "Only c2 available without separate Qwen3-8B-Instruct text-only run",
        })
    else:
        results.append({"model": "Qwen3-VL-8B", "status": "sweep not yet complete"})

    # ---- Gemma-3-4B (PCD 기존 데이터, L≡V-text) ----
    pcd_gemma = RESULTS / "pcd" / "gemma_family"
    if (pcd_gemma / "V-text" / "mean_diffs.pt").exists():
        vtext_g = load_direction(
            pcd_gemma / "V-text" / "mean_diffs.pt",
            pcd_gemma / "V-text" / "best_layer.json",
        )
        vblank_g = load_direction(
            pcd_gemma / "V-blank" / "mean_diffs.pt",
            pcd_gemma / "V-blank" / "best_layer.json",
        )
        c2_g = cosine(vtext_g, vblank_g)
        results.append({
            "model": "Gemma-3-4B",
            "c1_llm_vs_vtext": "N/A (L≡V-text, no VL alignment shift)",
            "c2_vtext_vs_vblank": round(c2_g, 4),
            "c3_llm_vs_vblank": "N/A",
        })
    else:
        results.append({"model": "Gemma-3-4B", "status": "PCD data not found"})

    out_path = OUT_DIR / "cross_modal_alignment.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved to {out_path}")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
