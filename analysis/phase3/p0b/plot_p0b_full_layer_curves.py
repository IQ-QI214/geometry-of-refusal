"""
P0-B Visualization: Full-Layer Amplitude & Cosine Curves
Generates publication-quality figures from exp_3a_results_full.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJ_ROOT / "results" / "phase3"
SAVE_DIR = Path(__file__).resolve().parent / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["llava_7b", "qwen2vl_7b", "internvl2_8b", "instructblip_7b"]
LABELS = {
    "llava_7b": "LLaVA-1.5-7B\n(CLIP ViT-L)",
    "qwen2vl_7b": "Qwen2.5-VL-7B\n(Custom ViT)",
    "internvl2_8b": "InternVL2-8B\n(InternViT)",
    "instructblip_7b": "InstructBLIP-7B\n(BLIP-2 ViT-G + Q-Former)",
}
LABELS_SHORT = {
    "llava_7b": "LLaVA-1.5",
    "qwen2vl_7b": "Qwen2.5-VL",
    "internvl2_8b": "InternVL2",
    "instructblip_7b": "InstructBLIP",
}
COLORS = {
    "llava_7b": "#2563EB",        # blue
    "qwen2vl_7b": "#DC2626",      # red
    "internvl2_8b": "#059669",     # green
    "instructblip_7b": "#D97706",  # amber
}
# Group A = solid, Group B = dashed
LINESTYLES = {
    "llava_7b": "-",
    "qwen2vl_7b": "--",
    "internvl2_8b": "--",
    "instructblip_7b": "-",
}
MARKERS = {
    "llava_7b": "o",
    "qwen2vl_7b": "s",
    "internvl2_8b": "D",
    "instructblip_7b": "^",
}

def load_all_results():
    data = {}
    for model in MODELS:
        path = RESULTS_DIR / model / "exp_3a_results_full.json"
        with open(path) as f:
            data[model] = json.load(f)
    return data


def fig1_norm_ratio_curves(data):
    """Figure 1: Norm Ratio (amplitude reversal) curves for all 4 models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS:
        r = data[model]
        depths = [lr["relative_depth"] for lr in r["layer_results"]]
        ratios = [lr["norm_ratio"] for lr in r["layer_results"]]
        ax.plot(depths, ratios, color=COLORS[model], linestyle=LINESTYLES[model],
                marker=MARKERS[model], markersize=5, linewidth=2.2,
                label=LABELS_SHORT[model], alpha=0.9)

    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.fill_between([0, 1], 0, 1, alpha=0.06, color='blue', label='_nolegend_')
    ax.fill_between([0, 1], 1, 2, alpha=0.06, color='red', label='_nolegend_')

    ax.text(0.02, 0.55, 'Visual modality\nSUPPRESSES\nrefusal signal', fontsize=9,
            color='#2563EB', alpha=0.7, va='center', style='italic')
    ax.text(0.02, 1.35, 'Visual modality\nAMPLIFIES\nrefusal signal', fontsize=9,
            color='#DC2626', alpha=0.7, va='center', style='italic')

    ax.set_xlabel("Relative Layer Depth", fontsize=13)
    ax.set_ylabel("Norm Ratio (mm / text-only)", fontsize=13)
    ax.set_title("Amplitude Reversal: How Visual Modality Modulates Refusal Signal Strength",
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.2, 1.6)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(SAVE_DIR / f"fig1_norm_ratio_curves.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Fig1] Saved: fig1_norm_ratio_curves.pdf/png")


def fig2_cosine_curves(data):
    """Figure 2: Cosine similarity (direction alignment) curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS:
        r = data[model]
        depths = [lr["relative_depth"] for lr in r["layer_results"]]
        cosines = [lr["cos_text_mm"] for lr in r["layer_results"]]
        ax.plot(depths, cosines, color=COLORS[model], linestyle=LINESTYLES[model],
                marker=MARKERS[model], markersize=5, linewidth=2.2,
                label=LABELS_SHORT[model], alpha=0.9)

    # Narrow waist annotations
    for model in MODELS:
        r = data[model]
        nw_depth = r["narrow_waist_relative_depth"]
        nw_cos = r["narrow_waist_cos"]
        ax.annotate(f'NW', xy=(nw_depth, nw_cos),
                    fontsize=8, fontweight='bold', color=COLORS[model],
                    ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')

    ax.axhline(y=0.85, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.95, 0.86, 'cos=0.85 threshold', fontsize=8, color='gray',
            ha='right', alpha=0.7)

    ax.set_xlabel("Relative Layer Depth", fontsize=13)
    ax.set_ylabel("cos(v_text, v_mm)", fontsize=13)
    ax.set_title("Cross-Modal Refusal Direction Alignment by Layer",
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(SAVE_DIR / f"fig2_cosine_curves.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Fig2] Saved: fig2_cosine_curves.pdf/png")


def fig3_dual_panel(data):
    """Figure 3: Combined 2-panel (norm_ratio + cos) — paper Figure 1 candidate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for model in MODELS:
        r = data[model]
        depths = [lr["relative_depth"] for lr in r["layer_results"]]
        ratios = [lr["norm_ratio"] for lr in r["layer_results"]]
        cosines = [lr["cos_text_mm"] for lr in r["layer_results"]]

        ax1.plot(depths, ratios, color=COLORS[model], linestyle=LINESTYLES[model],
                 marker=MARKERS[model], markersize=5, linewidth=2.2,
                 label=LABELS_SHORT[model], alpha=0.9)
        ax2.plot(depths, cosines, color=COLORS[model], linestyle=LINESTYLES[model],
                 marker=MARKERS[model], markersize=5, linewidth=2.2,
                 label=LABELS_SHORT[model], alpha=0.9)

    # Panel A: norm ratio
    ax1.axhline(y=1.0, color='black', linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.fill_between([0, 1], 0, 1, alpha=0.05, color='blue')
    ax1.fill_between([0, 1], 1, 2, alpha=0.05, color='red')
    ax1.set_xlabel("Relative Layer Depth", fontsize=12)
    ax1.set_ylabel("Norm Ratio (mm / text-only)", fontsize=12)
    ax1.set_title("(a) Amplitude: Visual Modality Effect on Refusal Strength", fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.02, 1.0)
    ax1.set_ylim(0.2, 1.6)
    ax1.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Panel B: cosine
    ax2.axhline(y=0.85, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    for model in MODELS:
        r = data[model]
        nw_depth = r["narrow_waist_relative_depth"]
        nw_cos = r["narrow_waist_cos"]
        if nw_cos > 0.6:
            ax2.plot(nw_depth, nw_cos, marker='*', markersize=14,
                     color=COLORS[model], zorder=5, markeredgecolor='black', markeredgewidth=0.5)

    ax2.set_xlabel("Relative Layer Depth", fontsize=12)
    ax2.set_ylabel("cos(v_text, v_mm)", fontsize=12)
    ax2.set_title("(b) Direction: Cross-Modal Refusal Alignment", fontsize=12, fontweight='bold')
    ax2.set_xlim(-0.02, 1.0)
    ax2.set_ylim(0.3, 1.02)
    ax2.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(SAVE_DIR / f"fig3_dual_panel.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Fig3] Saved: fig3_dual_panel.pdf/png")


def fig4_norm_decomposition(data):
    """Figure 4: Raw norm decomposition (norm_text and norm_mm separately)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        r = data[model]
        depths = [lr["relative_depth"] for lr in r["layer_results"]]
        norm_text = [lr["norm_text"] for lr in r["layer_results"]]
        norm_mm = [lr["norm_mm"] for lr in r["layer_results"]]

        ax.plot(depths, norm_text, 'b-o', linewidth=2, markersize=4, label='Text-only', alpha=0.8)
        ax.plot(depths, norm_mm, 'r-s', linewidth=2, markersize=4, label='Multimodal', alpha=0.8)

        # Shade the gap
        ax.fill_between(depths, norm_text, norm_mm, alpha=0.15,
                         color='green' if any(m > t for m, t in zip(norm_mm, norm_text)) else 'gray')

        # Find crossover
        for i in range(len(depths) - 1):
            if (norm_text[i] > norm_mm[i]) and (norm_text[i+1] < norm_mm[i+1]):
                ax.axvline(x=(depths[i] + depths[i+1]) / 2, color='purple',
                           linestyle='--', alpha=0.5, linewidth=1.5)
                ax.text((depths[i] + depths[i+1]) / 2, ax.get_ylim()[1] * 0.9,
                        'crossover', fontsize=8, color='purple', ha='center')

        ax.set_xlabel("Relative Layer Depth", fontsize=11)
        ax.set_ylabel("Refusal Direction Norm", fontsize=11)
        ax.set_title(LABELS[model], fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Refusal Signal Norm Decomposition: Text-only vs Multimodal", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(SAVE_DIR / f"fig4_norm_decomposition.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Fig4] Saved: fig4_norm_decomposition.pdf/png")


def fig5_group_comparison_radar(data):
    """Figure 5: Safety Geometry Portrait — radar chart for Group A vs Group B."""
    import matplotlib.patches as mpatches

    # Compute summary metrics for each model
    metrics = {}
    for model in MODELS:
        r = data[model]
        lrs = r["layer_results"]
        depths = [lr["relative_depth"] for lr in lrs]
        ratios = [lr["norm_ratio"] for lr in lrs]
        cosines = [lr["cos_text_mm"] for lr in lrs]

        # 1. Crossover depth (0 if no crossover)
        crossover_depth = 0.0
        for i in range(len(ratios) - 1):
            if ratios[i] < 1.0 and ratios[i+1] >= 1.0:
                crossover_depth = (depths[i] + depths[i+1]) / 2
                break

        # 2. Max norm ratio
        max_ratio = max(ratios)

        # 3. Min norm ratio
        min_ratio = min(ratios)

        # 4. Narrow waist depth
        nw_depth = r["narrow_waist_relative_depth"]

        # 5. Peak cos
        peak_cos = max(cosines)

        # 6. Min cos (trough)
        min_cos = min(cosines)

        # 7. Cos range (peak - trough)
        cos_range = peak_cos - min_cos

        # 8. Amplitude reversal magnitude (deep_mean - shallow_mean)
        reversal_mag = r["deep_mean_ratio"] - r["shallow_mean_ratio"]

        metrics[model] = {
            "crossover_depth": crossover_depth,
            "max_ratio": max_ratio,
            "min_ratio": min_ratio,
            "nw_depth": nw_depth,
            "peak_cos": peak_cos,
            "min_cos": min_cos,
            "cos_range": cos_range,
            "reversal_mag": reversal_mag,
        }

    # Grouped bar chart instead of radar (cleaner)
    metric_names = ["Reversal\nMagnitude", "Max\nNorm Ratio", "Min\nNorm Ratio",
                    "Crossover\nDepth", "NW\nDepth", "Peak\ncos", "Min\ncos", "cos\nRange"]
    metric_keys = ["reversal_mag", "max_ratio", "min_ratio", "crossover_depth",
                   "nw_depth", "peak_cos", "min_cos", "cos_range"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metric_names))
    width = 0.18

    for i, model in enumerate(MODELS):
        vals = [metrics[model][k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, color=COLORS[model],
                      label=LABELS_SHORT[model], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Safety Geometry Portrait: Cross-Model Comparison", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(SAVE_DIR / f"fig5_safety_geometry_portrait.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Fig5] Saved: fig5_safety_geometry_portrait.pdf/png")


def fig6_cosine_shape_analysis(data):
    """Figure 6: Cosine curve shape classification (inverted-U vs V-shape)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Group A: inverted-U
    ax = axes[0]
    for model in ["llava_7b", "instructblip_7b"]:
        r = data[model]
        depths = [lr["relative_depth"] for lr in r["layer_results"]]
        cosines = [lr["cos_text_mm"] for lr in r["layer_results"]]
        ax.plot(depths, cosines, color=COLORS[model], linestyle='-',
                marker=MARKERS[model], markersize=5, linewidth=2.2,
                label=LABELS_SHORT[model], alpha=0.9)
    ax.set_title("Group A: CLIP-family ViT\n(Inverted-U / Flat-low cos shape)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Relative Layer Depth", fontsize=11)
    ax.set_ylabel("cos(v_text, v_mm)", fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Group B: V-shape
    ax = axes[1]
    for model in ["qwen2vl_7b", "internvl2_8b"]:
        r = data[model]
        depths = [lr["relative_depth"] for lr in r["layer_results"]]
        cosines = [lr["cos_text_mm"] for lr in r["layer_results"]]
        ax.plot(depths, cosines, color=COLORS[model], linestyle='-',
                marker=MARKERS[model], markersize=5, linewidth=2.2,
                label=LABELS_SHORT[model], alpha=0.9)
    ax.set_title("Group B: Custom ViT\n(V-shape cos: dip in mid-layers, recover in deep)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Relative Layer Depth", fontsize=11)
    ax.set_ylabel("cos(v_text, v_mm)", fontsize=11)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Cosine Curve Shape Taxonomy: Two Distinct Safety Geometry Patterns", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(SAVE_DIR / f"fig6_cosine_shape_taxonomy.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Fig6] Saved: fig6_cosine_shape_taxonomy.pdf/png")


def fig7_crossover_analysis(data):
    """Figure 7: Crossover point analysis — where norm_ratio crosses 1.0."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS:
        r = data[model]
        depths = [lr["relative_depth"] for lr in r["layer_results"]]
        ratios = [lr["norm_ratio"] for lr in r["layer_results"]]

        # Deviation from 1.0
        deviation = [ratio - 1.0 for ratio in ratios]
        ax.plot(depths, deviation, color=COLORS[model], linestyle=LINESTYLES[model],
                marker=MARKERS[model], markersize=5, linewidth=2.2,
                label=LABELS_SHORT[model], alpha=0.9)

    ax.axhline(y=0, color='black', linewidth=2, alpha=0.8)
    ax.fill_between([0, 1], -1, 0, alpha=0.05, color='blue')
    ax.fill_between([0, 1], 0, 1, alpha=0.05, color='red')
    ax.text(0.02, -0.25, 'MM suppresses refusal', fontsize=9, color='#2563EB', alpha=0.7, style='italic')
    ax.text(0.02, 0.1, 'MM amplifies refusal', fontsize=9, color='#DC2626', alpha=0.7, style='italic')

    ax.set_xlabel("Relative Layer Depth", fontsize=13)
    ax.set_ylabel("Norm Ratio Deviation (ratio - 1.0)", fontsize=13)
    ax.set_title("Crossover Analysis: Where Visual Modality Flips from Suppressor to Amplifier",
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.7, 0.6)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(SAVE_DIR / f"fig7_crossover_analysis.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Fig7] Saved: fig7_crossover_analysis.pdf/png")


if __name__ == "__main__":
    data = load_all_results()
    fig1_norm_ratio_curves(data)
    fig2_cosine_curves(data)
    fig3_dual_panel(data)
    fig4_norm_decomposition(data)
    fig5_group_comparison_radar(data)
    fig6_cosine_shape_analysis(data)
    fig7_crossover_analysis(data)
    print(f"\nAll figures saved to: {SAVE_DIR}/")
