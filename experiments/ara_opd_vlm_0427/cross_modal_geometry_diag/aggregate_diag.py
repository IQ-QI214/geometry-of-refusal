#!/usr/bin/env python3
"""aggregate_diag.py — 汇总诊断结果，输出 target_modules.json 和 HANDOFF.md。

Run from project root:
  python3 experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/aggregate_diag.py
"""
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
DIAG = ROOT / "results" / "ara_opd_vlm_0427" / "cross_modal_geometry_diag"
OUT_MODULES = DIAG / "target_modules.json"
OUT_HANDOFF = Path(__file__).parent / "HANDOFF.md"


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"status": f"NOT FOUND: {path}"}


def decide_target_modules(probe: dict, causal: dict) -> dict:
    """根据 heretic probe 和 projector 因果测试输出 ARA 目标模块配置。"""
    include_projector = causal.get("ara_target_modules_decision") == "include_projector"

    # 从 Qwen3-VL sweep 结果读最优层
    sweep_best = DIAG / "qwen3vl" / "V-blank" / "sweep" / "best_layer.json"
    best = json.loads(sweep_best.read_text()) if sweep_best.exists() else {"layer": 17}
    best_layer = best["layer"]

    # 连续 5 层（最优层为中心，向两侧扩展）
    layer_range = list(range(max(0, best_layer - 2), best_layer + 3))

    modules = {
        "Qwen3-VL-8B": {
            "llm_backbone": {
                "layers": layer_range,
                "module_types": ["attn.o_proj", "mlp.down_proj"],
                "hook_name_pattern": "model.layers.{layer}.{type}",
            },
            "include_projector": include_projector,
            "projector_modules": ["visual_merger", "visual_proj"] if include_projector else [],
            "decision_basis": {
                "projector_delta_asr_lg3": causal.get("delta_b_minus_a", "N/A"),
                "best_layer": best_layer,
            },
        },
        "Gemma-3-4B": {
            "llm_backbone": {
                "layers": list(range(25, 30)),  # Gemma V-text best layer=29，保守范围
                "module_types": ["self_attn.o_proj", "mlp.down_proj"],
                "hook_name_pattern": "model.layers.{layer}.{type}",
            },
            "include_projector": False,
            "projector_modules": [],
            "decision_basis": {
                "note": "Gemma L≡V-text, no VL alignment shift. Projector test not run.",
            },
        },
    }

    # heretic probe 참고 정보（不直接影响模块选择）
    modules["_heretic_probe_reference"] = {
        "asr_kw": probe.get("asr_kw"),
        "asr_lg3": probe.get("asr_lg3"),
        "asr_sr": probe.get("asr_sr") or probe.get("asr_strongreject"),
        "note": "MoE architecture, qualitative reference only",
    }

    return modules


def main():
    probe = load_json(DIAG / "heretic_probe" / "heretic_probe_n50.json")
    alignment = load_json(DIAG / "cross_modal_alignment.json")
    causal = load_json(DIAG / "projector_causal_test.json")

    modules = decide_target_modules(probe, causal)
    OUT_MODULES.write_text(json.dumps(modules, indent=2, ensure_ascii=False))
    print(f"Saved target_modules.json to {OUT_MODULES}")

    # HANDOFF.md 생성
    handoff = f"""# ARA-OPD-VLM 诊断阶段 Handoff

**更新时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}
**状态**：诊断完成，ARA 阶段就绪

---

## 快速结论

### heretic probe（ARA 单独效果参考）
- ASR_kw: {probe.get('asr_kw', 'N/A')}
- ASR_LG3: {probe.get('asr_lg3', 'N/A')}
- ASR_SR: {probe.get('asr_sr') or probe.get('asr_strongreject', 'N/A')}
- 注：gemma-4-heretic 为 MoE 架构，定性参考

### Cross-modal cosine alignment
{json.dumps(alignment, indent=2, ensure_ascii=False)}

### Projector 因果测试（Qwen3-VL）
- 条件 A（仅 backbone）ASR_LG3: {causal.get('asr_lg3_condition_a_backbone_only', 'N/A')}
- 条件 B（backbone + projector）ASR_LG3: {causal.get('asr_lg3_condition_b_backbone_plus_projector', 'N/A')}
- Delta B-A: {causal.get('delta_b_minus_a', 'N/A')}
- **ARA 目标模块决策**: {causal.get('ara_target_modules_decision', 'N/A')}

---

## ARA 阶段配置

目标模块已写入：`results/ara_opd_vlm_0427/cross_modal_geometry_diag/target_modules.json`

下一步：参考 ARA 实验 plan（待写）启动 ara_vlm/ 实验。
"""
    OUT_HANDOFF.write_text(handoff)
    print(f"Saved HANDOFF.md to {OUT_HANDOFF}")
    print("\n=== target_modules.json ===")
    print(json.dumps(modules, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
