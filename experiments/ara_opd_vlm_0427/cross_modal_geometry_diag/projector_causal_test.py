#!/usr/bin/env python3
"""projector_causal_test.py — Projector causal contribution test for Qwen3-VL.

비교 두 조건의 ASR_LG3：
  조건 A：仅消融 LLM backbone 方向（与标准 DIM 相同）
  조건 B：조건 A + hook projector 输出，减去 harmful/harmless 均值差（置零 projector 漂移）

판단 규칙：
  B - A > 10pp → projector 有独立因果贡献，ARA 必须包含 projector
  B - A ≤ 5pp  → projector 贡献不显著，ARA 只改 LLM backbone

Run from project root (GPU required):
  CUDA_VISIBLE_DEVICES=0 conda run -n qwen3-vl python3 \
      experiments/ara_opd_vlm_0427/cross_modal_geometry_diag/projector_causal_test.py
"""
import sys
import os
import json
import torch
import functools
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_REFUSAL = _ROOT / "refusal_direction"
sys.path.insert(0, str(_REFUSAL))
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_PATH = "/inspire/hdd/global_user/wenming-253108090054/models/Qwen3-VL-8B"
DIAG_ROOT = _ROOT / "results" / "ara_opd_vlm_0427" / "cross_modal_geometry_diag" / "qwen3vl"
OUT_PATH = _ROOT / "results" / "ara_opd_vlm_0427" / "cross_modal_geometry_diag" / "projector_causal_test.json"
DATA_DIR = _ROOT / "data" / "saladbench_splits"
LG3_PATH = "/inspire/hdd/global_user/wenming-253108090054/models/llama-guard-3-8b"

N_EVAL = 128


def load_data(n: int):
    harmful = json.loads((DATA_DIR / "harmful_val.json").read_text())[:n]
    harmless = json.loads((DATA_DIR / "harmless_val.json").read_text())[:n]
    return harmful, harmless


def run_condition_a(model_base, harmful_instructions, best_layer, best_pos):
    """标准 DIM 消融：仅正交化 backbone 方向，与 PCD 完全一致。"""
    from pipeline.submodules.generate_directions import get_mean_diff
    from pipeline.submodules.generate import generate_completions

    # 提取方向（仅用 train 集提取，这里用 val 集近似，数量够用）
    direction = get_mean_diff(
        model_base,
        harmful_instructions,
        layer=best_layer,
        pos=best_pos,
    )

    # 临时正交化权重
    import copy
    from pipeline.utils.utils import get_orthogonalized_matrix
    orig_weights = {}
    for name, module in model_base.model.named_modules():
        if hasattr(module, "weight") and f"layers.{best_layer}" in name:
            orig_weights[name] = module.weight.data.clone()
            module.weight.data = get_orthogonalized_matrix(
                module.weight.data, direction
            )

    responses = generate_completions(model_base, harmful_instructions, max_new_tokens=200)

    # 恢复权重
    for name, module in model_base.model.named_modules():
        if name in orig_weights:
            module.weight.data = orig_weights[name]

    return responses


def compute_projector_mean_diff(model_base, harmful_instructions, harmless_instructions):
    """계산 projector 输出의 harmful/harmless 균값 차 벡터."""
    hooks = []
    harmful_acts = []
    harmless_acts = []

    def _hook_collect(storage, m, inp, out):
        # projector 输出 shape: (batch, n_vis_tokens, d_model)
        storage.append(out.detach().mean(dim=1).cpu())  # 对 vis tokens 取平均

    # 找到 projector 模块（Qwen3-VL 的 visual_merger 或 visual_proj）
    proj_module = None
    for name, module in model_base.model.named_modules():
        if "visual" in name and ("merger" in name or "proj" in name or "mlp" in name):
            proj_module = module
            proj_name = name
            break

    if proj_module is None:
        raise RuntimeError("Cannot find projector module. Check model architecture.")

    print(f"[projector_test] Found projector: {proj_name}")

    device = next(model_base.model.parameters()).device

    # 收集 harmful activations
    h = proj_module.register_forward_hook(
        functools.partial(_hook_collect, harmful_acts)
    )
    with torch.no_grad():
        for instr in harmful_instructions[:32]:
            inputs = model_base.tokenize_instructions_fn([instr], image_mode="blank")
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            model_base.model(**inputs)
    h.remove()

    # 收集 harmless activations
    h = proj_module.register_forward_hook(
        functools.partial(_hook_collect, harmless_acts)
    )
    with torch.no_grad():
        for instr in harmless_instructions[:32]:
            inputs = model_base.tokenize_instructions_fn([instr], image_mode="blank")
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            model_base.model(**inputs)
    h.remove()

    harmful_mean = torch.stack(harmful_acts).mean(0)
    harmless_mean = torch.stack(harmless_acts).mean(0)
    return harmful_mean - harmless_mean, proj_module, proj_name


def run_condition_b(model_base, harmful_instructions, best_layer, best_pos,
                    proj_mean_diff, proj_module):
    """조건 A + 置零 projector 漂移。"""
    from pipeline.submodules.generate_directions import get_mean_diff
    from pipeline.submodules.generate import generate_completions
    from pipeline.utils.utils import get_orthogonalized_matrix

    direction = get_mean_diff(model_base, harmful_instructions,
                               layer=best_layer, pos=best_pos)

    # 注册 hook：proj 输出减去均值差
    drift = proj_mean_diff.to("cuda:0")

    def _subtract_drift(m, inp, out):
        return out - drift.unsqueeze(0)

    h = proj_module.register_forward_hook(_subtract_drift)

    orig_weights = {}
    for name, module in model_base.model.named_modules():
        if hasattr(module, "weight") and f"layers.{best_layer}" in name:
            orig_weights[name] = module.weight.data.clone()
            module.weight.data = get_orthogonalized_matrix(
                module.weight.data, direction
            )

    responses = generate_completions(model_base, harmful_instructions, max_new_tokens=200)

    h.remove()
    for name, module in model_base.model.named_modules():
        if name in orig_weights:
            module.weight.data = orig_weights[name]

    return responses


def score_lg3(responses, instructions):
    from experiments.category_a.common.judge_utils import LlamaGuard3Judge
    judge = LlamaGuard3Judge(LG3_PATH)
    n_unsafe = sum(
        1 for instr, resp in zip(instructions, responses)
        if judge.judge_response(instr, resp).get("is_harmful", False)
    )
    del judge
    torch.cuda.empty_cache()
    return n_unsafe / len(responses)


def main():
    from pipeline.model_utils.model_factory import construct_model_base

    best = json.loads((DIAG_ROOT / "V-blank" / "sweep" / "best_layer.json").read_text())
    best_layer, best_pos = best["layer"], best.get("pos", -5)
    print(f"[projector_test] Using layer={best_layer}, pos={best_pos}")

    harmful, harmless = load_data(N_EVAL)
    harmful_instructions = [x["instruction"] for x in harmful]
    harmless_instructions = [x["instruction"] for x in harmless]

    model_base = construct_model_base(MODEL_PATH, model_name="qwen3vl_8b")

    # 계산 projector 균값 차
    proj_mean_diff, proj_module, proj_name = compute_projector_mean_diff(
        model_base, harmful_instructions, harmless_instructions
    )
    print(f"[projector_test] proj_mean_diff norm: {proj_mean_diff.norm():.4f}")

    # 조건 A
    print("[projector_test] Running condition A (backbone only)...")
    resp_a = run_condition_a(model_base, harmful_instructions, best_layer, best_pos)
    asr_a = score_lg3(resp_a, harmful_instructions)
    print(f"[projector_test] Condition A ASR_LG3: {asr_a:.3f}")

    # 조건 B
    print("[projector_test] Running condition B (backbone + projector drift removal)...")
    resp_b = run_condition_b(model_base, harmful_instructions, best_layer, best_pos,
                              proj_mean_diff, proj_module)
    asr_b = score_lg3(resp_b, harmful_instructions)
    print(f"[projector_test] Condition B ASR_LG3: {asr_b:.3f}")

    delta = asr_b - asr_a
    decision = "include_projector" if delta > 0.10 else "backbone_only"
    print(f"[projector_test] Delta B-A: {delta:+.3f} → ARA decision: {decision}")

    result = {
        "model": "Qwen3-VL-8B",
        "best_layer": best_layer,
        "best_pos": best_pos,
        "projector_module": proj_name,
        "projector_drift_norm": float(proj_mean_diff.norm()),
        "asr_lg3_condition_a_backbone_only": round(asr_a, 4),
        "asr_lg3_condition_b_backbone_plus_projector": round(asr_b, 4),
        "delta_b_minus_a": round(delta, 4),
        "ara_target_modules_decision": decision,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[projector_test] Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
