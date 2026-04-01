"""
Exp 3C: Cross-Model Narrow Waist Ablation Attack

核心问题:
  1. Phase 2 发现的 narrow waist 单层 ablation 优于全层 ablation (89.7% > 74.1%) 是否跨模型通用?
  2. Group A (LLaVA, InstructBLIP, 有 reversal) vs Group B (Qwen2.5-VL, InternVL2, 无 reversal)
     的 ablation 效果是否不同?

方法:
  在 model.generate() 中通过 forward hook 移除指定层 hidden state 在 refusal direction 上的投影,
  使用 Exp 3A 保存的 v_text / v_mm 作为 ablation 方向.

5 种配置:
  baseline_text  : text-only, 无 ablation (负对照)
  baseline_mm    : blank image, 无 ablation (基准)
  ablation_nw_vmm: blank image, ablate narrow waist 层, v_mm 方向
  ablation_all_vmm: blank image, ablate 全部层, v_mm 方向
  ablation_nw_vtext: blank image, ablate narrow waist 层, v_text 方向 (验证跨模态稳定性)

用法:
  python exp_3c_ablation_attack.py --model llava_7b --device cuda:0
  python exp_3c_ablation_attack.py --model qwen2vl_7b --device cuda:0
  python exp_3c_ablation_attack.py --model internvl2_8b --device cuda:0
  python exp_3c_ablation_attack.py --model instructblip_7b --device cuda:0
"""

import os
import sys
import argparse
import json
import contextlib
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
# phase3 最后插入 → index 0 → common.* 优先找 phase3/common/
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase3"))
# phase2/common 直接加入 → eval_utils / llava_utils 作为顶层模块导入，避免与 phase3/common 冲突
sys.path.insert(0, str(_PROJ_ROOT / "experiments" / "phase2" / "common"))

from common.model_configs import MODEL_CONFIGS, load_model_by_name
from common.model_adapters import create_adapter, ModelAdapter

def load_test_prompts(n=100):
    """从 SaladBench harmful_test.json 加载前 n 条 prompts。"""
    data_path = _PROJ_ROOT / "data" / "saladbench_splits" / "harmful_test.json"
    with open(data_path) as f:
        data = json.load(f)
    # data 格式: [{"instruction": "...", "target": "...", "category": "..."}, ...]
    prompts = [item["instruction"] for item in data[:n]]
    print(f"[load_test_prompts] Loaded {len(prompts)} prompts from SaladBench")
    return prompts

TEST_PROMPTS = load_test_prompts(n=100)

from eval_utils import evaluate_response, compute_attack_metrics


# ── 通用 Ablation Hook ────────────────────────────────────────────────────────

def _make_pre_hook(direction: torch.Tensor):
    """返回 forward pre-hook: 从 layer 输入移除 direction 投影。"""
    def hook_fn(module, args):
        activation = args[0] if isinstance(args, tuple) else args
        d = direction.to(activation.device).to(activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_act = activation - proj
        return (new_act,) + args[1:] if isinstance(args, tuple) else new_act
    return hook_fn


def _make_output_hook(direction: torch.Tensor):
    """返回 forward hook: 从 module 输出移除 direction 投影。"""
    def hook_fn(module, input, output):
        activation = output[0] if isinstance(output, tuple) else output
        d = direction.to(activation.device).to(activation.dtype)
        proj = (activation @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        new_act = activation - proj
        return (new_act,) + output[1:] if isinstance(output, tuple) else new_act
    return hook_fn


@contextlib.contextmanager
def ablation_context(adapter: ModelAdapter, direction: torch.Tensor,
                     target_layers: Optional[List[int]] = None):
    """
    在 adapter.get_llm_layers() 的指定层上注册 ablation hooks。
    退出时自动移除所有 hooks。

    在每层注册:
      - pre-hook (layer 输入)
      - output-hook on self_attn (如果存在)
      - output-hook on mlp (如果存在)
    """
    llm_layers = adapter.get_llm_layers()
    n_layers = len(llm_layers)
    if target_layers is None:
        target_layers = list(range(n_layers))

    handles = []
    try:
        for idx in target_layers:
            layer = llm_layers[idx]
            handles.append(layer.register_forward_pre_hook(_make_pre_hook(direction)))
            if hasattr(layer, 'self_attn'):
                handles.append(layer.self_attn.register_forward_hook(_make_output_hook(direction)))
            if hasattr(layer, 'mlp'):
                handles.append(layer.mlp.register_forward_hook(_make_output_hook(direction)))
        yield
    finally:
        for h in handles:
            h.remove()


# ── Direction 加载 ─────────────────────────────────────────────────────────────

def load_directions(model_name: str) -> dict:
    """
    加载 Exp 3A 保存的 directions.
    返回 {
        "narrow_waist_layer": int,
        "v_text": tensor (hidden_dim,),
        "v_mm": tensor (hidden_dim,),
    }
    """
    directions_path = (_PROJ_ROOT / "results" / "phase3" / model_name
                       / "exp_3a_directions.pt")
    if not directions_path.exists():
        raise FileNotFoundError(f"exp_3a_directions.pt not found: {directions_path}")

    data = torch.load(directions_path, map_location="cpu")
    nw_layer = data["narrow_waist_layer"]
    v_text = data["directions"][nw_layer]["v_text"]
    v_mm = data["directions"][nw_layer]["v_mm"]

    print(f"[directions] narrow_waist_layer={nw_layer}")
    print(f"[directions] v_text: shape={v_text.shape}, norm={v_text.norm():.4f}")
    print(f"[directions] v_mm:   shape={v_mm.shape}, norm={v_mm.norm():.4f}")

    return {"narrow_waist_layer": nw_layer, "v_text": v_text, "v_mm": v_mm}


# ── 核心生成函数 ──────────────────────────────────────────────────────────────

def generate_response(adapter: ModelAdapter, prompt: str, image,
                      direction: Optional[torch.Tensor],
                      target_layers: Optional[List[int]],
                      max_new_tokens: int = 200) -> str:
    """
    生成回复，可选地应用 ablation hooks.
    image=None: text-only 模式; image 为 PIL.Image: mm 模式.
    direction=None: 不做 ablation.
    """
    torch.cuda.empty_cache()

    if direction is not None:
        ctx = ablation_context(adapter, direction, target_layers)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        if image is not None:
            return adapter.generate_mm(prompt, image, max_new_tokens=max_new_tokens)
        else:
            return adapter.generate_text(prompt, max_new_tokens=max_new_tokens)


# ── 主实验函数 ─────────────────────────────────────────────────────────────────

def run_exp_3c(model_name: str, device: str, max_new_tokens: int = 200,
               n_prompts: int = 100) -> dict:
    """Exp 3C 完整流程。"""
    cfg = MODEL_CONFIGS[model_name]
    total_layers = cfg["total_layers"]

    test_prompts = TEST_PROMPTS[:n_prompts]
    print(f"[exp_3c] 使用 {len(test_prompts)} 条测试 prompts")

    # ── 1. 加载模型 ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Exp 3C: Narrow Waist Ablation Attack — {model_name}")
    print(f"{'='*60}")
    model, processor = load_model_by_name(model_name, device)
    adapter = create_adapter(model_name, model, processor, device)

    blank_image = Image.new("RGB", cfg["blank_image_size"], color=(128, 128, 128))

    # ── 2. 加载 directions ───────────────────────────────────────────────────
    dir_data = load_directions(model_name)
    nw_layer = dir_data["narrow_waist_layer"]
    v_text = dir_data["v_text"]
    v_mm = dir_data["v_mm"]

    all_layers = list(range(total_layers))

    # ── 3. 实验配置 ──────────────────────────────────────────────────────────
    # (config_name, image, direction, target_layers)
    configs = [
        ("baseline_text",      None,        None,   None),
        ("baseline_mm",        blank_image, None,   None),
        ("ablation_nw_vmm",    blank_image, v_mm,   [nw_layer]),
        ("ablation_all_vmm",   blank_image, v_mm,   all_layers),
        ("ablation_nw_vtext",  blank_image, v_text, [nw_layer]),
    ]

    print(f"\nNarrow waist layer: {nw_layer} (rel_depth={nw_layer/total_layers:.2f})")
    print(f"Total configs: {len(configs)}, prompts: {len(test_prompts)}")
    print(f"max_new_tokens: {max_new_tokens}")

    # ── 4. 快速 sanity check: 第一个 prompt 验证 generation 正常 ─────────────
    print(f"\n[sanity] Running baseline_mm on first prompt...")
    sample_response = generate_response(
        adapter, test_prompts[0], blank_image, None, None, max_new_tokens=50
    )
    print(f"[sanity] Response: {sample_response[:200]}")

    # ── 5. 主循环 ─────────────────────────────────────────────────────────────
    all_config_results = {}

    for config_name, image, direction, target_layers in configs:
        print(f"\n{'─'*50}")
        print(f"Config: {config_name}")
        if target_layers is not None:
            print(f"  Target layers: {target_layers}")
        if direction is not None:
            print(f"  Direction norm: {direction.norm():.4f}")

        config_responses = []

        for i, prompt in enumerate(test_prompts):
            short_prompt = prompt[:50]
            response = generate_response(
                adapter, prompt, image, direction, target_layers, max_new_tokens
            )
            eval_result = evaluate_response(response)
            eval_result["prompt"] = prompt
            eval_result["response"] = response[:500]
            config_responses.append(eval_result)

            status = ("FULL_HARMFUL" if eval_result["full_harmful_completion"]
                      else ("bypassed" if eval_result["initial_bypass"] else "refused"))
            if eval_result["self_correction_found"]:
                status += f"+sc@{eval_result['self_correction_pos_chars']}"
            print(f"  [{i+1}/{len(test_prompts)}] {short_prompt}... → {status}")

        metrics = compute_attack_metrics(config_responses)
        all_config_results[config_name] = {
            "metrics": metrics,
            "responses": config_responses,
        }

        print(f"  bypass={metrics['initial_bypass_rate']:.3f}  "
              f"self_corr={metrics['self_correction_rate_overall']:.3f}  "
              f"full_harmful={metrics['full_harmful_completion_rate']:.3f}")

    # ── 6. 汇总 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — {model_name}")
    print(f"{'='*60}")
    print(f"{'Config':<22} {'Bypass':>8} {'SelfCorr':>10} {'FullHarm':>10}")
    print(f"{'─'*52}")
    for cfg_name, cfg_data in all_config_results.items():
        m = cfg_data["metrics"]
        print(f"{cfg_name:<22} {m['initial_bypass_rate']:>8.3f} "
              f"{m['self_correction_rate_overall']:>10.3f} "
              f"{m['full_harmful_completion_rate']:>10.3f}")

    # ── 7. 保存结果 ───────────────────────────────────────────────────────────
    results = {
        "model": model_name,
        "total_layers": total_layers,
        "narrow_waist_layer": nw_layer,
        "narrow_waist_relative_depth": round(nw_layer / total_layers, 3),
        "n_prompts": len(test_prompts),
        "max_new_tokens": max_new_tokens,
        "configs": {
            cfg_name: {
                "metrics": cfg_data["metrics"],
                "responses": cfg_data["responses"],
            }
            for cfg_name, cfg_data in all_config_results.items()
        },
    }

    save_dir = _PROJ_ROOT / "results" / "phase3" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "exp_3c_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[{model_name}] Results saved to: {save_path}")

    return results


# ── CLI 入口 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Exp 3C: Cross-Model Ablation Attack")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()), help="模型名称")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--n_prompts", type=int, default=100, help="测试 prompt 数量")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    run_exp_3c(args.model, args.device, args.max_new_tokens, args.n_prompts)


if __name__ == "__main__":
    main()
