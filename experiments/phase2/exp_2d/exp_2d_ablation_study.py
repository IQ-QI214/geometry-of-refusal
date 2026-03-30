"""
Exp 2D: Ablation Study — 系统消融各个设计选择的效果

消融轴：
  1. Single-step (t=1 only) vs Sequence-level (全序列)
  2. With vs without Component 2 (harmful direction alignment)
  3. Layer targeting: shallow / deep / all / best layer only
  4. Epsilon budget: 4/255, 8/255, 16/255, 32/255

使用方法：
  cd geometry-of-refusal
  python experiments/phase2/exp_2d/exp_2d_ablation_study.py
  python experiments/phase2/exp_2d/exp_2d_ablation_study.py --gpu 0 --config_group layers
"""

import argparse
import subprocess
import json
import os
import sys
import dotenv

dotenv.load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Exp 2D: Ablation study orchestrator")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config_group", type=str, default="all",
                        choices=["all", "timestep", "component2", "layers", "epsilon"],
                        help="Which ablation group to run")
    parser.add_argument("--steps", type=int, default=100, help="PGD steps (reduced for ablation)")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()


# ============================================================
# 消融配置
# ============================================================

def get_ablation_configs(args):
    """返回要运行的消融配置列表。"""

    configs = []
    save_dir = args.save_dir or os.getenv("SAVE_DIR", "./results")

    base = {
        "epsilon": 16/255,
        "alpha": 1/255,
        "steps": args.steps,
        "lambda_harmful": 0.1,
        "target_layers_mode": "all",
        "direction_mode": "auto",
        "n_train_prompts": 8,  # 减少以加速
        "gpu": args.gpu,
    }

    if args.config_group in ("all", "timestep"):
        # 轴 1: Single-step vs Sequence-level
        # Single-step: 只用 1 层 (layer 16)，模拟只关注初始 token
        configs.append(("single_step_best", {**base, "target_layers_mode": "best", "steps": args.steps}))
        configs.append(("sequence_level_all", {**base, "target_layers_mode": "all", "steps": args.steps}))

    if args.config_group in ("all", "component2"):
        # 轴 2: With vs without Component 2
        configs.append(("no_harmful_align", {**base, "lambda_harmful": 0.0}))
        configs.append(("with_harmful_align", {**base, "lambda_harmful": 0.1}))

    if args.config_group in ("all", "layers"):
        # 轴 3: Layer targeting
        configs.append(("layers_shallow", {**base, "target_layers_mode": "shallow"}))
        configs.append(("layers_deep", {**base, "target_layers_mode": "deep"}))
        configs.append(("layers_all", {**base, "target_layers_mode": "all"}))
        configs.append(("layers_best", {**base, "target_layers_mode": "best"}))

    if args.config_group in ("all", "epsilon"):
        # 轴 4: Epsilon budget
        for eps_num in [4, 8, 16, 32]:
            configs.append((f"epsilon_{eps_num}", {**base, "epsilon": eps_num / 255}))

    return configs


def run_single_config(name, config, save_dir, use_4bit):
    """运行单个消融配置。"""
    exp_2c_script = os.path.join(os.path.dirname(__file__), "..", "exp_2c", "exp_2c_visual_perturbation.py")

    cmd = [
        sys.executable, exp_2c_script,
        "--epsilon", str(config["epsilon"]),
        "--alpha", str(config["alpha"]),
        "--steps", str(config["steps"]),
        "--lambda_harmful", str(config["lambda_harmful"]),
        "--target_layers_mode", config["target_layers_mode"],
        "--direction_mode", config["direction_mode"],
        "--n_train_prompts", str(config["n_train_prompts"]),
        "--gpu", str(config["gpu"]),
        "--save_dir", save_dir,
    ]
    if use_4bit:
        cmd.append("--use_4bit")

    print(f"\n{'='*60}")
    print(f"Running ablation config: {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return None

    # 读取结果
    results_path = os.path.join(save_dir, "exp_2c_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return None


def run_exp_2d(args):
    save_dir = args.save_dir or os.getenv("SAVE_DIR", "./results")
    ablation_dir = os.path.join(save_dir, "exp_2d_ablations")
    os.makedirs(ablation_dir, exist_ok=True)

    configs = get_ablation_configs(args)
    print(f"\nExp 2D: Running {len(configs)} ablation configurations")
    print(f"Config group: {args.config_group}")

    all_results = {}

    for name, config in configs:
        # 每个配置用自己的子目录
        config_save_dir = os.path.join(ablation_dir, name)
        os.makedirs(config_save_dir, exist_ok=True)

        # 复制必要的输入文件到子目录
        for fname in ["exp_a_directions.pt", "exp_2a_results.json",
                       "exp_2a_controlled_directions.pt", "exp_2b_harmful_completions.json"]:
            src = os.path.join(save_dir, fname)
            dst = os.path.join(config_save_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

        result = run_single_config(name, config, config_save_dir, args.use_4bit)

        if result is not None:
            all_results[name] = {
                "config": config,
                "perturbed_metrics": result.get("perturbed_metrics", {}),
                "blank_metrics": result.get("blank_metrics", {}),
            }
            pm = result.get("perturbed_metrics", {})
            print(f"  {name}: ASR={pm.get('initial_bypass_rate', 0):.3f}, "
                  f"FullHarm={pm.get('full_harmful_completion_rate', 0):.3f}")
        else:
            all_results[name] = {"config": config, "error": True}

    # 汇总
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'Bypass':>8} {'SelfCorr':>10} {'FullHarm':>10}")
    print("-" * 55)
    for name in all_results:
        r = all_results[name]
        if "error" in r:
            print(f"{name:<25} {'ERROR':>8}")
            continue
        pm = r["perturbed_metrics"]
        print(f"{name:<25} {pm.get('initial_bypass_rate', 0):>8.3f} "
              f"{pm.get('self_correction_rate_overall', 0):>10.3f} "
              f"{pm.get('full_harmful_completion_rate', 0):>10.3f}")

    # 保存
    out_path = os.path.join(save_dir, "exp_2d_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAblation results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_exp_2d(args)
