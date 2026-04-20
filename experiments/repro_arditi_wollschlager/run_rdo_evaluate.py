"""
在 RDO/Cone 训练好的 direction 上生成 eval completions。

用法（从 repo root）：
  python experiments/repro_arditi_wollschlager/run_rdo_evaluate.py \\
      --model qwen2.5_7b \\
      --direction results/repro_arditi_wollschlager/rdo/Qwen2.5-7B-Instruct/rdo_direction.pt \\
      --config rdo_k1
"""
import sys, os, json, random, argparse, torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "../../refusal_direction"))
sys.path.insert(0, _SCRIPT_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks
from dataset.load_dataset import load_dataset_split
from common.model_paths import MODEL_PATHS
from common.eval_judges import compute_asr_keyword

N_TEST = 128
SAVE_ROOT = "results/repro_arditi_wollschlager"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
    parser.add_argument("--direction", required=True, help="path to direction .pt file")
    parser.add_argument("--config", required=True, help="config label, e.g. rdo_k1, cone_k5")
    args = parser.parse_args()

    random.seed(42)
    model_path = MODEL_PATHS[args.model]
    model_alias = os.path.basename(model_path)
    comp_dir = os.path.join(SAVE_ROOT, "rdo", model_alias, args.config, "completions")
    os.makedirs(comp_dir, exist_ok=True)

    print(f"=== RDO Eval: {model_alias} | {args.config} ===")
    print(f"  direction: {args.direction}")
    print(f"  out:       {comp_dir}")

    harmful_test = load_dataset_split("harmful", "test")[:N_TEST]
    print(f"  test set : {len(harmful_test)} prompts")

    print("Loading model...")
    model_base = construct_model_base(model_path)

    direction = torch.load(args.direction, map_location="cpu").to(model_base.model.dtype)
    ablation_pre_hooks, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)

    print("Generating baseline completions...")
    baseline_comps = model_base.generate_completions(
        harmful_test, fwd_pre_hooks=[], fwd_hooks=[], max_new_tokens=512, batch_size=16
    )
    with open(os.path.join(comp_dir, "saladbench_baseline_completions.json"), "w") as f:
        json.dump(baseline_comps, f, indent=2)

    print("Generating ablation completions...")
    ablation_comps = model_base.generate_completions(
        harmful_test, fwd_pre_hooks=ablation_pre_hooks, fwd_hooks=ablation_hooks,
        max_new_tokens=512, batch_size=16
    )
    with open(os.path.join(comp_dir, "saladbench_ablation_completions.json"), "w") as f:
        json.dump(ablation_comps, f, indent=2)

    asr_base = compute_asr_keyword(baseline_comps)
    asr_abl  = compute_asr_keyword(ablation_comps)
    print(f"\n  ASR_kw baseline: {asr_base:.3f}")
    print(f"  ASR_kw ablation: {asr_abl:.3f}")
    print(f"  Delta:           {asr_abl - asr_base:+.3f}")
    print(f"  Saved to: {comp_dir}")


if __name__ == "__main__":
    main()
