"""
RDO k=1 training wrapper for repro-arditi-wollschlager.
Calls rdo.py with correct env vars set inside Python (not via shell).

Usage:
  python experiments/repro_arditi_wollschlager/run_rdo_wrapper.py --model qwen2.5_7b
  python experiments/repro_arditi_wollschlager/run_rdo_wrapper.py --model llama3.1_8b
"""
import sys
import os
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATHS = {
    "qwen2.5_7b":  "/inspire/hdd/global_user/wenming-253108090054/models/Qwen2.5-7B-Instruct",
    "llama3.1_8b": "/inspire/hdd/global_user/wenming-253108090054/models/Llama-3.1-8B-Instruct",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODEL_PATHS.keys()), required=True)
parser.add_argument("--train_cone", action="store_true")
parser.add_argument("--min_cone_dim", type=int, default=2)
parser.add_argument("--max_cone_dim", type=int, default=5)
args = parser.parse_args()

model_path = MODEL_PATHS[args.model]

env = os.environ.copy()
env["SAVE_DIR"]              = os.path.join(REPO_ROOT, "results/repro_arditi_wollschlager")
env["DIM_DIR"]               = ""
env["HF_HUB_OFFLINE"]        = "1"
env["TRANSFORMERS_OFFLINE"]  = "1"
env["WANDB_MODE"]            = "offline"
env["PYTHONUNBUFFERED"]      = "1"

cmd = [sys.executable, "-u", "rdo.py", "--model", model_path, "--splits", "saladbench"]
if args.train_cone:
    cmd += ["--train_cone", "--min_cone_dim", str(args.min_cone_dim), "--max_cone_dim", str(args.max_cone_dim)]
else:
    cmd += ["--train_direction"]

print(f"[rdo_wrapper] cwd={REPO_ROOT}")
print(f"[rdo_wrapper] cmd={' '.join(cmd)}")
print(f"[rdo_wrapper] SAVE_DIR={env['SAVE_DIR']}")

result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
sys.exit(result.returncode)
