# ARA-SAPP Probe — Environment Setup

Setup for `exp_gemma4_heretic_probe.py` (gemma-4-heretic refusal/capability probe).

## Topology

Two containers share `/inspire/hdd/global_user/wenming-253108090054/`.

| Container | Base image | Python | Role |
|---|---|---|---|
| CPU | NGC 24.05 (Ubuntu 22.04) | 3.10 | Online download of py312 wheels; no install here |
| GPU | `docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3` | 3.12 | Offline install + run probe |

**Do NOT** `pip install` in the CPU container trying to "prepare" the env — the py310 site-packages are incompatible with the GPU's py312. CPU only runs `pip download`.

## CPU-side (one-time, already done in this repo)

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

# 1) Download py312 wheels
pip download \
  transformers==5.5.4 accelerate peft sentencepiece \
  openai datasets litellm \
  -d pip_wheels_py312/ \
  --python-version 3.12 \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  --only-binary=:all:

# 2) Clone strong_reject source
git clone --depth=1 https://github.com/dsbowen/strong_reject.git vendored/strong_reject

# 3) (auto by plan) requirements.lock was reverse-parsed from wheel names;
#    torch / triton / nvidia-cu12 wheels were stripped (NGC provides them).
```

## GPU-side (run this)

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

# 1) Offline install (runs verify_env.py at the end)
bash install_offline.sh

# 2) Runtime smoke
CUDA_VISIBLE_DEVICES=0 python3 experiments/ara_sapp/smoke_test.py
# expect "READY FOR FULL RUN."

# 3) Full probe
CUDA_VISIBLE_DEVICES=0 python3 experiments/ara_sapp/exp_gemma4_heretic_probe.py all --n 50
```

**Note:** The docstring inside `exp_gemma4_heretic_probe.py` shows `conda run -n qwen3-vl python ...`. NGC containers have no conda — use bare `python3 ...` as shown above. Script unchanged to keep diffs minimal.

## What's installed (GPU py312)

| Package | Why |
|---|---|
| transformers==5.5.4 | Gemma4ForConditionalGeneration |
| accelerate | device_map |
| peft | StrongReject LoRA adapter |
| sentencepiece | Gemma tokenizer |
| openai / datasets / litellm | strong_reject module-load hard imports (never called at runtime) |
| strong_reject (vendored, `--no-deps`) | SR scorer |
| (transitive deps) | safetensors, tokenizers, huggingface_hub, pyarrow, pandas, numpy, pydantic, ... |

Full list in `../../requirements.lock`.

## Known footguns

1. **Tokenizer `extra_special_tokens` list→dict bug.** `gemma-4-heretic/tokenizer_config.json` ships a list where transformers 5.x expects a dict. The probe script patches this into a scratch dir at runtime (`_patch_tokenizer_config`). Do not edit the upstream model files.

2. **Do NOT upgrade torch / flash-attn / triton in the GPU container.** NGC 25.02 pins them against specific CUDA 12.8 builds. A plain `pip install torch` breaks CUDA extensions silently. The wheel backup was deliberately stripped of torch + nvidia-cu12-* for this reason.

3. **strong_reject module-load imports.** Even with `--no-deps`, `from strong_reject.evaluate import ...` runs `import openai`, `from datasets import ...`, `from litellm import completion`. These three must be installed. The wheels are in `pip_wheels_py312/`.

4. **Single-GPU memory.** gemma-4-heretic + one 8B judge at bf16 ≈ 25 GB. The probe script unloads the model before loading judges — keep this order.

5. **CPU and GPU Python versions differ.** `verify_env.py` only works on GPU side (py312 + installed packages). Running it on CPU will fail on imports — that's expected; don't "fix" it.

## Related files

- `../../pip_wheels_py312/MANIFEST.txt` — expected wheel names (the .whl binaries themselves are gitignored)
- `../../vendored/strong_reject/` — strong_reject source (gitignored)
- `../../requirements.lock` — pinned deps for offline install
- `../../install_offline.sh` — GPU-side installer
- `../../verify_env.py` — import + config validator (GPU-only)
- `smoke_test.py` — GPU runtime check
- `exp_gemma4_heretic_probe.py` — the actual probe
