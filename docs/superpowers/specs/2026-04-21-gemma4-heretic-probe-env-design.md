# Gemma-4-heretic Probe — CPU-Download / GPU-Install Environment Design (v2)

> **日期**：2026-04-21（v2 pivot：CPU 和 GPU 不是同一镜像）
> **范围**：为 `experiments/ara_sapp/exp_gemma4_heretic_probe.py` 准备可在 GPU 容器 (NGC 25.02, py312) 离线安装的 Python 环境。CPU 容器（NGC 24.05, py310）仅用于联网下载 py312 wheels。
> **不在范围**：实际跑 ASR 实验、SAPP pair 构造、attack method 实现。

---

## 1. 背景

按 `plan/0421-Atom+ARA+GRP.md` PreExp-1 要求，需要在 gemma-4-heretic 上做 refusal bypass vs capability loss 诊断。现有 probe 脚本已写好：
- 加载 `/inspire/hdd/global_user/wenming-253108090054/models/gemma-4-heretic`
- 对 50 个 harmful prompts 做文本生成
- 4 层 ASR 评估：keyword / StrongReject / Qwen3Guard / LlamaGuard3（全部本地模型）

但 GPU 容器离线，需要用 CPU 容器联网下载 wheels 预置到共享存储。

## 2. 环境约束

**CPU container（当前，仅用于下载打包）**：
- Ubuntu 22.04，Python 3.10.12
- NGC PyTorch 24.05（torch 2.4.0a0+nv24.05、flash-attn 2.4.2 预装）
- PyPI / GitHub / HuggingFace 联网可达
- 无 GPU
- **角色**：联网 `pip download` py312 wheels 与 clone strong_reject 源码；不在此容器 install 或运行

**GPU container（目标）**：
- `docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3`
- Python 3.12，torch 2.5+，CUDA 12.8
- 离线、手动启动实验
- **角色**：离线 install + 运行 probe

**共享存储**：两边 mount 同一个 `/inspire/hdd/global_user/wenming-253108090054/`。CPU 写 `pip_wheels_py312/`，GPU 读。**无需拷贝或 docker commit**。

**GPU 容器里不要升级**：torch、flash-attn、triton、torchvision、torch_tensorrt（NGC 25.02 预装、CUDA ABI 绑死）。

## 3. 依赖清单

| 包 | 版本 | 来源 | 备注 |
|---|---|---|---|
| transformers | 5.5.4 | PyPI py312 wheel | Gemma4ForConditionalGeneration 支持 |
| accelerate | 最新兼容 | PyPI py312 wheel | `device_map={"": device}` |
| peft | 最新兼容 | PyPI py312 wheel | StrongReject LoRA adapter |
| sentencepiece | 最新 | PyPI py312 wheel | Gemma tokenizer |
| strong_reject | git main | GitHub clone 到 `vendored/` | SR 判分器入口 |
| 其它（safetensors / tokenizers / hf_hub / openai / datasets / litellm 等）| 由 `pip download` 传递 | PyPI py312 wheel | strong_reject 运行时硬 import openai / datasets / litellm（模块加载时），必须在 wheels 里 |

**不包含**：nnsight、bitsandbytes、wandb、matplotlib（probe 脚本不用到）。

## 4. 执行步骤

**Step 1 — CPU 侧下载 py312 wheels**

```bash
cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal

pip download \
  transformers==5.5.4 accelerate peft sentencepiece \
  openai datasets litellm \
  -d pip_wheels_py312/ \
  --python-version 3.12 \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  --only-binary=:all:
```

strong_reject 在 Task 2 的 CPU 尝试里被发现**模块加载时硬 import** openai / datasets / litellm — 即使不调用外部 API，这些必须是可 import 的。所以把它们显式列出来一起下。

**Step 2 — clone strong_reject 源码**

```bash
git clone --depth=1 https://github.com/dsbowen/strong_reject.git vendored/strong_reject
```

**Step 3 — 从 wheels 反推 `requirements.lock`**

无法在 CPU py310 上 `pip freeze` py312 包。改为从 `pip_wheels_py312/` 文件名解析：

```bash
ls pip_wheels_py312/ | python3 -c "
import re, sys
seen = {}
for line in sys.stdin:
    m = re.match(r'^([A-Za-z0-9_\.\-]+)-(\d+[^-]*)-.*\.whl$', line.strip())
    if m:
        pkg = m.group(1).replace('_', '-').lower()
        ver = m.group(2)
        seen[pkg] = ver
for pkg, ver in sorted(seen.items()):
    print(f'{pkg}=={ver}')
" > requirements.lock
```

排除 torch / flash-attn 等：`pip download` 指定了具体包名，不会误拉这些，默认干净。

**Step 4 — CPU 侧 static sanity**

CPU py310 不能真 import py312 包。只做文件级检查：

`verify_env_cpu.py`（可选，轻量）：
- `pip_wheels_py312/` 存在且 >= 30 wheels
- `vendored/strong_reject/setup.py` 或 `pyproject.toml` 存在
- `requirements.lock` 每行能匹配到 `pip_wheels_py312/` 里的 wheel
- `gemma-4-heretic/config.json` architecture 合法

Task 1 写的 `verify_env.py` 保留，但**只能在 GPU 容器跑**（py312 + 已安装）。在 ENV_SETUP.md 里讲清楚。

**Step 5 — `install_offline.sh`（GPU 侧运行）**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
pip install --no-index --find-links="$ROOT/pip_wheels_py312" -r "$ROOT/requirements.lock"
pip install --no-deps "$ROOT/vendored/strong_reject"
python3 "$ROOT/verify_env.py"
```

**Step 6 — GPU 侧 smoke**

`experiments/ara_sapp/smoke_test.py`：
- load gemma-4-heretic → 16 tokens 生成
- Qwen3Guard 判一次
- LlamaGuard3 判一次
- StrongReject 判一次
- 全通过打印 "READY FOR FULL RUN"

**Step 7 — 文档**

`experiments/ara_sapp/ENV_SETUP.md`：
- CPU 和 GPU 容器角色分工
- CPU 侧流程（download + clone，不 install）
- GPU 侧流程（install_offline.sh → verify_env.py → smoke_test.py → probe）
- 共享 `/inspire/hdd/` 免拷贝
- 已知坑：tokenizer `extra_special_tokens` list→dict（脚本内已 patch）
- 已知坑：strong_reject 硬 import openai/datasets/litellm，所以 wheels 必须带

## 5. 产物

```
geometry-of-refusal/
├── requirements.lock                    # 从 wheels 反推
├── pip_wheels_py312/                    # py312 wheels（CPU 下载，GPU 消费）
├── vendored/strong_reject/              # git clone，--no-deps 装
├── install_offline.sh                   # GPU 侧安装入口
├── verify_env.py                        # GPU 侧 import 验证（Task 1 写过，保留）
├── verify_env_cpu.py                    # CPU 侧 static 文件检查（可选）
└── experiments/ara_sapp/
    ├── smoke_test.py                    # GPU 侧 1-prompt 运行验证
    └── ENV_SETUP.md                     # 环境说明
```

## 6. 验收标准

**CPU 侧（现在能验）**：
- `ls pip_wheels_py312/*.whl | wc -l` ≥ 30
- `ls vendored/strong_reject/` 含 setup.py / pyproject.toml 与 `strong_reject/` 子目录
- `requirements.lock` 每行对应到 `pip_wheels_py312/` 里一个具体 wheel（无缺件）
- `python3 verify_env_cpu.py` PASS（如果写了）

**GPU 侧（用户手验）**：
- `bash install_offline.sh` 成功且末尾 `verify_env.py` 打印 "ALL 6 CHECKS PASS"
- `CUDA_VISIBLE_DEVICES=0 python3 experiments/ara_sapp/smoke_test.py` 打印 "READY FOR FULL RUN"
- `python3 exp_gemma4_heretic_probe.py all --n 50` 可启动

## 7. 风险与备案

| 风险 | 概率 | 应对 |
|---|---|---|
| `pip download` 对 py312 wheel 解析失败（某个包只有 sdist） | 中 | 先试 `--only-binary=:all:`；若失败去掉该 flag 针对那个包单独 `pip download` 取 sdist |
| wheels 里缺 transitive deps（strong_reject 运行时 import 失败） | 中 | 在 GPU 侧 smoke 时即暴露；CPU 补装对应包再重装 |
| GPU 容器 torch 2.5+ 与 transformers 5.5.4 新兼容问题 | 低 | transformers 5.5.4 本来就为 torch 2.5+ 写的（CPU 遇到的 custom_op 问题源于 torch 2.4）；GPU 侧应无此问题 |
| pip_wheels 体积过大导致 shared storage 配额紧张 | 低 | 预估 500 MB - 1.5 GB；若超标 prune 冗余 platform tag |

## 8. 不做

- 不在 CPU 容器 `pip install`（已试过，py310 包对 GPU py312 无用）
- 不升级 GPU 的 torch / flash-attn
- 不构造 SAPP pairs、不跑 attack method
- 不改 probe 脚本逻辑

## 9. Task 2 遗留（from v1 尝试）

v1 时在 CPU 侧执行过 `pip install transformers==5.5.4 accelerate peft sentencepiece openai datasets litellm` 以及 `pip install --no-deps strong_reject`，CPU 侧 site-packages 被污染。**不清理**：

- 这些包只影响当前 CPU container，不出镜像边界
- 用户的 CPU container 后续也不会跑 Python 工作负载
- 清理增加风险（可能误删预装项）

记录在此以备后查。

