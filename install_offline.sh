#!/usr/bin/env bash
# install_offline.sh — GPU-side offline installer for gemma-4-heretic probe env.
#
# Run this INSIDE the GPU container (NGC 25.02 / py312):
#   cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
#   bash install_offline.sh
#
# Creates an ISOLATED venv at ./.venv_gemma_probe/ with --system-site-packages,
# so NGC's torch / flash-attn / CUDA libs stay visible but our packages live in
# the venv, not root site-packages. Other tasks in the same GPU container are
# unaffected.
#
# Prerequisites (produced by CPU container):
#   - pip_wheels_py312/      populated via pip download (py312 target) +
#                            strong_reject-0.0.1-py3-none-any.whl
#   - requirements.lock      69 pinned packages
#
# Does NOT install torch / flash-attn / triton / sympy / networkx / jinja2 /
# filelock / fsspec / typing_extensions / mpmath / markupsafe — NGC provides
# them (torch 2.7.0a0 pins sympy==1.13.1 exactly, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WHEELS="$ROOT/pip_wheels_py312"
LOCK="$ROOT/requirements.lock"
VENV="$ROOT/.venv_gemma_probe"

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYVER" != "3.12" ]]; then
  echo "WARNING: Python is $PYVER, expected 3.12. Proceeding anyway." >&2
fi

[[ -d "$WHEELS" ]] || { echo "ERROR: $WHEELS missing" >&2; exit 1; }
[[ -f "$LOCK" ]]   || { echo "ERROR: $LOCK missing"   >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Create venv (with --system-site-packages so NGC torch is visible)
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "[install_offline] creating venv at $VENV ..."
  python3 -m venv --system-site-packages "$VENV"
else
  echo "[install_offline] reusing existing venv at $VENV"
fi

VENV_PY="$VENV/bin/python"
VENV_PIP="$VENV/bin/pip"

# ---------------------------------------------------------------------------
# 2. Install our 69 packages into the venv (NOT into root)
# ---------------------------------------------------------------------------
echo "[install_offline] installing 69 packages into venv ..."
# --no-deps: bypass pip's cross-package constraint solver. Our lock enumerates
# the full set; NGC-provided deps (sympy, jinja2, networkx, torch, ...) remain
# untouched via --system-site-packages and satisfy runtime imports.
"$VENV_PIP" install --no-index --find-links="$WHEELS" --no-deps \
    --root-user-action=ignore -r "$LOCK"

# ---------------------------------------------------------------------------
# 3. Sanity check via the venv's python
# ---------------------------------------------------------------------------
echo "[install_offline] running verify_env.py in venv ..."
"$VENV_PY" "$ROOT/verify_env.py"

echo
echo "[install_offline] DONE."
echo
echo "To use the env, EITHER activate it:"
echo "    source $VENV/bin/activate"
echo "    python3 experiments/ara_sapp/smoke_test.py"
echo
echo "OR call the venv's python directly (no activation needed):"
echo "    $VENV_PY experiments/ara_sapp/smoke_test.py"
echo "    $VENV_PY experiments/ara_sapp/exp_gemma4_heretic_probe.py all --n 50"
