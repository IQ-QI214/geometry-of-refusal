#!/usr/bin/env bash
# install_offline.sh — GPU-side offline installer for gemma-4-heretic probe env.
#
# Run this INSIDE the GPU container (NGC 25.02 / py312):
#   cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
#   bash install_offline.sh
#
# Prerequisites (produced by CPU container):
#   - pip_wheels_py312/      populated via pip download (py312 target) +
#                            strong_reject-0.0.1-py3-none-any.whl built via pip wheel
#   - requirements.lock      includes strong-reject==0.0.1
#
# Does NOT install torch / flash-attn / triton / sympy / networkx / jinja2 /
# filelock / fsspec / typing_extensions / mpmath / markupsafe — those come
# from NGC 25.02 base image (torch 2.7.0a0 pins sympy==1.13.1 exactly, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WHEELS="$ROOT/pip_wheels_py312"
LOCK="$ROOT/requirements.lock"

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYVER" != "3.12" ]]; then
  echo "WARNING: Python is $PYVER, expected 3.12. Proceeding anyway." >&2
fi

[[ -d "$WHEELS" ]] || { echo "ERROR: $WHEELS missing" >&2; exit 1; }
[[ -f "$LOCK" ]]   || { echo "ERROR: $LOCK missing"   >&2; exit 1; }

echo "[install_offline] installing from $WHEELS ..."
# --no-deps: bypass pip's cross-package constraint solver. Our lock enumerates
# the full set we want; NGC-provided deps (sympy, jinja2, networkx, torch, ...)
# remain untouched and satisfy runtime imports.
# --root-user-action=ignore: silence pip's venv-preference warning (NGC ships
# with system Python and no conda; the container is the isolation boundary).
pip install --no-index --find-links="$WHEELS" --no-deps \
    --root-user-action=ignore -r "$LOCK"

echo "[install_offline] running verify_env.py ..."
python3 "$ROOT/verify_env.py"

echo "[install_offline] DONE."
