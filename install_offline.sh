#!/usr/bin/env bash
# install_offline.sh — GPU-side offline installer for gemma-4-heretic probe env.
#
# Run this INSIDE the GPU container (NGC 25.02 / py312):
#   cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
#   bash install_offline.sh
#
# Prerequisites (produced by CPU container):
#   - pip_wheels_py312/      populated via pip download (py312 target)
#   - vendored/strong_reject/ git clone of dsbowen/strong_reject
#   - requirements.lock      reverse-parsed from wheels
#
# Does NOT install torch / flash-attn / triton — those come from NGC 25.02 base image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WHEELS="$ROOT/pip_wheels_py312"
VENDORED="$ROOT/vendored/strong_reject"
LOCK="$ROOT/requirements.lock"

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYVER" != "3.12" ]]; then
  echo "WARNING: Python is $PYVER, expected 3.12. Proceeding anyway." >&2
fi

[[ -d "$WHEELS" ]]   || { echo "ERROR: $WHEELS missing"   >&2; exit 1; }
[[ -d "$VENDORED" ]] || { echo "ERROR: $VENDORED missing" >&2; exit 1; }
[[ -f "$LOCK" ]]     || { echo "ERROR: $LOCK missing"     >&2; exit 1; }

echo "[install_offline] installing from $WHEELS ..."
pip install --no-index --find-links="$WHEELS" -r "$LOCK"

echo "[install_offline] installing strong_reject (no-deps) from $VENDORED ..."
pip install --no-deps "$VENDORED"

echo "[install_offline] running verify_env.py ..."
python3 "$ROOT/verify_env.py"

echo "[install_offline] DONE."
