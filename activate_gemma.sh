#!/usr/bin/env bash
# activate_gemma.sh — conda-like activation for the gemma probe venv.
#
# Usage (MUST source, not bash):
#   cd /inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal
#   source activate_gemma.sh         # like "conda activate gemma"
#   # ... do your work ...
#   deactivate                        # same as conda deactivate

# Guard against being run (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: this script must be SOURCED, not executed." >&2
  echo "       Use: source activate_gemma.sh" >&2
  exit 1
fi

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VENV="$_ROOT/.venv_gemma_probe"

if [[ ! -d "$_VENV" ]]; then
  echo "ERROR: venv not found at $_VENV" >&2
  echo "       Run: bash install_offline.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$_VENV/bin/activate"

# Visual feedback like conda's "(gemma) user@host $" prompt
echo "[gemma] activated $(python3 --version) at $_VENV"
echo "[gemma] to leave: deactivate"
unset _ROOT _VENV
