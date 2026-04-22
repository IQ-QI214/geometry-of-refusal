#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/download_gemma3.sh [target_dir]
# Default target: /inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it
TARGET="${1:-/inspire/hdd/global_user/wenming-253108090054/models/gemma-3-4b-it}"
mkdir -p "$TARGET"

echo "Downloading google/gemma-3-4b-it to $TARGET ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/gemma-3-4b-it',
    local_dir='$TARGET',
    local_dir_use_symlinks=False,
)
print('Done: $TARGET')
"

if [ ! -f "$TARGET/config.json" ]; then
    echo "ERROR: download may have failed; config.json not found in $TARGET" >&2
    exit 1
fi
echo "Verified: config.json present at $TARGET"
