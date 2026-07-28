#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/run_t15md_limited_replay_dataset.py \
  --config configs/T15MD.toml \
  --boundary-mode equilibrium_lcfs \
  "$@"
