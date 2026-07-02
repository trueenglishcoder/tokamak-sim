#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

SHOTS=(3856 3857 3858 3863 3864)
DATA_ROOT="data/t15_data_new_trim50_idealized_matched"
REPLAY_ROOT="runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6"
PARAM_ROOT="output/t15_boundary_parameters_trim50_idealized_matched_gpu_plain_1e6"
INITIAL_PREFIX="T15MD_new_data_trim50_idealized_matched"

python3 scripts/idealize_t15_coil_actions.py \
  --input-root data/t15_data_new_trim50 \
  --output-root "${DATA_ROOT}" \
  --shots "${SHOTS[@]}" \
  --knot-step-s 0.05

python3 scripts/run_t15md_limited_replay_dataset.py \
  --shots "${SHOTS[@]}" \
  --config configs/T15MD_new_data.toml \
  --data-root "${DATA_ROOT}" \
  --initial-prefix "${INITIAL_PREFIX}" \
  --out "${REPLAY_ROOT}" \
  --angles 32 \
  --compute-backend gpu \
  --gpu-device cuda:0 \
  --legacy-precision-index2 1e-6 \
  --no-video

python3 scripts/fit_t15_boundary_parameters.py \
  --runs-root "${REPLAY_ROOT}" \
  --run-glob 't15md_limited_replay_*' \
  --out "${PARAM_ROOT}"

find "${DATA_ROOT}" -maxdepth 2 -type f | sort
find "${REPLAY_ROOT}" -maxdepth 1 -name 'lqr_boundary_reference_*.npz' | sort
find "${PARAM_ROOT}" -maxdepth 1 -name '*_boundary_params.csv' | sort
