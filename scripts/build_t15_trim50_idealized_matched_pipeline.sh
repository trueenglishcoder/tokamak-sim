#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${TOKAMAK_PIPELINE_INSIDE_CONTAINER:-0}" != "1" ]]; then
  if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
import torch
PY
  then
    IMAGE=${TOKAMAK_CONTAINER_IMAGE:-/scratch/$USER/tokamak/tokamak-rl-v2.sqsh}
    WORKSPACE_ROOT=$(cd .. && pwd)
    if [[ ! -f "${IMAGE}" ]]; then
      echo "host python lacks numpy/torch and container image is missing: ${IMAGE}" >&2
      echo "run this on a node with the project container, or set TOKAMAK_CONTAINER_IMAGE=/path/to/tokamak-rl-v2.sqsh" >&2
      exit 2
    fi
    if ! command -v srun >/dev/null 2>&1; then
      echo "host python lacks numpy/torch and srun is unavailable; cannot enter the project container automatically" >&2
      exit 2
    fi

    echo "host python lacks numpy/torch; running matched idealized pipeline inside ${IMAGE}"
    exec srun \
      --partition=batch \
      --nodes=1 \
      --ntasks=1 \
      --gres=gpu:1 \
      --cpus-per-task="${TOKAMAK_PIPELINE_CPUS:-8}" \
      --container-image "${IMAGE}" \
      --container-mounts "${WORKSPACE_ROOT}:/workspace,/dev:/dev,/lib/x86_64-linux-gnu:/host-libs" \
      bash -lc '
        set -euo pipefail
        mkdir -p /tmp/nvidia-lib
        ln -sf /host-libs/libcuda.so.1 /tmp/nvidia-lib/libcuda.so.1
        ln -sf /tmp/nvidia-lib/libcuda.so.1 /tmp/nvidia-lib/libcuda.so
        export LD_LIBRARY_PATH=/tmp/nvidia-lib:${LD_LIBRARY_PATH:-}
        export PYTHONPATH=/workspace/tokamak-sim:/workspace/tokamak-rl-v2:${PYTHONPATH:-}
        export TOKAMAK_PIPELINE_INSIDE_CONTAINER=1
        cd /workspace/tokamak-sim
        bash scripts/build_t15_trim50_idealized_matched_pipeline.sh
      '
  fi
fi

SHOTS=(3856 3857 3858 3863 3864)
DATA_ROOT="data/t15_data_new_trim50_idealized_matched"
REPLAY_ROOT="runs/t15md_limited_replay_dataset_trim50_idealized_matched_gpu_plain_1e6"
PARAM_ROOT="output/t15_boundary_parameters_trim50_idealized_matched_gpu_plain_1e6"
INITIAL_PREFIX="T15MD_new_data_trim50_idealized_matched"

python3 scripts/idealize_t15_coil_actions.py \
  --input-root data/t15_data_new_trim50 \
  --output-root "${DATA_ROOT}" \
  --shots "${SHOTS[@]}" \
  --method bounded_smooth_jdot \
  --smooth-window-steps 21 \
  --max-current-deviation-a 250

python3 - <<'PY'
from pathlib import Path

import numpy as np

shots = ("3856", "3857", "3858", "3863", "3864")
source_root = Path("data/t15_data_new_trim50")
ideal_root = Path("data/t15_data_new_trim50_idealized_matched")
cap_a = 250.0 + 1.0e-6

for shot in shots:
    src_ip = np.loadtxt(source_root / "ip" / f"t15md_{shot}_ip.csv", delimiter=";")
    out_ip = np.loadtxt(ideal_root / "ip" / f"t15md_{shot}_ip.csv", delimiter=";")
    src_coils = np.loadtxt(source_root / "coils" / f"t15md_{shot}_coils.csv", delimiter=";")
    out_coils = np.loadtxt(ideal_root / "coils" / f"t15md_{shot}_coils.csv", delimiter=";")

    if src_ip.shape != out_ip.shape:
        raise SystemExit(f"{shot}: idealized Ip shape {out_ip.shape} != trim50 shape {src_ip.shape}")
    if src_coils.shape != out_coils.shape:
        raise SystemExit(f"{shot}: idealized coil shape {out_coils.shape} != trim50 shape {src_coils.shape}")
    if not np.allclose(src_ip, out_ip, rtol=0.0, atol=1.0e-9):
        raise SystemExit(f"{shot}: idealized Ip table is not an exact trim50 copy")
    if not np.allclose(src_coils[:, 0], out_coils[:, 0], rtol=0.0, atol=1.0e-12):
        raise SystemExit(f"{shot}: idealized coil time grid differs from trim50")
    if not np.allclose(src_coils[0, 1:], out_coils[0, 1:], rtol=0.0, atol=1.0e-7):
        raise SystemExit(f"{shot}: idealized first coil row differs from trim50")
    if not np.allclose(src_coils[-1, 1:], out_coils[-1, 1:], rtol=0.0, atol=1.0e-7):
        raise SystemExit(f"{shot}: idealized final coil row differs from trim50")

    max_diff = float(np.max(np.abs(out_coils[:, 1:] - src_coils[:, 1:])))
    if max_diff > cap_a:
        raise SystemExit(f"{shot}: idealized coil table deviates by {max_diff:.3f} A > {cap_a:.3f} A")

    src_delta = np.diff(src_coils[:, 1:], axis=0)
    out_delta = np.diff(out_coils[:, 1:], axis=0)
    first = min(50, src_delta.shape[0])
    src_first = np.max(np.abs(src_delta[:first]), axis=0)
    out_first = np.max(np.abs(out_delta[:first]), axis=0)
    flat = (src_first > 1.0) & (out_first < 0.2 * src_first)
    if np.any(flat):
        cols = np.where(flat)[0].tolist()
        raise SystemExit(
            f"{shot}: idealized first {first} steps are suspiciously flat for current columns {cols}; "
            f"trim50_delta={src_first}, ideal_delta={out_first}"
        )

    print(
        f"{shot}: idealized matched alignment ok; "
        f"max_abs_current_diff={max_diff:.3f} A, "
        f"first{first}_delta_ratio_min={float(np.min(out_first / (src_first + 1.0e-12))):.3f}"
    )
PY

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
