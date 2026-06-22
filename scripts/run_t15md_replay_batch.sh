#!/usr/bin/env bash
set -euo pipefail

SHOTS=(3854 3855 3856 3857 3858 3859 3862 3863 3864)

CONFIG="${CONFIG:-configs/T15MD_new_data.toml}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"
FRAME_DPI="${FRAME_DPI:-100}"
FPS="${FPS:-20}"
ANGLES="${ANGLES:-32}"

python - <<'PY'
from pathlib import Path
import tomllib

cfg_path = Path("configs/T15MD_new_data.toml")
cfg = tomllib.loads(cfg_path.read_text())
mode = cfg.get("boundary", {}).get("mode")
if mode != "legacy_contour_limited":
    raise SystemExit(f"{cfg_path} must use boundary.mode='legacy_contour_limited', got {mode!r}")
PY

for SHOT in "${SHOTS[@]}"; do
  COILS="data/t15_data_new/coils/t15md_${SHOT}_coils.csv"
  IP="data/t15_data_new/ip/t15md_${SHOT}_ip.csv"
  INIT="configs/initial_currents/T15MD_new_data_${SHOT}.toml"
  OUT="${RUNS_ROOT}/t15md_limited_replay_${SHOT}"

  test -f "$COILS"
  test -f "$IP"
  test -f "$INIT"

  STEPS="$(wc -l < "$IP")"
  echo "===== T15MD limited replay shot ${SHOT}: steps=${STEPS} ====="

  python scripts/run_simulation_artifacts.py \
    --config "$CONFIG" \
    --initial-currents "$INIT" \
    --steps "$STEPS" \
    --controller t15md_replay \
    --controller-arg "replay_path=$COILS" \
    --angles "$ANGLES" \
    --scenario ip_table \
    --scenario-arg "ip_csv=$IP" \
    --out "$OUT" \
    --compute-backend cpu \
    --video \
    --frame-stride "$FRAME_STRIDE" \
    --frame-dpi "$FRAME_DPI" \
    --fps "$FPS" \
    --verbose \
    --no-progress
done

python scripts/summarize_t15md_limited_replays.py \
  --runs-root "$RUNS_ROOT" \
  --out "${RUNS_ROOT}/t15md_limited_replay_summary.csv" \
  --shots "${SHOTS[@]}"
