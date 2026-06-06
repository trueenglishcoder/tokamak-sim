#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3854.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3854_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3854_ip.csv \
  --out runs/simulated_replay_3854 \
  --video \
  --frame-stride 5 \
  --frame-dpi 90 \
  --fps 20 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3855.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3855_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3855_ip.csv \
  --out runs/simulated_replay_3855 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3856.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3856_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3856_ip.csv \
  --out runs/simulated_replay_3856 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3857.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3857_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3857_ip.csv \
  --out runs/simulated_replay_3857 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3858.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3858_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3858_ip.csv \
  --out runs/simulated_replay_3858 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3859.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3859_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3859_ip.csv \
  --out runs/simulated_replay_3859 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3862.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3862_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3862_ip.csv \
  --out runs/simulated_replay_3862 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3863.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3863_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3863_ip.csv \
  --out runs/simulated_replay_3863 \
  --verbose

docker compose run --rm tokamak-sim python scripts/run_simulation_artifacts.py \
  --config configs/T15MD_new_data.toml \
  --initial-currents configs/initial_currents/T15MD_new_data_3864.toml \
  --steps 1439 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new/coils/t15md_3864_coils.csv \
  --angles 32 \
  --scenario ip_follow \
  --scenario-arg ip_csv=data/t15_data_new/ip/t15md_3864_ip.csv \
  --out runs/simulated_replay_3864 \
  --verbose
