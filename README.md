# Tokamak Sim

Pure-Python tokamak plasma simulation and control toolkit. The repository runs
closed-loop simulations from TOML machine configs and shot tables, writes
reproducible artifacts, and loads exported learned controllers from
`tokamak-rl-v2`.

See [CURRENT_PIPELINE.md](CURRENT_PIPELINE.md) for the active plant and learned
policy contracts.

## Setup

```bash
cd ~/tokamak/tokamak-sim
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Local configs, shot data, and generated runs are not source-controlled. Provide
them separately:

```text
configs/
data/
runs/
```

## Active Plant Contract

Controllers return absolute next-step currents. The simulator derives current
derivatives internally:

```text
Jdot = (J_next - J_now) / t_step
```

The learned-controller export contract is:

```text
absolute_jdot_command_v1
```

The actor chooses normalized absolute Jdot; `LearnedMagneticController` converts
that to absolute next currents before stepping the plant.

## Trim50 Replay Reference Generation

The final RL pipeline uses 6-PFC T15 trim50 data and plain GPU fixed-angle
boundary extraction with `legacy_precision_index2 = 1e-6`.

The canonical local/server artifacts are:

```text
data/t15_data_new_trim50/
runs/t15md_trim50_plain_gpu_1e6_setup/
runs/t15md_limited_replay_dataset_trim50_gpu_plain_1e6/
```

Those replay references are consumed by `tokamak-rl-v2` to build 0.1 s
replay-window oracle targets.

## Run A T15 Replay

```bash
python scripts/run_simulation_artifacts.py \
  --config runs/t15md_trim50_plain_gpu_1e6_setup/T15MD_new_data_trim50_plain_gpu_1e6_3864.toml \
  --initial-state runs/t15md_trim50_plain_gpu_1e6_setup/T15MD_new_data_trim50_3864.toml \
  --steps 100 \
  --controller t15md_replay \
  --controller-arg replay_path=data/t15_data_new_trim50/coils/t15md_3864_coils.csv \
  --angles 32 \
  --scenario ip_table \
  --scenario-arg ip_csv=data/t15_data_new_trim50/ip/t15md_3864_ip.csv \
  --out runs/example_trim50_replay_3864 \
  --no-progress
```

Add `--video --frame-stride 10 --frame-dpi 140 --fps 20` when you need a video.

## Run An Exported Learned Policy

```bash
python scripts/run_simulation_artifacts.py \
  --config <run>/generated_configs/T15MD_new_data.toml \
  --initial-state runs/t15md_trim50_plain_gpu_1e6_setup/T15MD_new_data_trim50_3864.toml \
  --steps 500 \
  --controller learned_magnetic_controller \
  --controller-arg export_dir=<run>/exports/best_actor \
  --controller-arg episode_norm_steps=100 \
  --controller-arg rolling_episode_norm=true \
  --angles 32 \
  --scenario t15_replay_reference \
  --scenario-arg reference_npz=runs/t15md_limited_replay_dataset_trim50_gpu_plain_1e6/lqr_boundary_reference_3864.npz \
  --out runs/learned_policy_example_3864 \
  --compute-backend gpu \
  --no-progress
```

For presentation videos, use presentation mode options exposed by
`scripts/run_simulation_artifacts.py` and keep the generated frames/video under a
clearly named `runs/PRESENTATION_*` folder.

## Fitting And Diagnostics

Sigma/L fitting and boundary comparison scripts remain supported diagnostics:

```bash
python scripts/fit_sigma_L_grid.py --help
python scripts/fit_sigma_L_gradient.py --help
python scripts/compare_cpu_gpu_boundary_t15_replay.py --help
```

LQR/Zaitsev controllers remain available for diagnostic comparison. They are not
the learned-policy action contract.

## Tests

Focused checks:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_learned_magnetic_controller.py \
  tests/test_batched_gpu_training_path.py \
  tests/test_legacy_boundary.py \
  tests/test_lqr_t15_zaitsev.py
```

Some tests require local ignored `configs/`, `data/`, or GPU availability and
will skip or fail clearly when those inputs are missing.

## Documentation

- [Repository Layout](docs/repository-layout.md)
- [Workflows](docs/workflows.md)
- [Architecture](docs/architecture.md)
- [LittleScope/Parity Audit](docs/tokamak-sim-0-audit/)
