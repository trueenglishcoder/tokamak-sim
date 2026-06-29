# Workflows

This file records maintained simulator workflows. Commands assume local ignored
`configs/`, `data/`, and `runs/` inputs exist.

## Run A Simulation With Artifacts

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

The artifact runner writes a manifest, time-series CSV, events CSV, boundary
plot, time-series plot, and optional frames/video.

## Generate Trim50 Plain GPU 1e-6 Replay References

The current RL path expects replay references generated from
`data/t15_data_new_trim50/` using the trim50 setup configs under:

```text
runs/t15md_trim50_plain_gpu_1e6_setup/
```

The resulting dataset should live at:

```text
runs/t15md_limited_replay_dataset_trim50_gpu_plain_1e6/
```

Each shot reference should include:

```text
lqr_boundary_reference_<SHOT>.npz
lqr_boundary_reference_<SHOT>.json
run_timeseries*.csv
events*.csv
manifest*.json
```

This dataset is the boundary reference source for the final `tokamak-rl-v2`
replay-window/oracle-target builder.

## Run A Learned Policy In Tokamak-Sim

```bash
python scripts/run_simulation_artifacts.py \
  --config <rl-run>/generated_configs/T15MD_new_data.toml \
  --initial-state runs/t15md_trim50_plain_gpu_1e6_setup/T15MD_new_data_trim50_3864.toml \
  --steps 500 \
  --controller learned_magnetic_controller \
  --controller-arg export_dir=<rl-run>/exports/best_actor \
  --controller-arg episode_norm_steps=100 \
  --controller-arg rolling_episode_norm=true \
  --angles 32 \
  --scenario t15_replay_reference \
  --scenario-arg reference_npz=runs/t15md_limited_replay_dataset_trim50_gpu_plain_1e6/lqr_boundary_reference_3864.npz \
  --out runs/learned_policy_example_3864 \
  --compute-backend gpu \
  --no-progress
```

The learned controller uses `absolute_jdot_command_v1`; old delta-Jdot exports
are rejected.

## Sigma/L Fits

Grid fit:

```bash
python scripts/fit_sigma_L_grid.py \
  --config configs/T15MD_new_data.toml \
  --ip-dir data/t15_data_new_trim50/ip \
  --coils-dir data/t15_data_new_trim50/coils \
  --sigma-min 1.5e6 \
  --sigma-max 3.5e6 \
  --sigma-points 31 \
  --L-min 5e-7 \
  --L-max 5e-6 \
  --L-points 31 \
  --top-k 100 \
  --out output \
  --name t15_trim50_sigma_L_refined \
  --plot
```

Gradient/SPSA alternative:

```bash
python scripts/fit_sigma_L_gradient.py --help
```

## Programmatic Simulation Session

External tools can use the bridge without artifact generation:

```python
import numpy as np

from tokamak_control.bridge import CurrentAction, SimulationSession

session = SimulationSession.from_paths(
    config_path="configs/T15MD_new_data.toml",
    initial_state_path="configs/initial_states/T15MD_new_data_3864.toml",
    scenario_name="nominal",
    scenario_args={},
    angles=32,
    steps=100,
)
reset = session.reset()
action = CurrentAction(
    active_currents_next=reset.observation_snapshot.true_active_currents.copy(),
)
step = session.step_currents(action)
```

The bridge accepts absolute next currents. Normalization, RL, search, or
optimization logic belongs outside this repository.

## Tests

Focused simulator checks:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_learned_magnetic_controller.py \
  tests/test_batched_gpu_training_path.py \
  tests/test_legacy_boundary.py \
  tests/test_lqr_t15_zaitsev.py
```

The full workflow suite may require local ignored datasets under `data/`.
