# LabUtopia Benchmark

This directory is a standalone LabUtopia simulator and benchmark checkout prepared for LabVLA evaluation. It contains the Isaac Sim task definitions, Franka controllers, scene assets, task configurations, data collectors, and the OpenPI-compatible inference client.

## Requirements

- Ubuntu 24.04
- Python 3.11
- Isaac Sim 5.1
- NVIDIA GPU with CUDA support (Isaac Sim does not support A100/A800)

Install Python dependencies after Isaac Sim is available:

```bash
pip install -r requirements.txt
pip install -e packages/openpi-client
```

## Run a benchmark task

Run from this directory so Hydra resolves `config/` and relative asset paths correctly:

```bash
python main.py --headless --no-video --config-name level3_TransportBeaker
```

Set `mode: infer`, `infer.type: remote`, the server `host` and `port`, and `max_episodes` in the selected configuration. `--no-video` disables video output. The current entry point also leaves video saving disabled by default when the flag is omitted.

Results are written under `outputs/infer/<date>/<run>/`, including the Hydra configuration and `inference_results.json`. Add `--save-action-trajectory` to save per-episode action JSON files.

## Collect demonstrations

Set `mode: collect` in a task configuration and run:

```bash
python main.py --config-name level1_pick
```

Collected HDF5 episodes are written under the configured output directory. Convert them to LeRobot format with:

```bash
python scripts/convert_labsim_data_to_lerobot.py --data_dir <episode-directory> --num_processes 8 --fps 60 --repo_name <repo-id>
```

LabUtopia uses an eight-dimensional Franka vector: seven arm joints followed by one gripper-width channel. For LabVLA training, verify `meta/info.json` and select the matching schema: legacy exports use `state`/`actions`, while canonical exports use `observation.state`/`action`.

## Layout

- `assets/`: Isaac Sim scenes and robot assets
- `config/`: Level 1–5 task configurations
- `controllers/`: task, robot, and inference controllers
- `factories/`: task, robot, controller, and collector factories
- `tasks/`: task implementations and success checks
- `data_collectors/`: HDF5 demonstration writer
- `packages/openpi-client/`: remote policy client
- `tests/`: configuration and smoke tests

See `LICENSE` for the code license and the upstream LabUtopia project for asset terms.
