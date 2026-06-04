# data_process — LabVLA_v3 dataset pipeline

All data-side tooling for LeRobot **v2.1** datasets lives here:

- **Cleanup**: detect problematic episodes, produce filtered symlink-copies
- **Stats**: compute normalization statistics (openpi-style, with q01/q99)
- **Migrate**: convert legacy JSON manifests → Python schema modules

> LabVLA_v3 is v2.1-canonical. v3.0 LeRobot was dropped 2026-04-20; use `*_old`-suffix datasets under `/all-flash-data/lerobot/.cache/LabUtopia/` for the v2.1 source.

## Unified CLI

```bash
python -m data_process <scan|clean|stats|migrate|vqa-clean|validate|preflight> [...args]
```

Run `python -m data_process -h` for the router help.

The full router entry set (see `cli.py`):

| Subcommand | Purpose |
|---|---|
| `scan` | run detectors, produce JSON report |
| `clean` | apply cleanup (scan report or manual drop-list) |
| `stats` | compute normalization stats (openpi-style, with q01/q99) |
| `migrate` | convert legacy JSON manifest → Python schema module |
| `vqa-clean` | remove bad RoboInter-VQA records and write a clean manifest |
| `validate` | run dataset-level validators across one or more repos |
| `preflight` | gate training launch on data-integrity invariants |

## Subcommands

### `scan` — run detectors, produce JSON report

```bash
python -m data_process scan \
    --root /all-flash-data/Pretrain_Data/robointer_droid \
    --out  /tmp/robointer_report.json \
    [--detectors video_concat_hazards,stats_orphans]  # optional subset
    [--action-cap 32 --state-cap 32]                   # for action_dim_cap
```

Detectors registered:

| Detector | What it catches | Example hit |
|---|---|---|
| `video_stubs` | Videos with ≤ N packets (tiny/truncated) | robointer eps 114193-4 |
| `video_concat_hazards` | Videos with outlier `first_dts` that break ffconcat | robointer eps 114193-4 (first_dts=0 vs -2048) |
| `video_decode` | mp4s PyAV can't open / demux | — |
| `missing_videos` | Episodes in episodes.jsonl but no mp4 on disk | — |
| `stats_orphans` | Entries in episodes_stats.jsonl with no episodes.jsonl row | robointer 21368 orphans |
| `action_dim_cap` | action_dim or state_dim > model cap (dataset-wide flag) | RoboCOIN leju_robot (118) |

### `clean` — apply cleanup

```bash
# A) detector-driven (uses report from scan)
python -m data_process clean \
    --src    /all-flash-data/Pretrain_Data/robointer_droid \
    --dst    /all-flash-data/Pretrain_Data/robointer_droid_clean \
    --report /tmp/robointer_report.json

# B) manual drop list
python -m data_process clean \
    --src /path/to/v21 --dst /path/to/v21_clean \
    --drop-episodes "1,2,114193"
```

Produces a **symlink-copy**: `<dst>` is ~1MB of rewritten meta JSONs + symlinks into `<src>`. The original dataset is never modified.

Episodes are renumbered contiguously in the cleaned copy (0..N-1), and parquet/mp4 filenames are updated via symlink rename to match — required so downstream tooling's filename-sort order still aligns with episodes.jsonl.

### `stats` — compute normalization statistics

```bash
python -m data_process stats \
    --dataset /all-flash-data/Pretrain_Data/robointer_droid_clean \
    --schema  robointer_droid \
    [--chunk_size 50] \
    [--no-quantile]    # skip q01/q99 (faster, smaller output)
```

Output: `<dataset>/meta/stats.json`. Same math as openpi's `RunningStats` (Welford + 5000-bin adaptive histogram for quantiles); runs the DeltaActions transform first, then aggregates over the transformed vector (so `mask=True` dims give delta distribution, `mask=False` dims give absolute — no separate two-pass).

### `migrate` — JSON manifest → Python schema

```bash
python -m data_process migrate \
    /all-flash-data/Pretrain_Data/robointer_droid \
    /all-flash-data/lerobot/.cache/LabUtopia/Level3_open_old \
    --out schemas/
```

One-shot utility: takes each dataset's `meta/labvla_manifest.json` and emits an equivalent `schemas/<name>.py` using the closest preset (`franka8` / `franka_split` / `aloha14`) or raw `DatasetSchema`.

## Typical end-to-end flow (robointer_droid example)

```bash
# 1. Scan
python -m data_process scan \
    --root /all-flash-data/Pretrain_Data/robointer_droid \
    --out  /tmp/robointer_report.json

# 2. Clean
python -m data_process clean \
    --src    /all-flash-data/Pretrain_Data/robointer_droid \
    --dst    /all-flash-data/Pretrain_Data/robointer_droid_clean \
    --report /tmp/robointer_report.json

# 3. Stats
python -m data_process stats \
    --dataset /all-flash-data/Pretrain_Data/robointer_droid_clean \
    --schema  robointer_droid

# 4. Train (points at cleaned dataset + its stats.json)
bash launch/labvla_pretrain.sh
```

## Architecture

```
data_process/
├── cli.py              # subcommand dispatcher (python -m data_process)
├── __main__.py         # invokes cli.main()
├── detectors/          # issue scanners (all subclass Detector)
├── cleanup/            # scan.py + apply.py
├── stats/              # running.py (RunningStats) + compute.py (CLI)
└── migrate/            # manifest_to_py.py (one-shot schema migration)
```

Adding a new detector: drop a file under `detectors/`, subclass `Detector`, implement `scan(root) -> DetectorResult`, register in `detectors/__init__.py::ALL_DETECTORS`.
