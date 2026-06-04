"""LeRobot v3.0 stats reader — shard-sliced layout.

v3 packs many episodes into a few large parquet shards (`data/chunk-XXX/
file-NNN.parquet`). Per-episode row ranges live in `meta/episodes/chunk-XXX/
file-NNN.parquet` as `(data/chunk_index, data/file_index, dataset_from_index,
dataset_to_index)`.

Compared to v2.1 this opens ~O(shard_count) files instead of O(episode_count).
For a large v2.1 dataset that's typically a few shards vs tens of thousands of per-episode parquets — ~100x
fewer file opens, so the DPC filesystem latency bottleneck disappears.

Column-presence tolerance: if a shard's columns don't include all schema
keys, the whole shard (and every episode inside) is dropped — matching v2's
per-episode drop semantics. In practice v3 shards are produced by the same
converter per dataset so mixed-column shards are unusual, but the safety net
lets us run any half-written dataset without a special flag.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

import os

from data_process.stats.v2.reader import _stack_column_pa
from data_process.stats._reader_base import (
    build_target_dims,
    is_missing_column_error,
    open_and_read,
    stack_keys_to_array,
)

# Honor the SAME env flag as the adapter's _pad_row (which raises on overwide
# columns by default, gated by LABVLA_ALLOW_TRUNCATE=1) so stats and training
# agree: either both fail loud on schema mismatch, or both truncate. A
# truncate=True default would hide data-quality problems at stats time that then
# surface as adapter raises during training.
_v3_stack = partial(
    _stack_column_pa,
    truncate_overwide=os.environ.get("LABVLA_ALLOW_TRUNCATE") == "1",
)


class _ShardTag(Enum):
    OK = "ok"
    DROPPED = "dropped"
    MISSING = "missing"


def _read_episodes_meta(ds_root: Path) -> pa.Table:
    """Concat every meta/episodes/chunk-*/file-*.parquet into one table.

    Only keeps the columns we need for shard dispatch (episode_index is for
    debugging; the rest drive the slicing).
    """
    meta_root = ds_root / "meta" / "episodes"
    if not meta_root.is_dir():
        raise FileNotFoundError(
            f"v3 meta/episodes dir missing: {meta_root}. Is this really a v3 dataset?"
        )
    paths = sorted(meta_root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No meta/episodes parquet under {meta_root}")
    cols = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    tables = [pq.read_table(p, columns=cols) for p in paths]
    return pa.concat_tables(tables)


def read_action_state(ds_root: Path, schema, workers: int = 16):
    """Return (acts_ep, states_ep) per-episode arrays from a v3 dataset.

    `dataset_from_index` / `dataset_to_index` in meta/episodes are *global*
    cumulative frame indices, but each shard parquet's rows are numbered
    locally `0..num_rows-1`. We must subtract the shard's offset (= the
    smallest `dataset_from_index` of any episode in that shard) before
    slicing, mirroring `LeRobotV30Adapter._load_ep_parquet`. Without the
    offset, every shard past the first reads garbage or empty: pyarrow's
    `Table.slice(start, length)` silently truncates when `start >= num_rows`.
    """
    state_targets, action_targets, need_cols, read_state_keys, read_action_keys = build_target_dims(schema)

    ep_meta = _read_episodes_meta(ds_root)
    # Group episode slice ranges by (chunk_index, file_index) so each shard
    # is opened exactly once.
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for ci, fi, a, b, ei in zip(
        ep_meta["data/chunk_index"].to_pylist(),
        ep_meta["data/file_index"].to_pylist(),
        ep_meta["dataset_from_index"].to_pylist(),
        ep_meta["dataset_to_index"].to_pylist(),
        ep_meta["episode_index"].to_pylist(),
    ):
        groups.setdefault((int(ci), int(fi)), []).append((int(ei), int(a), int(b)))

    # Iterate groups (not a filesystem scan) so orphan parquets don't trip us up.
    shard_path = lambda ci, fi: ds_root / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"

    def _read_shard(key: tuple[int, int]):
        ci, fi = key
        p = shard_path(ci, fi)
        if not p.exists():
            return (_ShardTag.MISSING, key)
        try:
            table = open_and_read(p, need_cols)
        except Exception as e:
            if is_missing_column_error(e):
                return (_ShardTag.DROPPED, key)
            raise
        # Per-shard global→local offset mirrors LeRobotV30Adapter._shard_offsets.
        offset = min(a for _ei, a, _b in groups[key])
        n_rows = table.num_rows
        eps = []
        for ei, a, b in groups[key]:
            local_a = a - offset
            local_b = b - offset
            if not (0 <= local_a <= local_b <= n_rows):
                raise ValueError(
                    f"v3 shard {p.name}: episode {ei} slice [{local_a}:{local_b}] "
                    f"out of bounds for {n_rows}-row parquet "
                    f"(global a={a}, b={b}, offset={offset}). Dataset's "
                    f"meta/episodes is inconsistent with on-disk data."
                )
            sub = table.slice(local_a, local_b - local_a)
            state = stack_keys_to_array(
                sub, read_state_keys, state_targets, _v3_stack
            )
            act = stack_keys_to_array(
                sub, read_action_keys, action_targets, _v3_stack
            )
            eps.append((ei, act, state))
        return (_ShardTag.OK, eps)

    acts_ep: list = []
    states_ep: list = []
    dropped_shards = 0
    dropped_episodes = 0
    dropped_paths: list = []
    missing = 0
    missing_paths: list = []
    t0 = time.monotonic()
    n_shards = len(groups)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_read_shard, k) for k in groups]
        done = 0
        for fut in futures:
            tag, payload = fut.result()
            if tag is _ShardTag.DROPPED:
                dropped_shards += 1
                dropped_episodes += len(groups[payload])
                if len(dropped_paths) < 10:
                    dropped_paths.append(str(shard_path(*payload)))
            elif tag is _ShardTag.MISSING:
                missing += 1
                if len(missing_paths) < 10:
                    missing_paths.append(str(shard_path(*payload)))
            else:
                for ei, act, state in payload:
                    acts_ep.append(act)
                    states_ep.append(state)
            done += 1
            if done == n_shards or done % max(1, n_shards // 20) == 0:
                dt = time.monotonic() - t0
                print(f"[stats] read {done}/{n_shards} shards ({dt:.0f}s elapsed)")
    if dropped_shards:
        sample_str = "; ".join(dropped_paths[:5]) + (" ..." if len(dropped_paths) > 5 else "")
        print(f"[stats] dropped {dropped_shards} shards ({dropped_episodes} eps) "
              f"lacking required columns ({need_cols}). Examples: {sample_str}")
    # A shard referenced by meta/episodes but absent on disk means the dataset's
    # declared sample face is NOT fully readable; returning the partial set would
    # compute stats over a silently-shrunk frame population that still looks
    # "successful". Hard-fail so the operator fixes it before stats are written.
    if missing:
        sample_m = "; ".join(missing_paths[:5]) + (" ..." if len(missing_paths) > 5 else "")
        raise FileNotFoundError(
            f"v3 stats reader: {missing} shard(s) listed in meta/episodes are "
            f"absent on disk — refusing to return partial samples (would yield "
            f"stats over an incomplete frame set). Examples: {sample_m}. "
            f"Fix the dataset (restore the missing data/chunk-*/file-*.parquet) "
            f"or regenerate meta/episodes so it only references shards that exist."
        )
    return acts_ep, states_ep
