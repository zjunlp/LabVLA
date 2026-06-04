"""LeRobot v2.1 stats reader.

v2.1 layout: `data/chunk-XXX/episode_NNNNNN.parquet` — one parquet per episode.
Reading 100k+ episodes means 100k+ file opens, so filesystem latency dominates.
ThreadPoolExecutor with 32 workers overlaps the IO since pyarrow releases the
GIL during parquet footer reads (~20-30× speedup on remote DPC).

Missing-column tolerance: robocoin has 3 chunks (chunk-041/042/043) that lack
`gripper_open_scale_*`. Such parquets raise `ArrowInvalid: No match for
FieldRef.Name(...)` on `pq.read_table(p, columns=need_cols)`. We catch and
return None → the drain loop counts them as dropped without aborting.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_process.stats._reader_base import (
    build_target_dims,
    is_missing_column_error,
    open_and_read,
    stack_keys_to_array,
)


class OverwideColumnError(ValueError):
    """Signal that an overwide column should drop the parquet (v2/v21 path).

    v21 chunk-init validation drops bad chunks; v2 stats reader catches this
    and drops the parquet to keep its sample set in lock-step. v3 stats reader
    does NOT use this path: ``LeRobotV30Adapter._pad_row`` truncates samples,
    so v3 stats reader passes ``truncate_overwide=True`` and never raises.
    """


def _pad_last_dim(
    arr: np.ndarray, target_dim: int, *, truncate_overwide: bool = False
) -> np.ndarray:
    """Zero-pad to target_dim. Overwide → truncate or raise per caller policy."""
    cur = arr.shape[-1]
    if cur == target_dim:
        return arr
    if cur > target_dim:
        if truncate_overwide:
            return arr[..., :target_dim]
        raise OverwideColumnError(
            f"parquet column has last dim {cur} > schema target {target_dim} "
            f"(losing {cur - target_dim} dims). Adapter drops such chunks; "
            f"reader now drops them too. Fix the schema or the data."
        )
    pad = np.zeros(arr.shape[:-1] + (target_dim - cur,), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


def _stack_column_pa(
    table: pa.Table, key: str, target_dim: int | None = None,
    *, truncate_overwide: bool = False,
) -> np.ndarray:
    """pyarrow fast-path column → (T, D) float32 numpy array.

    Three cases:
      (a) fixed_size_list<float>[D]   → flatten + reshape (zero-copy when
          pyarrow owns a contiguous buffer).
      (b) list<float> (variable)      → flatten + offsets; if all rows share
          one length, reshape directly; otherwise vectorized scatter into a
          zero-padded output.
      (c) scalar numeric              → to_numpy().reshape(-1, 1).

    Beats pandas-object-row loops by ~50-500×.
    """
    col = table.column(key).combine_chunks()
    t = col.type
    T = len(col)

    if pa.types.is_fixed_size_list(t):
        D = t.list_size
        flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        arr = flat.reshape(T, D)
    elif pa.types.is_list(t) or pa.types.is_large_list(t):
        offsets = col.offsets.to_numpy()
        values = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        lens = np.diff(offsets)
        if T == 0:
            arr = np.zeros((0, target_dim or 0), dtype=np.float32)
        elif (lens == lens[0]).all():
            D = int(lens[0])
            arr = values.reshape(T, D) if values.size == T * D else np.zeros((T, D), dtype=np.float32)
        else:
            D = int(target_dim if target_dim is not None else lens.max())
            row_ids = np.repeat(np.arange(T, dtype=np.int64), lens)
            col_ids = np.arange(len(values), dtype=np.int64) \
                      - np.repeat(offsets[:-1].astype(np.int64), lens)
            mask = col_ids < D
            arr = np.zeros((T, D), dtype=np.float32)
            arr[row_ids[mask], col_ids[mask]] = values[mask]
    else:
        arr = col.to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(T, 1)

    if target_dim is not None and arr.shape[-1] != target_dim:
        arr = _pad_last_dim(arr, target_dim, truncate_overwide=truncate_overwide)
    return arr


def read_action_state(ds_root: Path, schema, workers: int = 32):
    """Read v2.1 per-episode parquets → (list[action_T_D], list[state_T_D]).

    Multi-key schemas (e.g. robocoin v2: `action + gripper_open_scale_action`)
    use per-key target dim lookup (`schema.action_dims[i]`).
    """
    parquets = sorted((ds_root / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files under {ds_root}/data/")

    # Shared schema → per-key target dims + deduped need_cols.
    state_targets, action_targets, need_cols, read_state_keys, read_action_keys = build_target_dims(schema)

    def _read_one(p: Path):
        try:
            table = open_and_read(p, need_cols)
        except Exception as e:
            if is_missing_column_error(e):
                return None
            raise
        try:
            state = stack_keys_to_array(
                table, read_state_keys, state_targets, _stack_column_pa
            )
            act = stack_keys_to_array(
                table, read_action_keys, action_targets, _stack_column_pa
            )
        except OverwideColumnError:
            return None
        return act, state

    N = len(parquets)
    acts_ep: list = []
    states_ep: list = []
    dropped = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_read_one, p): i for i, p in enumerate(parquets)}
        done = 0
        for fut, i in futures.items():
            res = fut.result()
            if res is None:
                dropped += 1
            else:
                act, state = res
                acts_ep.append(act)
                states_ep.append(state)
            done += 1
            if done % 2000 == 0 or done == N:
                dt = time.monotonic() - t0
                print(f"[stats] read {done}/{N} episodes ({done/dt:.0f} eps/s, {dt:.0f}s elapsed)")
    if dropped:
        print(f"[stats] dropped {dropped} parquets lacking required columns ({need_cols})")
    return acts_ep, states_ep
