"""Compute normalization stats for a LeRobot v2.1 or v3.0 dataset.

Single-pass pipeline:
  1. Read info.json, detect `codebase_version`.
  2. Dispatch to v2/reader or v3/reader → (acts_ep, states_ep) per episode.
  3. Apply DeltaActions (per schema.delta_mask) and accumulate into a
     RunningStats for mean/std (+ optional q01/q99).
  4. Write `<dataset>/meta/stats.json`.

For mask=True dims the action pipeline yields delta distribution; for
mask=False dims it yields absolute — one pass, no second sweep.

Usage:
  python -m data_process stats \\
      --dataset /path/to/dataset \\
      --schema  <name>_schema_file_stem \\
      [--chunk_size 50] \\
      [--no-quantile]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data_process.stats.running import RunningStats, save as save_stats

# Expose src/ and repo root so schema.resolve + schemas/ autoload work.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import resolve as resolve_schema  # noqa: E402


def _iter_delta_action_chunks(acts_ep, states_ep, mask, chunk_size):
    """DeltaActions: mask=True dims have state subtracted; mask=False stay
    absolute. Yields ONE big (N, D) batch per episode where
    N = Σ_{t=0}^{T-1} min(chunk_size, T-t). RunningStats.update is
    numpy-vectorized, so yielding per-episode (~5-10k rows) is far faster
    than per-chunk (~50 rows).

    When ``state_dim < action_dim`` (e.g. 7-dim joint state paired with 8-dim
    joint+gripper action), zero-pad ``st`` to action width before subtracting.
    Mirrors the runtime ``DeltaActionTransformFn`` in
    ``src/transforms/core.py`` and avoids a silent shape-broadcast error when
    ``mask`` has length ``action_dim`` but state is shorter.
    """
    mask = np.asarray(mask, dtype=bool)
    for act, st in zip(acts_ep, states_ep):
        T = act.shape[0]
        if T == 0:
            continue
        # Pad state to action width so `mask` (length action_dim) and `st[t]`
        # align. Matches src/transforms/core.py.
        if st.shape[-1] < act.shape[-1]:
            pad_width = act.shape[-1] - st.shape[-1]
            st = np.concatenate(
                [st, np.zeros(st.shape[:-1] + (pad_width,), dtype=st.dtype)],
                axis=-1,
            )
        elif st.shape[-1] > act.shape[-1]:
            st = st[..., :act.shape[-1]]
        pieces = []
        for t in range(T):
            K = min(chunk_size, T - t)
            piece = act[t:t + K] - np.where(mask, st[t], 0.0)[None, :]
            pieces.append(piece)
        yield np.concatenate(pieces, axis=0)


def _iter_abs_action_chunks(acts_ep, states_ep, chunk_size):
    """Absolute action chunks (no state subtraction — ignores delta_mask).

    For training π0.5-style abs mode: stats reflect the raw target-pose
    distribution directly, regardless of what the schema's delta_mask says.
    Same chunking shape as delta version so q01/q99 are comparable.
    """
    for act in acts_ep:
        T = act.shape[0]
        if T == 0:
            continue
        pieces = []
        for t in range(T):
            K = min(chunk_size, T - t)
            pieces.append(act[t:t + K])
        yield np.concatenate(pieces, axis=0)


def _clear_stats_invalidated(ds_root: Path) -> None:
    """Remove the ``stats_invalidated`` block left by ``cleanup.apply`` after a
    successful stats recompute.

    Mirrors the inverse of ``data_process/cleanup/apply.py:rewrite_meta``:
      * deletes ``info.json["stats_invalidated"]`` (if present),
      * deletes the sibling ``meta/stats.invalidated.json`` marker.

    Best-effort and quiet on a clean dataset (no flag → nothing to do). The
    info.json rewrite is in-place; the file already exists because the reader
    parsed it earlier in this run.
    """
    meta = ds_root / "meta"
    info_path = meta / "info.json"
    if info_path.is_file():
        try:
            with open(info_path) as f:
                info = json.load(f)
        except (OSError, ValueError):
            info = None
        if isinstance(info, dict) and "stats_invalidated" in info:
            del info["stats_invalidated"]
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            print("[stats] cleared info.json['stats_invalidated'] "
                  "(stats successfully recomputed)")
    marker = meta / "stats.invalidated.json"
    if marker.is_file():
        marker.unlink()
        print(f"[stats] removed stale {marker.name}")


def _read_by_version(ds_root: Path, schema):
    """Inspect info.json, dispatch to the matching reader.

    Returns ``(acts_ep, states_ep, info)`` so the caller can validate frame
    coverage against ``info["total_frames"]`` without re-reading info.json.
    """
    info_path = ds_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json missing at {info_path}")
    with open(info_path) as f:
        info = json.load(f)
    ver = info.get("codebase_version", "")
    print(f"[stats] detected codebase_version={ver!r}")
    if ver.startswith("v2"):
        from data_process.stats.v2.reader import read_action_state
    elif ver.startswith("v3"):
        from data_process.stats.v3.reader import read_action_state
    else:
        raise ValueError(
            f"Unsupported codebase_version={ver!r}. Expected v2.x or v3.x."
        )
    acts_ep, states_ep = read_action_state(ds_root, schema)
    return acts_ep, states_ep, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=str,
                    help="schema name (file stem in schemas/) or path/to/file.py")
    ap.add_argument("--chunk_size", type=int, default=50)
    ap.add_argument("--no-quantile", action="store_true",
                    help="skip q01/q99 computation")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <dataset>/meta/stats.json")
    args = ap.parse_args()

    schema = resolve_schema(args.schema)
    print(f"[stats] schema={schema.schema_id} "
          f"state_dim={sum(schema.state_dims)} action_dim={sum(schema.action_dims)}")
    mask = schema.to_bool_mask().cpu().numpy().astype(bool)
    print(f"[stats] delta_mask={mask.tolist()}")

    print(f"[stats] reading {args.dataset}/data/ ...")
    acts_ep, states_ep, info = _read_by_version(args.dataset, schema)
    total_frames = sum(a.shape[0] for a in acts_ep)
    print(f"[stats] {len(acts_ep)} episodes, {total_frames} frames")

    # Fail-loud safety net for partial-coverage reader bugs (e.g. the v3
    # global/local index miss). 1% slack absorbs legitimate overwide /
    # missing-column drops; bigger gaps abort before writing stats.
    expected_frames = int(info.get("total_frames", 0))
    if expected_frames > 0:
        coverage = total_frames / expected_frames
        if coverage < 0.99:
            raise RuntimeError(
                f"[stats] partial-coverage abort: read {total_frames} frames but "
                f"info.json declares {expected_frames} (coverage={coverage:.2%}). "
                f"Aborting before writing stats — re-run after fixing the reader "
                f"or manifest."
            )
        print(f"[stats] frame coverage = {coverage:.2%} of info.json::total_frames")

    rs_state = RunningStats()
    for st in states_ep:
        rs_state.update(st)

    # Dual accumulator: delta-transformed (current "action" key) and raw abs
    # ("action_abs" key, used by action_mode="abs" training).
    rs_action_delta = RunningStats()
    for chunk in _iter_delta_action_chunks(
        acts_ep, states_ep, mask, args.chunk_size,
    ):
        rs_action_delta.update(chunk)

    rs_action_abs = RunningStats()
    for chunk in _iter_abs_action_chunks(acts_ep, states_ep, args.chunk_size):
        rs_action_abs.update(chunk)

    stats = {
        "observation.state": rs_state.get_statistics(),
        "action": rs_action_delta.get_statistics(),
        "action_abs": rs_action_abs.get_statistics(),
    }
    if args.no_quantile:
        for s in stats.values():
            s.q01 = None
            s.q99 = None

    out = args.out or args.dataset / "meta" / "stats.json"
    # Persist chunk_size into stats.json so training can verify the stats were
    # computed against the chunk_size it actually uses.
    save_stats(out, stats, chunk_size=int(args.chunk_size))
    print(f"[stats] wrote {out} (chunk_size={int(args.chunk_size)})")

    # Clear the cleanup-set ``stats_invalidated`` flag now that a correct
    # recompute has landed: cleanup.apply sets it (+ a stats.invalidated.json
    # marker) after dropping episodes, and dataset_build.py refuses to start
    # while it's set, so without this the dataset stays permanently invalid even
    # after the operator re-ran stats. Only clear when we wrote this dataset's
    # canonical meta/stats.json (not a custom --out), to avoid touching others.
    _wrote_canonical = Path(out).resolve() == (
        args.dataset / "meta" / "stats.json"
    ).resolve()
    if _wrote_canonical:
        _clear_stats_invalidated(args.dataset)
    for k, s in stats.items():
        print(f"  {k}: mean[:4]={s.mean[:4].round(4)} std[:4]={s.std[:4].round(4)}"
              + (f" q01[:4]={s.q01[:4].round(4)}" if s.q01 is not None else ""))


if __name__ == "__main__":
    main()
