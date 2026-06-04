"""Produce a cleaned symlink-copy of a v2.1 dataset.

Two drop-list sources (mutually exclusive):
  --report    JSON produced by `python -m data_process scan`
  --drop-episodes   comma-separated indices (manual, e.g. "114193,114194")

Usage:

  # A) detector-driven (recommended):
  python -m data_process clean \\
      --src    /all-flash-data/Pretrain_Data/robointer_droid \\
      --dst    /all-flash-data/Pretrain_Data/robointer_droid_clean \\
      --report /tmp/robointer_cleanup.json

  # B) manual drop list:
  python -m data_process clean \\
      --src  /path/to/v21_dataset --dst /path/to/v21_clean \\
      --drop-episodes "1,2,114193"

Architecture (single pass + threadpool):
  1. Parse drop list, read src/meta/episodes.jsonl → valid_eps + remap (old→new).
  2. `stage_and_remap`: walk src/{data,videos,images}, for each eligible file
     compute dst path with *new chunk-dir* (new_ep // chunks_size) *and* new
     filename, symlink in parallel via ThreadPoolExecutor.
  3. `rewrite_meta`: rewrite episodes.jsonl / episodes_stats.jsonl / info.json
     totals; symlink other meta files unchanged.

Invariant: LeRobot v2.1 layout requires episode N live in
`chunk-{N//chunks_size:03d}/`; the adapter computes this from new_ep and fails
with FileNotFoundError if files stay in their old chunk dirs. The single pass
above renumbers chunk-dir and filename together so this holds.

The original src/ is NEVER modified.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Accept any digit width — some datasets use 4-digit episode numbers (e.g.
# 10k-episode Bridge v1). Output is zfill(6)'d to the canonical 6-digit form.
_EP_RE = re.compile(r"episode_(\d+)\.")
_CHUNK_RE = re.compile(r"chunk-(\d+)")

logger = logging.getLogger(__name__)


def ep_from_name(name: str) -> int | None:
    m = _EP_RE.search(name)
    return int(m.group(1)) if m else None


def _symlink(src: Path, dst: Path) -> None:
    """Atomically replace dst with symlink to src. `src` should be an absolute
    path (caller-resolved) to avoid redundant readlink in thread workers.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def build_remap(
    src_meta: Path, drop_eps: set[int]
) -> tuple[list[int], dict[int, int], int]:
    """Read src/meta/episodes.jsonl, apply drop filter, return:
      kept_old   — sorted list of original episode_index values that survive
      remap      — old_ep → new_ep (contiguous 0..N-1)
      total_kept — len(kept_old)
    """
    kept_old: list[int] = []
    with open(src_meta / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if ep["episode_index"] in drop_eps:
                continue
            kept_old.append(int(ep["episode_index"]))
    kept_old.sort()
    remap = {old: new for new, old in enumerate(kept_old)}
    return kept_old, remap, len(kept_old)


def _compute_new_rel(rel: Path, new_ep: int, chunks_size: int) -> Path:
    """Replace the `chunk-XXX` path component with `chunk-{new_ep//chunks_size:03d}`
    and the filename's `episode_OLD` with `episode_{new_ep:06d}`. All other
    path parts intact.

    Path patterns handled:
      data/chunk-XXX/episode_NNN.parquet       → data/chunk-NEW/episode_NEWEP.parquet
      videos/chunk-XXX/<cam>/episode_NNN.mp4   → videos/chunk-NEW/<cam>/episode_NEWEP.mp4
      images/chunk-XXX/<cam>/episode_NNN.jpg   → images/chunk-NEW/<cam>/episode_NEWEP.jpg

    Chunking contract: LeRobot v2.1 places episode N inside
    `chunk-{N//chunks_size:03d}/`, with `chunks_size` declared per-dataset in
    `meta/info.json` (read by ``main()`` and passed in). `chunks_size` must be
    a positive integer — 0 is undefined and huge values degenerate to
    all-in-chunk-0; both collide episode files across datasets, so enforce it.
    """
    if chunks_size <= 0:
        raise ValueError(
            f"_compute_new_rel: chunks_size must be > 0 (got {chunks_size}). "
            f"Check the source dataset's meta/info.json for a sensible "
            f"'chunks_size' key (LeRobot v2.1 default is 1000)."
        )
    new_chunk = new_ep // chunks_size
    parts = list(rel.parts)
    # Replace ALL chunk-XXX components, not just the first: v3-like layouts have
    # `data/chunk-AAA/...file-chunk-BBB.parquet` where BBB is a shard index —
    # both need the canonical 3-digit form.
    for i, part in enumerate(parts):
        if _CHUNK_RE.fullmatch(part):
            parts[i] = f"chunk-{new_chunk:03d}"
    parts[-1] = _EP_RE.sub(f"episode_{new_ep:06d}.", parts[-1])
    return Path(*parts)


def _fast_symlink(src: Path, dst: Path) -> None:
    """symlink without mkdir+exists check — assumes parent dir already exists
    and dst doesn't. Caller is responsible for both preconditions (saves 2
    syscalls per file, critical on NFS)."""
    try:
        dst.symlink_to(src)
    except FileExistsError:
        dst.unlink()
        dst.symlink_to(src)


def stage_and_remap(
    src_root: Path,
    dst_root: Path,
    drop_eps: set[int],
    valid_eps: set[int] | None,
    remap: dict[int, int],
    chunks_size: int,
    workers: int = 32,
) -> dict:
    """Single-pass clone: for each eligible file under src/{data,videos,images},
    resolve its new path (new chunk-dir + new filename) and symlink in parallel.

    Optimization: plan first (walk + compute new paths + collect unique parent
    dirs), batch mkdir all parents serially (hundreds of dirs, trivial cost),
    then symlink in parallel without per-file mkdir (saves ~50% NFS time on
    datasets with many frame-level files).

    Eligibility:
      - drop_eps:  file's ep index is in drop list → skip
      - valid_eps: if given, file's ep index must be in valid_eps (orphan
                   filter — protects against upstream contamination where
                   ep has files on disk but no episodes.jsonl entry)
      - remap:     ep must be in remap (belt-and-suspenders to valid_eps)
    """
    n_dropped = 0
    n_orphan = 0

    # Phase 1: walk + classify + compute final dst paths.
    jobs: list[tuple[Path, Path]] = []    # (full_src_abs, dst_abs)
    for sub in ("data", "videos", "images"):
        src_dir = src_root / sub
        if not src_dir.is_dir():
            continue
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = Path(root) / fn
                rel = full.relative_to(src_root)
                ep = ep_from_name(fn)
                if ep is not None and ep in drop_eps:
                    n_dropped += 1
                    continue
                if ep is not None and (
                    (valid_eps is not None and ep not in valid_eps)
                    or ep not in remap
                ):
                    n_orphan += 1
                    continue
                if ep is None:
                    new_rel = rel
                else:
                    new_rel = _compute_new_rel(rel, remap[ep], chunks_size)
                jobs.append((full.resolve(), dst_root / new_rel))

    # Phase 2: pre-create all unique dst parent dirs (small set, serial fast).
    unique_parents = {dst.parent for _, dst in jobs}
    for p in unique_parents:
        p.mkdir(parents=True, exist_ok=True)

    # Phase 3: parallel symlink creation — no mkdir in hot path.
    n_linked = 0
    counter_lock = threading.Lock()

    def _link_one(src: Path, dst: Path) -> None:
        nonlocal n_linked
        _fast_symlink(src, dst)
        with counter_lock:
            n_linked += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_link_one, s, d) for s, d in jobs]
        # Drain + surface exceptions.
        for fut in futures:
            fut.result()

    # Top-level non-meta files — serially, tiny volume.
    for entry in src_root.iterdir():
        if entry.name in ("data", "videos", "images", "meta"):
            continue
        if entry.is_file():
            _symlink(entry.resolve(), dst_root / entry.name)
            n_linked += 1

    # Prune empty camera subdirs under videos/ (contamination-only cam dirs).
    pruned = 0
    videos_dst = dst_root / "videos"
    if videos_dst.is_dir():
        for root, _dirs, _files in os.walk(videos_dst, topdown=False):
            p = Path(root)
            if p == videos_dst:
                continue
            try:
                if not any(p.iterdir()):
                    p.rmdir()
                    pruned += 1
            except OSError:
                pass

    return {
        "linked": n_linked,
        "dropped_files": n_dropped,
        "orphan_files": n_orphan,
        "pruned_dirs": pruned,
    }


def rewrite_meta(
    src_root: Path,
    dst_root: Path,
    kept_old: list[int],
    remap: dict[int, int],
    drop_eps: set[int],
    orphan_stats_eps: set[int],
    preserve_stats: bool = False,
) -> dict:
    """Rebuild meta/episodes.jsonl + meta/episodes_stats.jsonl using the
    pre-built remap. Also rewrites info.json totals. Symlinks tasks.jsonl
    / other meta files unchanged.

    stats.json is NOT symlinked by default: cleanup removes/renumbers episodes
    but the source stats.json was computed over the original set, so keeping it
    silently leaks dropped episodes' distribution into downstream
    train/deploy normalization. Default writes a ``stats.invalidated.json``
    marker and records ``stats_invalidated=true`` in ``info.json`` so callers
    fail loud (or re-run ``python -m data_process stats``). Pass
    ``preserve_stats=True`` only after verifying the dropped episodes do not
    move the quantile distribution materially.
    """
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # episodes.jsonl
    eps_in: list[dict] = []
    total_frames = 0
    with open(src_meta / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            old = ep["episode_index"]
            if old in drop_eps:
                continue
            ep["episode_index"] = remap[old]
            eps_in.append(ep)
            total_frames += int(ep.get("length", 0))
    eps_in.sort(key=lambda e: e["episode_index"])
    with open(dst_meta / "episodes.jsonl", "w") as f:
        for ep in eps_in:
            f.write(json.dumps(ep) + "\n")

    # episodes_stats.jsonl (if present): drop orphans, renumber.
    es_src = src_meta / "episodes_stats.jsonl"
    n_orphan_skipped = 0
    if es_src.exists():
        kept_set = set(kept_old)
        es_in: list[dict] = []
        with open(es_src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                es = json.loads(line)
                old = es["episode_index"]
                if old in drop_eps:
                    continue
                if old in orphan_stats_eps or old not in kept_set:
                    n_orphan_skipped += 1
                    continue
                es["episode_index"] = remap[old]
                es_in.append(es)
        es_in.sort(key=lambda e: e["episode_index"])
        with open(dst_meta / "episodes_stats.jsonl", "w") as f:
            for es in es_in:
                f.write(json.dumps(es) + "\n")

    # tasks.jsonl: symlink as-is.
    tasks_src = src_meta / "tasks.jsonl"
    if tasks_src.exists():
        _symlink(tasks_src.resolve(), dst_meta / "tasks.jsonl")

    # stats.json: only preserve under explicit opt-in. But if no episode was
    # dropped, the source distribution is identical to the cleaned one, so
    # invalidating would force a needless recompute — only invalidate when the
    # per-episode set actually changed.
    stats_src = src_meta / "stats.json"
    stats_invalidated = False
    _distribution_changed = len(drop_eps) > 0
    if stats_src.exists():
        if preserve_stats:
            _symlink(stats_src.resolve(), dst_meta / "stats.json")
            logging.warning(
                "[cleanup] --preserve-stats: re-linking source stats.json. "
                "It still reflects the pre-clean distribution and may bias "
                "normalization. Verify before training."
            )
        elif not _distribution_changed:
            # No episode dropped: on-disk distribution is byte-identical to
            # source, so symlink stats through without an invalidation marker.
            _symlink(stats_src.resolve(), dst_meta / "stats.json")
            logging.info(
                "[cleanup] no episodes dropped; preserving source stats.json "
                "(distribution unchanged)."
            )
        else:
            invalidation_payload = {
                "reason": "data_process.cleanup dropped episodes; quantile/mean stats "
                          "computed from original dataset are no longer valid.",
                "dropped_episode_count": len(drop_eps),
                "kept_episode_count": len(kept_old),
                "remediation": "Run `python -m data_process stats --dataset <dst>` "
                               "to recompute, or pass --preserve-stats at cleanup "
                               "time if you've manually verified the distribution.",
            }
            with open(dst_meta / "stats.invalidated.json", "w") as f:
                json.dump(invalidation_payload, f, indent=2)
            stats_invalidated = True
            logging.info(
                "[cleanup] stats.json invalidated; wrote stats.invalidated.json. "
                "Recompute with `python -m data_process stats`."
            )

    # info.json: update totals (total_episodes / total_frames).
    with open(src_meta / "info.json") as _f:
        info = json.load(_f)
    info["total_episodes"] = len(kept_old)
    info["total_frames"] = total_frames
    if stats_invalidated:
        info["stats_invalidated"] = True
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Any other file under meta/ — symlink as-is.
    for e in src_meta.iterdir():
        if e.name in {"episodes.jsonl", "episodes_stats.jsonl", "info.json",
                      "tasks.jsonl", "stats.json"}:
            continue
        dst_path = dst_meta / e.name
        if not dst_path.exists() and not dst_path.is_symlink():
            _symlink(e.resolve(), dst_path)

    return {"kept": len(kept_old), "dropped": len(drop_eps),
            "orphan_stats_skipped": n_orphan_skipped}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", type=Path,
                       help="JSON produced by `python -m data_process scan`")
    group.add_argument("--drop-episodes", type=str,
                       help="Comma-separated episode indices to drop (manual)")
    ap.add_argument("--allow-existing-dst", action="store_true")
    ap.add_argument("--clear-dst", action="store_true",
                    help="F14 fix: delete dst contents before re-emitting. Pairs "
                         "with --allow-existing-dst to avoid stale residue when "
                         "re-cleaning the same target directory.")
    ap.add_argument("--preserve-stats", action="store_true",
                    help="P0-01 escape hatch: keep the source stats.json verbatim "
                         "(legacy behavior). Default is to mark stats invalidated "
                         "after cleaning so downstream callers regenerate them.")
    ap.add_argument("--allow-incompatible", action="store_true",
                    help="D10 override: proceed even when the scan report flags "
                         "the dataset as model-incompatible (e.g. action/state "
                         "dim exceeds the model cap). Default refuses, because a "
                         "per-episode symlink-clean cannot fix a dataset-wide "
                         "dimension violation.")
    ap.add_argument("--workers", type=int, default=32,
                    help="Thread pool size for parallel symlink (default 32).")
    args = ap.parse_args()

    # 1. Parse drop set.
    if args.report:
        with open(args.report) as _f:
            report = json.load(_f)
        # A dataset flagged incompatible by a dataset-level detector
        # (action_dim_cap) cannot be salvaged by dropping episodes — the
        # violation is the whole dataset's action/state width. Refuse by default
        # so we never emit a surface-"clean" copy of an unusable dataset.
        if report.get("dataset_incompatible") and not args.allow_incompatible:
            reasons = report.get("dataset_incompatible_reasons", [])
            raise SystemExit(
                f"Scan report marks {report.get('root', args.src)} as "
                f"DATASET-INCOMPATIBLE: {reasons}. A cleanup symlink-copy "
                f"cannot fix a dataset-wide dimension violation. Re-run with "
                f"--allow-incompatible only if you understand the cleaned "
                f"dataset will still exceed the model cap."
            )
        drop_eps = {int(x) for x in report.get("bad_episodes", [])}
        orphan_stats_eps = {int(x) for x in report.get("orphan_stats_episodes", [])}
    else:
        drop_eps = {int(x) for x in args.drop_episodes.split(",") if x.strip()}
        orphan_stats_eps = set()
    logger.info("drop_eps: %d, orphan_stats: %d", len(drop_eps), len(orphan_stats_eps))

    if args.dst.exists():
        if not args.allow_existing_dst:
            raise SystemExit(f"{args.dst} exists; pass --allow-existing-dst to overwrite")
        # --allow-existing-dst needs --clear-dst when the destination is
        # non-empty, else stale files from a previous cleanup leak in. Removing
        # the managed subdirs (data/videos/images/meta) gives a clean re-emit
        # while leaving anything the operator dropped in manually alone.
        managed_subdirs = ("data", "videos", "images", "meta")
        has_managed_residue = any((args.dst / sub).exists() for sub in managed_subdirs)
        if has_managed_residue and not args.clear_dst:
            raise SystemExit(
                f"{args.dst} contains managed subdirs from a previous cleanup run "
                f"({', '.join(s for s in managed_subdirs if (args.dst / s).exists())}); "
                "pass --clear-dst to delete them before re-emitting, or pick a fresh "
                "destination directory."
            )
        if args.clear_dst:
            import shutil
            for sub in managed_subdirs:
                p = args.dst / sub
                if p.exists() or p.is_symlink():
                    if p.is_symlink() or p.is_file():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
            # Remove the stats invalidation marker so a subsequent run can
            # write a fresh one.
            inv = args.dst / "meta" / "stats.invalidated.json"
            if inv.exists():
                inv.unlink()
            logger.info("[cleanup] --clear-dst removed previous managed subdirs in %s", args.dst)
    else:
        args.dst.mkdir(parents=True)

    # 2. Build valid_eps + remap from episodes.jsonl.
    src_meta = args.src / "meta"
    valid_eps: set[int] = set()
    with open(src_meta / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep_idx = json.loads(line).get("episode_index")
            if ep_idx is not None:
                valid_eps.add(int(ep_idx))
    logger.info("episodes.jsonl: %d valid episode indices", len(valid_eps))

    kept_old, remap, total_kept = build_remap(src_meta, drop_eps)
    logger.info("remap: %d kept episodes (new range 0..%d)",
                total_kept, total_kept - 1 if total_kept else -1)

    # 3. Read chunks_size from src info.json. It defines how many episodes share
    # one `chunk-XXX/` directory (LeRobot v2.1 default: 1000); the clone reuses
    # the SAME value so renumbered episodes group as the source did. Missing →
    # fall back to 1000; present-but-invalid → fail loud.
    with open(src_meta / "info.json") as _f:
        info = json.load(_f)
    chunks_size_raw = info.get("chunks_size", 1000)
    try:
        chunks_size = int(chunks_size_raw)
    except (TypeError, ValueError):
        raise SystemExit(
            f"apply: meta/info.json has non-integer 'chunks_size' "
            f"{chunks_size_raw!r}. Expected a positive int (default 1000)."
        )
    if chunks_size <= 0:
        raise SystemExit(
            f"apply: meta/info.json 'chunks_size'={chunks_size} must be a "
            f"positive integer. LeRobot v2.1 uses 1000 by default."
        )
    logger.info("chunks_size=%d (from src info.json)", chunks_size)

    # 4. Single-pass parallel stage.
    t0 = time.monotonic()
    tree_stats = stage_and_remap(
        args.src, args.dst, drop_eps, valid_eps,
        remap, chunks_size, workers=args.workers,
    )
    dt = time.monotonic() - t0
    logger.info(
        "stage: linked=%d dropped=%d orphan=%d pruned=%d (%.1fs @ %d workers)",
        tree_stats["linked"], tree_stats["dropped_files"],
        tree_stats["orphan_files"], tree_stats["pruned_dirs"],
        dt, args.workers,
    )

    # 5. Rewrite meta.
    meta_stats = rewrite_meta(
        args.src, args.dst, kept_old, remap, drop_eps, orphan_stats_eps,
        preserve_stats=args.preserve_stats,
    )
    logger.info("meta: kept=%d dropped=%d orphan_stats_skipped=%d",
                meta_stats["kept"], meta_stats["dropped"],
                meta_stats["orphan_stats_skipped"])


if __name__ == "__main__":
    main()
