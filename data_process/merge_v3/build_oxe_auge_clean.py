"""Merge 180 oxe-auge v3.0 sub-repos into one clean repo via symlink + meta rewrite.

Product: /all-flash-data/Pretrain_Data/oxe-auge_clean/ (v3.0 LeRobot layout).

Design principle: data and video shards are symlinked (not copied) to conserve
storage. Only meta/episodes/*.parquet are physically rewritten, with chunk/file
indices remapped to globally unique values per (data, per-camera-video, meta-
episodes) namespace. `dataset_from_index` and `dataset_to_index` are PRESERVED
exactly — they are shard-internal row offsets, not global frame indices.

Heterogeneity handled:
  - joints_dim: 18/180 sub-repos are 7-dim, rest are 8-dim. Merged info.json
    declares 8; v30 adapter right-pads 7-dim reads at load (Phase 4).
  - fps: 7 distinct values across sub-repos. Merged info.json declares a single
    nominal fps (default 10, the majority). Timestamps in data/video stay
    native; downstream delta_timestamps computation uses nominal fps — this is
    a deliberate resampling for non-matching sub-repos (dominant-fps strategy).
  - camera keys: 7 distinct combinations. All include `observation.images.image`
    (schema key). Additional per-sub-repo cameras are available but unused
    unless schema declares them.

Usage:
    python -m data_process.merge_v3.build_oxe_auge_clean \\
        --src /all-flash-data/Pretrain_Data/oxe-auge \\
        --dst /all-flash-data/Pretrain_Data/oxe-auge_clean \\
        --nominal-fps 10 --workers 32
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# Declarative merge rules: the schema module owns the source → canonical layout
# mapping via a MergeRecipe. See schemas/oxe_auge.py (MERGE_RECIPE).
from schemas.oxe_auge import MERGE_RECIPE as _MERGE_RECIPE

# Per-subset gripper value canonicalization. See
# data_process/merge_v3/gripper_canonicalize.py for the rule registry.
from data_process.merge_v3.gripper_canonicalize import (
    EXCLUDED_SUBSETS as _GRIPPER_EXCLUDED,
    INCLUDED_SUBSETS as _GRIPPER_INCLUDED,
    canonicalize_gripper as _canonicalize_gripper,
    get_base_subset_name as _get_base_subset_name,
)


def _is_six_dof_source(sub_name: str) -> bool:
    """Back-compat wrapper around `MERGE_RECIPE.needs_rewrite` used by
    existing code paths below. Any source whose raw layout differs from
    the canonical (7-DoF arm + grip at dim 7) must be physically rewritten."""
    return _MERGE_RECIPE.needs_rewrite(sub_name)


def _rewrite_7dim_shard_to_8dim(
    src: Path,
    dst: Path,
    sub_name: str,
    apply_gripper_canonicalize: bool = True,
) -> None:
    """Physically rewrite a parquet shard with two layered transforms:

      1. **Layout promotion** (existing): promote each row's
         `observation.joints` to the canonical layout declared by
         MERGE_RECIPE — for 6-DoF sources this places arm joints in
         dims [0 .. arm_dof-1], zero-pads remaining arm slots, and
         puts gripper at canonical `gripper_index_in_raw`. For already-
         canonical 8-DoF rows this is a no-op pass-through.

      2. **Gripper value canonicalize** (new — oxe-auge_clean_v2):
         apply per-subset rule from
         `data_process/merge_v3/gripper_canonicalize.py` to dim 7
         (gripper) so all included subsets converge to the
         "0=close, 1=open ∈ [0, 1]" convention.

    Args:
        src:      source parquet path (resolved absolute).
        dst:      destination parquet path (will be created with parents).
        sub_name: sub-repo directory name — used both as MERGE_RECIPE key
                  and as the gripper_canonicalize rule lookup key.
        apply_gripper_canonicalize:
                  Pass False to disable Step 2 entirely (legacy
                  oxe-auge_clean v1 behavior). Default True for v2.

    Raises:
        ValueError: if `sub_name` resolves to an EXCLUDED subset (callers
                    must filter excluded subsets at the scan stage so they
                    never reach this rewriter — see scan_all_sub_repos).
    """
    import numpy as _np
    table = pq.read_table(src)
    df = table.to_pandas()
    if "observation.joints" in df.columns:
        canonical_total = _MERGE_RECIPE.canonical.arm_dof + 1

        def _promote(row):
            # `np.array(...)` (not asarray) ensures a writable copy — pandas
            # often hands us a read-only view, which would crash the in-place
            # gripper update at Step 2 below.
            arr = _np.array(row, dtype=_np.float32, copy=True)
            # Step 1: layout promotion (idempotent for already-canonical rows)
            if arr.shape[-1] == canonical_total:
                pass
            else:
                arr = _MERGE_RECIPE.transform_row(arr, sub_name)
                if arr.shape[-1] != canonical_total:
                    # Fail loud: a misconfigured MergeRecipe (wrong arm_dof or
                    # source layout) would otherwise emit wrong-width rows that
                    # adapters read as if-canonical, garbling gripper semantics
                    # across the whole merged sub-repo.
                    raise ValueError(
                        f"MergeRecipe.transform_row({sub_name!r}) produced "
                        f"shape {arr.shape}, expected (..., {canonical_total})."
                        f" Check MERGE_RECIPE.source_layouts / default_layout."
                    )
                arr = _np.asarray(arr, dtype=_np.float32)
            # Step 2: gripper value canonicalize on dim 7 (canonical gripper).
            if apply_gripper_canonicalize:
                grip_dim = _MERGE_RECIPE.canonical.gripper_index_in_raw
                canon_g = _canonicalize_gripper(arr[grip_dim:grip_dim + 1], sub_name)
                arr[grip_dim] = float(canon_g[0])
            return arr

        df["observation.joints"] = df["observation.joints"].map(_promote)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, engine="pyarrow", compression="snappy")


_V21_EP_RE = __import__("re").compile(r"episode_(\d+)\.")


def _scan_sub_repo_v3(sub: Path, info: dict) -> dict:
    """Gather v3.0 sub-repo shard inventory."""
    data_files: list[tuple[int, int, Path]] = []
    data_chunks: set[int] = set()
    for shard in (sub / "data").rglob("file-*.parquet"):
        ci = int(shard.parent.name.split("-")[1])
        fi = int(shard.stem.split("-")[1])
        data_files.append((ci, fi, shard))
        data_chunks.add(ci)

    video_files_per_cam: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
    video_chunks_per_cam: dict[str, set[int]] = defaultdict(set)
    vid_root = sub / "videos"
    if vid_root.is_dir():
        for cam_dir in vid_root.iterdir():
            if not cam_dir.is_dir():
                continue
            cam = cam_dir.name
            for shard in cam_dir.rglob("file-*.mp4"):
                ci = int(shard.parent.name.split("-")[1])
                fi = int(shard.stem.split("-")[1])
                video_files_per_cam[cam].append((ci, fi, shard))
                video_chunks_per_cam[cam].add(ci)

    meta_ep_files: list[tuple[int, int, Path]] = []
    meta_ep_chunks: set[int] = set()
    ep_root = sub / "meta/episodes"
    if ep_root.is_dir():
        for shard in ep_root.rglob("file-*.parquet"):
            ci = int(shard.parent.name.split("-")[1])
            fi = int(shard.stem.split("-")[1])
            meta_ep_files.append((ci, fi, shard))
            meta_ep_chunks.add(ci)

    return {
        "format": "v3",
        "n_data_chunks": (max(data_chunks) + 1) if data_chunks else 0,
        "n_meta_ep_chunks": (max(meta_ep_chunks) + 1) if meta_ep_chunks else 0,
        "n_video_chunks_per_cam": {
            cam: (max(chs) + 1) if chs else 0
            for cam, chs in video_chunks_per_cam.items()
        },
        "data_files": data_files,
        "video_files_per_cam": dict(video_files_per_cam),
        "meta_ep_files": meta_ep_files,
    }


def _scan_sub_repo_v21(sub: Path, info: dict) -> dict:
    """Gather v2.1 sub-repo inventory and compute v2→v3 remap plan.

    v2.1 layout: data/chunk-XXX/episode_NNNNNN.parquet (one ep per file),
    videos/chunk-XXX/<cam>/episode_NNNNNN.mp4, meta/episodes.jsonl.

    In the merged v3 repo, each v2.1 episode becomes ONE v3 shard (1 ep / shard).
    - data shard symlinked from original per-ep parquet to oxe-auge_clean/
      data/chunk-GC/file-GF.parquet
    - video shards symlinked with cam-before-chunk path reordering
    - meta/episodes gets ONE row per ep, all rows packed into a single
      synthesized meta parquet for the sub-repo
    """
    chunks_size = int(info.get("chunks_size", 1000))
    fps = float(info.get("fps", 10.0))

    # Read episodes.jsonl → {ep_idx: (length, tasks)}.
    eps_meta: list[dict] = []
    with open(sub / "meta/episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            ep_idx = int(ep["episode_index"])
            length = int(ep.get("length", 0))
            tasks = ep.get("tasks") or []
            if not isinstance(tasks, list):
                tasks = [tasks]
            eps_meta.append({
                "ep_idx": ep_idx,
                "length": length,
                "tasks": [str(t) for t in tasks if t is not None],
            })
    eps_meta.sort(key=lambda e: e["ep_idx"])

    # Index per-episode data parquet + videos by episode index.
    data_by_ep: dict[int, Path] = {}
    for p in (sub / "data").rglob("*.parquet"):
        m = _V21_EP_RE.search(p.name)
        if m:
            data_by_ep[int(m.group(1))] = p

    # v2.1 video path: videos/chunk-XXX/<cam>/episode_NNN.mp4
    videos_by_ep: dict[str, dict[int, Path]] = defaultdict(dict)
    vid_root = sub / "videos"
    if vid_root.is_dir():
        for chunk_dir in vid_root.iterdir():
            if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
                continue
            for cam_dir in chunk_dir.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam = cam_dir.name
                for mp4 in cam_dir.glob("*.mp4"):
                    m = _V21_EP_RE.search(mp4.name)
                    if m:
                        videos_by_ep[cam][int(m.group(1))] = mp4

    # Build conversion plan: each ep gets a slot (within_sub_idx i).
    # Slot i → (chunk_i, file_i) = (i // chunks_size, i % chunks_size).
    # Total data chunks for this sub = ceil(len(eps_meta) / chunks_size).
    # Same layout for each camera's videos.
    plan_eps: list[dict] = []
    for i, ep in enumerate(eps_meta):
        ep_idx = ep["ep_idx"]
        chunk_i = i // chunks_size
        file_i = i % chunks_size
        entry = {
            "within_sub_idx": i,
            "ep_idx_orig": ep_idx,
            "length": ep["length"],
            "tasks": ep["tasks"],
            "chunk_i": chunk_i,
            "file_i": file_i,
            "data_path": data_by_ep.get(ep_idx),
            "videos": {cam: v.get(ep_idx) for cam, v in videos_by_ep.items()},
        }
        plan_eps.append(entry)

    n_data_chunks = (len(eps_meta) + chunks_size - 1) // chunks_size if eps_meta else 0
    n_meta_ep_chunks = 1  # one synthesized meta-ep parquet per sub-repo
    n_video_chunks_per_cam = {cam: n_data_chunks for cam in videos_by_ep.keys()}

    return {
        "format": "v21",
        "fps": fps,
        "chunks_size": chunks_size,
        "plan_eps": plan_eps,
        "n_data_chunks": n_data_chunks,
        "n_meta_ep_chunks": n_meta_ep_chunks,
        "n_video_chunks_per_cam": n_video_chunks_per_cam,
        # For uniformity with v3 scan output; empty lists since v21 has no
        # pre-existing shards to symlink in the v3 walker.
        "data_files": [],
        "video_files_per_cam": {},
        "meta_ep_files": [],
    }


def _scan_sub_repo(sub: Path) -> dict:
    """Gather shard inventory + info.json for one sub-repo (v2.1 or v3.0)."""
    info = json.loads((sub / "meta/info.json").read_text())
    version = str(info.get("codebase_version", ""))
    if version.startswith("v2"):
        specific = _scan_sub_repo_v21(sub, info)
    elif version.startswith("v3"):
        specific = _scan_sub_repo_v3(sub, info)
    else:
        raise ValueError(f"{sub}: unsupported codebase_version={version!r}")

    joints_shape = info.get("features", {}).get("observation.joints", {}).get("shape", [None])
    common = {
        "sub_name": sub.name,
        "sub_path": sub,
        "info": info,
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
        "fps": info.get("fps"),
        "joints_dim": joints_shape[0] if joints_shape else None,
    }
    common.update(specific)
    return common


def scan_all_sub_repos(
    src_root: Path,
    workers: int = 8,
    whitelist_only: bool = True,
) -> list[dict]:
    """Scan oxe-auge sub-repos and (optionally) filter by gripper audit whitelist.

    Args:
        src_root:        oxe-auge directory containing per-source sub-repos.
        workers:         parallel scan workers.
        whitelist_only:  if True (default for oxe-auge_clean_v2), drop
                         sub-repos whose base subset name is in
                         `gripper_canonicalize.EXCLUDED_SUBSETS` or is unknown.
                         Pass False to keep legacy v1 behavior (scan all).

    Returns:
        List of report dicts in alphabetical sub-repo name order, ONLY for
        sub-repos that pass the whitelist filter.
    """
    all_subs = sorted([p for p in src_root.iterdir()
                       if p.is_dir() and (p / "meta/info.json").exists()])

    if whitelist_only:
        kept: list[Path] = []
        excluded: list[tuple[Path, str]] = []
        unknown: list[Path] = []
        for p in all_subs:
            base = _get_base_subset_name(p.name)
            if base is None:
                unknown.append(p)
            elif base in _GRIPPER_EXCLUDED:
                excluded.append((p, base))
            elif base in _GRIPPER_INCLUDED:
                kept.append(p)
            else:
                # Defensive: shouldn't happen if get_base_subset_name returns
                # only INCLUDED ∪ EXCLUDED, but guard anyway.
                unknown.append(p)
        if excluded:
            logger.warning(
                f"Whitelist: SKIPPING {len(excluded)} sub-repos in "
                f"EXCLUDED_SUBSETS list: "
                f"{sorted({base for _, base in excluded})}"
            )
        if unknown:
            logger.warning(
                f"Whitelist: SKIPPING {len(unknown)} sub-repos with UNKNOWN "
                f"base name (not in INCLUDED or EXCLUDED audit lists): "
                f"{[p.name for p in unknown[:5]]}"
                f"{'...' if len(unknown) > 5 else ''}"
            )
        logger.info(
            f"Whitelist: keeping {len(kept)}/{len(all_subs)} sub-repos "
            f"(in INCLUDED_SUBSETS list)."
        )
        subs = kept
    else:
        subs = all_subs
        logger.info(f"Whitelist disabled — scanning all {len(subs)} sub-repos.")

    if not subs:
        raise RuntimeError(
            f"After whitelist filter, no sub-repos remain in {src_root}. "
            f"Check that --src points at oxe-auge and that the canonicalize "
            f"INCLUDED_SUBSETS list contains the expected base names."
        )

    logger.info(f"Scanning {len(subs)} sub-repos (parallel workers={workers})...")
    out = [None] * len(subs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_scan_sub_repo, sub): i for i, sub in enumerate(subs)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            done += 1
            if done % 20 == 0:
                logger.info(f"  scanned {done}/{len(subs)}")
    return out


def compute_offsets(reports: list[dict]) -> None:
    """In-place: each sub-repo gets cumulative offsets for (ep, data_chunk,
    meta_ep_chunk, per-camera video_chunk)."""
    ep_off = 0
    data_off = 0
    meta_ep_off = 0
    video_offs: dict[str, int] = defaultdict(int)
    for r in reports:
        r["ep_offset"] = ep_off
        r["data_chunk_offset"] = data_off
        r["meta_ep_chunk_offset"] = meta_ep_off
        r["video_chunk_offsets"] = dict(video_offs)  # snapshot of running offsets
        ep_off += r["total_episodes"]
        data_off += r["n_data_chunks"]
        meta_ep_off += r["n_meta_ep_chunks"]
        for cam, n in r["n_video_chunks_per_cam"].items():
            video_offs[cam] += n

    for r in reports:
        fmt = r.get("format", "v3")
        if fmt != "v21":
            continue
        d_off = r["data_chunk_offset"]
        for cam, v_off in r["video_chunk_offsets"].items():
            if d_off != v_off:
                logger.warning(
                    "v2.1 sub '%s': data_chunk_offset (%d) != "
                    "video_chunk_offset[%s] (%d) — "
                    "data/video shard numbering diverged",
                    r["sub_name"], d_off, cam, v_off,
                )


def _fast_symlink(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src)
    except FileExistsError:
        dst.unlink()
        dst.symlink_to(src)


def symlink_shards(
    reports: list[dict],
    dst_root: Path,
    workers: int = 32,
    apply_gripper_canonicalize: bool = True,
) -> int:
    """Symlink (or physically rewrite) every data + video shard into dst_root.

    For v3 sub-repos: preserve (chunk, file) and offset chunk_index globally.
    For v2.1 sub-repos: each ep becomes a 1-ep v3 shard; (chunk, file) assigned
    from the plan_eps list (chunk_i = i // chunks_size, file_i = i % chunks_size).

    **Data shard routing**:
      - When `apply_gripper_canonicalize=True` (oxe-auge_clean_v2 default):
        ALL data shards go through `_rewrite_7dim_shard_to_8dim` so that
        every included subset's gripper dim 7 is canonicalized to the
        unified [0, 1] / 1=open convention. Layout-promotion path stays
        active for 6-DoF sources (berkeley_autolab_ur5, bridge, jaco_play).
      - When `apply_gripper_canonicalize=False` (legacy v1 behavior):
        only 6-DoF sources are rewritten (layout only); other sources are
        symlinked.

    Video shards are always symlinked (no value transform needed).
    """
    sym_jobs: list[tuple[Path, Path]] = []                        # (src, dst)
    rewrite_jobs: list[tuple[Path, Path, str]] = []               # (src, dst, sub_name)

    # Decide per-shard routing: when canonicalize is on, every data shard
    # of every included sub-repo must be rewritten so we can apply the
    # per-subset gripper transform (even if MergeRecipe says no layout
    # rewrite is needed). Video shards never need value rewriting.
    def _data_needs_rewrite(sub_name: str) -> bool:
        if apply_gripper_canonicalize:
            return True
        return _MERGE_RECIPE.needs_rewrite(sub_name)

    for r in reports:
        fmt = r.get("format", "v3")
        sub_name = r["sub_name"]
        needs_rewrite = _data_needs_rewrite(sub_name)
        if fmt == "v3":
            for ci, fi, path in r["data_files"]:
                new_ci = ci + r["data_chunk_offset"]
                dst = dst_root / "data" / f"chunk-{new_ci:03d}" / f"file-{fi:03d}.parquet"
                if needs_rewrite:
                    rewrite_jobs.append((path.resolve(), dst, sub_name))
                else:
                    sym_jobs.append((path.resolve(), dst))
            for cam, files in r["video_files_per_cam"].items():
                off = r["video_chunk_offsets"].get(cam, 0)
                for ci, fi, path in files:
                    new_ci = ci + off
                    dst = dst_root / "videos" / cam / f"chunk-{new_ci:03d}" / f"file-{fi:03d}.mp4"
                    sym_jobs.append((path.resolve(), dst))
        elif fmt == "v21":
            for ep in r["plan_eps"]:
                new_ci = ep["chunk_i"] + r["data_chunk_offset"]
                fi = ep["file_i"]
                if ep["data_path"] is not None:
                    dst = dst_root / "data" / f"chunk-{new_ci:03d}" / f"file-{fi:03d}.parquet"
                    if needs_rewrite:
                        rewrite_jobs.append((ep["data_path"].resolve(), dst, sub_name))
                    else:
                        sym_jobs.append((ep["data_path"].resolve(), dst))
                for cam, vpath in ep["videos"].items():
                    if vpath is None:
                        continue
                    vi = ep["chunk_i"] + r["video_chunk_offsets"].get(cam, 0)
                    dst = dst_root / "videos" / cam / f"chunk-{vi:03d}" / f"file-{fi:03d}.mp4"
                    sym_jobs.append((vpath.resolve(), dst))

    # Pre-create parent dirs (serial, cheap).
    parents = {dst.parent for _, dst in sym_jobs}
    parents.update(dst.parent for _, dst, _ in rewrite_jobs)
    for p in parents:
        p.mkdir(parents=True, exist_ok=True)

    # Fast path: parallel symlink.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_fast_symlink, s, d) for s, d in sym_jobs]
        for fut in futs:
            fut.result()

    # Slow path: physical rewrite (layout promotion + gripper canonicalize).
    # When canonicalize is on, this covers all data shards; otherwise only
    # the 6-DoF sources flagged by MergeRecipe.
    if rewrite_jobs:
        rewritten_sources = sorted({sn for _, _, sn in rewrite_jobs})
        logger.info(
            f"Rewriting {len(rewrite_jobs)} data shards "
            f"(apply_gripper_canonicalize={apply_gripper_canonicalize}, "
            f"sources: {rewritten_sources})..."
        )
        with ThreadPoolExecutor(max_workers=min(workers, 16)) as pool:
            futs = [
                pool.submit(
                    _rewrite_7dim_shard_to_8dim,
                    s, d, sn,
                    apply_gripper_canonicalize,
                )
                for s, d, sn in rewrite_jobs
            ]
            for i, fut in enumerate(as_completed(futs)):
                fut.result()
                if (i + 1) % 100 == 0:
                    logger.info(f"  rewrote {i+1}/{len(rewrite_jobs)} shards")

    return len(sym_jobs) + len(rewrite_jobs)


def _rewrite_one_meta_ep(
    r: dict, ci: int, fi: int, path: Path, dst_root: Path,
) -> Path:
    """Rewrite ONE meta/episodes parquet: remap ep idx + data/video chunk idx."""
    table = pq.read_table(path)
    df = table.to_pandas()
    new_ci = ci + r["meta_ep_chunk_offset"]

    # Rewrite columns.
    df["episode_index"] = df["episode_index"].astype("int64") + r["ep_offset"]
    df["data/chunk_index"] = (
        df["data/chunk_index"].astype("int64") + r["data_chunk_offset"]
    )
    # data file_index is preserved as-is.
    for col in list(df.columns):
        if col.startswith("videos/") and col.endswith("/chunk_index"):
            cam = col[len("videos/"):-len("/chunk_index")]
            off = r["video_chunk_offsets"].get(cam, 0)
            df[col] = df[col].astype("int64") + off
    # Self-reference to the meta/episodes shard location (present in some v3 repos).
    if "meta/episodes/chunk_index" in df.columns:
        df["meta/episodes/chunk_index"] = new_ci
    if "meta/episodes/file_index" in df.columns:
        df["meta/episodes/file_index"] = fi

    dst = dst_root / "meta/episodes" / f"chunk-{new_ci:03d}" / f"file-{fi:03d}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, engine="pyarrow", compression="snappy")
    return dst


def _synthesize_v21_meta_ep(r: dict, dst_root: Path) -> int:
    """Build a v3-shaped meta/episodes parquet from v2.1 episodes.jsonl.

    One synthesized parquet per v2.1 sub-repo, at meta/episodes/chunk-MEC/file-0.parquet
    where MEC = meta_ep_chunk_offset (since n_meta_ep_chunks=1 for v2 sub-repos).
    Each row maps one v2.1 episode to its converted v3 shard (chunk_i, file_i),
    with dataset_from_index=0, dataset_to_index=length (full parquet = one ep),
    and per-camera video chunk/file pointing at the same (chunk_i, file_i) slot.
    """
    rows: list[dict] = []
    fps = r["fps"]
    ep_off = r["ep_offset"]
    data_off = r["data_chunk_offset"]
    video_offs = r["video_chunk_offsets"]

    # Collect camera keys that this sub-repo actually has videos for.
    cam_keys = sorted({cam for ep in r["plan_eps"] for cam in ep["videos"].keys()})

    for ep in r["plan_eps"]:
        new_ep_idx = ep_off + ep["within_sub_idx"]
        gc_data = ep["chunk_i"] + data_off
        fi = ep["file_i"]
        length = ep["length"]
        row = {
            "episode_index": int(new_ep_idx),
            "data/chunk_index": int(gc_data),
            "data/file_index": int(fi),
            "dataset_from_index": 0,
            "dataset_to_index": int(length),
            "tasks": list(ep["tasks"]) if ep["tasks"] else [""],
            "length": int(length),
        }
        for cam in cam_keys:
            gc_vid = ep["chunk_i"] + video_offs.get(cam, 0)
            row[f"videos/{cam}/chunk_index"] = int(gc_vid)
            row[f"videos/{cam}/file_index"] = int(fi)
            row[f"videos/{cam}/from_timestamp"] = 0.0
            row[f"videos/{cam}/to_timestamp"] = float(length) / fps if fps else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    mec = r["meta_ep_chunk_offset"]
    dst = dst_root / "meta/episodes" / f"chunk-{mec:03d}" / f"file-000.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, engine="pyarrow", compression="snappy")
    return len(rows)


def rewrite_meta_episodes(reports: list[dict], dst_root: Path, workers: int = 16) -> int:
    v3_jobs: list[tuple[dict, int, int, Path]] = []
    v21_reports: list[dict] = []
    for r in reports:
        if r.get("format", "v3") == "v3":
            for ci, fi, path in r["meta_ep_files"]:
                v3_jobs.append((r, ci, fi, path))
        else:
            v21_reports.append(r)

    # v3: parallel rewrite.
    n_total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_rewrite_one_meta_ep, r, ci, fi, path, dst_root)
                for (r, ci, fi, path) in v3_jobs]
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0:
                logger.info(f"  rewrote {done}/{len(v3_jobs)} v3 meta/episodes parquets")
    n_total += len(v3_jobs)

    # v21: serial synthesis (only 11 sub-repos, trivial).
    n_rows_v21 = 0
    for r in v21_reports:
        n_rows_v21 += _synthesize_v21_meta_ep(r, dst_root)
    logger.info(f"  synthesized {len(v21_reports)} v21 meta/episodes parquets ({n_rows_v21} rows total)")
    n_total += len(v21_reports)
    return n_total


def merge_tasks(reports: list[dict], dst_root: Path) -> int:
    """Merge all sub-repos' tasks.parquet into one with globally-unique task_index.

    Handles two schema variants (v21 discovery.py comments):
      (a) columns = {"task_index": int64, "task": string}
      (b) columns = {"task_index": int64, "__index_level_0__": string (= task)}
    """
    all_tasks: set[str] = set()
    for r in reports:
        tk_parquet = r["sub_path"] / "meta/tasks.parquet"
        tk_jsonl = r["sub_path"] / "meta/tasks.jsonl"
        if tk_parquet.exists():
            df = pq.read_table(tk_parquet).to_pandas()
            if "task" in df.columns:
                all_tasks.update(str(t) for t in df["task"].tolist())
            elif "__index_level_0__" in df.columns:
                all_tasks.update(str(t) for t in df["__index_level_0__"].tolist())
            else:
                all_tasks.update(str(idx) for idx in df.index)
        elif tk_jsonl.exists():
            # v2.1 sub-repos ship tasks.jsonl instead of parquet.
            with open(tk_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    task = json.loads(line).get("task")
                    if task:
                        all_tasks.add(str(task))

    tasks_sorted = sorted(all_tasks)
    out_df = pd.DataFrame({
        "task_index": list(range(len(tasks_sorted))),
        "task": tasks_sorted,
    })
    out_df.to_parquet(dst_root / "meta/tasks.parquet", engine="pyarrow")
    return len(tasks_sorted)


def write_info_json(
    reports: list[dict],
    dst_root: Path,
    nominal_fps: float,
    chunks_size: int = 1000,
) -> dict:
    """Union features across sub-repos; override observation.joints to 8-dim."""
    total_eps = sum(r["total_episodes"] for r in reports)
    total_frames = sum(r["total_frames"] for r in reports)

    all_feats: dict[str, dict] = {}
    for r in reports:
        feats = r["info"].get("features", {}) or {}
        for k, v in feats.items():
            if k not in all_feats:
                all_feats[k] = v

    # Force unified joints dim (runtime pad handles 7-dim sources).
    if "observation.joints" in all_feats:
        all_feats["observation.joints"] = {
            **all_feats["observation.joints"],
            "shape": [8],
            "fps": nominal_fps,
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "oxe_auge_merged",
        "fps": nominal_fps,
        "total_episodes": total_eps,
        "total_frames": total_frames,
        "total_tasks": len(reports),  # rough; actual tasks set in merge_tasks
        "chunks_size": chunks_size,
        "data_files_size_in_mb": 500,
        "features": all_feats,
    }
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta/info.json").write_text(json.dumps(info, indent=2))
    return info


def write_labvla_manifest(dst_root: Path) -> None:
    """Tier-1 schema discovery manifest (see src/schema/manifest.py).

    Must match the oxe_auge DatasetSchema exactly; the merge declares 8-dim
    joints for both state and action (joints-as-action BC on OXE-AugE),
    all 8 dims delta, no absolute-gripper carve-out.
    """
    manifest = {
        "version": 1,
        "schema_id": "oxe_auge_v1",
        "robot_type": "oxe_auge_merged",
        # state_keys distinct from action_keys to avoid the delta_timestamps
        # write-back collision — see schemas/oxe_auge.py docstring.
        "state": {"keys": ["observation.state"], "dims": [8]},
        "action": {
            "keys": ["observation.joints"],
            "dims": [8],
            # arm (dim 0..6) delta, gripper (dim 7) absolute — matches
            # robointer_droid schema for a consistent pretraining mix.
            "delta": [True, True, True, True, True, True, True, False],
            "gripper_dims": [7],
        },
        "images": {"observation.images.image": "image0"},
        "allow_extra_cameras": True,
        # arm_layout: after merge, all 6-DoF sources had 0-pad inserted at dim 6,
        # so every sample's gripper lands at canonical dim 7 (7-DoF single-arm
        # layout). Pretrain/finetune/deploy all use this.
        "arm_layout": {
            "arm_count": "single",
            "arm_dof": 7,
            "gripper_index_in_raw": 7,
        },
    }
    (dst_root / "meta/labvla_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser(
        prog="data_process.merge_v3.build_oxe_auge_clean",
        description="Symlink-merge oxe-auge sub-repos into one v3.0 clean repo.",
    )
    ap.add_argument("--src", required=True, type=Path,
                    help="Parent dir containing sub-repos (each with meta/info.json).")
    ap.add_argument("--dst", required=True, type=Path, help="Merged clean repo output.")
    ap.add_argument("--nominal-fps", type=float, default=10.0,
                    help="Declared fps in merged info.json. Should match majority.")
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--scan-workers", type=int, default=8)
    ap.add_argument("--allow-existing-dst", action="store_true")
    ap.add_argument(
        "--apply-gripper-canonicalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-subset gripper canonicalize (oxe-auge_clean_v2 "
             "default). Use --no-apply-gripper-canonicalize for legacy v1.",
    )
    ap.add_argument(
        "--whitelist-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only merge sub-repos in INCLUDED_SUBSETS list (audit "
             "whitelist). Default True for v2; pass --no-whitelist-only "
             "for legacy v1.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[merge %(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dst.exists():
        if not args.allow_existing_dst:
            logger.error(f"{args.dst} exists; pass --allow-existing-dst to overwrite")
            sys.exit(1)
    else:
        args.dst.mkdir(parents=True)

    t0 = time.monotonic()
    # Phase 1: scan (with optional whitelist filter).
    reports = scan_all_sub_repos(
        args.src,
        workers=args.scan_workers,
        whitelist_only=args.whitelist_only,
    )
    logger.info(
        f"Pipeline flags: whitelist_only={args.whitelist_only}, "
        f"apply_gripper_canonicalize={args.apply_gripper_canonicalize}"
    )
    total_eps = sum(r["total_episodes"] for r in reports)
    total_frames = sum(r["total_frames"] for r in reports)
    fps_distrib = Counter(r["fps"] for r in reports)
    joints_distrib = Counter(r["joints_dim"] for r in reports)
    logger.info(
        f"Scan done: {len(reports)} sub-repos, {total_eps:,} eps, "
        f"{total_frames:,} frames ({time.monotonic()-t0:.1f}s)"
    )
    logger.info(f"fps distribution: {dict(fps_distrib)}")
    logger.info(f"joints_dim distribution: {dict(joints_distrib)}")
    if len(fps_distrib) > 1:
        logger.warning(
            f"MIXED FPS — {len(fps_distrib)} distinct values. "
            f"Merged info.json declares fps={args.nominal_fps}; adapter reads this. "
            f"Non-matching sub-repos effectively get temporal resampling."
        )

    # Phase 2: compute offsets.
    compute_offsets(reports)

    # Phase 3: symlink (or rewrite, when canonicalize is on) data + video shards.
    t = time.monotonic()
    n = symlink_shards(
        reports,
        args.dst,
        workers=args.workers,
        apply_gripper_canonicalize=args.apply_gripper_canonicalize,
    )
    logger.info(f"Linked/rewrote {n} shards ({time.monotonic()-t:.1f}s)")

    # Phase 4: rewrite meta/episodes parquets.
    t = time.monotonic()
    n = rewrite_meta_episodes(reports, args.dst, workers=args.workers)
    logger.info(f"Rewrote {n} meta/episodes parquets ({time.monotonic()-t:.1f}s)")

    # Phase 5: merge tasks.
    t = time.monotonic()
    n_tasks = merge_tasks(reports, args.dst)
    logger.info(f"Merged {n_tasks} unique tasks ({time.monotonic()-t:.1f}s)")

    # Phase 6: write info.json + labvla_manifest.json.
    info = write_info_json(
        reports, args.dst,
        nominal_fps=args.nominal_fps,
        chunks_size=args.chunks_size,
    )
    write_labvla_manifest(args.dst)
    logger.info(
        f"Merged info.json: total_episodes={info['total_episodes']:,}, "
        f"total_frames={info['total_frames']:,}, fps={info['fps']}, "
        f"n_features={len(info['features'])}"
    )

    logger.info(f"DONE — {args.dst} ({time.monotonic()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
