#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from dataset.compute_stats import aggregate_stats
from dataset.lerobot_dataset import LeRobotDatasetMetadata
from dataset.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from dataset.video_utils import concatenate_video_files, get_video_duration_in_s


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """Validates that all dataset metadata have consistent properties.

    Ensures all datasets have the same fps, robot_type, and features to guarantee
    compatibility when aggregating them into a single dataset.

    Args:
        all_metadata: List of LeRobotDatasetMetadata objects to validate.

    Returns:
        tuple: A tuple containing (fps, robot_type, features) from the first metadata.

    Raises:
        ValueError: If any metadata has different fps, robot_type, or features
                   than the first metadata in the list.
    """

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    def _feature_shape(feat_dict, key):
        """Return shape tuple of a feature if present, else None."""
        if not isinstance(feat_dict, dict):
            return None
        entry = feat_dict.get(key)
        if not isinstance(entry, dict):
            return None
        shape = entry.get("shape")
        return tuple(shape) if shape is not None else None

    def _feature_names(feat_dict, key):
        """Return per-dim names tuple if the feature carries a `names` list.

        The `names` list lives in info.json `features[key]["names"]` and
        encodes the per-dim semantic label (e.g. ``["joint_0", ..., "gripper"]``).
        Two repos with the same dim count but different `names` ordering
        carry incompatible per-dim semantics — mixing them at aggregation
        time silently rewires action/state channels.
        """
        if not isinstance(feat_dict, dict):
            return None
        entry = feat_dict.get(key)
        if not isinstance(entry, dict):
            return None
        names = entry.get("names")
        if names is None:
            return None
        return tuple(names) if isinstance(names, (list, tuple)) else None

    def _video_target_set(feat_dict):
        """Return the set of video feature keys (image target slots).

        Aggregation requires every source repo to expose the same image
        slots; if one repo has `observation.images.cam_high` and another
        only has `observation.images.image0`, the merged repo's downstream
        unified-keys layer would be unable to satisfy both.
        """
        if not isinstance(feat_dict, dict):
            return frozenset()
        return frozenset(
            k for k, v in feat_dict.items()
            if isinstance(v, dict) and v.get("dtype") == "video"
        )

    # Early targeted checks: action/state dim must match across repos. The
    # generic `features != meta.features` check below catches this too, but its
    # error message is opaque; isolate the dim-mismatch case for an actionable
    # diagnostic before the full dict diff.
    ref_action_shape = _feature_shape(features, "action")
    ref_state_shape = _feature_shape(features, "observation.state")
    # Capture semantic labels to flag dim-equal-but-semantics-different
    # mismatches. Single-repo aggregation is unaffected (the cross-repo loop is
    # a no-op for one entry).
    ref_action_names = _feature_names(features, "action")
    ref_state_names = _feature_names(features, "observation.state")
    ref_video_keys = _video_target_set(features)

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        this_action_shape = _feature_shape(meta.features, "action")
        if ref_action_shape is not None and this_action_shape is not None \
                and ref_action_shape != this_action_shape:
            raise ValueError(
                f"Cross-repo action shape mismatch: reference repo has "
                f"action shape {ref_action_shape}, current repo has "
                f"{this_action_shape}. Multi-repo aggregation requires a "
                f"common action layout — partition repos by robot/action "
                f"schema before aggregating."
            )
        this_state_shape = _feature_shape(meta.features, "observation.state")
        if ref_state_shape is not None and this_state_shape is not None \
                and ref_state_shape != this_state_shape:
            raise ValueError(
                f"Cross-repo observation.state shape mismatch: reference "
                f"{ref_state_shape} vs current {this_state_shape}. Partition "
                f"repos by state schema before aggregating."
            )

        # Reject dim-equal-but-semantics-different repos. `names` ordering
        # encodes per-dim semantics (gripper position, joint labels); two repos
        # with equal action-shape but reordered joint names would otherwise pass
        # and silently teach the model contradictory per-dim conventions.
        this_action_names = _feature_names(meta.features, "action")
        if (ref_action_names is not None and this_action_names is not None
                and ref_action_names != this_action_names):
            raise ValueError(
                f"Cross-repo action names mismatch: reference repo has "
                f"action names {ref_action_names}, current repo has "
                f"{this_action_names}. Equal dim count but different "
                f"per-dim semantics — partition repos by action schema "
                f"before aggregating."
            )
        this_state_names = _feature_names(meta.features, "observation.state")
        if (ref_state_names is not None and this_state_names is not None
                and ref_state_names != this_state_names):
            raise ValueError(
                f"Cross-repo observation.state names mismatch: reference "
                f"{ref_state_names} vs current {this_state_names}. Equal "
                f"dim count but different per-dim semantics — partition "
                f"repos by state schema before aggregating."
            )
        this_video_keys = _video_target_set(meta.features)
        if this_video_keys != ref_video_keys:
            raise ValueError(
                f"Cross-repo video-feature mismatch: reference repo exposes "
                f"image slots {sorted(ref_video_keys)} vs current "
                f"{sorted(this_video_keys)}. All sub-datasets must declare "
                f"the same image_mapping target set."
            )
        # Structural equality only — compare feature dtype/shape per key.
        # Exact-equality wrongly rejected multi-repo when one repo carried an
        # extra metadata field (e.g. a joint names list) the other lacked, even
        # though the data layout matched.
        def _structurally_equal(a: dict, b: dict) -> bool:
            if set(a.keys()) != set(b.keys()):
                return False
            for k in a.keys():
                av, bv = a[k], b[k]
                if not isinstance(av, dict) or not isinstance(bv, dict):
                    if av != bv:
                        return False
                    continue
                if av.get("dtype") != bv.get("dtype"):
                    return False
                if av.get("shape") != bv.get("shape"):
                    return False
            return True

        if not _structurally_equal(features, meta.features):
            raise ValueError(
                f"Structural feature mismatch (dtype/shape/keys). "
                f"reference keys={sorted(features.keys())}, "
                f"current keys={sorted(meta.features.keys())}."
            )

    return fps, robot_type, features


def update_data_df(df, src_meta, dst_meta):
    """Updates a data DataFrame with new indices and task mappings for aggregation.

    Adjusts episode indices, frame indices, and task indices to account for
    previously aggregated data in the destination dataset.

    Args:
        df: DataFrame containing the data to be updated.
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices.
    """

    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]
    df["index"] = df["index"] + dst_meta.info["total_frames"]

    src_task_names = src_meta.tasks.index.take(df["task_index"].to_numpy())
    df["task_index"] = dst_meta.tasks.loc[src_task_names, "task_index"].to_numpy()

    return df


def update_meta_data(
    df,
    dst_meta,
    meta_idx,
    data_idx,
    videos_idx,
):
    """Updates metadata DataFrame with new chunk, file, and timestamp indices.

    Adjusts all indices and timestamps to account for previously aggregated
    data and videos in the destination dataset.

    Args:
        df: DataFrame containing the metadata to be updated.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary containing current metadata chunk and file indices.
        data_idx: Dictionary containing current data chunk and file indices.
        videos_idx: Dictionary containing current video indices and timestamps.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices and timestamps.
    """

    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]

    # Remap each episode's data shard pointer through the per-source-file map
    # built by aggregate_data, rather than a single constant offset. A constant
    # offset is only correct when the whole source repo collapses to exactly one
    # destination shard; with append/rotation across multiple data parquets it
    # points earlier episodes at the wrong shard. Mirrors the video remapping below.
    data_src_to_dst = data_idx.get("src_to_dst", {}) if isinstance(data_idx, dict) else {}
    if data_src_to_dst:
        orig_data_chunk = df["data/chunk_index"].copy()
        orig_data_file = df["data/file_index"].copy()
        for row_idx in df.index:
            src_key = (int(orig_data_chunk.at[row_idx]), int(orig_data_file.at[row_idx]))
            dst_chunk, dst_file = data_src_to_dst.get(
                src_key, (data_idx["chunk"], data_idx["file"])
            )
            df.at[row_idx, "data/chunk_index"] = dst_chunk
            df.at[row_idx, "data/file_index"] = dst_file
    else:
        # Backward-compatible fallback (e.g. callers that didn't populate the
        # map): legacy constant offset.
        df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
        df["data/file_index"] = df["data/file_index"] + data_idx["file"]
    for key, video_idx in videos_idx.items():
        # Snapshot original video file indices before remapping.
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        src_to_offset = video_idx.get("src_to_offset", {})
        src_to_dst = video_idx.get("src_to_dst", {})

        if src_to_dst:
            # Map each episode to its destination file and apply its offset.
            # int() casts avoid numpy/dict key-type mismatch on lookup.
            for idx in df.index:
                src_key = (int(df.at[idx, "_orig_chunk"]), int(df.at[idx, "_orig_file"]))
                dst_chunk, dst_file = src_to_dst.get(src_key, (video_idx["chunk"], video_idx["file"]))
                df.at[idx, orig_chunk_col] = dst_chunk
                df.at[idx, orig_file_col] = dst_file

                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        elif src_to_offset:
            # Fallback: single destination, per-file offsets only.
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            for idx in df.index:
                src_key = (int(df.at[idx, "_orig_chunk"]), int(df.at[idx, "_orig_file"]))
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        else:
            # Backward-compatible simple constant offset.
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]

        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info["total_frames"]
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info["total_frames"]
    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]

    return df


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: float | None = None,
    video_files_size_in_mb: float | None = None,
    chunk_size: int | None = None,
):
    """Aggregates multiple LeRobot datasets into a single unified dataset.

    This is the main function that orchestrates the aggregation process by:
    1. Loading and validating all source dataset metadata
    2. Creating a new destination dataset with unified tasks
    3. Aggregating videos, data, and metadata from all source datasets
    4. Finalizing the aggregated dataset with proper statistics

    Args:
        repo_ids: List of repository IDs for the datasets to aggregate.
        aggr_repo_id: Repository ID for the aggregated output dataset.
        roots: Optional list of root paths for the source datasets.
        aggr_root: Optional root path for the aggregated dataset.
        data_files_size_in_mb: Maximum size for data files in MB (defaults to DEFAULT_DATA_FILE_SIZE_IN_MB)
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)
    """
    logging.info("Start aggregate_datasets")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    fps, robot_type, features = validate_all_metadata(all_metadata)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
        use_videos=len(video_keys) > 0,
        chunks_size=chunk_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    logging.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    dst_meta.episodes = {}

    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx = aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size)
        data_idx = aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size)

        meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx)

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"] += src_meta.total_frames

    finalize_aggregation(dst_meta, all_metadata)
    logging.info("Aggregation complete.")


def aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """Aggregates video chunks from a source dataset into the destination dataset.

    Handles video file concatenation and rotation based on file size limits.
    Creates new video files when size limits are exceeded.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        videos_idx: Dictionary tracking video chunk and file indices.
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)

    Returns:
        dict: Updated videos_idx with current chunk and file indices.
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0
        # Per source (chunk, file): timestamp offset and destination (chunk, file).
        videos_idx[key]["src_to_offset"] = {}
        videos_idx[key]["src_to_dst"] = {}
        # dst_file_durations tracks the duration of each destination file.
        if "dst_file_durations" not in videos_idx[key]:
            videos_idx[key]["dst_file_durations"] = {}

    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs = {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }
        unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        dst_file_durations = video_idx["dst_file_durations"]

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            # int() casts ensure consistent dict keys.
            src_chunk_idx = int(src_chunk_idx)
            src_file_idx = int(src_file_idx)

            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            src_duration = get_video_duration_in_s(src_path)
            dst_key = (chunk_idx, file_idx)

            if not dst_path.exists():
                # New destination file: offset is 0.
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                dst_file_durations[dst_key] = src_duration
                videos_idx[key]["episode_duration"] += src_duration
                continue

            src_size = get_file_size_in_mb(src_path)
            dst_size = get_file_size_in_mb(dst_path)

            if dst_size + src_size >= video_files_size_in_mb:
                # Rotate to a new file: offset is 0.
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                dst_key = (chunk_idx, file_idx)
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                dst_file_durations[dst_key] = src_duration
            else:
                # Append to existing file: offset is its current duration.
                current_dst_duration = dst_file_durations.get(dst_key, 0)
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_dst_duration
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                )
                dst_file_durations[dst_key] = current_dst_duration + src_duration

            videos_idx[key]["episode_duration"] += src_duration

        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx

    return videos_idx


def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """Aggregates data chunks from a source dataset into the destination dataset.

    Reads source data files, updates indices to match the aggregated dataset,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        data_idx: Dictionary tracking data chunk and file indices.

    Returns:
        dict: Updated data_idx with current chunk and file indices. The dict also
            carries ``data_idx["src_to_dst"]``: a per-source-repo mapping
            ``{(src_chunk, src_file): (dst_chunk, dst_file)}`` recording where
            each source data shard landed. ``update_meta_data`` uses it to
            remap ``data/chunk_index``/``data/file_index`` per row, mirroring
            the per-source-file video mapping.
    """
    unique_chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["data/chunk_index"], src_meta.episodes["data/file_index"], strict=False
        )
    }

    unique_chunk_file_ids = sorted(unique_chunk_file_ids)

    # Build a fresh per-source-repo src->dst data-shard map. A single source repo
    # can span multiple data parquets that get appended/rotated into different
    # destination shards; a constant offset would point earlier episodes at the
    # wrong shard.
    src_to_dst = {}

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        data_idx, dst_chunk, dst_file = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )
        src_to_dst[(int(src_chunk_idx), int(src_file_idx))] = (int(dst_chunk), int(dst_file))

    data_idx["src_to_dst"] = src_to_dst
    return data_idx


def aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx):
    """Aggregates metadata from a source dataset into the destination dataset.

    Reads source metadata files, updates all indices and timestamps,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary tracking metadata chunk and file indices.
        data_idx: Dictionary tracking data chunk and file indices.
        videos_idx: Dictionary tracking video indices and timestamps.

    Returns:
        dict: Updated meta_idx with current chunk and file indices.
    """
    chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["meta/episodes/chunk_index"],
            src_meta.episodes["meta/episodes/file_index"],
            strict=False,
        )
    }

    chunk_file_ids = sorted(chunk_file_ids)
    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(src_path)
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
        )

        # The meta-parquet destination (chunk, file) is already encoded per-row
        # via the meta_idx offset applied in update_meta_data, so only the
        # updated index is needed here.
        meta_idx, _meta_dst_chunk, _meta_dst_file = append_or_create_parquet_file(
            df,
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )

    # Increment latest_duration by the total duration added from this source dataset
    for k in videos_idx:
        videos_idx[k]["latest_duration"] += videos_idx[k]["episode_duration"]

    return meta_idx


def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
):
    """Appends data to an existing parquet file or creates a new one based on size constraints.

    Manages file rotation when size limits are exceeded to prevent individual files
    from becoming too large. Handles both regular parquet files and those containing images.

    Args:
        df: DataFrame to write to the parquet file.
        src_path: Path to the source file (used for size estimation).
        idx: Dictionary containing current 'chunk' and 'file' indices.
        max_mb: Maximum allowed file size in MB before rotation.
        chunk_size: Maximum number of files per chunk before incrementing chunk index.
        default_path: Format string for generating file paths.
        contains_images: Whether the data contains images requiring special handling.
        aggr_root: Root path for the aggregated dataset.

    Returns:
        tuple[dict, int, int]: ``(idx, dst_chunk, dst_file)`` where ``idx`` is the
            updated index dictionary and ``(dst_chunk, dst_file)`` is the
            destination chunk/file that *this* ``df`` was actually written to.
            The destination differs from the pre-call ``idx`` only on rotation,
            so callers must use the returned destination (not a constant offset)
            when remapping per-row metadata.
    """
    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx, idx["chunk"], idx["file"]

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx, idx["chunk"], idx["file"]


def finalize_aggregation(aggr_meta, all_metadata):
    """Finalizes the dataset aggregation by writing summary files and statistics.

    Writes the tasks file, info file with total counts and splits, and
    aggregated statistics from all source datasets.

    Args:
        aggr_meta: Aggregated dataset metadata.
        all_metadata: List of all source dataset metadata objects.
    """
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    # aggregate_stats() drops q01/q99: cross-dataset quantile aggregation via
    # weighted-mean is mathematically wrong. Mean/std/min/max remain correct
    # (Welford-style closed-form). Re-run data_process.stats.compute over the
    # aggregated root to recover exact q01/q99 via RunningStats histograms.
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)