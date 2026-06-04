"""Adapter factory — dispatches by info.json codebase_version.

  v2.x → LeRobotV21Adapter   (one parquet + one mp4 per episode)
  v3.x → LeRobotV30Adapter   (many episodes per shard — slice ranges live
                              in meta/episodes/*.parquet)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_adapter(
    repo_id: str,
    root: str | Path | None = None,
    data_root: str | Path | None = None,
    delta_timestamps: dict | None = None,
    image_transforms=None,
    video_backend: str | None = None,
    external_stats: dict | None = None,
    override_schema=None,
    episode_filter: list[int] | tuple[int, ...] | None = None,
):
    """Create a LeRobotV21Adapter for the given repo_id.

    Args:
        repo_id: Dataset repository ID (e.g., "robointer_droid_clean").
        root: Direct path to dataset root. If None, uses data_root/repo_id.
        data_root: Parent directory containing repo_id subdirectory.
        delta_timestamps: Action chunk timestamps for temporal indexing.
        image_transforms: Optional image augmentation transforms.
        video_backend: Video decoding backend (default "pyav").
        external_stats: Override stats dict (for reusing pretrain stats during finetune).
        override_schema: Optional DatasetSchema to use instead of discovery.
    """
    if root is None and data_root is not None:
        candidate = Path(data_root) / repo_id
        if candidate.exists():
            root = candidate
    if root is None:
        raise ValueError(
            f"Cannot find dataset root for {repo_id!r}. "
            f"Provide --data_root or --root."
        )

    root = Path(root)
    # VQA repos carry vqa_manifest.json instead of a LeRobot info.json.
    vqa_manifest = root / "meta" / "vqa_manifest.json"
    info_path = root / "meta" / "info.json"
    if vqa_manifest.exists():
        from .vqa_adapter import VQAAdapter
        return VQAAdapter(
            repo_id=repo_id,
            root=str(root),
            override_schema=override_schema,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            external_stats=external_stats,
            video_backend=video_backend,
            episode_filter=episode_filter,
        )
    if not info_path.exists():
        raise FileNotFoundError(
            f"Neither {info_path} nor {vqa_manifest} found. "
            "create_adapter expects a LeRobot dataset (info.json) or a "
            "VQA dataset (vqa_manifest.json)."
        )
    with open(info_path) as f:
        version = json.load(f).get("codebase_version", "unknown")

    kwargs = dict(
        repo_id=repo_id,
        root=str(root),
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        external_stats=external_stats,
        override_schema=override_schema,
        video_backend=video_backend or "pyav",
        episode_filter=episode_filter,
    )
    if version.startswith("v2"):
        from .lerobot_v21 import LeRobotV21Adapter
        return LeRobotV21Adapter(**kwargs)
    if version.startswith("v3"):
        from .lerobot_v30 import LeRobotV30Adapter
        return LeRobotV30Adapter(**kwargs)
    raise ValueError(
        f"Unsupported codebase_version {version!r} at {info_path}. "
        f"Expected v2.x or v3.x."
    )
