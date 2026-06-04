"""Thin adapter wrapper around `RoboInterVQADataset` so it can be plumbed
through the existing `build_dataset` → `TransformedAdapterDataset` pipeline.

VQA repos do not have a LeRobot info.json. Detection in `create_adapter`:
the repo root must contain a `meta/vqa_manifest.json` file describing which
JSON files to load and which task family they belong to.

`vqa_manifest.json` schema (minimal):
{
  "task_family": "understanding" | "generation_short" | "generation_traj"
                 | "task_planning",
  "image_root_subdir": "Understanding/image/train/droid",  // relative to repo root
  "json_files": [
    "Understanding/meta/train/droid/contact_decide.json",
    "Understanding/meta/train/droid/grounding_choice.json"
  ]
}

The image_root_subdir resolves relative paths in the JSON `images` field
(LLaVA-style path). Image paths inside JSON typically include the same prefix
already, so `image_root_subdir` is usually the repo root itself ("").
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseAdapter


@dataclass
class _VQAAdapterMeta:
    """Mimic the ``DatasetMeta`` interface that build_dataset uses.

    Satisfies the full surface (``total_episodes``, ``total_frames``,
    ``camera_keys``, ``episodes``, ``require_schema()``) with minimal correct
    values: VQA has no robot trajectory (one "episode" per sample, no shard
    frame ranges, cameras from the schema's ``image_mapping``).
    """
    schema: Any
    stats: dict
    fps: float = 1.0
    robot_type: str = "vqa_no_robot"
    total_episodes: int = 0
    total_frames: int = 0
    video_keys: tuple = ()
    camera_keys: tuple = ()
    features: dict = None
    # MultiLeRobotDataset reads meta.episodes["dataset_from_index"/"to_index"].
    # VQA samples aren't packed into shards, so expose empty parallel arrays.
    episodes: dict = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}
        if self.episodes is None:
            self.episodes = {
                "dataset_from_index": [],
                "dataset_to_index": [],
            }

    def require_schema(self):
        """Return ``self.schema`` or raise if it is still ``None``.

        Mirrors ``DatasetMeta.require_schema`` so generic code can enforce the
        schema-present invariant uniformly across LeRobot and VQA adapters.
        """
        if self.schema is None:
            from schema.errors import SchemaDiscoveryError

            raise SchemaDiscoveryError(
                "_VQAAdapterMeta.schema is None after adapter construction. "
                "VQAAdapter must resolve a schema (override_schema, or "
                "manifest schema_name) before handing meta to the pipeline."
            )
        return self.schema


class VQAAdapter(BaseAdapter):
    """Adapter wrapping RoboInterVQADataset, satisfying the LeRobot adapter
    surface that build_dataset uses (.meta with .schema/.stats, .num_episodes,
    .num_frames, __len__/__getitem__). A ``BaseAdapter`` subclass so generic
    factory consumers get the full documented contract.
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        override_schema: Any = None,
        # Other adapter kwargs accepted for API compatibility (ignored by VQA):
        delta_timestamps: dict | None = None,
        image_transforms=None,
        external_stats: dict | None = None,
        use_external_stats: bool = False,
        video_backend: str | None = None,
        episode_filter=None,
    ):
        self.repo_id = repo_id
        self.root = Path(root)

        # VQA normalization stats are the built-in identity VQA_STATS
        # (state/action are zero-filled). Fail loud rather than silently ignore
        # an external-stats override so callers don't believe it took effect.
        if external_stats or use_external_stats:
            raise ValueError(
                "VQAAdapter does not support external_stats: VQA samples have "
                "zero-filled state/action and MUST use the built-in identity "
                f"VQA_STATS (got external_stats={external_stats!r}, "
                f"use_external_stats={use_external_stats!r}). Remove the "
                "external-stats override for VQA repos, or exclude this repo "
                "from external-stats resolution."
            )

        manifest_path = self.root / "meta" / "vqa_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"VQAAdapter expects {manifest_path} (vqa_manifest.json missing). "
                f"See src/adapters/vqa_adapter.py docstring for schema."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Resolve schema: caller override > manifest `schema_name`. Require an
        # explicit schema_name (no silent default) so a corpus with different
        # semantics isn't bound to the wrong schema.
        if override_schema is not None:
            schema = override_schema
        else:
            from schema.registry import resolve as _resolve_schema
            schema_name = manifest.get("schema_name")
            if not schema_name:
                raise ValueError(
                    f"VQA manifest {manifest_path} is missing required "
                    "'schema_name'. Add an explicit schema_name (e.g. "
                    "\"robointer_vqa\") to the manifest, or pass "
                    "override_schema. Refusing to default silently so a "
                    "non-robointer_vqa corpus is not bound to the wrong schema."
                )
            schema = _resolve_schema(schema_name)

        from dataset.adapters.robointer_vqa_adapter import RoboInterVQADataset

        json_paths = [
            os.path.join(root, jp) for jp in manifest["json_files"]
        ]
        image_root = os.path.join(root, manifest.get("image_root_subdir", ""))
        task_family = manifest["task_family"]

        self._dataset = RoboInterVQADataset(
            json_paths=json_paths,
            image_root=image_root,
            task_family=task_family,
            schema=schema,
        )
        # One "episode" per VQA sample so PI0Mixture n^0.43 weighting and the
        # inherited num_episodes/num_frames properties (read from meta) work.
        n = len(self._dataset)
        # VQA has no on-disk video; the only image slot is the schema's
        # image_mapping target(s) (e.g. observation.images.primary).
        camera_keys = tuple(
            getattr(schema, "image_mapping", None) or {}
        )
        # Identity stats: state/action are zero-filled so identity normalization
        # leaves them at zero. Populating observation.state/action/action_abs
        # also satisfies the action_mode='abs' guard.
        from schemas.robointer_vqa import VQA_STATS as _VQA_STATS
        self.meta = _VQAAdapterMeta(
            schema=schema,
            stats=dict(_VQA_STATS),
            total_episodes=n,
            total_frames=n,
            camera_keys=camera_keys,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        return self._dataset[idx]
