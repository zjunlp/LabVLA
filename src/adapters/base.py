"""
Base adapter interface.

All dataset adapters must provide:
  - meta: DatasetMeta (robot_type, stats, fps, features, ...)
  - __getitem__(idx) → dict with raw column-name keys
  - __len__() → total frames
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import torch.utils.data

if TYPE_CHECKING:
    from schema import DatasetSchema


@dataclass
class DatasetMeta:
    """Minimal metadata interface that the Transform chain needs."""
    robot_type: str
    stats: dict[str, dict] | None  # per-key normalization stats
    fps: int
    total_episodes: int
    total_frames: int
    features: dict[str, Any]
    video_keys: list[str] = field(default_factory=list)
    camera_keys: list[str] = field(default_factory=list)
    # Optional only for the brief pre-discovery window inside adapter __init__;
    # once the adapter returns, schema MUST be non-None. Use require_schema().
    schema: Optional["DatasetSchema"] = None
    # MultiLeRobotDataset expects `meta.episodes["dataset_from_index"/"to_index"]`
    # to build combined frame indices across multi-repo concatenation.
    episodes: dict = field(default_factory=lambda: {
        "dataset_from_index": [],
        "dataset_to_index": [],
    })

    def require_schema(self) -> "DatasetSchema":
        """Return ``self.schema`` or raise if it is still ``None``.

        Code paths after adapter construction should use this accessor rather
        than carrying a ``None`` schema forward: a missing schema means
        hydrate_all has no state/action keys, no delta mask, and no image
        mapping — silently training on raw features with zero normalization.
        """
        if self.schema is None:
            from schema.errors import SchemaDiscoveryError

            raise SchemaDiscoveryError(
                "DatasetMeta.schema is None after adapter construction. "
                "Every adapter __init__ must populate meta.schema via "
                "schema.discover_schema(root, robot_type=...). If this meta "
                "was constructed manually (tests, ablations), attach a "
                "DatasetSchema before handing it to the transform/dataset "
                "pipeline."
            )
        return self.schema


class BaseAdapter(torch.utils.data.Dataset):
    """Abstract base for all dataset format adapters."""

    meta: DatasetMeta
    repo_id: str
    root: Path

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    @property
    def num_frames(self) -> int:
        return self.meta.total_frames
