"""EpisodeMeta dataclass + MetaReader protocol.

A MetaReader hides the v2 vs v3 storage difference:
  - v2.1: meta/episodes.jsonl (one dict per line)
  - v3.0: meta/episodes/chunk-*/file-*.parquet (sharded)

Both produce a list of EpisodeMeta with the minimum common fields used by
mixture task discovery and per-task adapter instantiation.

Note: the *adapter* still does its own meta read internally (needs v3-specific
sharding pointers like `data/chunk_index`). This reader is intentionally
minimal — only the fields discovery/builder require.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EpisodeMeta:
    """Per-episode metadata, format-agnostic."""
    episode_index: int
    length: int
    tasks: tuple[str, ...]             # language instructions (≥1)
    source: str | None = None          # optional ABot-style task-folder name
    # Any extra repo-specific fields a discovery policy might need.
    extra: dict = field(default_factory=dict)


class MetaReader(Protocol):
    """Reads per-episode metadata for ONE LeRobot repo, regardless of
    codebase_version."""
    def read(self, meta_root: Path) -> list[EpisodeMeta]:
        ...
