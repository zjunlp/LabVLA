"""Format-agnostic LeRobot meta reader.

Public API:
    from data_process.meta_reader import EpisodeMeta, read_episode_meta
    episodes = read_episode_meta(repo_root)

The reader dispatches internally by `meta/info.json::codebase_version`:
    - v2.x → V21MetaReader (reads meta/episodes.jsonl)
    - v3.x → V30MetaReader (reads meta/episodes/chunk-*/file-*.parquet)

Each returns `list[EpisodeMeta]` with uniform schema regardless of format.
"""
from .base import EpisodeMeta, MetaReader
from .registry import detect_codebase_version, get_reader, read_episode_meta
from .v21 import V21MetaReader
from .v30 import V30MetaReader

__all__ = [
    "EpisodeMeta", "MetaReader",
    "V21MetaReader", "V30MetaReader",
    "detect_codebase_version", "get_reader", "read_episode_meta",
]
