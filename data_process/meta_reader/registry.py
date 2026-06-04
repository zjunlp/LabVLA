"""Dispatch MetaReader by info.json codebase_version.

Single entry point: `read_episode_meta(root) -> list[EpisodeMeta]`.
Callers don't need to care about v2 vs v3 on-disk layout.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import EpisodeMeta, MetaReader
from .v21 import V21MetaReader
from .v30 import V30MetaReader


def detect_codebase_version(repo_root: Path) -> str:
    """Read `meta/info.json::codebase_version`. Raises if missing."""
    info_path = Path(repo_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found at {info_path}")
    info = json.loads(info_path.read_text())
    v = info.get("codebase_version")
    if not v:
        raise ValueError(f"{info_path} missing 'codebase_version' field")
    return str(v)


def get_reader(codebase_version: str) -> MetaReader:
    """Return the MetaReader instance for a given codebase_version string."""
    if codebase_version.startswith("v2"):
        return V21MetaReader()
    if codebase_version.startswith("v3"):
        return V30MetaReader()
    raise ValueError(
        f"Unsupported codebase_version={codebase_version!r}. "
        f"Supported: v2.x | v3.x"
    )


def read_episode_meta(repo_root: Path) -> list[EpisodeMeta]:
    """One-shot: read a repo's per-episode metadata, format-agnostic."""
    version = detect_codebase_version(repo_root)
    reader = get_reader(version)
    return reader.read(Path(repo_root) / "meta")
