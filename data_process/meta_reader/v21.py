"""MetaReader for LeRobot v2.1 (meta/episodes.jsonl)."""
from __future__ import annotations

import json
from pathlib import Path

from .base import EpisodeMeta


class V21MetaReader:
    """Reads meta/episodes.jsonl, one episode per line.

    Expected fields per line: {episode_index, tasks[], length, [source], ...}.
    """

    def read(self, meta_root: Path) -> list[EpisodeMeta]:
        ep_jsonl = Path(meta_root) / "episodes.jsonl"
        if not ep_jsonl.exists():
            raise FileNotFoundError(f"v2.1 episodes.jsonl not found at {ep_jsonl}")
        out: list[EpisodeMeta] = []
        with open(ep_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                # Guard `tasks`: a scalar string must NOT split into a
                # per-character tuple (`tuple("abc") == ('a','b','c')`). Wrap any
                # non-list as a single-element list before tupling.
                tasks_raw = raw.get("tasks", []) or []
                if not isinstance(tasks_raw, list):
                    tasks_raw = [tasks_raw]
                out.append(EpisodeMeta(
                    episode_index=int(raw["episode_index"]),
                    length=int(raw["length"]),
                    tasks=tuple(tasks_raw),
                    source=raw.get("source"),
                    # Stash anything else the caller might want (e.g. robot_type).
                    extra={k: v for k, v in raw.items()
                           if k not in ("episode_index", "length", "tasks", "source")},
                ))
        out.sort(key=lambda e: e.episode_index)
        return out
