"""Flag orphan entries in episodes_stats.jsonl (indices missing from episodes.jsonl).

Not an episode filter in the usual sense — these entries reference episodes
that no longer exist in the dataset, and must be dropped from
episodes_stats.jsonl (otherwise the LeRobot v21→v30 converter's
`generate_episode_metadata_dict` zip-by-position cross-check fails).

Emits ORPHAN episode indices into `extra["orphan_stats_eps"]` — the cleaner
uses this to strip orphan lines from episodes_stats.jsonl without flagging
the main bad_episodes set.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import Detector, DetectorResult, Reason


class StatsOrphanDetector(Detector):
    name = "stats_orphans"
    description = "episodes_stats.jsonl entries missing from episodes.jsonl"

    def scan(self, root: Path) -> DetectorResult:
        eps_path = root / "meta" / "episodes.jsonl"
        stats_path = root / "meta" / "episodes_stats.jsonl"
        if not (eps_path.exists() and stats_path.exists()):
            return DetectorResult(self.name, set())

        eps_idx: set[int] = set()
        with open(eps_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ep = json.loads(line).get("episode_index")
                    if ep is not None:
                        eps_idx.add(ep)
        stats_idx: set[int] = set()
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ep = json.loads(line).get("episode_index")
                    if ep is not None:
                        stats_idx.add(ep)

        orphans = stats_idx - eps_idx
        reasons: dict[int, list[Reason]] = {}
        for ep in orphans:
            reasons[ep] = [Reason(self.name, kind="orphan_stats_entry",
                                  detail={"note": "stats entry without episode.jsonl entry"})]
        return DetectorResult(
            self.name,
            bad_episodes=set(),         # these don't ban episodes — they ban stats rows
            reasons={},
            warnings=reasons,
            extra={"orphan_stats_eps": sorted(orphans)},
        )
