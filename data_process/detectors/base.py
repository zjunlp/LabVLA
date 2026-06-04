"""Detector ABC — a uniform interface for dataset issue scanners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Reason:
    """Structured reason for flagging an episode."""
    detector: str            # e.g. "VideoStubDetector"
    kind: str                # e.g. "too_few_packets"
    detail: dict[str, Any] = field(default_factory=dict)  # e.g. {"npkt": 2, "camera": "primary"}


@dataclass
class DetectorResult:
    """What a detector returns."""
    detector_name: str
    bad_episodes: set[int]                       # episode indices to filter out
    reasons: dict[int, list[Reason]] = field(default_factory=dict)  # ep -> [Reason]
    # Optional: episodes with issues that DON'T warrant filtering but are worth
    # logging (e.g. stats orphans that the cleaner should also filter at the
    # stats-only level, not the whole episode).
    warnings: dict[int, list[Reason]] = field(default_factory=dict)
    # Optional extra data the detector wants to carry (e.g. orphan stats keys).
    extra: dict[str, Any] = field(default_factory=dict)
    # Optional: episode_index → (chunk_idx, file_idx) provenance. Populated by
    # v3 detectors where shards hold multiple episodes and the cleaner needs more
    # than an episode index to locate the offending rows. v2.1 is one episode per
    # file and leaves this None.
    shard_info: dict[int, tuple[int, int]] | None = None


class Detector(ABC):
    """Scans a v2.1 LeRobot dataset root and identifies problematic episodes.

    Subclasses implement `scan()` and set a human-readable `name` + `description`.
    """
    name: str = "detector"
    description: str = ""

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    def scan(self, root: Path) -> DetectorResult:
        """Walk the dataset at `root` and return a DetectorResult.

        `root` points at a v2.1 LeRobot directory containing `meta/`, `data/`,
        `videos/`. Implementations must be SAFE (read-only) and SHOULD be
        parallelizable where scanning is IO-bound.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(options={self.options})"
