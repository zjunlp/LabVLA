"""Flag episodes listed in episodes.jsonl but missing mp4 on disk for any
REQUIRED camera. The set of required cameras is read from info.json features
(dtype=video), NOT inferred from the filesystem — otherwise stray residual
camera dirs left over from past merges (with only a handful of files) will
make every episode look "missing" those cameras.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .base import Detector, DetectorResult, Reason


_EP_RE = re.compile(r"episode_(\d{6})\.mp4$")


class MissingVideoDetector(Detector):
    """Cross-check episodes.jsonl against on-disk videos/*/*/episode_*.mp4.

    Uses info.json's `features` (dtype == "video") as the authoritative list
    of required cameras — residual subdirs under `videos/chunk-XXX/` that
    aren't in features are ignored.
    """
    name = "missing_videos"
    description = "episodes listed in episodes.jsonl but lacking an mp4 on disk"

    def scan(self, root: Path) -> DetectorResult:
        videos_root = root / "videos"
        info_path = root / "meta" / "info.json"
        eps_path = root / "meta" / "episodes.jsonl"
        if not (videos_root.is_dir() and eps_path.exists() and info_path.exists()):
            return DetectorResult(self.name, set())

        info = json.loads(info_path.read_text())
        required_cams = sorted(
            k for k, v in info.get("features", {}).items()
            if isinstance(v, dict) and v.get("dtype") == "video"
        )
        if not required_cams:
            # No video features in info.json means no baseline to cross-check
            # against. Returning plain-clean would disguise "cannot determine"
            # as "nothing missing", so surface it explicitly via `extra`.
            return DetectorResult(
                self.name, set(),
                extra={
                    "required_camera_baseline_unavailable": True,
                    "reason": (
                        "info.json declares no video features (dtype=='video'); "
                        "cannot establish required-camera baseline — "
                        "missing-video check is INCONCLUSIVE, not clean."
                    ),
                },
            )

        per_cam_eps: dict[str, set[int]] = {}
        for cam in required_cams:
            have: set[int] = set()
            for mp4 in videos_root.glob(f"chunk-*/{cam}/*.mp4"):
                m = _EP_RE.search(mp4.name)
                if m:
                    have.add(int(m.group(1)))
            per_cam_eps[cam] = have

        bad: set[int] = set()
        reasons: dict[int, list[Reason]] = {}
        with open(eps_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line).get("episode_index")
                if ep is None:
                    continue
                missing_cams = [c for c in required_cams if ep not in per_cam_eps[c]]
                if missing_cams:
                    bad.add(ep)
                    reasons.setdefault(ep, []).append(Reason(
                        detector=self.name, kind="missing_mp4",
                        detail={"missing_cameras": missing_cams,
                                "required_cameras": required_cams},
                    ))
        return DetectorResult(
            self.name, bad, reasons,
            extra={"required_cameras": required_cams},
        )
