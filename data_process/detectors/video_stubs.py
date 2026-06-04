"""Flag videos with abnormally few packets (truncated/broken recordings).

Short-duration video stubs (e.g. 2-3 frames) often indicate a failed/cut-short
recording that should not be used for training. As a bonus, 2-packet stubs
tend to have `first_dts=0` which trips LeRobot's v21→v30 ffconcat (see
VideoConcatHazardDetector for the direct trigger).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import av

from .base import Detector, DetectorResult, Reason

_EP_RE = re.compile(r"episode_(\d{6})\.mp4$")


def _probe(mp4_path: Path, max_pkts: int) -> tuple[int, int] | None:
    """Return (npkt, first_dts) — counts up to max_pkts+1 packets for speed."""
    try:
        with av.open(str(mp4_path)) as c:
            vs = c.streams.video[0]
            npkt, first_dts = 0, None
            for pkt in c.demux(vs):
                if pkt.dts is None:
                    continue
                if first_dts is None:
                    first_dts = int(pkt.dts)
                npkt += 1
                if npkt > max_pkts + 1:
                    break
            if first_dts is None:
                return None
            return npkt, first_dts
    except Exception:
        return None


class VideoStubDetector(Detector):
    """Flag episodes whose ANY camera mp4 has npkt ≤ `threshold` packets."""
    name = "video_stubs"
    description = "episodes with at least one camera video of suspiciously few packets"

    def __init__(self, threshold: int = 3, workers: int = 16) -> None:
        super().__init__(threshold=threshold, workers=workers)
        self.threshold = threshold
        self.workers = workers

    def scan(self, root: Path) -> DetectorResult:
        videos = root / "videos"
        if not videos.is_dir():
            return DetectorResult(self.name, set())

        # Enumerate each camera's mp4 list.
        camera_keys = sorted({p.name for p in videos.glob("chunk-*/*") if p.is_dir()})
        all_tasks: list[tuple[int, str, Path]] = []
        for cam in camera_keys:
            for mp4 in videos.glob(f"chunk-*/{cam}/*.mp4"):
                m = _EP_RE.search(mp4.name)
                if not m:
                    continue
                ep = int(m.group(1))
                all_tasks.append((ep, cam, mp4))

        bad: set[int] = set()
        reasons: dict[int, list[Reason]] = {}
        warnings: dict[int, list[Reason]] = {}
        # `_probe` collapses a PyAV open/demux failure or a file with no usable
        # DTS into None; treating None as healthy would under-count broken
        # videos, so record each skip as a per-episode warning (no drop) plus a
        # total via `extra`.
        skipped: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {
                ex.submit(_probe, mp4, self.threshold): (ep, cam, mp4)
                for (ep, cam, mp4) in all_tasks
            }
            for fut in as_completed(futures):
                ep, cam, mp4 = futures[fut]
                res = fut.result()
                if res is None:
                    skipped.append({"episode": ep, "camera": cam, "mp4": str(mp4)})
                    warnings.setdefault(ep, []).append(Reason(
                        detector=self.name, kind="probe_failed",
                        detail={"camera": cam, "mp4": str(mp4),
                                "reason": "PyAV could not probe this mp4 or it "
                                          "yielded no DTS — stub check is "
                                          "INCONCLUSIVE for this file"},
                    ))
                    continue
                npkt, fd = res
                if npkt <= self.threshold:
                    bad.add(ep)
                    reasons.setdefault(ep, []).append(Reason(
                        detector=self.name, kind="too_few_packets",
                        detail={"camera": cam, "npkt": npkt, "first_dts": fd,
                                "threshold": self.threshold},
                    ))
        return DetectorResult(
            self.name, bad, reasons, warnings=warnings,
            extra={"probe_skipped_count": len(skipped),
                   "probe_skipped": skipped} if skipped else {},
        )
