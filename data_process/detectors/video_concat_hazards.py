"""Flag videos whose DTS layout trips LeRobot's v21→v30 ffconcat.

When `concatenate_video_files` uses FFmpeg's ffconcat demuxer to stream-copy
packets, it computes an inter-file DTS offset based on `start_time` rather
than the actual first DTS. If some files have `first_dts = -2048`
(B-frame pre-roll) while others have `first_dts = 0` (no pre-roll, e.g.
2-frame stubs), interleaving them produces back-stepping DTS at the
concat boundary:

    stub    first=0     last=1024       → output 0..1024
    normal  first=-2048 last=68608      → ffconcat shifts normal to start at 0
                                          instead of 1025 → back-step by 1024
                                          → muxer raises ArrowInvalid.

This detector finds the hazardous files: any mp4 whose first_dts != the
modal first_dts used by its camera family (and whose count is tiny — i.e.
it's an outlier, not a legitimate encoding variant).
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import av

from .base import Detector, DetectorResult, Reason

_EP_RE = re.compile(r"episode_(\d{6})\.mp4$")


def _first_dts(mp4_path: Path) -> int | None:
    try:
        with av.open(str(mp4_path)) as c:
            vs = c.streams.video[0]
            for pkt in c.demux(vs):
                if pkt.dts is not None:
                    return int(pkt.dts)
    except Exception:
        return None
    return None


class VideoConcatHazardDetector(Detector):
    """Flag episodes whose video first_dts is an outlier vs the modal value.

    Options:
        workers:     thread-pool size for PyAV probes.
        outlier_max: maximum fraction of files allowed to share a first_dts
                     value before it's considered "legitimate" (not an outlier).
                     Default 0.01 — a first_dts appearing in <1% of files
                     flags its episodes.
    """
    name = "video_concat_hazards"
    description = "outlier video first_dts values that crash v21→v30 concat"

    def __init__(self, workers: int = 16, outlier_max: float = 0.01) -> None:
        super().__init__(workers=workers, outlier_max=outlier_max)
        self.workers = workers
        self.outlier_max = outlier_max

    def scan(self, root: Path) -> DetectorResult:
        videos = root / "videos"
        if not videos.is_dir():
            return DetectorResult(self.name, set())

        camera_keys = sorted({p.name for p in videos.glob("chunk-*/*") if p.is_dir()})
        bad: set[int] = set()
        reasons: dict[int, list[Reason]] = {}
        warnings: dict[int, list[Reason]] = {}
        per_camera_modes: dict[str, int] = {}
        # `_first_dts` collapses a PyAV failure or a missing DTS into None; an
        # unprobeable file must not silently vanish from the distribution, so
        # record each skip as a per-episode warning plus a total via `extra`.
        skipped: list[dict] = []

        for cam in camera_keys:
            mp4_paths = sorted(videos.glob(f"chunk-*/{cam}/*.mp4"))
            if not mp4_paths:
                continue
            # Phase 1: probe all files, collect first_dts distribution.
            first_dts_by_ep: dict[int, int] = {}
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(_first_dts, p): p for p in mp4_paths}
                for f in as_completed(futs):
                    p = futs[f]
                    fd = f.result()
                    if fd is None:
                        m_skip = _EP_RE.search(p.name)
                        ep_skip = int(m_skip.group(1)) if m_skip else None
                        skipped.append({"episode": ep_skip, "camera": cam,
                                        "mp4": str(p)})
                        if ep_skip is not None:
                            warnings.setdefault(ep_skip, []).append(Reason(
                                detector=self.name, kind="probe_failed",
                                detail={"camera": cam, "mp4": str(p),
                                        "reason": "PyAV could not probe this "
                                                  "mp4 or it yielded no DTS — "
                                                  "concat-hazard check is "
                                                  "INCONCLUSIVE for this file"},
                            ))
                        continue
                    m = _EP_RE.search(p.name)
                    if m:
                        first_dts_by_ep[int(m.group(1))] = fd

            if not first_dts_by_ep:
                continue
            counts = Counter(first_dts_by_ep.values())
            modal_dts, modal_n = counts.most_common(1)[0]
            per_camera_modes[cam] = modal_dts

            # Phase 2: flag outlier files (those whose first_dts count < threshold).
            total = sum(counts.values())
            for ep, fd in first_dts_by_ep.items():
                if fd == modal_dts:
                    continue
                fd_frac = counts[fd] / total
                if fd_frac <= self.outlier_max:
                    bad.add(ep)
                    reasons.setdefault(ep, []).append(Reason(
                        detector=self.name, kind="outlier_first_dts",
                        detail={"camera": cam, "first_dts": fd,
                                "modal_first_dts": modal_dts,
                                "modal_frac": modal_n / total,
                                "this_dts_frac": fd_frac},
                    ))
        extra: dict = {"per_camera_modal_first_dts": per_camera_modes}
        if skipped:
            extra["probe_skipped_count"] = len(skipped)
            extra["probe_skipped"] = skipped
        return DetectorResult(
            self.name, bad, reasons, warnings=warnings, extra=extra,
        )
