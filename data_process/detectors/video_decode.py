"""Flag videos that fail to open/demux via PyAV.

Corrupted mp4s (incomplete header, truncated moov atom, codec mismatch etc.)
silently pass the glob walk but crash any downstream consumer. Detect early.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import av

from .base import Detector, DetectorResult, Reason

_EP_RE = re.compile(r"episode_(\d{6})\.mp4$")


def _probe_decodable(mp4_path: Path) -> str | None:
    """Return None if the file opens + decodes ≥1 real frame; else an error.

    A demux-only probe (one packet + timestamp) misses mp4s that demux cleanly
    but fail in the decoder (truncated/corrupt first GOP, unsupported
    codec/profile, missing reference frames), which then crash the real
    training/eval consumer's ``decode()``. So do a BOUNDED real decode:
    single-thread, stop at the first yielded frame (cost ~one-GOP, not
    whole-file). No frames or a raise → reported bad.
    """
    try:
        with av.open(str(mp4_path)) as c:
            vs = c.streams.video[0] if c.streams.video else None
            if vs is None:
                return "no_video_stream"
            # Single-thread decode to match the deadlock-avoidance used
            # elsewhere (lerobot_v21.py) and keep the probe deterministic.
            vs.thread_type = "NONE"
            vs.thread_count = 1
            for _frame in c.decode(vs):
                return None  # first real frame decoded — good enough
            return "no_decodable_frames"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:200]


class VideoDecodeDetector(Detector):
    """Flag episodes with unreadable/undecodable video."""
    name = "video_decode"
    description = "videos that PyAV cannot open or demux"

    def __init__(self, workers: int = 16) -> None:
        super().__init__(workers=workers)
        self.workers = workers

    def scan(self, root: Path) -> DetectorResult:
        videos = root / "videos"
        if not videos.is_dir():
            return DetectorResult(self.name, set())

        all_mp4 = list(videos.glob("chunk-*/*/*.mp4"))
        bad: set[int] = set()
        reasons: dict[int, list[Reason]] = {}
        # shard_info records (chunk_idx, file_idx) provenance per bad episode so
        # the cleaner can skip re-scanning. v2.1 is one episode per file, so
        # file_idx is always 0; kept as a 2-tuple for v3 forward-compat.
        shard_info: dict[int, tuple[int, int]] = {}
        # A bad mp4 whose filename doesn't match the per-episode regex can't be
        # tied to an episode index. Collect these as dataset-level warnings (raw
        # path + error) so the bad file is still surfaced, just not as a drop.
        unattributed_bad: list[dict] = []
        _chunk_re = re.compile(r"chunk-(\d+)")
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(_probe_decodable, p): p for p in all_mp4}
            for fut in as_completed(futs):
                p = futs[fut]
                err = fut.result()
                if err is None:
                    continue
                m = _EP_RE.search(p.name)
                if not m:
                    unattributed_bad.append({"path": str(p), "error": err})
                    continue
                ep = int(m.group(1))
                cam = p.parent.name
                bad.add(ep)
                # chunk_idx is in the grandparent dir: p =
                # videos/chunk-XXX/<cam>/episode_NNN.mp4 → p.parent.parent.
                _cm = _chunk_re.fullmatch(p.parent.parent.name)
                if _cm is not None and ep not in shard_info:
                    shard_info[ep] = (int(_cm.group(1)), 0)  # file_idx 0: one ep/file
                reasons.setdefault(ep, []).append(Reason(
                    detector=self.name, kind="decode_error",
                    detail={"camera": cam, "error": err, "path": str(p)},
                ))
        extra: dict = {}
        if unattributed_bad:
            extra["unattributed_bad_videos"] = unattributed_bad
            extra["unattributed_bad_count"] = len(unattributed_bad)
        return DetectorResult(self.name, bad, reasons, extra=extra,
                              shard_info=shard_info or None)
