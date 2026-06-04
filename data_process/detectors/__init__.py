"""Per-issue detectors. Each subclass scans a dataset and returns bad episodes.

Registered detectors:
  * `VideoStubDetector`        — videos with ≤ threshold packets (tiny / truncated)
  * `VideoConcatHazardDetector`— videos with abnormal first_dts that break
                                 LeRobot's v21→v30 video concatenation
  * `VideoDecodeDetector`      — videos that fail to open/demux via PyAV
  * `MissingVideoDetector`     — episode has jsonl entry but no mp4 on disk
  * `StatsOrphanDetector`      — entries in episodes_stats.jsonl missing from
                                 episodes.jsonl (breaks the converter's zip
                                 positional cross-check)
  * `ActionDimCapDetector`     — action_dim / state_dim exceeds model cap
                                 (dataset-level flag, not per-episode)

Video detectors do `import av` at module level. PyAV is heavy and not always
available in lightweight tooling envs that only need preflight / stats, and the
eager import used to be triggered just by `import data_process.preflight`. So
the video detectors load lazily via PEP 562 ``__getattr__``; non-av detectors
and the base classes stay eager (zero overhead, no av).
"""
from .base import Detector, DetectorResult, Reason  # noqa: F401
from .missing_videos import MissingVideoDetector  # noqa: F401
from .stats_orphans import StatsOrphanDetector  # noqa: F401
from .action_dim_cap import ActionDimCapDetector  # noqa: F401


# Lazy attributes: importing pyav-backed detectors only happens on first
# attribute access. ALL_DETECTORS is also lazy because building the list
# would force every detector module to load.
_LAZY_DETECTORS = {
    "VideoStubDetector": (".video_stubs", "VideoStubDetector"),
    "VideoConcatHazardDetector": (".video_concat_hazards", "VideoConcatHazardDetector"),
    "VideoDecodeDetector": (".video_decode", "VideoDecodeDetector"),
}


def __getattr__(name: str):  # PEP 562 module-level __getattr__
    if name in _LAZY_DETECTORS:
        modname, clsname = _LAZY_DETECTORS[name]
        from importlib import import_module
        mod = import_module(modname, package=__name__)
        cls = getattr(mod, clsname)
        # Cache on the module so subsequent lookups skip __getattr__.
        globals()[name] = cls
        return cls
    if name == "ALL_DETECTORS":
        # Build on demand; uses __getattr__ recursively for the lazy ones.
        all_dets = [
            __getattr__("VideoStubDetector"),
            __getattr__("VideoConcatHazardDetector"),
            __getattr__("VideoDecodeDetector"),
            MissingVideoDetector,
            StatsOrphanDetector,
            ActionDimCapDetector,
        ]
        globals()["ALL_DETECTORS"] = all_dets
        return all_dets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # tab-completion / inspect.getmembers friendliness
    return sorted(list(globals().keys()) + list(_LAZY_DETECTORS.keys()) + ["ALL_DETECTORS"])


__all__ = [
    "Detector", "DetectorResult", "Reason",
    "VideoStubDetector", "VideoConcatHazardDetector", "VideoDecodeDetector",
    "MissingVideoDetector", "StatsOrphanDetector", "ActionDimCapDetector",
    "ALL_DETECTORS",
]
