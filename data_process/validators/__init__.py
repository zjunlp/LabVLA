"""Dataset-level validators — peer module to detectors/ (per-episode) and
cleanup/ (rewriter). Validators report cross-repo or whole-repo invariant
violations that the caller decides how to handle.

Registered validators:
  * `FPSConsistencyValidator`    — all repos declare same fps
  * `TimestampColumnValidator`   — parquet has a timestamp column
  * `FrameCountValidator`        — sampled mp4 frame count matches jsonl length

Intended use:
    python -m data_process validate --repos r1 r2 [--validators fps_consistency,...]
"""
from .base import Validator, ValidationIssue  # noqa: F401
from .fps_consistency import FPSConsistencyValidator
from .timestamp_column import TimestampColumnValidator
from .frame_count import FrameCountValidator

ALL_VALIDATORS = [
    FPSConsistencyValidator,
    TimestampColumnValidator,
    FrameCountValidator,
]

__all__ = [
    "Validator", "ValidationIssue",
    "FPSConsistencyValidator",
    "TimestampColumnValidator",
    "FrameCountValidator",
    "ALL_VALIDATORS",
]
