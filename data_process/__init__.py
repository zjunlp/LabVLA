"""Pretrain data cleaning utilities.

Use this package to scan a LeRobot v2.1 dataset (or similar) for episodes that
are unsafe to use for training or v3.0 conversion, and to produce a cleaned
symlink-copy of the dataset with those episodes filtered out.

Typical flow:

    # 1. scan for issues:
    python -m data_process.process.scan_dataset \\
        --root /all-flash-data/Pretrain_Data/robointer_droid \\
        --out  /tmp/robointer_cleanup.json

    # 2. produce a cleaned, renumbered symlink copy:
    python -m data_process.process.apply_cleanup \\
        --src   /all-flash-data/Pretrain_Data/robointer_droid \\
        --dst   /all-flash-data/Pretrain_Data/robointer_droid_clean \\
        --report /tmp/robointer_cleanup.json

    # 3. pass the cleaned dir to the v21→v30 converter.

Detectors live under `data_process.detectors`; each knows how to scan a
dataset and return a set of bad episode indices with per-episode reasons.
"""
from .detectors.base import Detector, Reason, DetectorResult  # noqa: F401
