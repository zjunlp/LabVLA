"""Run all detectors against a v2.1 dataset and emit a JSON report.

Usage:

  python -m data_process scan \\
      --root /all-flash-data/Pretrain_Data/robointer_droid \\
      --out  /tmp/robointer_cleanup.json

  # or select specific detectors:
  python -m data_process scan \\
      --root ... --out ... \\
      --detectors video_concat_hazards,stats_orphans

The output JSON has shape:
  {
    "root": "...",
    "total_episodes": N,
    "bad_episodes": [ep_idx, ...],           # union across detectors
    "per_episode_reasons": {"45": [{...}], ...},
    "orphan_stats_episodes": [ep_idx, ...],  # for stats_orphans detector
    "detector_summaries": {"video_stubs": {"n_bad": 2, "n_reasons": 2}, ...}
  }

apply_cleanup.py consumes this report directly.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from data_process.detectors import ALL_DETECTORS, Detector


def _select_detectors(name_csv: str | None) -> list[type[Detector]]:
    if not name_csv:
        return ALL_DETECTORS
    wanted = {x.strip() for x in name_csv.split(",") if x.strip()}
    out = [D for D in ALL_DETECTORS if D.name in wanted]
    missing = wanted - {D.name for D in out}
    if missing:
        raise SystemExit(
            f"Unknown detector(s): {missing}. "
            f"Available: {[D.name for D in ALL_DETECTORS]}"
        )
    return out


def _count_episodes(root: Path) -> int:
    eps_path = root / "meta" / "episodes.jsonl"
    if not eps_path.exists():
        return 0
    with open(eps_path) as f:
        return sum(1 for line in f if line.strip())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--detectors", type=str, default=None,
                    help="Comma-separated detector names; default = all.")
    ap.add_argument("--stub-threshold", type=int, default=3,
                    help="VideoStubDetector: flag ep if any camera mp4 has "
                         "≤ this many packets (default 3).")
    ap.add_argument("--action-cap", type=int, default=32,
                    help="ActionDimCapDetector: max allowed action_dim (default 32).")
    ap.add_argument("--state-cap", type=int, default=32,
                    help="ActionDimCapDetector: max allowed state_dim (default 32).")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not (args.root / "meta" / "info.json").exists():
        raise SystemExit(f"{args.root} doesn't look like a LeRobot v2.1 dataset (no meta/info.json)")

    det_classes = _select_detectors(args.detectors)
    logging.info("Running detectors: %s", [D.name for D in det_classes])

    report = {
        "root": str(args.root),
        "total_episodes": _count_episodes(args.root),
        "bad_episodes": [],
        "per_episode_reasons": {},
        "orphan_stats_episodes": [],
        # Top-level blocking signal. Some detectors (e.g. action_dim_cap) decide
        # the WHOLE dataset is incompatible and only set
        # ``extra["dataset_incompatible"]``, leaving ``bad_episodes`` empty (no
        # per-episode fix). Hoisting the flag here lets ``apply`` refuse by
        # default; ``dataset_incompatible_reasons`` records which tripped it.
        "dataset_incompatible": False,
        "dataset_incompatible_reasons": [],
        "detector_summaries": {},
    }
    bad_union: set[int] = set()

    for D in det_classes:
        logging.info("--- %s ---", D.name)
        opts = {}
        if D.name == "video_stubs":
            opts["threshold"] = args.stub_threshold
        if D.name == "action_dim_cap":
            opts["action_cap"] = args.action_cap
            opts["state_cap"] = args.state_cap
        if D.name in ("video_stubs", "video_concat_hazards", "video_decode", "missing_videos"):
            opts["workers"] = args.workers
        det = D(**opts)
        res = det.scan(args.root)
        report["detector_summaries"][D.name] = {
            "n_bad": len(res.bad_episodes),
            "n_reasons": sum(len(v) for v in res.reasons.values()),
            "n_warnings": sum(len(v) for v in res.warnings.values()),
            "extra": res.extra,
        }
        logging.info("  bad=%d reasons=%d warnings=%d",
                     len(res.bad_episodes),
                     sum(len(v) for v in res.reasons.values()),
                     sum(len(v) for v in res.warnings.values()))
        bad_union |= res.bad_episodes
        for ep, rs in res.reasons.items():
            report["per_episode_reasons"].setdefault(str(ep), []).extend([asdict(r) for r in rs])
        if D.name == "stats_orphans":
            report["orphan_stats_episodes"].extend(res.extra.get("orphan_stats_eps", []))
        # Hoist any detector's dataset-level incompatibility flag to the top of
        # the report so the cleaner can block on it.
        if res.extra.get("dataset_incompatible"):
            report["dataset_incompatible"] = True
            report["dataset_incompatible_reasons"].append({
                "detector": D.name,
                "violations": res.extra.get("violations", []),
                "extra": {k: v for k, v in res.extra.items()
                          if k not in ("dataset_incompatible", "violations")},
            })
            logging.warning(
                "  %s flagged DATASET-INCOMPATIBLE: %s",
                D.name, res.extra.get("violations", []),
            )

    report["bad_episodes"] = sorted(bad_union)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: if scan is killed mid-dump, callers must not see partial JSON.
    import os as _os
    import tempfile as _tempfile
    _fd, _tmp = _tempfile.mkstemp(
        prefix=args.out.name + ".", suffix=".tmp", dir=str(args.out.parent)
    )
    try:
        with _os.fdopen(_fd, "w") as f:
            json.dump(report, f, indent=2)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(_tmp, str(args.out))
    except Exception:
        try:
            _os.unlink(_tmp)
        except FileNotFoundError:
            pass
        raise
    logging.info("Wrote report: %s", args.out)
    logging.info("Summary: %d bad episodes union, %d stats orphans",
                 len(bad_union), len(report["orphan_stats_episodes"]))


if __name__ == "__main__":
    main()
