"""TimestampColumnValidator — ensure per-frame timestamp info exists.

Most downstream tools (e.g. video-sync, episode trimming, action-interp
probes) rely on a timestamp column in the per-episode parquet. LabVLA's
training pipeline does NOT strictly require it (delta_timestamps are
derived from fps), so this is a MED-severity heads-up rather than a
blocker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import Validator, ValidationIssue


class TimestampColumnValidator(Validator):
    name = "timestamp_column"
    description = "Verify each repo's parquet schema exposes a timestamp column."

    def validate(self, repos: Iterable[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            return [
                ValidationIssue(
                    severity="MED",
                    repo="<validator>",
                    message=(
                        "pyarrow is not installed; timestamp_column validator "
                        f"skipped ({e}). Install pyarrow to inspect parquet metadata."
                    ),
                    context={},
                )
            ]

        for repo in repos:
            repo_path = Path(repo)
            if (repo_path / "meta" / "vqa_manifest.json").exists():
                # VQA-only repos are JSON/JSONL+image datasets, not LeRobot
                # parquet repos. Preflight validates their manifest separately.
                continue
            parquet = next(iter(repo_path.rglob("data/**/*.parquet")), None)
            if parquet is None:
                issues.append(ValidationIssue(
                    severity="HIGH",
                    repo=str(repo_path),
                    message="no parquet file found under data/",
                    context={},
                ))
                continue
            try:
                cols = pq.read_metadata(parquet).schema.names
            except Exception as e:
                issues.append(ValidationIssue(
                    severity="HIGH",
                    repo=str(repo_path),
                    message=f"failed to read parquet metadata: {e}",
                    context={"path": str(parquet)},
                ))
                continue
            if not any("timestamp" in c.lower() or "time" in c.lower() for c in cols):
                issues.append(ValidationIssue(
                    severity="MED",
                    repo=str(repo_path),
                    message=(
                        f"no timestamp/time column found (columns: {cols}). "
                        "Training works via fps-derived delta_timestamps, but "
                        "downstream tooling (video sync, episode trim) may break."
                    ),
                    context={"columns": cols, "sample_parquet": str(parquet)},
                ))
        return issues
