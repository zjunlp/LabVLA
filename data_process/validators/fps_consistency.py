"""FPSConsistencyValidator — detect fps mismatch across repos.

Mixed-fps training is not fatal (the adapter builds per-repo delta_timestamps
from each repo's own fps), but it means `chunk_size=50` represents different
wall-clock horizons (e.g. 1.67 s at 30 fps vs 5 s at 10 fps). The model's
temporal learning signal is heterogeneous — worth flagging so the user makes
an explicit choice (resample, per-repo chunk size, or accept).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from .base import Validator, ValidationIssue


class FPSConsistencyValidator(Validator):
    name = "fps_consistency"
    description = "Verify all repos declare the same fps in meta/info.json."

    def validate(self, repos: Iterable[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        fps_map: dict[str, float | None] = {}

        for repo in repos:
            repo_path = Path(repo)
            if (repo_path / "meta" / "vqa_manifest.json").exists():
                # VQA-only repos have no time-aligned action chunks and no fps.
                # They are valid inputs for VQA co-train but are outside this
                # LeRobot fps validator's contract.
                continue
            info_path = repo_path / "meta" / "info.json"
            if not info_path.exists():
                issues.append(ValidationIssue(
                    severity="HIGH",
                    repo=str(repo_path),
                    message=f"meta/info.json not found at {info_path}",
                    context={"path": str(info_path)},
                ))
                fps_map[str(repo_path)] = None
                continue
            try:
                info = json.loads(info_path.read_text())
            except Exception as e:
                issues.append(ValidationIssue(
                    severity="HIGH",
                    repo=str(repo_path),
                    message=f"failed to parse info.json: {e}",
                    context={"path": str(info_path)},
                ))
                fps_map[str(repo_path)] = None
                continue
            fps_value = info.get("fps")

            # Missing fps is itself a HIGH issue: dropping fps==None from the
            # cross-repo distinct set would let it pass preflight, then blow up
            # when the adapter builds delta_timestamps with `None / fps`.
            if fps_value is None:
                fps_map[str(repo_path)] = None
                issues.append(ValidationIssue(
                    severity="HIGH",
                    repo=str(repo_path),
                    message=(
                        f"meta/info.json missing required 'fps' field. "
                        f"Adapter delta_timestamps and chunk-horizon math "
                        f"both depend on a numeric fps."
                    ),
                    context={"path": str(info_path)},
                ))
                continue

            # fps must be finite and positive. Otherwise 0, negatives, NaN/Inf,
            # bools, and non-numeric strings poison every downstream `i / fps`
            # conversion (ZeroDivisionError, negative/NaN horizons, wrong frame
            # indexing). Reject fail-loud and keep the value OUT of the distinct
            # set so it can't mask a real fps mismatch.
            fps_is_valid = (
                isinstance(fps_value, (int, float))
                and not isinstance(fps_value, bool)
                and math.isfinite(fps_value)
                and fps_value > 0
            )
            if not fps_is_valid:
                fps_map[str(repo_path)] = None
                issues.append(ValidationIssue(
                    severity="HIGH",
                    repo=str(repo_path),
                    message=(
                        f"meta/info.json 'fps' must be a finite number > 0, "
                        f"got {fps_value!r} ({type(fps_value).__name__}). "
                        f"Adapter delta_timestamps and chunk-horizon math "
                        f"both depend on a valid positive fps."
                    ),
                    context={"path": str(info_path), "fps": fps_value},
                ))
                continue

            fps_map[str(repo_path)] = fps_value

        distinct = {fps for fps in fps_map.values() if fps is not None}
        if len(distinct) > 1:
            issues.append(ValidationIssue(
                severity="MED",
                repo="<multi>",
                message=(
                    f"Inconsistent fps across repos: {fps_map}. "
                    "With fixed chunk_size, each repo's 50-frame chunk covers a "
                    "different wall-clock horizon (5 s at 10 fps vs 1.67 s at 30 fps). "
                    "Training continues but model must implicitly handle both."
                ),
                context={"fps_map": fps_map},
            ))
        return issues
