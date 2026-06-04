"""Validator ABC — dataset-level invariant checks.

Distinction vs `detectors/`:

    Detector                          Validator
    --------                          ---------
    scope: per-episode                scope: per-repo (or across repos)
    returns: set of bad episode       returns: list of ValidationIssue
             indices                           (free-form dataset-level issues)
    use:    filter-then-cleanup       use:    fail-fast pre-merge / pre-train
                                              sanity check

A Validator never rewrites or deletes data; it inspects and reports.
Callers decide whether an issue blocks execution (HIGH/CRIT severity) or
just warns (MED/LOW).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

Severity = Literal["CRIT", "HIGH", "MED", "LOW"]


@dataclass
class ValidationIssue:
    severity: Severity
    repo: str                 # either a single repo name or "<multi>" for
                              # cross-repo issues (e.g. FPS mismatch across repos)
    message: str              # human-readable explanation
    context: dict[str, Any] = field(default_factory=dict)  # structured payload for tools


class Validator(ABC):
    """Check a dataset-level invariant across one or more LeRobot repos.

    Subclasses implement `validate(repos) -> list[ValidationIssue]`. They
    must be READ-ONLY. Cross-repo validators (like fps consistency) expect
    multiple roots; single-repo validators must still handle the list form
    (iterate internally).
    """
    name: str = "validator"
    description: str = ""

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    def validate(self, repos: Iterable[Path]) -> list[ValidationIssue]:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(options={self.options})"
