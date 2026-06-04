"""Flag datasets whose action_dim or state_dim exceeds the model cap.

Unlike per-episode detectors, action_dim is a dataset-wide feature —
any violation means the whole dataset is incompatible with the current
model.max_action_dim / max_state_dim, so we emit a dataset-level flag
via `extra["dataset_incompatible"]` rather than per-episode bad_episodes.

Consumers (e.g. `data_process scan`) can then decide to refuse cleanup
or skip the dataset entirely.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import Detector, DetectorResult, Reason


class ActionDimCapDetector(Detector):
    name = "action_dim_cap"
    description = "action_dim or state_dim exceeds model cap"

    def __init__(self, action_cap: int = 32, state_cap: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.action_cap = action_cap
        self.state_cap = state_cap

    def scan(self, root: Path) -> DetectorResult:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            return DetectorResult(self.name, set())

        with open(info_path) as _f:
            info = json.load(_f)
        feats = info.get("features", {}) or {}

        incompat: list[str] = []
        for key, cap in [
            ("action", self.action_cap),
            ("observation.state", self.state_cap),
            ("state", self.state_cap),
        ]:
            entry = feats.get(key, {}) or {}
            shape = entry.get("shape") or []
            if shape and int(shape[0]) > cap:
                incompat.append(f"{key} dim {shape[0]} > cap {cap}")

        warnings: dict[int, list[Reason]] = {}
        if incompat:
            # Use synthetic episode -1 to signal dataset-level reject; keeps the
            # DetectorResult schema simple without adding a new field.
            warnings[-1] = [
                Reason(
                    detector=self.name,
                    kind="dataset_incompatible",
                    detail={"violations": incompat,
                            "caps": {"action": self.action_cap,
                                     "state": self.state_cap}},
                )
            ]

        # Dataset-level detector (action/state dims are a dataset-wide feature),
        # so shard_info is always None: the decision is reject-or-accept the
        # whole dataset, not per-episode quarantine.
        return DetectorResult(
            detector_name=self.name,
            bad_episodes=set(),
            reasons={},
            warnings=warnings,
            extra={
                "dataset_incompatible": bool(incompat),
                "violations": incompat,
                "caps": {"action": self.action_cap, "state": self.state_cap},
            },
            shard_info=None,
        )
