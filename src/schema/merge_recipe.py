"""Declarative merge recipes for building multi-source clean datasets.

`oxe-auge_clean` is assembled by pulling shards from 180 sub-source OXE repos
into one canonical layout. Some sources are 6-DoF (UR5, Jaco, WidowX/Bridge)
with gripper at raw index 6, while the dominant majority are 7-DoF with
gripper at raw index 7. The merge script used to hardcode that split as a
string-prefix list; this module lets the schema itself describe the split
declaratively.

Usage (in a schema module):

    from schema.arm_layout import ArmLayoutSpec, ArmCount
    from schema.merge_recipe import MergeRecipe, SourceArmLayout

    MERGE_RECIPE = MergeRecipe(
        canonical=ArmLayoutSpec(ArmCount.SINGLE, arm_dof=7, gripper_index_in_raw=7),
        source_layouts={
            "berkeley_autolab_ur5": SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
            "bridge":               SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
            "jaco_play":            SourceArmLayout(arm_dof=6, raw_gripper_idx=6),
        },
        default_layout=SourceArmLayout(arm_dof=7, raw_gripper_idx=7),
    )

The merge script consumes `recipe.needs_rewrite(source_family)` and
`recipe.transform_row(row, source_family)` to promote each row to the
canonical layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .arm_layout import ArmLayoutSpec, ArmCount


@dataclass(frozen=True)
class SourceArmLayout:
    """Raw parquet layout of one source in a merged multi-source dataset."""

    arm_dof: int              # joints in source's raw joints column (6 or 7)
    raw_gripper_idx: int      # index of gripper within the raw row

    def __post_init__(self) -> None:
        if self.arm_dof not in (6, 7):
            raise ValueError(f"arm_dof must be 6 or 7, got {self.arm_dof}")
        # Gripper should sit immediately after the last arm joint.
        if self.raw_gripper_idx != self.arm_dof:
            raise ValueError(
                f"raw_gripper_idx ({self.raw_gripper_idx}) must equal arm_dof "
                f"({self.arm_dof}) — gripper lives right after the arm joints."
            )


@dataclass(frozen=True)
class MergeRecipe:
    """Declarative rule for promoting per-source raw rows to canonical layout.

    Fields:
        canonical:
            Target layout after merge. For oxe-auge this is
            single-arm 7-DoF + 1 gripper at index 7 (8-dim total).
        source_layouts:
            Prefix string → per-source layout. The longest matching prefix
            wins; order in the mapping doesn't matter.
        default_layout:
            Fallback for source families not listed in ``source_layouts``.
            Typically equal to ``canonical`` layout's (arm_dof, gripper_index_in_raw).
    """

    canonical: ArmLayoutSpec
    source_layouts: Mapping[str, SourceArmLayout]
    default_layout: SourceArmLayout

    def __post_init__(self) -> None:
        if self.canonical.arm_count != ArmCount.SINGLE:
            # Dual-arm merge is not in scope for v1 — no sources require it.
            raise NotImplementedError(
                "MergeRecipe currently only supports single-arm canonical layout."
            )
        if self.canonical.arm_dof is None or self.canonical.gripper_index_in_raw is None:
            raise ValueError(
                "MergeRecipe.canonical must have arm_dof and gripper_index_in_raw set."
            )
        # Detect overlapping source prefixes. `layout_for` uses
        # longest-prefix match, so two prefixes like "bridge" and "bridge_v2"
        # superficially work — but an author who adds them both is almost
        # certainly intending "bridge (v1)" vs "bridge_v2" as distinct
        # families, and silently folding v1 under the longer match is
        # surprising. Force the author to pick a disambiguating pair
        # ("bridge_v1" + "bridge_v2", or "bridge/" + "bridge_v2/") instead.
        prefixes = list(self.source_layouts.keys())
        for i, a in enumerate(prefixes):
            for b in prefixes[i + 1:]:
                if a == b:
                    # Dict-keyed, can't actually happen, but guard anyway.
                    raise ValueError(
                        f"MergeRecipe.source_layouts has duplicate prefix {a!r}."
                    )
                if a.startswith(b) or b.startswith(a):
                    shorter, longer = (b, a) if len(a) > len(b) else (a, b)
                    raise ValueError(
                        f"MergeRecipe.source_layouts has overlapping prefixes "
                        f"{shorter!r} and {longer!r}: every source family that "
                        f"matches {shorter!r} would also match {longer!r}, so "
                        f"the shorter entry is unreachable for sources whose "
                        f"family name starts with {longer!r}. Disambiguate "
                        f"with a trailing separator (e.g. {shorter + '/'!r}) "
                        f"or rename one prefix."
                    )

    def layout_for(self, source_family: str) -> SourceArmLayout:
        """Return the SourceArmLayout for the longest matching prefix (or default)."""
        best_match: tuple[int, SourceArmLayout] = (-1, self.default_layout)
        for prefix, lay in self.source_layouts.items():
            if source_family.startswith(prefix) and len(prefix) > best_match[0]:
                best_match = (len(prefix), lay)
        return best_match[1]

    def needs_rewrite(self, source_family: str) -> bool:
        """True if the source's raw layout differs from canonical and must be
        physically rewritten (as opposed to symlinked unchanged)."""
        lay = self.layout_for(source_family)
        return (lay.arm_dof != self.canonical.arm_dof
                or lay.raw_gripper_idx != self.canonical.gripper_index_in_raw)

    def transform_row(self, raw_row: np.ndarray, source_family: str) -> np.ndarray:
        """Promote one raw row to canonical layout.

        For the only case in scope — 6-DoF source (arm_dof=6, grip_idx=6) →
        7-DoF canonical (arm_dof=7, grip_idx=7) — we zero-pad the arm slot
        at canonical dim 6 and move gripper from raw[6] to canonical[7].
        """
        lay = self.layout_for(source_family)
        canonical_total = self.canonical.arm_dof + 1  # + gripper
        if not self.needs_rewrite(source_family):
            # No rewrite needed: the source declares the canonical layout, so a
            # correct raw row is already canonical-width and is returned as-is.
            if raw_row.shape[-1] == canonical_total:
                return raw_row
            # A row shorter than canonical here is NOT a benign mixed-width
            # episode — the source's real layout disagrees with the declared
            # (no-rewrite) layout, e.g. a mis-/under-configured ``source_layouts``
            # entry. Zero-padding it would fabricate a canonical row whose
            # gripper sits at the wrong index and whose true gripper dim is
            # dropped, with no error. Fail loud instead; zero-padding only
            # happens in the controlled 6→7 promotion path below where the
            # gripper is explicitly relocated.
            raise ValueError(
                f"transform_row: source family {source_family!r} is declared "
                f"as already-canonical (no rewrite) but a raw row has width "
                f"{raw_row.shape[-1]} != canonical width {canonical_total}. "
                f"This means the source's real arm/gripper layout disagrees "
                f"with its source_layouts/default_layout entry "
                f"(arm_dof={lay.arm_dof}, raw_gripper_idx={lay.raw_gripper_idx}). "
                f"Fix the MergeRecipe layout for this source instead of "
                f"promoting a short row blindly."
            )

        # In-scope rewrite: 6-DoF arm + grip at raw[6] → 7-DoF arm (dim 6 = 0)
        # + grip at canonical[7].
        if lay.arm_dof != 6 or self.canonical.arm_dof != 7:
            raise NotImplementedError(
                f"transform_row supports only 6→7 DoF promotion at this time; "
                f"got source arm_dof={lay.arm_dof}, canonical={self.canonical.arm_dof}"
            )
        out = np.zeros(canonical_total, dtype=raw_row.dtype)
        out[: lay.arm_dof] = raw_row[: lay.arm_dof]         # arm joints 0..5
        out[self.canonical.gripper_index_in_raw] = raw_row[lay.raw_gripper_idx]  # grip → dim 7
        # Dim 6 stays 0 (the zero-pad introduced by the promotion).
        return out
