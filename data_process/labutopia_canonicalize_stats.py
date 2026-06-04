"""Generate a canonical-gripper stats.json for LabUtopia tasks.

Reads the source `meta/stats.json` from a LabUtopia v3.0 dataset and writes a
new stats file (default: `meta/stats_canonical_grip.json`) where:
  - All non-gripper dims are copied verbatim.
  - Gripper dim (index 7) q01/q99 are overridden to {0, max_width} so that
    the training-side `SnapGripperToEndpointsFn` snap output {0, max_width}
    maps cleanly through `NormalizeTransformFn(q01_q99)` to {-1, +1}, and
    the deploy-side q01/q99 inverse maps model output {-1, +1} back to
    {0, max_width} = physical width.

The matching `--external_stats_path` should point at the new file in the
launcher.

Usage:
  PYTHONPATH=src:. python -m data_process.labutopia_canonicalize_stats \\
      --src /all-flash-data/lerobot/.cache/LabUtopia/Level3_TransportBeaker/meta/stats.json \\
      --dst /all-flash-data/lerobot/.cache/LabUtopia/Level3_TransportBeaker/meta/stats_canonical_grip.json \\
      --gripper-dim 7 \\
      --max-width 0.04
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def patch_stats(stats: dict, gripper_dim: int, max_width: float) -> dict:
    """In-place patch q01/q99 for gripper dim across action/observation.state.

    Returns the modified dict (same object).
    """
    keys_present = 0
    keys_patched = 0
    for key in ("action", "action_abs", "observation.state"):
        if key not in stats:
            continue
        keys_present += 1
        entry = stats[key]
        # A key counts as patched only once BOTH q01 and q99 gripper endpoints
        # are written. A drifted entry (non-dict, or q01/q99 missing/not a list)
        # must fail loud, not be skipped while main() still writes "wrote": the
        # training-side SnapGripperToEndpointsFn would then read un-patched
        # endpoints and mis-normalize the gripper.
        if not isinstance(entry, dict):
            raise ValueError(
                f"stats[{key!r}] is {type(entry).__name__}, not a dict; cannot "
                f"patch gripper q01/q99. The source stats.json is malformed."
            )
        for stat_field, override_val in (("q01", 0.0), ("q99", max_width)):
            arr = entry.get(stat_field)
            if not isinstance(arr, list):
                raise ValueError(
                    f"stats[{key!r}][{stat_field!r}] is "
                    f"{type(arr).__name__ if arr is not None else 'missing'}, "
                    f"not a list; cannot patch gripper endpoint. Re-run "
                    f"`python -m data_process stats` so the source has proper "
                    f"q01/q99 quantile lists before canonicalizing."
                )
            if gripper_dim < 0 or gripper_dim >= len(arr):
                raise ValueError(
                    f"gripper_dim={gripper_dim} out of bounds for "
                    f"{key}.{stat_field} (len={len(arr)})"
                )
            arr[gripper_dim] = float(override_val)
        keys_patched += 1
    if keys_present == 0:
        raise ValueError(
            "patch_stats found none of the expected v3 keys (action, action_abs, "
            "observation.state) in stats. This script is designed for v3.0 datasets. "
            "If using v2.1 legacy stats with keys 'state'/'actions', you must first "
            "rename them to v3 keys before running canonicalize_stats."
        )
    # Defensive: keys_present>0 but nothing patched would mean every present key
    # was skipped — impossible now (each present key either patches both fields
    # or raises above), but guard anyway so we never claim success on no-op.
    if keys_patched == 0:
        raise ValueError(
            "patch_stats found expected v3 keys but patched no gripper endpoint; "
            "refusing to write a stats file that claims canonicalization without "
            "actually applying it."
        )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(prog="labutopia_canonicalize_stats")
    ap.add_argument("--src", required=True, type=Path, help="Source stats.json")
    ap.add_argument("--dst", required=True, type=Path, help="Output stats path")
    ap.add_argument("--gripper-dim", type=int, default=7)
    ap.add_argument("--max-width", type=float, default=0.04)
    args = ap.parse_args()

    if not args.src.exists():
        print(f"ERROR: source stats.json not found: {args.src}", file=sys.stderr)
        return 1

    with open(args.src) as f:
        stats = json.load(f)

    patched = patch_stats(stats, args.gripper_dim, args.max_width)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w") as f:
        json.dump(patched, f, indent=2)

    # Sanity print
    for key in ("action", "action_abs", "observation.state"):
        if key not in patched:
            continue
        e = patched[key]
        q01 = e.get("q01", [None] * 8)
        q99 = e.get("q99", [None] * 8)
        if (
            args.gripper_dim < len(q01)
            and args.gripper_dim < len(q99)
        ):
            print(
                f"  {key} gripper q01={q01[args.gripper_dim]}, "
                f"q99={q99[args.gripper_dim]}"
            )
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
