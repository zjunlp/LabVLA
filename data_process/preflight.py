"""Pre-flight data integrity check.

Composes validators + stats-resolver into a single gate that a launch
script runs BEFORE `accelerate launch`. Exit 0 = training may proceed;
exit 1 = HIGH/CRIT issue found or stats unresolvable — abort.

Usage:
    python -m data_process preflight \\
        --repos RoboCOIN robointer_droid_clean \\
        --data_root /all-flash-data/Pretrain_Data \\
        --external_stats_map "RoboCOIN=/all-flash-data/Pretrain_Data/robocoin_clean/meta/stats.json"

Checks performed:
    V1 (topology):     each repo_id auto-detects as FlatRepo or Collection
    V2 (meta readable): per sub-repo (Collection) or top-level (Flat),
                        meta/info.json + episodes parseable
    V3 (stats):        resolve_stats() succeeds per source_root
    V4 (validators):   FPS / timestamp / frame-count — HIGH blocks
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from data_process.validators import (
    FPSConsistencyValidator, TimestampColumnValidator, FrameCountValidator,
    ValidationIssue,
)


_SEVERITY_ORDER = {"LOW": 0, "MED": 1, "HIGH": 2, "CRIT": 3}


def _is_vqa_repo(root: Path) -> bool:
    return (root / "meta" / "vqa_manifest.json").exists()


def _check_vqa_manifest(repo_id: str, root: Path) -> list[ValidationIssue]:
    """Validate the lightweight VQA-only repo contract."""
    issues: list[ValidationIssue] = []
    manifest_path = root / "meta" / "vqa_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        return [
            ValidationIssue(
                severity="HIGH",
                repo=repo_id,
                message=f"cannot parse vqa_manifest.json: {e}",
                context={"path": str(manifest_path)},
            )
        ]

    json_files = manifest.get("json_files")
    if not isinstance(json_files, list) or not json_files:
        issues.append(ValidationIssue(
            severity="HIGH",
            repo=repo_id,
            message="vqa_manifest.json must contain a non-empty json_files list",
            context={"path": str(manifest_path)},
        ))
        return issues

    missing = []
    bad_jsonl_offsets = []
    for rel in json_files:
        if not isinstance(rel, str) or not rel:
            issues.append(ValidationIssue(
                severity="HIGH",
                repo=repo_id,
                message=f"vqa_manifest.json contains invalid json_files entry: {rel!r}",
                context={"path": str(manifest_path)},
            ))
            continue
        meta_path = root / rel
        if not meta_path.is_file():
            missing.append(rel)
            continue
        if rel.endswith(".jsonl"):
            offsets_path = Path(f"{meta_path}.offsets.npy")
            if not offsets_path.is_file():
                bad_jsonl_offsets.append(f"{rel}: missing {offsets_path.name}")
            else:
                try:
                    import numpy as _np
                    offsets = _np.load(offsets_path, mmap_mode="r")
                    if getattr(offsets, "ndim", None) != 1 or len(offsets) <= 0:
                        bad_jsonl_offsets.append(
                            f"{rel}: invalid offsets shape={getattr(offsets, 'shape', None)}"
                        )
                except Exception as e:
                    bad_jsonl_offsets.append(f"{rel}: cannot read offsets: {e}")

    if missing:
        issues.append(ValidationIssue(
            severity="HIGH",
            repo=repo_id,
            message=f"vqa_manifest.json references missing metadata files: {missing[:5]}",
            context={"path": str(manifest_path), "missing_count": len(missing)},
        ))

    if bad_jsonl_offsets:
        issues.append(ValidationIssue(
            severity="HIGH",
            repo=repo_id,
            message=f"vqa_manifest.json references JSONL files with bad offset indexes: {bad_jsonl_offsets[:5]}",
            context={"path": str(manifest_path), "bad_offset_count": len(bad_jsonl_offsets)},
        ))

    logging.info(
        f"[preflight] {repo_id}: VQA-only repo, manifest entries={len(json_files)}"
    )
    return issues


def _check_topology_and_meta(repo_id: str, root: Path) -> list[ValidationIssue]:
    """Verify the repo is a flat LeRobot dir (v2.x or v3.x) with readable meta.

    After removal of CollectionTopology, every supported top-level repo is a
    flat LeRobot repo: `meta/info.json` at root, `codebase_version` in v2.x
    or v3.x. Multi-folder datasets (oxe-auge, RoboCOIN) are pre-merged via
    `data_process/merge_v3/build_*_clean.py` before training.
    """
    issues: list[ValidationIssue] = []

    if _is_vqa_repo(root):
        return _check_vqa_manifest(repo_id, root)

    from data_process.meta_reader import read_episode_meta
    from data_process.meta_reader.registry import detect_codebase_version

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        issues.append(ValidationIssue(
            severity="HIGH", repo=repo_id,
            message=f"meta/info.json missing; not a LeRobot repo",
            context={"root": str(root)},
        ))
        return issues

    try:
        version = detect_codebase_version(root)
    except Exception as e:
        issues.append(ValidationIssue(
            severity="HIGH", repo=repo_id,
            message=f"cannot detect codebase_version: {e}",
            context={"root": str(root)},
        ))
        return issues

    logging.info(f"[preflight] {repo_id}: flat LeRobot repo, codebase_version={version}")

    try:
        episodes = read_episode_meta(root)
        if not episodes:
            issues.append(ValidationIssue(
                severity="HIGH", repo=repo_id,
                message="repo has 0 episodes",
                context={"root": str(root)},
            ))
    except Exception as e:
        issues.append(ValidationIssue(
            severity="HIGH", repo=repo_id,
            message=f"cannot read meta: {e}",
            context={"root": str(root)},
        ))
    return issues


def _check_stats_resolution(
    repo_id: str, root: Path, data_root: Path,
    external_stats_map: dict[str, str],
    schema=None,
    action_mode: str = "",
) -> list[ValidationIssue]:
    """Probe stats resolution for the flat top-level repo.

    Beyond checking that stats.json is reachable, parse it and verify it carries
    at least one expected statistic shape (q01/q99, mean/std, or min/max) — a
    corrupted/truncated file would otherwise blow up at training launch.

    Optional ``schema`` enables shape + NaN/Inf checks: each canonical entry's
    arrays must have length == sum(state_dims) for ``observation.state`` and
    sum(action_dims) for ``action`` / ``action_abs``, and any NaN/Inf is HIGH
    (it would normalize to NaN and silently train on garbage). Without
    ``schema``, behavior is unchanged.

    Optional ``action_mode``: when ``"abs"`` and the parsed stats lacks a usable
    ``action_abs`` dict, emit HIGH. Otherwise ``hydrate_all`` raises KeyError at
    train step 0 (src/transforms/core.py), after paying for env + dataset spin-up.
    """
    import json as _json
    import math as _math
    issues: list[ValidationIssue] = []
    if _is_vqa_repo(root):
        logging.info(f"[preflight] {repo_id}: VQA-only repo; stats check skipped")
        return issues
    from data_process.stats.resolver import resolve_stats, StatsResolutionError
    try:
        resolved = resolve_stats(
            source_root=root,
            repo_id=repo_id,
            data_root=data_root,
            external_stats_map=external_stats_map,
        )
        logging.info(
            f"[preflight] {repo_id}: stats from {resolved.source} → {resolved.path}"
        )
    except StatsResolutionError as e:
        issues.append(ValidationIssue(
            severity="HIGH", repo=repo_id,
            message=f"no reachable stats: {e}",
            context={"root": str(root)},
        ))
        return issues
    # Parse + validate resolved stats.json.
    try:
        with open(resolved.path) as _f:
            parsed = _json.load(_f)
    except (OSError, ValueError) as e:
        issues.append(ValidationIssue(
            severity="HIGH", repo=repo_id,
            message=f"stats.json exists but failed to parse: {e}",
            context={"path": str(resolved.path)},
        ))
        return issues
    # Spot-check at least one canonical entry has usable fields.
    canonicals = ("observation.state", "action", "action_abs")
    touched = [c for c in canonicals if isinstance(parsed.get(c), dict)]
    if not touched:
        issues.append(ValidationIssue(
            severity="HIGH", repo=repo_id,
            message=(
                f"stats.json parsed but contains none of expected canonical "
                f"keys {canonicals}"
            ),
            context={"path": str(resolved.path),
                     "keys": list(parsed.keys())[:10]},
        ))
        return issues

    # action_mode=abs requires stats['action_abs'] (a dict); else the failure is
    # a KeyError in hydrate_all at train step 0, after env spin-up + NCCL
    # handshake + dataset construction. Catch it here.
    if action_mode == "abs" and not isinstance(parsed.get("action_abs"), dict):
        issues.append(ValidationIssue(
            severity="HIGH", repo=repo_id,
            message=(
                "action_mode='abs' requires stats['action_abs'] but it is "
                "missing or not a dict. Re-run "
                "`python -m data_process stats --dataset <root> --schema <name>` "
                "(current pipeline writes both 'action' and 'action_abs' "
                "since the dual-accumulator change in compute.py)."
            ),
            context={"path": str(resolved.path),
                     "action_abs_present": "action_abs" in parsed,
                     "action_abs_type": type(parsed.get("action_abs")).__name__,
                     "available_keys": [k for k in parsed.keys()
                                        if not k.startswith("_")][:10]},
        ))
    for c in touched:
        entry = parsed[c]
        has_q = "q01" in entry and "q99" in entry
        has_mean = "mean" in entry and "std" in entry
        has_minmax = "min" in entry and "max" in entry
        if not (has_q or has_mean or has_minmax):
            issues.append(ValidationIssue(
                severity="HIGH", repo=repo_id,
                message=(
                    f"stats.json[{c!r}] missing any usable statistic "
                    f"(q01/q99, mean/std, or min/max)"
                ),
                context={"path": str(resolved.path),
                         "available_keys": list(entry.keys())},
            ))

    # Schema-aware shape + NaN/Inf checks over every numeric stats array
    # (q01/q99, mean/std, min/max). When schema is None we skip the shape check.
    expected_dim_for: dict[str, int] = {}
    if schema is not None:
        try:
            state_total = int(sum(schema.state_dims))
            action_total = int(sum(schema.action_dims))
        except (AttributeError, TypeError):
            state_total = action_total = -1  # signal "schema is malformed"
        if state_total >= 0 and action_total >= 0:
            expected_dim_for["observation.state"] = state_total
            expected_dim_for["action"] = action_total
            expected_dim_for["action_abs"] = action_total

    # NaN / Inf check runs whether or not schema is supplied — catastrophic
    # data corruption is an issue at any layer.
    for c in touched:
        entry = parsed[c]
        for stats_field in ("mean", "std", "q01", "q99", "min", "max"):
            arr = entry.get(stats_field)
            if not isinstance(arr, list):
                continue
            try:
                bad_idx = next(
                    (i for i, v in enumerate(arr)
                     if not isinstance(v, (int, float))
                     or isinstance(v, bool)
                     or _math.isnan(float(v))
                     or _math.isinf(float(v))),
                    None,
                )
            except (TypeError, ValueError):
                bad_idx = 0
            if bad_idx is not None:
                bad_val = arr[bad_idx] if bad_idx < len(arr) else None
                issues.append(ValidationIssue(
                    severity="HIGH", repo=repo_id,
                    message=(
                        f"stats.json[{c!r}][{stats_field!r}] contains "
                        f"NaN/Inf/non-numeric at index {bad_idx} "
                        f"(value={bad_val!r}). Training on these would "
                        f"propagate NaN through normalize/denormalize."
                    ),
                    context={"path": str(resolved.path),
                             "key": c,
                             "stats_field": stats_field,
                             "index": bad_idx},
                ))

            # Schema-aware length check (only when schema supplied and
            # this canonical key has a known expected dim).
            if c in expected_dim_for and isinstance(arr, list):
                expected = expected_dim_for[c]
                if expected >= 0 and len(arr) != expected:
                    issues.append(ValidationIssue(
                        severity="HIGH", repo=repo_id,
                        message=(
                            f"stats.json[{c!r}][{stats_field!r}] length "
                            f"{len(arr)} does not match schema's "
                            f"expected dim {expected} (sum of "
                            f"{'state_dims' if c == 'observation.state' else 'action_dims'})."
                        ),
                        context={"path": str(resolved.path),
                                 "key": c,
                                 "stats_field": stats_field,
                                 "actual_len": len(arr),
                                 "expected_dim": expected},
                    ))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="data_process preflight",
        description="Gate training launch on data-integrity invariants.",
    )
    parser.add_argument("--repos", nargs="+", required=True,
                        help="Top-level repo_ids under --data_root.")
    parser.add_argument("--data_root", required=True,
                        help="Parent directory containing every --repos entry.")
    parser.add_argument("--external_stats_map", default=None,
                        help="repo_id=path,repo_id=path (fallback for repos "
                             "that have no own or parent stats.json).")
    parser.add_argument("--skip_validators", action="store_true",
                        help="Skip FPS/timestamp/frame-count validators (topology + "
                             "stats-resolution still run).")
    parser.add_argument("--sample_size", type=int, default=10,
                        help="FrameCountValidator sample size per repo.")
    parser.add_argument(
        "--schema", default=None,
        help=(
            "Optional --dataset_schema-style spec (registered name, dotted "
            "module:ATTR, or absolute /path/file.py[:ATTR]). When set, the "
            "stats-resolution check additionally verifies that every stats "
            "array length matches sum(state_dims) / sum(action_dims) and "
            "rejects NaN/Inf entries. Default = None (legacy behavior, "
            "stats key/shape spot-check only)."
        ),
    )
    parser.add_argument(
        "--schema_per_repo", default=None,
        help=(
            "Optional `repo_id=spec,repo_id=spec` map letting different "
            "repos use different schemas (e.g. mixed-robot pretrain). "
            "Takes precedence over --schema for matched repos."
        ),
    )
    parser.add_argument(
        "--action_mode", default="",
        help=(
            "Optional. When 'abs', preflight requires stats['action_abs'] "
            "to exist as a dict for every repo (matches "
            "src/transforms/core.py hydrate_all enforcement). Without this, "
            "missing action_abs only surfaces at train step 0."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[preflight] %(message)s",
    )

    data_root = Path(args.data_root)
    from data_process.stats.resolver import parse_external_stats_map
    ext_map = parse_external_stats_map(args.external_stats_map)

    repo_roots = {r: data_root / r for r in args.repos}
    missing_roots = [r for r, p in repo_roots.items() if not p.exists()]
    if missing_roots:
        for r in missing_roots:
            logging.error(f"[HIGH] {r}: path {repo_roots[r]} does not exist")
        sys.exit(1)

    # Optional per-repo schema map for the schema-aware stats check; the single
    # --schema flag is the default for repos not listed in --schema_per_repo.
    schema_map: dict[str, object] = {}
    default_schema = None
    if args.schema:
        try:
            from schema import resolve as _resolve_schema
            default_schema = _resolve_schema(args.schema)
        except Exception as e:
            logging.warning(
                "[preflight] --schema=%r failed to resolve: %r — schema-aware "
                "checks disabled for repos without a per-repo override.",
                args.schema, e,
            )
    if args.schema_per_repo:
        try:
            from schema import resolve as _resolve_schema
            for token in args.schema_per_repo.split(","):
                token = token.strip()
                if not token:
                    continue
                if "=" not in token:
                    raise ValueError(
                        f"--schema_per_repo entry {token!r} must be repo_id=spec"
                    )
                repo_id, spec = token.split("=", 1)
                schema_map[repo_id.strip()] = _resolve_schema(spec.strip())
        except Exception as e:
            logging.warning(
                "[preflight] --schema_per_repo failed to parse / resolve: %r",
                e,
            )

    all_issues: list[ValidationIssue] = []

    # V1 + V2
    logging.info(f"=== [V1/V2] Topology + meta readability ===")
    for r, root in repo_roots.items():
        all_issues.extend(_check_topology_and_meta(r, root))

    # V3
    logging.info(f"=== [V3] Stats resolution ===")
    for r, root in repo_roots.items():
        if _is_vqa_repo(root):
            logging.info(f"[preflight] {r}: VQA-only repo; LeRobot stats check skipped")
            continue
        # Bind the repo's OWN schema as the stats-check baseline. Precedence:
        # per-repo override > global --schema > auto-discover from disk. Without
        # this, a missing --schema skipped the shape check, letting stats built
        # against a wrong/stale schema pass. ``discover_schema`` is the same
        # resolver the training adapter uses, so preflight enforces what
        # training will. Discovery failure is non-fatal (NaN/Inf scan still runs).
        repo_schema = schema_map.get(r)
        if repo_schema is None:
            repo_schema = default_schema
        if repo_schema is None:
            try:
                from schema import discover_schema as _discover_schema
                repo_schema = _discover_schema(root)
                logging.info(
                    "[preflight] %s: stats baseline schema auto-discovered "
                    "(schema_id=%s)", r, getattr(repo_schema, "schema_id", "?"),
                )
            except Exception as e:
                logging.warning(
                    "[preflight] %s: could not auto-discover schema (%r) — "
                    "stats shape check skipped; NaN/Inf scan still runs.", r, e,
                )
                repo_schema = None
        all_issues.extend(_check_stats_resolution(
            r, root, data_root, ext_map, schema=repo_schema,
            action_mode=args.action_mode,
        ))

    # V4
    if not args.skip_validators:
        logging.info(f"=== [V4] Cross-repo validators ===")
        roots_list = [root for root in repo_roots.values() if not _is_vqa_repo(root)]
        if not roots_list:
            logging.info("[preflight] no LeRobot repos for FPS/timestamp/frame-count validators")
            roots_list = []

        # A validator that RAISES is a check FAILURE, not warning-and-continue:
        # a crash while scanning a genuinely broken dataset must not let
        # preflight exit 0, so we record a HIGH issue (forces final gate exit 1).
        _validators = [
            ("FPSConsistency", FPSConsistencyValidator()),
            ("TimestampColumn", TimestampColumnValidator()),
            ("FrameCount", FrameCountValidator(sample_size=args.sample_size)),
        ]
        for _name, _validator in _validators:
            try:
                all_issues.extend(_validator.validate(roots_list))
            except Exception as e:
                logging.error(f"[preflight] {_name} validator raised: {e!r}")
                all_issues.append(ValidationIssue(
                    severity="HIGH", repo="<all>",
                    message=(
                        f"{_name} validator raised an exception while running "
                        f"({e!r}); treating as a check failure rather than "
                        f"silently passing."
                    ),
                    context={"validator": _name, "error": repr(e)},
                ))

    # Report
    if not all_issues:
        logging.info("=== OK — no issues ===")
        sys.exit(0)

    all_issues.sort(key=lambda i: -_SEVERITY_ORDER.get(i.severity, -1))
    for issue in all_issues:
        logging.info(f"[{issue.severity}] {issue.repo}: {issue.message}")

    has_blocker = any(i.severity in ("HIGH", "CRIT") for i in all_issues)
    if has_blocker:
        logging.error("=== FAIL — HIGH/CRIT issues found ===")
        sys.exit(1)
    logging.info("=== PASS (with MED/LOW warnings) ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
