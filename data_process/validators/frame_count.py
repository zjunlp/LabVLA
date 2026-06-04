"""FrameCountValidator — verify mp4 frame count matches episodes.jsonl `length`.

Truncated mp4 or corrupt jsonl entries silently produce wrong-length
action chunks (clip-to-last-frame padding kicks in but there's no `is_pad`
signal because the mismatch is between LENGTH fields, not between delta_
timestamps). This validator samples N episodes per repo and decodes their
videos to count actual frames.

To keep runtime bounded (each mp4 decode is ~50-200 ms), we sample
`sample_size` random episodes per repo (default 20).

Multi-camera & v3 packed-video support:

* v2 layout (``videos/chunk-XXX/<camera>/episode_NNNNNN.mp4``): every camera
  subdir is checked (corruption in a non-first camera must not go silent).
* v3 layout (``videos/<camera>/chunk-XXX/file-YYY.mp4``): best-effort
  packed-shard verification. We can't pinpoint a single episode's frame range
  inside a packed shard, so we verify each camera has at least one shard that
  opens cleanly and whose frame count is at least the single-episode length.
  Per-episode mp4 lookup legitimately misses for v3, so the missing-mp4
  severity is LOW (the layout itself is not per-episode).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .base import Validator, ValidationIssue


class FrameCountValidator(Validator):
    name = "frame_count"
    description = (
        "Sample episodes per repo; verify mp4 actual frame count matches "
        "episodes.jsonl 'length' field (±1 frame tolerance)."
    )

    def __init__(self, sample_size: int = 20, tolerance: int = 1, **options):
        super().__init__(**options)
        self.sample_size = sample_size
        self.tolerance = tolerance

    def validate(self, repos: Iterable[Path]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for repo in repos:
            repo_path = Path(repo)
            if (repo_path / "meta" / "vqa_manifest.json").exists():
                # VQA-only repos have still images referenced by a manifest,
                # not LeRobot episode videos.
                continue

            # Probe codebase_version BEFORE checking `episodes.jsonl`: v3 repos
            # store episode metadata under meta/episodes/chunk-*/file-*.parquet
            # and never write a flat episodes.jsonl, so flagging HIGH "missing
            # episodes.jsonl" would be wrong. v3 gets a LOW informational issue.
            is_v3 = self._is_v3_repo(repo_path)
            ep_jsonl = repo_path / "meta" / "episodes.jsonl"
            if not ep_jsonl.exists():
                if is_v3:
                    issues.append(ValidationIssue(
                        severity="LOW",
                        repo=str(repo_path),
                        message=(
                            "v3 repo: per-episode frame-count check skipped "
                            "(no flat episodes.jsonl in v3 layout — episode "
                            "metadata lives under meta/episodes/chunk-*/)"
                        ),
                        context={"is_v3": True},
                    ))
                else:
                    issues.append(ValidationIssue(
                        severity="HIGH",
                        repo=str(repo_path),
                        message=f"missing episodes.jsonl at {ep_jsonl}",
                        context={"is_v3": False},
                    ))
                continue

            # v3 missing mp4 is LOW (per-episode lookup never applies to packed
            # layout); v2 missing mp4 is MED.
            missing_mp4_severity = "LOW" if is_v3 else "MED"

            # For v3, resolve each episode to the EXACT video shard holding its
            # frames (via meta/episodes parquet) instead of always decoding
            # file-000. None means metadata unavailable → first-shard heuristic.
            v3_shard_map = (
                self._build_v3_video_shard_map(repo_path) if is_v3 else None
            )

            ep_lines = [l for l in ep_jsonl.read_text().splitlines() if l.strip()]
            rng = random.Random(42)  # deterministic: same run → same sample
            n = min(self.sample_size, len(ep_lines))
            sampled = rng.sample(ep_lines, n) if n < len(ep_lines) else ep_lines

            for line in sampled:
                ep = json.loads(line)
                ei = int(ep["episode_index"])
                expected = int(ep["length"])
                # Scan ALL cameras (one (camera, mp4) tuple each); pass the v3
                # shard map so the v3 branch opens the CORRECT shard for ``ei``.
                mp4s = self._find_mp4s(
                    repo_path, ei, is_v3=is_v3, v3_shard_map=v3_shard_map,
                )
                # An authoritative shard map listing ``ei`` but no mp4 on disk
                # is a concrete missing-shard fault (the map only holds episodes
                # whose pointers resolved), so escalate above best-effort LOW.
                if not mp4s and v3_shard_map is not None and ei in v3_shard_map:
                    issues.append(ValidationIssue(
                        severity="MED",
                        repo=str(repo_path),
                        message=(
                            f"episode {ei}: meta/episodes references video "
                            f"shard(s) {v3_shard_map[ei]} but none exist on "
                            f"disk — missing v3 video shard"
                        ),
                        context={
                            "episode_index": ei,
                            "is_v3": True,
                            "referenced_shards": {
                                cam: list(idx)
                                for cam, idx in v3_shard_map[ei].items()
                            },
                        },
                    ))
                    continue
                if not mp4s:
                    issues.append(ValidationIssue(
                        severity=missing_mp4_severity,
                        repo=str(repo_path),
                        message=f"episode {ei}: no mp4 found for frame-count check",
                        context={"episode_index": ei, "is_v3": is_v3},
                    ))
                    continue
                for camera, mp4 in mp4s:
                    try:
                        actual = self._count_frames(mp4)
                    except Exception as e:
                        issues.append(ValidationIssue(
                            severity="HIGH",
                            repo=str(repo_path),
                            message=(
                                f"episode {ei} camera {camera!r}: mp4 decode "
                                f"failed: {e}"
                            ),
                            context={
                                "episode_index": ei,
                                "camera": camera,
                                "mp4": str(mp4),
                            },
                        ))
                        continue
                    # v3: `actual` is a packed shard's full frame count spanning
                    # many episodes, so only flag when STRICTLY LESS than this
                    # episode's length (shard can't hold it → truncated). v2 is
                    # strict ±tolerance (HIGH) since the mp4 is per-episode.
                    if is_v3:
                        if actual < expected:
                            # A shard resolved authoritatively from meta/episodes
                            # is the CORRECT one, so too-short → MED. Best-effort
                            # first-shard stays LOW (may be an unrelated shard).
                            mapped = (
                                v3_shard_map is not None
                                and ei in v3_shard_map
                            )
                            issues.append(ValidationIssue(
                                severity="MED" if mapped else "LOW",
                                repo=str(repo_path),
                                message=(
                                    f"episode {ei} camera {camera!r}: "
                                    f"{'mapped' if mapped else 'packed'} "
                                    f"shard has {actual} frames but episode "
                                    f"length={expected} — shard truncated"
                                    + ("" if mapped else " (best-effort: shard "
                                       "not resolved from meta/episodes)")
                                ),
                                context={
                                    "episode_index": ei,
                                    "camera": camera,
                                    "expected_frames": expected,
                                    "actual_frames": actual,
                                    "mp4": str(mp4),
                                    "is_v3": True,
                                    "shard_resolved_from_meta": mapped,
                                },
                            ))
                    else:
                        if abs(actual - expected) > self.tolerance:
                            issues.append(ValidationIssue(
                                severity="HIGH",
                                repo=str(repo_path),
                                message=(
                                    f"episode {ei} camera {camera!r}: jsonl "
                                    f"length={expected} but mp4 has {actual} "
                                    f"frames (diff={actual - expected})"
                                ),
                                context={
                                    "episode_index": ei,
                                    "camera": camera,
                                    "expected_frames": expected,
                                    "actual_frames": actual,
                                    "mp4": str(mp4),
                                },
                            ))
        return issues

    @staticmethod
    def _build_v3_video_shard_map(
        repo: Path,
    ) -> dict[int, dict[str, tuple[int, int]]] | None:
        """Map each episode → {camera: (video_chunk_idx, video_file_idx)}.

        v3 packs many episodes into a few large video shards
        (``videos/<camera>/chunk-XXX/file-YYY.mp4``). The per-episode pointer to
        the shard holding an episode's frames lives in
        ``meta/episodes/chunk-*/file-*.parquet`` as the per-camera
        ``videos/<camera>/chunk_index`` / ``file_index`` columns (mirrors
        ``src/dataset/lerobot_dataset.py`` video-path resolution). Without it,
        the v3 fallback decodes each camera's FIRST shard (``file-000``)
        regardless of the sampled episode — almost always the wrong shard.

        Returns ``None`` (caller falls back to the first-shard heuristic) when
        pyarrow is unavailable, the episodes dir is absent, or the pointer
        columns aren't present, so inputs lacking this metadata keep working.
        """
        ep_root = repo / "meta" / "episodes"
        if not ep_root.is_dir():
            return None
        try:
            import pyarrow.parquet as pq  # lazy: runtime dep, not import-time
        except Exception:
            return None
        paths = sorted(ep_root.rglob("*.parquet"))
        if not paths:
            return None

        out: dict[int, dict[str, tuple[int, int]]] = {}
        try:
            for p in paths:
                schema_names = pq.read_schema(p).names
                # Discover camera names from the pointer columns present.
                cams = [
                    c[len("videos/"):-len("/chunk_index")]
                    for c in schema_names
                    if c.startswith("videos/") and c.endswith("/chunk_index")
                ]
                if not cams:
                    continue
                want = ["episode_index"]
                for cam in cams:
                    want.append(f"videos/{cam}/chunk_index")
                    want.append(f"videos/{cam}/file_index")
                want = [c for c in want if c in schema_names]
                tbl = pq.read_table(p, columns=want).to_pydict()
                eis = tbl.get("episode_index")
                if eis is None:
                    continue
                for row in range(len(eis)):
                    ei = int(eis[row])
                    cam_map: dict[str, tuple[int, int]] = {}
                    for cam in cams:
                        ck = tbl.get(f"videos/{cam}/chunk_index")
                        fk = tbl.get(f"videos/{cam}/file_index")
                        if ck is None or fk is None:
                            continue
                        if ck[row] is None or fk[row] is None:
                            continue
                        cam_map[cam] = (int(ck[row]), int(fk[row]))
                    if cam_map:
                        out[ei] = cam_map
        except Exception:
            # Malformed metadata — degrade to the legacy heuristic rather
            # than crashing the whole validator (preflight already treats a
            # validator exception conservatively; we don't make it worse).
            return None
        return out or None

    @staticmethod
    def _is_v3_repo(repo: Path) -> bool:
        """Return True if `repo/meta/info.json` declares codebase_version >= 3.0.

        Treats unparseable / missing info.json as v2 (default behavior).
        """
        info_path = repo / "meta" / "info.json"
        if not info_path.exists():
            return False
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            return False
        version = str(info.get("codebase_version", "v2.1")).lstrip("v")
        # Best-effort major-version compare without distutils.
        try:
            major = int(version.split(".")[0])
        except (ValueError, IndexError):
            return False
        return major >= 3

    def _find_mp4(self, repo: Path, ei: int) -> Path | None:
        """Locate ONE per-episode mp4 (legacy single-camera helper).

        Kept for backward compatibility with tests that probe the finder
        directly (e.g. ``test_frame_count_validator_v3_finds_alternative_mp4_pattern``).
        Returns the first mp4 across all cameras; the validator's main
        loop now uses :meth:`_find_mp4s` to inspect every camera.
        """
        mp4s = self._find_mp4s(repo, ei, is_v3=self._is_v3_repo(repo))
        return mp4s[0][1] if mp4s else None

    def _find_mp4s(
        self, repo: Path, ei: int, *, is_v3: bool,
        v3_shard_map: dict[int, dict[str, tuple[int, int]]] | None = None,
    ) -> list[tuple[str, Path]]:
        """Locate per-camera mp4 paths for episode ``ei``.

        Returns a list of ``(camera_name, mp4_path)`` tuples — one entry
        per camera that has a relevant mp4 on disk. Empty list when the
        repo lacks a videos directory or no camera has a matching shard.

        v2 layout (per-episode):
            ``videos/chunk-XXX/<camera>/episode_NNNNNN.mp4``
            We accept three spellings: zero-padded ``episode_NNNNNN.mp4``
            (canonical), dash-spelled ``episode-NNNNNN.mp4`` (alt), and
            unpadded ``episode_<ei>.mp4`` (raw form).

        v3 layout (packed shards):
            ``videos/<camera>/chunk-XXX/file-YYY.mp4``
            When ``v3_shard_map`` is supplied (built from
            ``meta/episodes/chunk-*/*.parquet``), resolve the EXACT shard that
            holds ``ei`` per camera via its ``videos/<cam>/chunk_index`` /
            ``file_index`` pointers, so the check inspects the right shard
            instead of always decoding ``file-000``. When the map is
            unavailable, degrade to best-effort: the FIRST packed shard per camera.
        """
        videos_dir = repo / "videos"
        if not videos_dir.is_dir():
            return []

        # First pass: detect v2 per-episode mp4 across every camera subdir.
        # Done regardless of `is_v3` because some converted repos mis-declare
        # codebase_version but still emit per-episode mp4.
        per_episode_patterns = (
            f"episode_{ei:06d}.mp4",      # v2.1 canonical
            f"episode-{ei:06d}.mp4",      # alternate dash spelling
            f"episode_{ei}.mp4",          # un-padded
        )
        per_camera: dict[str, Path] = {}
        for pattern in per_episode_patterns:
            for p in videos_dir.rglob(pattern):
                cam = p.parent.name
                # First pattern hitting a camera wins; canonical zero-padded
                # is tried first, so it's preferred.
                if cam not in per_camera:
                    per_camera[cam] = p
        if per_camera:
            return sorted(per_camera.items())

        # No per-episode mp4 found. v3 packed layout:
        # ``videos/<camera>/chunk-XXX/file-YYY.mp4``.
        if is_v3:
            # Preferred: resolve the EXACT shard for this episode from the map.
            ep_shards = (v3_shard_map or {}).get(ei)
            if ep_shards:
                mapped: dict[str, Path] = {}
                for cam, (ci, fi) in ep_shards.items():
                    shard = (
                        videos_dir / cam
                        / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
                    )
                    if shard.exists():
                        mapped[cam] = shard
                # May be empty → caller treats a mapped-but-absent shard as a
                # concrete missing-shard fault.
                return sorted(mapped.items())

            # Best-effort fallback (no map / episode not in map): one shard/cam.
            packed: dict[str, Path] = {}
            for cam_dir in sorted(videos_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                # Recurse to find any mp4 (covers chunk-* nesting).
                shards = sorted(cam_dir.rglob("*.mp4"))
                if shards:
                    packed[cam_dir.name] = shards[0]
            return sorted(packed.items())

        return []

    def _count_frames(self, mp4: Path) -> int:
        import av
        with av.open(str(mp4)) as container:
            stream = container.streams.video[0]
            # Match lerobot_v21.py deadlock-avoidance: single-thread decode.
            stream.thread_type = "NONE"
            stream.thread_count = 1
            return sum(1 for _ in container.decode(stream))
