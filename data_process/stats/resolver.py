"""Stats resolution — find a usable `meta/stats.json` for any source_root.

Fallback chain, applied in order:

    1. <source_root>/meta/stats.json                    (own stats — ideal)
    2. <parent_root>/meta/stats.json                    (Collection parent stats)
    3. external_stats_map[repo_id] = <path>             (user-supplied override)
    4. raise StatsResolutionError                       (fail fast, don't train
                                                         without normalization)

`repo_id` is the top-level name a user wrote in launch-script REPO_IDS
(e.g. "RoboCOIN"). `source_root` is the actual LeRobot-repo directory
(for Collection this is a sub-directory of the repo_id).

`parent_root` is used only for Collection sub-repos — i.e. when
source_root != data_root/repo_id.

A successful resolution returns a **path**, not the loaded dict. Callers
decide how to read it (so we don't double-parse).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class StatsResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedStats:
    path: Path
    source: str             # "own" | "parent" | "external_map"


def resolve_stats(
    source_root: Path,
    repo_id: str,
    data_root: Path | None = None,
    external_stats_map: dict[str, str] | None = None,
) -> ResolvedStats:
    """Search the fallback chain. Raises StatsResolutionError on no hit.

    Args:
        source_root: actual LeRobot repo path (for Collection sub-repos this
            is the sub-directory; for FlatRepo this is data_root/repo_id).
        repo_id: top-level name used in launch-script REPO_IDS. Used to
            index external_stats_map.
        data_root: parent of repo_id. Used to construct parent_root for
            Collection sub-repos. Optional — if None we derive as
            source_root's grandparent when source_root != data_root/repo_id.
        external_stats_map: user-supplied map from launch-script CLI
            (--external_stats_map="repo=path,...").
    """
    source_root = Path(source_root)

    # Level 1: own stats.
    own = source_root / "meta" / "stats.json"
    if own.exists():
        return ResolvedStats(path=own, source="own")

    # Level 2: Collection parent stats (parent = data_root/repo_id).
    if data_root is not None:
        parent_root = Path(data_root) / repo_id
        # Only fall back to the parent's stats when source_root lives under (or
        # is) parent_root, else a mis-paired (repo_id, source_root) would bind
        # an unrelated repo's stats.
        if parent_root != source_root and source_root.is_relative_to(parent_root):
            parent_stats = parent_root / "meta" / "stats.json"
            if parent_stats.is_file():
                return ResolvedStats(path=parent_stats, source="parent")

    # Level 3: external_stats_map.
    if external_stats_map and repo_id in external_stats_map:
        # Anchor before touching disk so resolution never depends on the process
        # CWD and any ``..``/symlink components are normalized.
        ext_path = Path(external_stats_map[repo_id]).expanduser().resolve()
        # Require an actual file: a directory or ``Path("")``==CWD passes .exists().
        if ext_path.is_file():
            return ResolvedStats(path=ext_path, source="external_map")
        raise StatsResolutionError(
            f"external_stats_map[{repo_id!r}] = {ext_path} is not a file."
        )

    # Level 4: no stats found.
    raise StatsResolutionError(
        f"No stats.json reachable for repo_id={repo_id!r} "
        f"(source_root={source_root}). Searched: "
        f"(a) {source_root}/meta/stats.json, "
        f"(b) {data_root}/{repo_id}/meta/stats.json (parent), "
        f"(c) external_stats_map[{repo_id!r}]. "
        f"Provide --external_stats_map=\"{repo_id}=/path/to/stats.json\" "
        f"or compute stats via `python -m data_process stats --dataset ...`."
    )


def parse_external_stats_map(arg: str | None) -> dict[str, str]:
    """Parse '--external_stats_map' CLI form: 'repo1=path1,repo2=path2'.

    Rejects malformed entries rather than letting them resolve to a bogus "hit":
    an empty key, an empty/blank path (which becomes ``Path("")``==CWD with a
    True ``.exists()``), or a duplicate repo key (silent last-write-wins) all
    raise. The stored path must be absolute so resolution doesn't depend on the
    launching CWD; ``~`` is expanded and a relative path is a hard error.
    """
    if not arg:
        return {}
    out: dict[str, str] = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--external_stats_map entry must be 'repo=path', got {item!r}"
            )
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(
                f"--external_stats_map entry has an empty repo key: {item!r}"
            )
        if not v:
            raise ValueError(
                f"--external_stats_map[{k!r}] has an empty/blank path."
            )
        if k in out:
            raise ValueError(
                f"--external_stats_map has a duplicate repo key {k!r} "
                f"(previous={out[k]!r}, new={v!r}); refusing silent "
                f"last-write-wins."
            )
        anchored = Path(v).expanduser()
        if not anchored.is_absolute():
            raise ValueError(
                f"--external_stats_map[{k!r}] path must be absolute, got "
                f"{v!r}. A relative path resolves differently depending on the "
                f"launching CWD; pass an absolute path (e.g. $(realpath {v}))."
            )
        out[k] = str(anchored)
    return out
