"""MetaReader for LeRobot v3.0 (meta/episodes/chunk-*/file-*.parquet)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .base import EpisodeMeta


def _maybe_decode_ascii_byte_array(seq):
    """Return decoded string if `seq` looks like a NUL-padded UTF-8 byte array
    (TFDS-style fixed-length string encoding used by OXE `language_table_*`
    sub-repos), else None.

    Faithful copy of the heuristic in
    ``src/adapters/lerobot_v30.py::_maybe_decode_ascii_byte_array`` so this
    read-only meta path accepts EXACTLY the same byte arrays as the training
    adapter (and rejects genuine low-id tokenizer output). Accept ONLY when the
    array structurally resembles a TFDS fixed-length string tensor:
      (a) every element in [0, 127] (ASCII range);
      (b) there is evidence of NUL padding (trailing zeros), OR the entire
          array is printable;
      (c) after stripping NULs, >= 90% of remaining bytes are printable ASCII
          (0x20..0x7E) or whitespace (\\t \\n \\r);
      (d) the decoded result is at least 2 chars long.
    Returns None on any failure — caller then fail-louds.
    """
    if seq is None or len(seq) == 0:
        return None
    try:
        ints = [int(x) for x in seq]
    except (TypeError, ValueError):
        return None
    if any(not (0 <= b <= 127) for b in ints):
        return None
    n_trailing_zeros = 0
    for b in reversed(ints):
        if b == 0:
            n_trailing_zeros += 1
        else:
            break
    printable_or_ws = lambda c: (0x20 <= c <= 0x7E) or c in (0x09, 0x0A, 0x0D)
    non_pad = ints[: len(ints) - n_trailing_zeros] if n_trailing_zeros > 0 else ints
    if not non_pad:
        return None
    all_printable = all(printable_or_ws(c) for c in non_pad)
    if n_trailing_zeros == 0 and not all_printable:
        return None
    n_printable = sum(1 for c in non_pad if printable_or_ws(c))
    if n_printable < 0.9 * len(non_pad):
        return None
    decoded = bytes(non_pad).decode("ascii", errors="replace").rstrip()
    if len(decoded) < 2:
        return None
    return decoded


def _normalize_tasks_col(val):
    """Merged oxe-auge_clean has heterogeneous `tasks` column types
    (list<string> for image datasets vs list<list<int>> for language_table
    tokenized tasks). Normalize to list<string> before concat.

    Aligned with the training adapter (``src/adapters/lerobot_v30.py``):
    tokenized ``list<int>`` task elements are decoded to text ONLY when they
    pass the ASCII-byte-array heuristic; any other ``list<int>`` element
    fail-louds instead of silently coercing to ``str(list(...))``, so this
    read-only probe never reports a digit-string task as normal while training
    would fail on the same shard. No env opt-out: this is a correctness probe.
    """
    if val is None:
        return None
    if isinstance(val, (list, np.ndarray)):
        out = []
        for item in val:
            if isinstance(item, (list, np.ndarray)):
                item_list = list(item) if isinstance(item, np.ndarray) else item
                decoded = _maybe_decode_ascii_byte_array(item_list)
                if decoded is not None:
                    out.append(decoded)
                else:
                    raise ValueError(
                        "v30 meta_reader: encountered a tokenized (list<int>) "
                        "task element that is not a decodable ASCII byte array. "
                        "Coercing it to str() would misrepresent the task as "
                        "natural language. Decode the tokens back to text in the "
                        "upstream dataset or exclude tokenized-task shards. "
                        "(This matches src/adapters/lerobot_v30.py fail-loud "
                        "semantics.)"
                    )
            elif item is not None:
                out.append(str(item))
            else:
                out.append("")
        return out
    return val


class V30MetaReader:
    """Reads concatenated meta/episodes/chunk-*/file-*.parquet shards.

    v3 episodes carry extra shard-pointer columns (data/chunk_index,
    dataset_from_index, videos/<cam>/from_timestamp, ...). Those are
    adapter-internal; for task discovery we only need episode_index,
    length, tasks. Shard pointers are preserved in EpisodeMeta.extra
    so that callers who build v3 adapters can forward them if needed
    (though our current adapter re-reads them itself).
    """

    def read(self, meta_root: Path) -> list[EpisodeMeta]:
        ep_root = Path(meta_root) / "episodes"
        if not ep_root.is_dir():
            raise FileNotFoundError(f"v3 meta/episodes dir missing at {ep_root}")
        paths = sorted(ep_root.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no episode parquets under {ep_root}")

        # Heterogeneous-schema-safe concat (see lerobot_v30 adapter for same
        # treatment): go through pandas so camera column unions and tasks
        # type mismatch (list<string> vs list<list<int>>) don't crash concat.
        dfs: list[pd.DataFrame] = []
        for p in paths:
            t = pq.read_table(p)
            # Drop per-episode stats columns early — they're large and unused.
            keep_cols = [c for c in t.schema.names if not c.startswith("stats/")]
            dfs.append(t.select(keep_cols).to_pandas())
        for df in dfs:
            if "tasks" in df.columns:
                df["tasks"] = df["tasks"].map(_normalize_tasks_col)
        df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

        out: list[EpisodeMeta] = []
        for _, row in df.iterrows():
            raw = row.to_dict()
            tasks_raw = raw.get("tasks")
            if isinstance(tasks_raw, (list, np.ndarray)):
                tasks = tuple(str(t) for t in tasks_raw if t)
            elif tasks_raw:
                tasks = (str(tasks_raw),)
            else:
                tasks = ()
            out.append(EpisodeMeta(
                episode_index=int(raw["episode_index"]),
                length=int(raw["length"]),
                tasks=tasks,
                source=raw.get("source"),   # v3 generally lacks this field
                extra={k: v for k, v in raw.items()
                       if k not in ("episode_index", "length", "tasks", "source")},
            ))
        out.sort(key=lambda e: e.episode_index)
        return out
