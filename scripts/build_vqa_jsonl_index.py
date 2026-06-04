"""Convert large RoboInter-VQA JSON arrays to lazy JSONL + offset indexes.

The legacy VQA adapter can load small metadata files as JSON lists. The
Task_planning train metadata is multi-GB, so loading it in every distributed
rank expands into large Python object graphs and can destabilize launch before
the first step. This helper writes one JSON object per line plus a numpy int64
offset sidecar consumed by ``RoboInterVQADataset``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np


def iter_json_array_records(source_path: Path, *, chunk_size: int) -> Iterator[dict]:
    """Stream records from a top-level JSON array without materializing it."""
    decoder = json.JSONDecoder()
    buffer = ""
    cursor = 0
    reached_eof = False

    def read_more(handle) -> None:
        nonlocal buffer, reached_eof
        if reached_eof:
            return
        chunk = handle.read(chunk_size)
        if chunk:
            buffer += chunk
        else:
            reached_eof = True

    with source_path.open("r", encoding="utf-8") as source_handle:
        read_more(source_handle)
        while True:
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            if cursor < len(buffer):
                break
            if reached_eof:
                raise ValueError(f"{source_path} is empty; expected JSON array")
            buffer = ""
            cursor = 0
            read_more(source_handle)

        if buffer[cursor] != "[":
            raise ValueError(f"{source_path} top-level JSON must be an array")
        cursor += 1

        while True:
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    if reached_eof:
                        raise ValueError(f"{source_path} ended before closing array")
                    buffer = ""
                    cursor = 0
                    read_more(source_handle)
                    continue
                if buffer[cursor] == ",":
                    cursor += 1
                    continue
                if buffer[cursor] == "]":
                    return
                break

            if cursor > chunk_size:
                buffer = buffer[cursor:]
                cursor = 0

            while True:
                try:
                    record, end_cursor = decoder.raw_decode(buffer, cursor)
                    break
                except json.JSONDecodeError:
                    if reached_eof:
                        raise
                    read_more(source_handle)

            if not isinstance(record, dict):
                raise ValueError(
                    f"{source_path} contains {type(record).__name__}; expected dict records"
                )
            yield record
            cursor = end_cursor
            if cursor > chunk_size:
                buffer = buffer[cursor:]
                cursor = 0


def build_offsets_from_jsonl(jsonl_path: Path) -> np.ndarray:
    offsets: list[int] = []
    with jsonl_path.open("rb") as jsonl_handle:
        while True:
            offset = jsonl_handle.tell()
            line = jsonl_handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    if not offsets:
        raise ValueError(f"no records found in {jsonl_path}")
    return np.asarray(offsets, dtype=np.int64)


def validate_jsonl_offsets(jsonl_path: Path, offsets_path: Path) -> tuple[bool, str]:
    try:
        offsets = np.load(offsets_path, mmap_mode="r")
    except Exception as exc:
        return False, f"cannot load offsets: {type(exc).__name__}: {exc}"
    if offsets.ndim != 1:
        return False, f"offsets must be 1-D, got shape={offsets.shape}"
    if len(offsets) == 0:
        return False, "offsets are empty"

    expected_count = 0
    with jsonl_path.open("rb") as jsonl_handle:
        while True:
            offset = jsonl_handle.tell()
            line = jsonl_handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            if expected_count >= len(offsets):
                return False, "jsonl has more records than offsets"
            if int(offsets[expected_count]) != int(offset):
                return (
                    False,
                    f"offset mismatch at record {expected_count}: "
                    f"index={int(offsets[expected_count])}, jsonl={offset}",
                )
            expected_count += 1
    if expected_count != len(offsets):
        return False, f"offset count {len(offsets)} != jsonl records {expected_count}"
    return True, f"records={expected_count}"


def _source_fingerprint(source_json: Path) -> dict:
    """Cheap freshness fingerprint for the source JSON (size + mtime_ns)."""
    st = source_json.stat()
    return {
        "source": str(source_json),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _fingerprint_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_name(f"{output_jsonl.name}.source.json")


def write_source_fingerprint(output_jsonl: Path, source_json: Path) -> None:
    """Persist the source fingerprint next to the generated JSONL (atomic)."""
    fp_path = _fingerprint_path(output_jsonl)
    tmp_fp = fp_path.with_name(f"{fp_path.name}.tmp.{os.getpid()}")
    with tmp_fp.open("w", encoding="utf-8") as fp_handle:
        json.dump(_source_fingerprint(source_json), fp_handle)
    os.replace(tmp_fp, fp_path)


def check_source_fresh(output_jsonl: Path, source_json: Path) -> tuple[bool, str]:
    """Return whether the recorded fingerprint still matches ``source_json``.

    The JSONL+offsets can be internally consistent yet stale relative to a
    changed source; without this check the converter silently reuses the old
    JSONL. A missing/unreadable fingerprint is treated as not-fresh so reuse
    never happens on an unverifiable basis.
    """
    fp_path = _fingerprint_path(output_jsonl)
    if not fp_path.exists():
        return False, f"no source fingerprint at {fp_path}"
    try:
        with fp_path.open("r", encoding="utf-8") as fp_handle:
            recorded = json.load(fp_handle)
        current = _source_fingerprint(source_json)
    except Exception as exc:
        return False, f"cannot read fingerprint: {type(exc).__name__}: {exc}"
    for field in ("size", "mtime_ns"):
        if int(recorded.get(field, -1)) != int(current[field]):
            return (
                False,
                f"source changed ({field}: recorded={recorded.get(field)} "
                f"current={current[field]})",
            )
    return True, "source fingerprint matches"


def write_offsets(offsets_path: Path, offsets: np.ndarray) -> None:
    tmp_offsets = offsets_path.with_name(
        f"{offsets_path.name}.tmp.{os.getpid()}"
    )
    with tmp_offsets.open("wb") as offsets_handle:
        np.save(offsets_handle, offsets)
    os.replace(tmp_offsets, offsets_path)


def write_temp_offsets(offsets_path: Path, offsets: np.ndarray) -> Path:
    tmp_offsets = offsets_path.with_name(
        f"{offsets_path.name}.tmp.{os.getpid()}"
    )
    with tmp_offsets.open("wb") as offsets_handle:
        np.save(offsets_handle, offsets)
    return tmp_offsets


def convert_json_array_to_jsonl(
    source_json: Path,
    output_jsonl: Path,
    *,
    force: bool,
    chunk_size: int,
    progress_every: int,
) -> None:
    offsets_path = Path(f"{output_jsonl}.offsets.npy")
    # Before reusing an existing JSONL, confirm it is still fresh w.r.t.
    # source_json; require --force to regenerate (or override the check).
    if output_jsonl.exists() and not force:
        fresh, fresh_reason = check_source_fresh(output_jsonl, source_json)
        if not fresh:
            raise SystemExit(
                f"[stale] {output_jsonl} exists but is out of date with "
                f"{source_json}: {fresh_reason}.\n"
                "Re-run with --force to regenerate from the current source "
                "(this overwrites the JSONL + offsets + fingerprint)."
            )
    if output_jsonl.exists() and offsets_path.exists() and not force:
        valid, reason = validate_jsonl_offsets(output_jsonl, offsets_path)
        if valid:
            print(
                f"[skip] {output_jsonl} and {offsets_path} already exist; "
                f"{reason}"
            )
            return
        print(f"[reindex] existing offsets invalid: {reason}", flush=True)
        offsets = build_offsets_from_jsonl(output_jsonl)
        write_offsets(offsets_path, offsets)
        write_source_fingerprint(output_jsonl, source_json)
        print(f"[index] rebuilt {offsets_path}; records={len(offsets)}")
        return
    if output_jsonl.exists() and not force:
        offsets = build_offsets_from_jsonl(output_jsonl)
        write_offsets(offsets_path, offsets)
        write_source_fingerprint(output_jsonl, source_json)
        print(f"[index] rebuilt {offsets_path}; records={len(offsets)}")
        return

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = output_jsonl.with_name(f"{output_jsonl.name}.tmp.{os.getpid()}")
    offsets: list[int] = []
    # Snapshot the source fingerprint before reading so the persisted value
    # reflects the exact input consumed.
    source_fp = _source_fingerprint(source_json)
    try:
        with tmp_jsonl.open("wb") as output_handle:
            for record_index, record in enumerate(
                iter_json_array_records(source_json, chunk_size=chunk_size), start=1
            ):
                offsets.append(output_handle.tell())
                line = json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                output_handle.write(line)
                output_handle.write(b"\n")
                if progress_every > 0 and record_index % progress_every == 0:
                    print(
                        f"[progress] records={record_index} "
                        f"written_gb={output_handle.tell() / (1024 ** 3):.2f}",
                        flush=True,
                    )
        if not offsets:
            raise ValueError(f"no records converted from {source_json}")
        tmp_offsets = write_temp_offsets(offsets_path, np.asarray(offsets, dtype=np.int64))
        os.replace(tmp_jsonl, output_jsonl)
        os.replace(tmp_offsets, offsets_path)
        fp_path = _fingerprint_path(output_jsonl)
        tmp_fp = fp_path.with_name(f"{fp_path.name}.tmp.{os.getpid()}")
        with tmp_fp.open("w", encoding="utf-8") as fp_handle:
            json.dump(source_fp, fp_handle)
        os.replace(tmp_fp, fp_path)
        print(
            f"[done] jsonl={output_jsonl} offsets={offsets_path} "
            f"fingerprint={fp_path} records={len(offsets)}"
        )
    finally:
        if tmp_jsonl.exists():
            tmp_jsonl.unlink()
        tmp_offsets_path = offsets_path.with_name(
            f"{offsets_path.name}.tmp.{os.getpid()}"
        )
        if tmp_offsets_path.exists():
            tmp_offsets_path.unlink()
        tmp_fp_path = _fingerprint_path(output_jsonl).with_name(
            f"{_fingerprint_path(output_jsonl).name}.tmp.{os.getpid()}"
        )
        if tmp_fp_path.exists():
            tmp_fp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_json", type=Path)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Defaults to source_json with .jsonl suffix.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--progress-every", type=int, default=25_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = args.output_jsonl or args.source_json.with_suffix(".jsonl")
    convert_json_array_to_jsonl(
        args.source_json,
        output_jsonl,
        force=args.force,
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
