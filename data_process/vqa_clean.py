"""Clean RoboInter-VQA metadata by removing records with bad image references.

The cleaner is CPU-only. It validates the VQA manifest, metadata records,
referenced image paths, and full PIL image decodability, then writes a clean
JSONL mirror plus a manifest that can replace ``meta/vqa_manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


_USER_ROLES = {"human", "huamn", "user"}
_ASSISTANT_ROLES = {"gpt", "assistant"}


@dataclass
class RecordCheck:
    ok: bool
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceStats:
    source: str
    output: str
    seen: int = 0
    kept: int = 0
    dropped: int = 0
    reasons: Counter = field(default_factory=Counter)


def _extract_qa(record: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return question, answer
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).lower()
        value = turn.get("value", "")
        if role in _USER_ROLES and not question:
            question = str(value).replace("<image>\n", "").replace("<image>", "").strip()
        elif role in _ASSISTANT_ROLES and not answer:
            answer = str(value).strip()
    return question, answer


def _validate_images(value: Any) -> tuple[list[str], str | None]:
    """Strictly normalize the ``images`` field.

    The runtime VQA adapter selects the first/last entry from the *raw* list, so
    every element must be a non-empty string: silently dropping non-string
    elements would let a record with a ``null`` / numeric / object image
    reference survive cleaning and still trip the adapter at train time. Returns
    ``(clean_images, reason)`` with ``reason`` None on success, else a short
    invalid-reason tag. Valid all-string inputs pass through verbatim.
    """
    if isinstance(value, str):
        return ([value], None) if value else ([], "empty_image_entry")
    if isinstance(value, list):
        clean: list[str] = []
        for x in value:
            if not isinstance(x, str):
                return [], "non_string_image_entry"
            if not x:
                return [], "empty_image_entry"
            clean.append(x)
        return clean, (None if clean else "no_images")
    if value is None:
        return [], "no_images"
    return [], "non_string_image_entry"


def _resolve_relative_image(root: Path, rel: str) -> Path:
    if not rel or rel.strip() != rel:
        raise ValueError(f"invalid image path string: {rel!r}")
    candidate = Path(rel)
    if candidate.is_absolute():
        raise ValueError(f"absolute image path is not allowed: {rel!r}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"parent traversal image path is not allowed: {rel!r}")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"image path escapes VQA root: {rel!r}") from exc
    return resolved


def _check_image_file(path: Path, *, decode_images: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing image: {path}")
    if path.stat().st_size <= 0:
        raise OSError(f"empty image file: {path}")
    if not decode_images:
        return
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
            width, height = rgb.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise OSError(f"unreadable image: {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise OSError(f"invalid image dimensions: {path}: {width}x{height}")


@lru_cache(maxsize=500_000)
def _check_image_file_cached(path_str: str, decode_images: bool) -> None:
    _check_image_file(Path(path_str), decode_images=decode_images)


def check_vqa_record(
    record: Any,
    *,
    vqa_root: Path,
    source_rel: str,
    local_index: int,
    check_all_images: bool = True,
    decode_images: bool = True,
) -> RecordCheck:
    if not isinstance(record, dict):
        return RecordCheck(False, "record_type", {"type": type(record).__name__})

    question, answer = _extract_qa(record)
    if not question:
        return RecordCheck(False, "empty_question", {"id": record.get("id")})
    if not answer:
        return RecordCheck(False, "empty_answer", {"id": record.get("id")})

    images, image_reason = _validate_images(record.get("images"))
    if image_reason is not None:
        return RecordCheck(False, image_reason, {"id": record.get("id")})

    images_to_check = images if check_all_images else [images[-1] if len(images) > 1 else images[0]]
    for rel in images_to_check:
        try:
            path = _resolve_relative_image(vqa_root, rel)
            _check_image_file_cached(str(path), bool(decode_images))
        except Exception as exc:
            return RecordCheck(
                False,
                type(exc).__name__,
                {
                    "id": record.get("id"),
                    "source": source_rel,
                    "local_index": local_index,
                    "image": rel,
                    "error": str(exc),
                },
            )
    return RecordCheck(True)


def _iter_json_records(path: Path) -> Iterable[tuple[int, Any, str | None]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        yield 0, None, f"json_parse:{exc}"
        return
    if not isinstance(data, list):
        yield 0, None, f"json_top_level:{type(data).__name__}"
        return
    for idx, record in enumerate(data):
        yield idx, record, None


def _iter_jsonl_records(path: Path) -> Iterable[tuple[int, Any, str | None]]:
    with path.open("rb") as handle:
        for idx, raw in enumerate(handle):
            if not raw.strip():
                yield idx, None, "empty_jsonl_line"
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                yield idx, None, f"jsonl_parse:{exc}"
                continue
            yield idx, record, None


def _iter_records(path: Path) -> Iterable[tuple[int, Any, str | None]]:
    if path.suffix == ".json":
        yield from _iter_json_records(path)
    elif path.suffix == ".jsonl":
        yield from _iter_jsonl_records(path)
    else:
        yield 0, None, f"unsupported_metadata_suffix:{path.suffix}"


def _clean_output_rel(output_prefix: str, source_rel: str) -> str:
    rel = Path(source_rel)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "meta" and parts[1] == "labvla_clean":
        rel = Path(*parts[2:])
    out = Path(output_prefix) / rel
    if out.suffix != ".jsonl":
        out = out.with_suffix(".jsonl")
    return out.as_posix()


def _write_jsonl_record(handle, offsets: list[int], record: dict[str, Any]) -> None:
    offsets.append(handle.tell())
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handle.write(payload)
    handle.write(b"\n")


def clean_vqa_dataset(
    *,
    root: str | Path,
    manifest: str | Path | None = None,
    output_prefix: str = "meta/labvla_clean",
    workers: int = 16,
    max_pending: int = 4096,
    check_all_images: bool = True,
    decode_images: bool = True,
    replace_manifest: bool = False,
    progress_every: int = 10000,
    limit: int | None = None,
) -> dict[str, Any]:
    vqa_root = Path(root).resolve()
    manifest_path = Path(manifest) if manifest is not None else vqa_root / "meta" / "vqa_manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = vqa_root / manifest_path
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    json_files = manifest_data.get("json_files")
    if not isinstance(json_files, list) or not json_files:
        raise ValueError(f"{manifest_path} must contain a non-empty json_files list")

    clean_root = vqa_root / output_prefix
    clean_root.mkdir(parents=True, exist_ok=True)
    invalid_path = clean_root / "invalid_records.jsonl"
    report_path = clean_root / "clean_report.json"
    clean_manifest_path = clean_root / "vqa_manifest.json"

    aggregate = SourceStats(source="__all__", output="")
    clean_json_files: list[str] = []
    source_reports: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    with invalid_path.open("w", encoding="utf-8") as invalid_handle:
        for source_rel in json_files:
            if not isinstance(source_rel, str) or not source_rel:
                invalid_handle.write(json.dumps({
                    "source": source_rel,
                    "reason": "invalid_manifest_entry",
                }) + "\n")
                aggregate.dropped += 1
                aggregate.reasons["invalid_manifest_entry"] += 1
                continue

            source_path = vqa_root / source_rel
            output_rel = _clean_output_rel(output_prefix, source_rel)
            output_path = vqa_root / output_rel
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stats = SourceStats(source=source_rel, output=output_rel)
            offsets: list[int] = []
            pending: deque[tuple[int, Any, Future[RecordCheck]]] = deque()

            logging.info("[vqa-clean] scanning %s -> %s", source_rel, output_rel)
            if not source_path.is_file():
                stats.dropped += 1
                stats.reasons["missing_metadata"] += 1
                invalid_handle.write(json.dumps({
                    "source": source_rel,
                    "reason": "missing_metadata",
                    "path": str(source_path),
                }, ensure_ascii=False) + "\n")
                source_reports.append({
                    "source": stats.source,
                    "output": stats.output,
                    "seen": stats.seen,
                    "kept": stats.kept,
                    "dropped": stats.dropped,
                    "reasons": dict(stats.reasons),
                })
                continue

            def flush_ready(out_handle, *, block: bool = False) -> None:
                while pending and (block or pending[0][2].done()):
                    local_index, record, future = pending.popleft()
                    check = future.result()
                    if check.ok:
                        _write_jsonl_record(out_handle, offsets, record)
                        stats.kept += 1
                        aggregate.kept += 1
                    else:
                        stats.dropped += 1
                        stats.reasons[check.reason] += 1
                        aggregate.dropped += 1
                        aggregate.reasons[check.reason] += 1
                        invalid_handle.write(json.dumps({
                            "source": source_rel,
                            "local_index": local_index,
                            "reason": check.reason,
                            **check.detail,
                        }, ensure_ascii=False) + "\n")

            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                with output_path.open("wb") as output_handle:
                    for local_index, record, parse_error in _iter_records(source_path):
                        if limit is not None and aggregate.seen >= int(limit):
                            break
                        stats.seen += 1
                        aggregate.seen += 1
                        if parse_error is not None:
                            reason = parse_error.split(":", 1)[0]
                            stats.dropped += 1
                            stats.reasons[reason] += 1
                            aggregate.dropped += 1
                            aggregate.reasons[reason] += 1
                            invalid_handle.write(json.dumps({
                                "source": source_rel,
                                "local_index": local_index,
                                "reason": parse_error,
                            }, ensure_ascii=False) + "\n")
                            continue
                        pending.append((
                            local_index,
                            record,
                            pool.submit(
                                check_vqa_record,
                                record,
                                vqa_root=vqa_root,
                                source_rel=source_rel,
                                local_index=local_index,
                                check_all_images=check_all_images,
                                decode_images=decode_images,
                            ),
                        ))
                        if len(pending) >= max(1, int(max_pending)):
                            flush_ready(output_handle, block=True)
                        else:
                            flush_ready(output_handle, block=False)
                        if progress_every and stats.seen % int(progress_every) == 0:
                            logging.info(
                                "[vqa-clean] %s seen=%d kept=%d dropped=%d",
                                source_rel,
                                stats.seen,
                                stats.kept,
                                stats.dropped,
                            )
                    while pending:
                        flush_ready(output_handle, block=True)

            with open(f"{output_path}.offsets.npy", "wb") as offsets_handle:
                np.save(offsets_handle, np.asarray(offsets, dtype=np.int64))

            if stats.kept > 0:
                clean_json_files.append(output_rel)
            else:
                logging.warning("[vqa-clean] source %s produced zero valid records", source_rel)
            source_reports.append({
                "source": stats.source,
                "output": stats.output,
                "seen": stats.seen,
                "kept": stats.kept,
                "dropped": stats.dropped,
                "reasons": dict(stats.reasons),
            })
            logging.info(
                "[vqa-clean] done %s seen=%d kept=%d dropped=%d",
                source_rel,
                stats.seen,
                stats.kept,
                stats.dropped,
            )

    if not clean_json_files:
        raise RuntimeError("VQA cleaning produced zero valid metadata files")

    clean_manifest = dict(manifest_data)
    clean_manifest["json_files"] = clean_json_files
    clean_manifest["_labvla_clean"] = {
        "source_manifest": str(manifest_path),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "output_prefix": output_prefix,
        "check_all_images": bool(check_all_images),
        "decode_images": bool(decode_images),
        "seen": aggregate.seen,
        "kept": aggregate.kept,
        "dropped": aggregate.dropped,
        "reasons": dict(aggregate.reasons),
        "invalid_records": str(invalid_path),
    }
    clean_manifest_path.write_text(
        json.dumps(clean_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "root": str(vqa_root),
        "source_manifest": str(manifest_path),
        "clean_manifest": str(clean_manifest_path),
        "invalid_records": str(invalid_path),
        "aggregate": {
            "seen": aggregate.seen,
            "kept": aggregate.kept,
            "dropped": aggregate.dropped,
            "reasons": dict(aggregate.reasons),
        },
        "sources": source_reports,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if replace_manifest:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = manifest_path.with_name(f"{manifest_path.name}.bak.{stamp}")
        shutil.copy2(manifest_path, backup_path)
        tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
        tmp_path.write_text(
            json.dumps(clean_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, manifest_path)
        report["replaced_manifest"] = str(manifest_path)
        report["backup_manifest"] = str(backup_path)
        logging.info("[vqa-clean] replaced %s (backup %s)", manifest_path, backup_path)

    logging.info(
        "[vqa-clean] complete seen=%d kept=%d dropped=%d clean_manifest=%s",
        aggregate.seen,
        aggregate.kept,
        aggregate.dropped,
        clean_manifest_path,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean RoboInter-VQA metadata/images")
    parser.add_argument("--root", required=True, help="RoboInter-VQA dataset root")
    parser.add_argument("--manifest", default=None, help="Manifest path; defaults to root/meta/vqa_manifest.json")
    parser.add_argument("--output_prefix", default="meta/labvla_clean")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max_pending", type=int, default=4096)
    parser.add_argument("--check_all_images", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--decode_images", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--replace_manifest", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit across all records")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        report = clean_vqa_dataset(
            root=args.root,
            manifest=args.manifest,
            output_prefix=args.output_prefix,
            workers=args.workers,
            max_pending=args.max_pending,
            check_all_images=args.check_all_images,
            decode_images=args.decode_images,
            replace_manifest=args.replace_manifest,
            progress_every=args.progress_every,
            limit=args.limit,
        )
    except Exception as exc:
        logging.exception("[vqa-clean] failed: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
