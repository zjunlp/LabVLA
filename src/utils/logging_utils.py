"""Centralized warn-once / dedup logging utilities.

HIGH-27 + M-32: replaces ad-hoc module-level `_WARNED_*` sets throughout the
codebase with a single `logging.Filter` that deduplicates by
``(logger_name, dedup_key)``.

Usage
-----
At process startup (train.py / serve_labvla.py / data_process/cli.py)::

    from utils.logging_utils import install_dedupe_filter
    install_dedupe_filter()          # attaches to root logger

At warning sites::

    from utils.logging_utils import warn_once
    warn_once(logger, ("repo_col", repo_id, tuple(cols)),
              "dropped %d eps lacking %s", n, cols)

For distributed training (rank-0 only)::

    from utils.logging_utils import rank_zero_warn_once
    rank_zero_warn_once(logger, "device_switch",
                        "Device %r unavailable; switching to %r", dev, alt)

Scope
-----
- DOES replace: module-level ``_WARNED_*`` sets + ``global _WARNED_*`` guards
- DOES NOT replace: per-model-instance ``self._warned_*`` flags (different
  semantics — intentionally per-instance)
- DOES NOT replace: ``print("\\033[...")`` ANSI terminal controls (those are
  TUI escapes, not logging)
"""
from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Optional

_FILTER_ATTR = "_labvla_dedupe_filter"   # sentinel on Logger → detect double-install


class DedupeFilter(logging.Filter):
    """Drop LogRecords whose (logger name, dedup_key) pair was seen before.

    Attach a dedup_key via ``extra={"dedup_key": <hashable>}``.
    Records without the key are passed through unchanged.

    Thread safety: CPython's GIL makes ``set.add`` / ``set.__contains__``
    effectively atomic for a single operation. Worst case on parallel
    runtimes is one duplicate slip-through per race — acceptable for
    warn-once semantics. Memory grows unbounded in the set but each entry
    is O(bytes_of_key), and in practice distinct keys are bounded by
    distinct (repo_id, column_tuple) / (device, alt_device) / similar.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[tuple[str, Hashable]] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        key = getattr(record, "dedup_key", None)
        if key is None:
            return True
        tag = (record.name, key)
        if tag in self._seen:
            return False
        self._seen.add(tag)
        return True

    def reset(self) -> None:
        """Clear the seen-set — re-enables all suppressed warnings.

        Useful for tests and interactive debugging sessions.
        """
        self._seen.clear()


def install_dedupe_filter(logger_name: Optional[str] = None) -> DedupeFilter:
    """Idempotently attach a DedupeFilter to the named logger (root if None).

    Returns the (possibly pre-existing) filter instance so callers can call
    ``.reset()`` when needed.
    """
    target: logging.Logger = (
        logging.getLogger(logger_name) if logger_name else logging.getLogger()
    )
    existing = getattr(target, _FILTER_ATTR, None)
    if existing is not None and isinstance(existing, DedupeFilter):
        return existing                     # already installed; idempotent
    f = DedupeFilter()
    target.addFilter(f)
    setattr(target, _FILTER_ATTR, f)
    return f


# A DedupeFilter on the ROOT logger never sees records from module loggers:
# CPython runs a logger's own filters only in that logger's handle(), and on
# propagation only ancestor *handler* filters run, not ancestor logger filters.
# So the warn-once family dedupes via this process-level seen-set keyed by
# (logger_name, key), independent of where a DedupeFilter is attached.
_PROCESS_SEEN: set[tuple[str, Hashable]] = set()


def _seen_once(logger: logging.Logger, key: Hashable) -> bool:
    """Return True if ``(logger.name, key)`` was already emitted in this process.

    Records the pair as seen on first call. ``set.add`` / ``in`` are effectively
    atomic under CPython's GIL; worst case under a free-threaded runtime is one
    duplicate slip-through per race — acceptable for warn-once semantics.
    """
    tag = (logger.name, key)
    if tag in _PROCESS_SEEN:
        return True
    _PROCESS_SEEN.add(tag)
    return False


def reset_process_seen() -> None:
    """Clear the warn-once process seen-set (tests / interactive debugging)."""
    _PROCESS_SEEN.clear()


def warn_once(logger: logging.Logger, key: Hashable, msg: str, *args: object) -> None:
    """Emit a WARNING with dedup_key; suppressed after the first call per key."""
    if _seen_once(logger, key):
        return
    logger.warning(msg, *args, extra={"dedup_key": key})


def info_once(logger: logging.Logger, key: Hashable, msg: str, *args: object) -> None:
    """INFO-level dedup variant — occasional reminders that shouldn't spam."""
    if _seen_once(logger, key):
        return
    logger.info(msg, *args, extra={"dedup_key": key})


def rank_zero_warn_once(
    logger: logging.Logger, key: Hashable, msg: str, *args: object
) -> None:
    """Like warn_once, but silently skipped on non-zero ranks when dist is active."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except ImportError:
        pass
    warn_once(logger, key, msg, *args)
