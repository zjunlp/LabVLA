"""Retry helpers for transient shared-storage I/O failures.

The training cluster reads data and writes checkpoints through
``/all-flash-data``. Short storage hiccups surface as ``EIO``/``ESTALE`` or
transport-related exceptions inside DataLoader workers. If those exceptions
escape the worker, PyTorch tears down the worker and distributed training dies.

These helpers retry only errors that look transient. Permanent data problems
still fail loudly.
"""

from __future__ import annotations

import errno
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)

_TRANSIENT_ERRNOS = {
    errno.EIO,
    errno.ESTALE,
    errno.ENOTCONN,
    errno.ETIMEDOUT,
    errno.ECONNRESET,
    errno.ECONNABORTED,
    errno.EHOSTDOWN,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.ENETUNREACH,
    errno.EAGAIN,
    errno.EBUSY,
    errno.EINTR,
}

_TRANSIENT_MESSAGE_FRAGMENTS = (
    "input/output error",
    "errno 5",
    "stale file handle",
    "transport endpoint is not connected",
    "connection reset by peer",
    "connection timed out",
    "resource temporarily unavailable",
    "remote i/o error",
    "network is down",
    "network is unreachable",
    "no route to host",
    "software caused connection abort",
)

_FATAL_DIAG_ERRNOS = {
    errno.ENOSPC: "disk full",
    getattr(errno, "EDQUOT", -1): "quota exceeded",
    errno.EROFS: "read-only filesystem",
}
_FATAL_DIAG_ERRNOS.pop(-1, None)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.1f", name, raw, default)
        return default


def storage_retry_enabled() -> bool:
    return _env_bool("LABVLA_STORAGE_RETRY_ENABLE", True)


def default_total_seconds() -> float:
    return max(0.0, _env_float("LABVLA_STORAGE_RETRY_TOTAL_SECONDS", 1800.0))


def default_initial_sleep_seconds() -> float:
    return max(0.01, _env_float("LABVLA_STORAGE_RETRY_INITIAL_SLEEP", 1.0))


def default_max_sleep_seconds() -> float:
    return max(0.05, _env_float("LABVLA_STORAGE_RETRY_MAX_SLEEP", 30.0))


def default_not_found_seconds() -> float:
    return max(0.0, _env_float("LABVLA_STORAGE_RETRY_NOT_FOUND_SECONDS", 0.0))


def _walk_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_not_found(exc: BaseException) -> bool:
    return any(isinstance(item, FileNotFoundError) for item in _walk_exception_chain(exc))


def is_transient_storage_error(
    exc: BaseException,
    *,
    retry_not_found: bool = False,
) -> bool:
    """Return whether ``exc`` looks like a recoverable shared-storage hiccup."""

    if retry_not_found and _is_not_found(exc):
        return True

    for item in _walk_exception_chain(exc):
        # Only OSError-family exceptions are eligible: storage hiccups always
        # surface as OSError, and gating the message-fragment fallback here
        # stops an arbitrary error whose text happens to match from being
        # retried for the full budget.
        if isinstance(item, OSError):
            if item.errno in _TRANSIENT_ERRNOS:
                return True
            message = str(item).lower()
            if any(fragment in message for fragment in _TRANSIENT_MESSAGE_FRAGMENTS):
                return True
    return False


def _fatal_storage_diagnostic(exc: BaseException) -> tuple[int, str] | None:
    for item in _walk_exception_chain(exc):
        if isinstance(item, OSError) and item.errno in _FATAL_DIAG_ERRNOS:
            errno_value = int(item.errno)
            return errno_value, _FATAL_DIAG_ERRNOS[errno_value]
    return None


def _retry_budget_seconds(
    exc: BaseException,
    *,
    retry_not_found: bool,
    total_seconds: float,
    not_found_seconds: float,
) -> float:
    if _is_not_found(exc) and retry_not_found:
        return min(total_seconds, not_found_seconds)
    return total_seconds


def run_with_storage_retry(
    fn: Callable[P, T],
    *args: P.args,
    path: str | os.PathLike | None = None,
    description: str = "storage I/O",
    retry_not_found: bool = False,
    total_seconds: float | None = None,
    initial_sleep_seconds: float | None = None,
    max_sleep_seconds: float | None = None,
    not_found_seconds: float | None = None,
    on_retry: Callable[[BaseException, int], None] | None = None,
    **kwargs: P.kwargs,
) -> T:
    """Run ``fn`` and retry transient storage failures in-place.

    The same callable is retried with the same arguments; callers should use
    this only around deterministic file I/O, before random augmentation or loss
    construction.
    """

    if not storage_retry_enabled():
        return fn(*args, **kwargs)

    total = default_total_seconds() if total_seconds is None else float(total_seconds)
    initial_sleep = (
        default_initial_sleep_seconds()
        if initial_sleep_seconds is None
        else float(initial_sleep_seconds)
    )
    max_sleep = (
        default_max_sleep_seconds()
        if max_sleep_seconds is None
        else float(max_sleep_seconds)
    )
    not_found_budget = (
        default_not_found_seconds()
        if not_found_seconds is None
        else float(not_found_seconds)
    )
    started_at = time.monotonic()
    attempt = 0
    sleep_s = max(0.01, initial_sleep)
    target = str(path) if path is not None else "<unknown>"

    while True:
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            # Never swallow control-flow exceptions into the retry loop.
            raise
        except Exception as exc:
            fatal = _fatal_storage_diagnostic(exc)
            if fatal is not None:
                errno_value, label = fatal
                logger.error(
                    "%s fatal storage error for %s: %s (errno=%d). "
                    "Not retrying; operator intervention required. error=%r",
                    description,
                    target,
                    label,
                    errno_value,
                    exc,
                )
                raise
            if not is_transient_storage_error(exc, retry_not_found=retry_not_found):
                raise

            budget = _retry_budget_seconds(
                exc,
                retry_not_found=retry_not_found,
                total_seconds=total,
                not_found_seconds=not_found_budget,
            )
            elapsed = time.monotonic() - started_at
            if elapsed >= budget:
                logger.error(
                    "%s failed after %.1fs and %d retries for %s: %r",
                    description,
                    elapsed,
                    attempt,
                    target,
                    exc,
                )
                raise

            attempt += 1
            if on_retry is not None:
                try:
                    on_retry(exc, attempt)
                except Exception as callback_error:
                    logger.warning("storage retry callback failed: %r", callback_error)

            remaining = max(0.0, budget - elapsed)
            wait_s = min(max_sleep, sleep_s, remaining)
            wait_s *= random.uniform(0.85, 1.15)
            if attempt == 1 or attempt % 5 == 0:
                logger.warning(
                    "%s transient failure for %s; retry=%d elapsed=%.1fs "
                    "sleep=%.1fs budget=%.1fs error=%r",
                    description,
                    target,
                    attempt,
                    elapsed,
                    wait_s,
                    budget,
                    exc,
                )
            time.sleep(max(0.01, wait_s))
            sleep_s = min(max_sleep, max(initial_sleep, sleep_s * 2.0))


def storage_path_exists(path: str | os.PathLike) -> bool:
    """Like ``Path.exists`` but does not swallow transient storage errors."""

    target = Path(path)
    try:
        run_with_storage_retry(target.stat, path=target, description="stat")
        return True
    except FileNotFoundError:
        return False


def read_parquet_with_storage_retry(path: str | os.PathLike, **kwargs):
    import pandas as pd

    return run_with_storage_retry(
        pd.read_parquet,
        path,
        path=path,
        description="read_parquet",
        **kwargs,
    )
