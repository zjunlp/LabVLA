"""Pickle-safe DataLoader helpers for LabVLA training."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


logger = logging.getLogger(__name__)

_SKIP_RETRY_STRIDES = (1, 37, 1009)

# SkipBadSamplesDataset rewrites the sampling distribution when bad samples
# cluster (same base_idx always redirects to the same candidates -> hidden
# over/under-sampling). Mitigations: (1) a process-wide counter + rate-limited
# warning so aggregate skip volume is observable; (2) per-index randomized
# candidate selection (reproducible per index, but spreads a cluster's
# redirects across candidates).
_GLOBAL_SKIP_COUNT = 0
_GLOBAL_SKIP_LOG_EVERY = max(
    1,
    int(os.environ.get("LABVLA_DATA_SKIP_LOG_EVERY", "1000")),
)


def _skip_retry_offset(attempt: int, *, base_idx: int = 0) -> int:
    """Sparse retry offset for bad map-style samples.

    Deterministic for a given ``(base_idx, attempt)`` pair (so resume/replay is
    reproducible), but randomized across ``base_idx`` so clustered bad samples
    do not all redirect to one fixed candidate set. ``attempt == 0`` always
    returns ``0`` (try the requested index first).
    """
    if int(attempt) <= 0:
        return 0
    stride = _SKIP_RETRY_STRIDES[(int(attempt) - 1) % len(_SKIP_RETRY_STRIDES)]
    multiplier = ((int(attempt) - 1) // len(_SKIP_RETRY_STRIDES)) + 1
    base_offset = stride * multiplier
    # Deterministic per-(base_idx, attempt) jitter so adjacent bad indices fan
    # out to distinct candidates. Window is the largest stride (not `stride`) so
    # even the first retry (stride==1) spreads out rather than always redirecting
    # to base_idx+1. Bounded + reproducible; caller takes the result mod length.
    jitter_rng = random.Random((int(base_idx) << 16) ^ int(attempt))
    return base_offset + jitter_rng.randrange(_SKIP_RETRY_STRIDES[-1])


def _note_global_skip(label: str, exc: Exception) -> None:
    """Increment the process-wide skip counter and emit a rate-limited warning."""
    global _GLOBAL_SKIP_COUNT
    _GLOBAL_SKIP_COUNT += 1
    if _GLOBAL_SKIP_COUNT % _GLOBAL_SKIP_LOG_EVERY == 0:
        logger.warning(
            "[data-skip] process-wide bad-sample skips reached %d "
            "(most recent label=%s error=%s: %s). High counts mean the "
            "sampling distribution is being rewritten, not just point-skipped.",
            _GLOBAL_SKIP_COUNT,
            label,
            type(exc).__name__,
            exc,
        )


@dataclass(frozen=True)
class DataLoaderWorkerSeed:
    """Deterministically seed each DataLoader worker.

    This callable intentionally lives at module scope so it remains pickleable
    when PyTorch workers are launched with the ``spawn`` multiprocessing
    context. A nested ``worker_init_fn`` works under ``fork`` but fails under
    ``spawn`` before the first batch is produced.
    """

    base_seed: int
    rank: int
    num_workers: int

    def __call__(self, worker_id: int) -> None:
        seed = int(self.base_seed) + int(self.rank) * int(self.num_workers) + int(worker_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


class DeterministicBatchSampler(Sampler[list[int]]):
    """Deterministic map-style batch sampler with a resumable batch offset."""

    def __init__(
        self,
        dataset_length: int,
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 42,
        epoch: int = 0,
        start_batch_index: int = 0,
    ) -> None:
        if int(dataset_length) <= 0:
            raise ValueError(f"dataset_length must be positive, got {dataset_length}")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset_length = int(dataset_length)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_batch_index = max(0, int(start_batch_index))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_batch_index = 0

    def set_start_batch_index(self, start_batch_index: int) -> None:
        self.start_batch_index = max(0, int(start_batch_index))

    def __len__(self) -> int:
        if self.drop_last:
            total = self.dataset_length // self.batch_size
        else:
            total = (self.dataset_length + self.batch_size - 1) // self.batch_size
        return max(0, total - min(self.start_batch_index, total))

    def __iter__(self) -> Iterator[list[int]]:
        if self.drop_last:
            total_batches = self.dataset_length // self.batch_size
        else:
            total_batches = (self.dataset_length + self.batch_size - 1) // self.batch_size
        start = min(self.start_batch_index, total_batches)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003)
        order = torch.randperm(self.dataset_length, generator=generator).tolist()
        for batch_index in range(start, total_batches):
            batch_start = batch_index * self.batch_size
            batch = order[batch_start:batch_start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch


class SkipBadSamplesDataset(Dataset):
    """Retry adjacent samples when a map-style dataset item raises.

    The wrapper sits outside adapter + transform code, so the same policy covers
    VQA JSON/image corruption, LeRobot video/parquet read errors, and transform
    shape errors. Retries stay inside the already-selected child dataset, which
    preserves homogeneous mixture batches and source weighting.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        enabled: bool = True,
        max_attempts: int = 64,
        log_first: int = 20,
        label: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.enabled = bool(enabled)
        self.max_attempts = max(1, int(max_attempts))
        self.log_first = max(0, int(log_first))
        self.label = label or getattr(dataset, "repo_id", type(dataset).__name__)
        self._skip_log_count = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.enabled:
            return self.dataset[idx]
        length = len(self.dataset)
        if length <= 0:
            raise IndexError(f"{self.label}: cannot sample from empty dataset")
        base_idx = int(idx)
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            candidate = (base_idx + _skip_retry_offset(attempt, base_idx=base_idx)) % length
            try:
                return self.dataset[candidate]
            except Exception as exc:
                last_err = exc
                _note_global_skip(self.label, exc)
                self._skip_log_count += 1
                if self._skip_log_count <= self.log_first:
                    logger.warning(
                        "Skipping bad data sample label=%s requested_idx=%s "
                        "candidate_idx=%s attempt=%d/%d error=%s: %s",
                        self.label,
                        base_idx,
                        candidate,
                        attempt + 1,
                        self.max_attempts,
                        type(exc).__name__,
                        exc,
                    )
                continue
        raise RuntimeError(
            f"{self.label}: {self.max_attempts} retry candidate data samples failed "
            f"starting at idx={base_idx}; last error={last_err!r}"
        ) from last_err

    def __getattr__(self, name: str):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def should_persist_workers(
    *,
    num_workers: int,
    dataset: object,
    homogeneous_mixture_batches: bool,
) -> bool:
    """Return whether a DataLoader should keep workers across epochs.

    ``PI0MixtureDataset`` uses its own ``_epoch`` inside worker-side
    ``__getitem__`` on the default non-homogeneous path. Persistent workers
    hold a stale fork-time dataset copy and therefore miss parent-side
    ``set_epoch`` calls. The homogeneous batch-sampler path is safe because
    source/local index sampling happens in the main process and workers receive
    explicit tuple indices.
    """
    if int(num_workers) <= 0:
        return False
    if bool(homogeneous_mixture_batches):
        return True
    try:
        from dataset.pi0_mixture import PI0MixtureDataset
    except Exception:
        return False
    return not isinstance(dataset, PI0MixtureDataset)
