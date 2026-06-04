"""π0-style volume-weighted multi-dataset mixing.

Implements the processor-granularity n^0.43 formula from:
  - π0 paper, Section V-A: "we weight each task-robot combination by n^0.43"
  - Reference impl: /all-flash-data/LabVLA/pretrain_pi0/data_processors/
    pretrain_dataset.py:66-73 (weight formula) and :149-151 (sampling).

Unlike MultiLeRobotDataset (concat-by-length) or MultiStreamingLeRobotDataset
(IterableDataset), this is a map-style Dataset that performs weighted random
adapter selection per __getitem__. Each sub-dataset is assumed to be a
TransformedAdapterDataset around a LeRobot adapter — its __getitem__ returns
a transform-ready frame dict.

Caller contract:
  - `datasets`: list of map-style Datasets with a valid __len__
  - `weights`: non-negative floats, one per dataset; normalized internally
  - `repo_ids`: optional list[str] for logging
  - `seed`: base seed for deterministic sampling (default 42)

Sampling is deterministic w.r.t. (epoch, idx): per-call seed =
numpy.SeedSequence([seed, epoch, idx]) -> default_rng. (The Python `random`
module is non-deterministic across ranks/resume and leaks state across
DataLoader workers.) Use `set_epoch(epoch)` (DistributedSampler convention) to
advance the seed each epoch. `__len__` returns the sum of per-dataset lengths so
DataLoader sizing is unchanged.
"""
from __future__ import annotations

import logging
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class PI0MixtureDataset(Dataset):
    """Volume-weighted mixture of map-style Datasets."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float],
        repo_ids: Sequence[str] | None = None,
        seed: int = 42,
    ) -> None:
        if len(datasets) != len(weights):
            raise ValueError(
                f"PI0MixtureDataset: len(datasets)={len(datasets)} != "
                f"len(weights)={len(weights)}"
            )
        if not datasets:
            raise ValueError("PI0MixtureDataset: requires at least one dataset")
        self._datasets = list(datasets)
        self._lengths = [len(d) for d in self._datasets]
        self._total_len = sum(self._lengths)

        # Normalize weights defensively (caller should have normalized).
        weights_arr = np.asarray([float(w) for w in weights], dtype=np.float64)
        if (weights_arr < 0).any():
            raise ValueError(f"PI0MixtureDataset: weights must be non-negative, got {weights_arr.tolist()}")
        total = float(weights_arr.sum())
        if total <= 0:
            raise ValueError(f"PI0MixtureDataset: sum(weights)={total} <= 0")
        self._weights = weights_arr / total

        # Reject zero-length sub-datasets with positive weight. Both __getitem__
        # and PI0MixtureBatchSampler call rng.integers(0, len(ds)), which raises
        # "low >= high" the moment such a dataset is sampled; fail loud at
        # construction instead of mid-training. Zero-length with zero weight is
        # harmless (rng.choice with p=0 never selects it) and allowed.
        empty_weighted = [
            (repo_ids[i] if repo_ids is not None else f"dataset[{i}]")
            for i in range(len(self._datasets))
            if self._lengths[i] == 0 and self._weights[i] > 0
        ]
        if empty_weighted:
            raise ValueError(
                f"PI0MixtureDataset: zero-length sub-dataset(s) with positive "
                f"weight: {empty_weighted}. Drop these datasets (and renormalize "
                f"weights) or set their weight to 0 before constructing the "
                f"mixture — otherwise sampling one raises 'low >= high'."
            )

        self._repo_ids = list(repo_ids) if repo_ids is not None else [
            f"dataset[{i}]" for i in range(len(self._datasets))
        ]

        # Deterministic RNG state.
        self._base_seed = int(seed)
        self._epoch = 0

        pairs = ", ".join(
            f"{r}={w:.3f}" for r, w in zip(self._repo_ids, self._weights)
        )
        logger.info(f"[π0-mix] weights: [{pairs}]")
        logger.info(
            f"[π0-mix] total_len={self._total_len:,} "
            f"({[l for l in self._lengths]} frames per dataset) "
            f"seed={self._base_seed}"
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance the deterministic-sampling epoch.

        Mirrors ``torch.utils.data.distributed.DistributedSampler.set_epoch``
        — call before each new epoch so that per-(epoch, idx) RNG seeds
        differ across epochs while still being identical across ranks for
        the same epoch.
        """
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int | tuple[int, int]) -> dict:
        if isinstance(idx, tuple):
            dataset_index, local_index = idx
            dataset_index = int(dataset_index)
            local_index = int(local_index)
            if not 0 <= dataset_index < len(self._datasets):
                raise IndexError(
                    f"PI0MixtureDataset direct dataset index {dataset_index} "
                    f"out of range [0, {len(self._datasets)})"
                )
            if not 0 <= local_index < self._lengths[dataset_index]:
                raise IndexError(
                    f"PI0MixtureDataset direct local index {local_index} "
                    f"out of range [0, {self._lengths[dataset_index]}) "
                    f"for dataset {dataset_index}"
                )
            return self._datasets[dataset_index][local_index]

        # Deterministic per-(base_seed, epoch, idx) RNG: two ranks asking the
        # same idx in the same epoch get the same sample, making resume
        # reproducible. The DataLoader index is a nominal counter (π0 convention)
        # used only as a seed component, not as direct selection content, so
        # different idx values still draw different samples within an epoch.
        ss = np.random.SeedSequence([self._base_seed, self._epoch, int(idx)])
        rng = np.random.default_rng(ss)
        ds_idx = int(rng.choice(len(self._datasets), p=self._weights))
        ds = self._datasets[ds_idx]
        local_idx = int(rng.integers(0, len(ds)))
        return ds[local_idx]

    # Expose per-dataset metadata for downstream logging / checkpointing.
    @property
    def num_frames(self) -> int:
        return self._total_len

    @property
    def num_episodes(self) -> int:
        total = 0
        for ds in self._datasets:
            n = getattr(ds, "num_episodes", None)
            if n is None:
                adapter = getattr(ds, "adapter", None)
                n = getattr(adapter, "num_episodes", None) if adapter is not None else None
            if n is not None:
                total += int(n)
        return total

    @property
    def weights(self) -> list[float]:
        return list(self._weights)

    @property
    def repo_ids(self) -> list[str]:
        return list(self._repo_ids)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def base_seed(self) -> int:
        return self._base_seed


class PI0MixtureBatchSampler(Sampler[list[tuple[int, int]]]):
    """Homogeneous mini-batches for ``PI0MixtureDataset``.

    Phase-A memory fix: choose exactly one source dataset per mini-batch, then
    sample all local indices from that source. This preserves π0-style
    weighted mixture semantics at the batch level while preventing one
    heterogeneous collate from taking the union of every annotation/FAST field
    across unrelated dataset schemas.
    """

    def __init__(
        self,
        dataset: PI0MixtureDataset,
        batch_size: int,
        drop_last: bool = True,
        seed: int | None = None,
        start_batch_index: int = 0,
    ) -> None:
        if not isinstance(dataset, PI0MixtureDataset):
            raise TypeError(
                "PI0MixtureBatchSampler requires a PI0MixtureDataset, got "
                f"{type(dataset).__name__}"
            )
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self._base_seed = dataset.base_seed if seed is None else int(seed)
        self.start_batch_index = max(0, int(start_batch_index))

    def set_start_batch_index(self, start_batch_index: int) -> None:
        self.start_batch_index = max(0, int(start_batch_index))

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            total_batches = total // self.batch_size
        else:
            total_batches = (total + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - min(self.start_batch_index, total_batches))

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        epoch = int(self.dataset.epoch)
        if self.drop_last:
            total_batches = len(self.dataset) // self.batch_size
        else:
            total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        start = min(self.start_batch_index, total_batches)
        for batch_index in range(start, total_batches):
            current_batch_size = self.batch_size
            if not self.drop_last and batch_index == total_batches - 1:
                remainder = len(self.dataset) % self.batch_size
                if remainder:
                    current_batch_size = remainder

            seed_sequence = np.random.SeedSequence(
                [self._base_seed, epoch, int(batch_index), 271828]
            )
            rng = np.random.default_rng(seed_sequence)
            dataset_index = int(
                rng.choice(len(self.dataset._datasets), p=self.dataset._weights)
            )
            source_length = self.dataset._lengths[dataset_index]
            local_indices = rng.integers(
                0, source_length, size=current_batch_size
            )
            yield [
                (dataset_index, int(local_index))
                for local_index in local_indices
            ]
