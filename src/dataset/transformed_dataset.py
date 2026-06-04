from __future__ import annotations
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from dataset.lerobot_dataset import LeRobotDataset
from dataset.streaming_dataset import StreamingLeRobotDataset
from transforms.core import (
    DataTransformFn,
    DataDict,
    compose,
    hydrate_all,
)


def _hydrate_unified(transforms, obj):
    """Drive all hydration off obj.meta.schema (populated by the adapter).

    After Phase 4, the schema-driven path is the only path. If meta.schema
    is None we attempt to discover one on the fly from obj.root + robot_type,
    so the legacy StreamingLeRobotDataset path (which uses LeRobotDatasetMetadata
    without schema) still works.
    """
    meta = getattr(obj, "meta", None)
    schema = getattr(meta, "schema", None) if meta is not None else None

    if schema is None:
        # Discover on the fly — supports callers whose meta object wasn't
        # populated by the adapter chain (e.g. StreamingLeRobotDataset wrapped
        # directly). On failure, let SchemaDiscoveryError propagate directly
        # (callers catch it); do not wrap it in RuntimeError. A failed import of
        # the schema module should crash, not silently fall through.
        from schema import discover_schema, SchemaDiscoveryError
        import logging as _logging
        root = getattr(obj, "root", None)
        robot_type = getattr(meta, "robot_type", None) if meta else None
        if root is not None:
            schema = discover_schema(root, robot_type=robot_type)
            # Attach to meta so subsequent code can find it.
            try:
                object.__setattr__(meta, "schema", schema)
            except Exception:
                pass  # some meta types are frozen — discovery result used locally
            _logging.getLogger(__name__).info(
                "_hydrate_unified: late-discovered schema id=%s source=%s",
                schema.schema_id, schema.source,
            )

    if schema is None:
        from schema import SchemaDiscoveryError
        raise SchemaDiscoveryError(
            "_hydrate_unified: obj.meta.schema is None and no root path was "
            "available for late discovery. Your adapter didn't populate a "
            "DatasetSchema. Use LeRobotV3Adapter (which auto-discovers) or "
            "call schema.discover_schema(root, robot_type) manually and "
            "attach the result to meta.schema."
        )
    transforms = hydrate_all(transforms, schema=schema, stats=obj.meta.stats)
    # Drop video features not referenced by the schema's image_mapping.
    for key in list(obj.meta.video_keys):
        if key not in schema.image_mapping:
            obj.meta.features.pop(key, None)
    return transforms


class TransformedLeRobotDataset(LeRobotDataset):

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use TransformedLeRobotDataset.from_base(...) or .from_repo(...).")

    @classmethod
    def from_base(
        cls,
        base: LeRobotDataset,
        transforms: Sequence[DataTransformFn] | None = None,
        *,
        share_dict: bool = True,
    ) -> TransformedLeRobotDataset:
        obj = cls.__new__(cls)
        # Give the wrapper its own top-level attribute namespace (shallow copy)
        # even when share_dict=True. Value objects (hf_dataset, episode tables,
        # ...) stay shared by reference — no heavy duplication — but the `meta`
        # rebinding below stays local instead of writing through to the base.
        obj.__dict__ = base.__dict__.copy()

        # _hydrate_unified() pops schema-filtered video keys out of meta.features
        # (backed by meta.info["features"]). meta is aliased here (shallow
        # __dict__ copy), so that pop would mutate the base's feature/camera view
        # in place. Give the wrapper a private meta with copied info/features so
        # hydration cannot pollute the base; heavy sub-objects stay shared by
        # reference. writer/buffer are neutralized so the copy's destructor never
        # flushes/closes the base's parquet writer.
        meta = getattr(obj, "meta", None)
        if meta is not None and getattr(meta, "info", None) is not None:
            wrapper_meta = copy.copy(meta)
            wrapper_meta.info = dict(meta.info)
            if isinstance(wrapper_meta.info.get("features"), dict):
                wrapper_meta.info["features"] = dict(meta.info["features"])
            # Prevent the shallow copy's __del__ from touching shared write state.
            wrapper_meta.writer = None
            wrapper_meta.metadata_buffer = []
            obj.meta = wrapper_meta

        transforms = _hydrate_unified(transforms, obj)

        obj._transform = compose(transforms)
        obj._wrapped_base_cls = base.__class__.__name__
        obj._is_transformed_wrapper = True
        return obj

    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        *,
        root=None,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        transforms: Sequence[DataTransformFn] | None = None,
        share_dict: bool = True,
    ) -> TransformedLeRobotDataset:
        base = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        return cls.from_base(base, transforms=transforms, share_dict=share_dict)

    def __getitem__(self, idx: int) -> DataDict:
        sample = super().__getitem__(idx)
        return self._transform(sample)

    def __repr__(self) -> str:
        base = super().__repr__().rstrip("\n")
        return base + f" (Transformed from {getattr(self, '_wrapped_base_cls', 'LeRobotDataset')})\n"


@dataclass
class CombinedMeta:
    """Lightweight metadata container for MultiLeRobotDataset.

    Stores flattened episode_from/to indices plus per-repo stats/schemas for
    checkpoint and deployment code that needs to avoid "first repo wins".
    """
    episodes: Dict[str, List[int]]
    stats: Dict[str, Any] = field(default_factory=dict)
    schemas: Dict[str, Any] = field(default_factory=dict)


class MultiLeRobotDataset(Dataset):
    """
    A concatenation dataset that merges multiple TransformedLeRobotDataset instances.

    Assumptions:
    - Each underlying dataset already applies its own DataTransformFn pipeline.
    - All datasets return aligned keys (i.e., they are transform-aligned).
    
    This class only:
    - Locates which sub-dataset corresponds to a global index.
    - Calls the appropriate __getitem__.
    - Optionally attaches metadata such as dataset_index and repo_id.

    It is intentionally minimal so it can be used directly inside a PyTorch DataLoader.
    """

    def __init__(
        self, 
        datasets: Sequence[TransformedLeRobotDataset], 
        dataset_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()

        if not datasets:
            raise ValueError("MultiLeRobotDataset requires at least one dataset.")

        # Validate all sub-datasets share identical schema semantics, failing
        # loud at __init__ with a precise "field X differs" message rather than
        # an opaque collate-time crash 500 steps later. Dim equality alone is
        # insufficient: two repos can be action-vector compatible yet differ in
        # delta/abs masks, gripper position, image-mapping targets, or canonical
        # arm layout — mixing them silently teaches contradictory actions on the
        # same input distribution. So cross-check every semantic invariant below.
        schemas = [getattr(ds.meta, "schema", None) for ds in datasets]
        if len(datasets) > 1 and all(s is not None for s in schemas):
            ref = schemas[0]
            ref_label = (
                f"dataset[0] schema_id={ref.schema_id!r}"
            )
            for i, s in enumerate(schemas[1:], start=1):
                cur_label = f"dataset[{i}] schema_id={s.schema_id!r}"

                # (1) state/action dims.
                if s.state_dims != ref.state_dims:
                    raise ValueError(
                        f"MultiLeRobotDataset: schema mismatch on field "
                        f"'state_dims' between {ref_label} ({ref.state_dims}) "
                        f"and {cur_label} ({s.state_dims}). "
                        f"All sub-datasets must share identical state widths."
                    )
                if s.action_dims != ref.action_dims:
                    raise ValueError(
                        f"MultiLeRobotDataset: schema mismatch on field "
                        f"'action_dims' between {ref_label} ({ref.action_dims}) "
                        f"and {cur_label} ({s.action_dims}). "
                        f"All sub-datasets must share identical action widths."
                    )

                # (2) delta_mask — flat tuple of bool, must match per-dim.
                ref_delta = tuple(bool(b) for b in ref.delta_mask)
                cur_delta = tuple(bool(b) for b in s.delta_mask)
                if cur_delta != ref_delta:
                    raise ValueError(
                        f"MultiLeRobotDataset: schema mismatch on field "
                        f"'delta_mask' between {ref_label} ({ref_delta}) and "
                        f"{cur_label} ({cur_delta}). Mixing repos with "
                        f"different per-dim delta/abs semantics would teach "
                        f"the model contradictory action conventions."
                    )

                # (3) gripper_action_dims — flat tuple of int, must match.
                ref_grip = tuple(int(x) for x in ref.gripper_action_dims)
                cur_grip = tuple(int(x) for x in s.gripper_action_dims)
                if cur_grip != ref_grip:
                    raise ValueError(
                        f"MultiLeRobotDataset: schema mismatch on field "
                        f"'gripper_action_dims' between {ref_label} ({ref_grip}) "
                        f"and {cur_label} ({cur_grip}). Gripper indices must "
                        f"align across all sub-datasets."
                    )

                # (4) image_mapping RHS targets — set equality (key order /
                # source camera names may legitimately differ across robots,
                # but the target slot set the model sees must be identical
                # so the unified-keys batch carries the same image columns).
                ref_targets = set(ref.image_mapping.values())
                cur_targets = set(s.image_mapping.values())
                if cur_targets != ref_targets:
                    raise ValueError(
                        f"MultiLeRobotDataset: schema mismatch on field "
                        f"'image_mapping' targets between {ref_label} "
                        f"({sorted(ref_targets)}) and {cur_label} "
                        f"({sorted(cur_targets)}). Unified image targets "
                        f"must be identical across sub-datasets."
                    )

                # (5) arm_layout — both None or both equal. One-sided
                # presence is also a mismatch (can't reverse-map deploy
                # canonical→raw with a heterogeneous mix).
                if (ref.arm_layout is None) != (s.arm_layout is None):
                    raise ValueError(
                        f"MultiLeRobotDataset: schema mismatch on field "
                        f"'arm_layout' presence between {ref_label} "
                        f"(arm_layout={'set' if ref.arm_layout else 'None'}) "
                        f"and {cur_label} (arm_layout="
                        f"{'set' if s.arm_layout else 'None'}). Either both "
                        f"sub-datasets define a canonical arm layout or "
                        f"neither does."
                    )
                if (ref.arm_layout is not None
                        and s.arm_layout is not None
                        and ref.arm_layout != s.arm_layout):
                    raise ValueError(
                        f"MultiLeRobotDataset: schema mismatch on field "
                        f"'arm_layout' between {ref_label} ({ref.arm_layout}) "
                        f"and {cur_label} ({s.arm_layout}). Canonical arm "
                        f"layout must match exactly across sub-datasets."
                    )

                # (6) raw source columns feeding canonical state/action. If one
                # dataset canonicalizes from source_* columns and another does not
                # (or uses different source widths), the same canonical key can be
                # assembled from different physical semantics.
                for field_name in (
                    "source_state_keys",
                    "source_state_dims",
                    "source_action_keys",
                    "source_action_dims",
                ):
                    ref_value = tuple(getattr(ref, field_name, ()) or ())
                    cur_value = tuple(getattr(s, field_name, ()) or ())
                    if cur_value != ref_value:
                        raise ValueError(
                            f"MultiLeRobotDataset: schema mismatch on field "
                            f"{field_name!r} between {ref_label} ({ref_value}) "
                            f"and {cur_label} ({cur_value}). Source-column "
                            f"canonicalization must match exactly across "
                            f"sub-datasets in one map-style concat."
                        )

        # List of transformed datasets (one per robot / repo)
        self.datasets = list(datasets)

        # Pre-compute lengths for fast index routing
        self._lengths = [ds.num_frames for ds in self.datasets]

        # Cumulative lengths for O(N_datasets) lookup
        self._cum_lengths = []
        running = 0
        for length in self._lengths:
            running += length
            self._cum_lengths.append(running)
        
        self.meta = self._build_combined_metadata()

        if dataset_weights is None:
            self.dataset_weights = None
        else:
            # MultiLeRobotDataset (non-streaming) never consumes dataset_weights:
            # __getitem__ routes the index directly via cum_lengths, so weights
            # here would give the illusion of a weighted mix while emitting pure
            # cumulative-length order. Fail loud instead of accepting dead params;
            # use MultiStreamingLeRobotDataset for an actual weighted mix.
            raise NotImplementedError(
                "MultiLeRobotDataset(dataset_weights=...) is not implemented — "
                "the non-streaming path does not sample according to weights. "
                "Either (a) omit dataset_weights to get pure-concat mixing, or "
                "(b) use MultiLeRobotStreamingDataset which honors weights."
            )
    
    @property
    def num_frames(self) -> int:
        """Total number of frames across all robots."""
        return self._cum_lengths[-1]

    @property
    def num_episodes(self) -> int:
        """Total number of episodes across all underlying datasets."""
        return sum(ds.meta.total_episodes for ds in self.datasets)

    def _build_combined_metadata(self) -> CombinedMeta:
        """
        Construct a lightweight metadata object that mimics:
            dataset.meta.episodes["dataset_from_index"]
            dataset.meta.episodes["dataset_to_index"]

        Behavior:
        - Episodes from all robots are concatenated in order:
            robot1 episodes → robot2 episodes → ...
        - Frame indexing is continuous across robots.
        """
        episodes = {
            "dataset_from_index": [],
            "dataset_to_index": [],
        }

        running_frame = 0
        stats: Dict[str, Any] = {}
        schemas: Dict[str, Any] = {}

        for i, ds in enumerate(self.datasets):
            dataset_from_index = np.asarray(ds.meta.episodes["dataset_from_index"]) + running_frame
            dataset_to_index = np.asarray(ds.meta.episodes["dataset_to_index"]) + running_frame

            episodes["dataset_from_index"].extend(dataset_from_index.tolist())
            episodes["dataset_to_index"].extend(dataset_to_index.tolist())

            label = (
                getattr(ds, "repo_id", None)
                or getattr(ds.meta, "repo_id", None)
                or getattr(getattr(ds.meta, "root", None), "name", None)
                or f"dataset_{i}"
            )
            if getattr(ds.meta, "stats", None):
                stats[str(label)] = ds.meta.stats
            if getattr(ds.meta, "schema", None) is not None:
                schemas[str(label)] = ds.meta.schema

            running_frame += ds.num_frames

        if stats:
            stats = {"_schema": "multi_repo_v1", **stats}
        return CombinedMeta(episodes=episodes, stats=stats, schemas=schemas)


    def __len__(self):
        return self.num_frames

    def _locate_dataset(self, idx: int) -> tuple[int, int]:
        """
        Convert a global index into (dataset_index, local_index).

        Example:
            If dataset lengths = [1000, 800],
            global idx 1200 → dataset #1, local_idx 200.
        """
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range.")

        start = 0
        for ds_idx, length in enumerate(self._lengths):
            end = start + length
            if idx < end:
                return ds_idx, idx - start
            start = end

        # Should not happen
        raise RuntimeError("Index resolution failed in MultiLeRobotDataset._locate_dataset")

    def __getitem__(self, idx: int) -> dict:
        """
        Forward a global index to the correct robot-dataset.
        """
        ds_idx, local_idx = self._locate_dataset(idx)
        ds = self.datasets[ds_idx]

        # Already transformed sample.
        sample = ds[local_idx]
        return sample

    def __repr__(self):
        return (
            f"MultiLeRobotDataset(\n"
            f"  Robots: {len(self.datasets)} → {[ds.repo_id for ds in self.datasets]},\n"
            f"  Total frames: {self.num_frames},\n"
            f"  Total episodes: {self.num_episodes}\n"
            f")"
        )


class TransformedStreamingLeRobotDataset(IterableDataset):

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use .from_base(...)")

    @classmethod
    def from_base(
        cls,
        base: StreamingLeRobotDataset,
        transforms: Sequence[DataTransformFn] | None = None,
    ):
        obj = cls.__new__(cls)

        obj._base = base

        obj.meta = base.meta
        obj.stats = base.meta.stats
        obj.robot_type = base.meta.robot_type

        transforms = _hydrate_unified(transforms, obj)

        obj._transform = compose(transforms)
        return obj

    def __iter__(self):
        for x in self._base:
            yield self._transform(x)

    @property
    def num_frames(self):
        return self._base.num_frames
    
    @property
    def num_episodes(self):
        return self._base.num_episodes

    @property
    def fps(self):
        return self._base.fps

    @property
    def camera_keys(self):
        return self._base.meta.camera_keys

    def __repr__(self):
        return f"TransformedStreamingLeRobotDataset(base={self._base})"


class MultiStreamingLeRobotDataset(IterableDataset):
    """
    Streaming version of MultiLeRobotDataset.

    This dataset merges multiple StreamingLeRobotDataset or
    TransformedStreamingLeRobotDataset instances into a single
    IterableDataset for large-scale streaming training.

    Notes:
    - This class does NOT support random indexing (__getitem__).
    - It supports two merging modes:
        (1) Simple concatenation (no dataset_weights)
        (2) Weighted multi-stream sampling (with dataset_weights)
    - For extremely large datasets (100M+ frames), this avoids
      loading into memory and fully supports PyTorch multi-worker
      dataloading.
    """

    def __init__(
        self,
        datasets: Sequence[TransformedStreamingLeRobotDataset],
        *,
        dataset_weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        add_dataset_index: bool = True,
    ) -> None:
        super().__init__()

        if not datasets:
            raise ValueError("MultiStreamingLeRobotDataset requires at least one dataset.")

        self.datasets: List[TransformedStreamingLeRobotDataset] = list(datasets)
        self.add_dataset_index = add_dataset_index

        # Aggregate minimal metadata (episode boundaries only).
        self.meta = self._build_combined_metadata()

        # Track whether the caller actually supplied weights. The ctor coerces
        # None to length-proportional weights, which is otherwise
        # indistinguishable from explicit weights at __iter__ time — so an
        # explicit flag is needed to deliberately reach both the concat and
        # weighted-sampling paths.
        self._weights_explicit = dataset_weights is not None

        # Handle optional dataset sampling weights
        if dataset_weights is None:
            # No sampling weights → simple concatenation
            self.dataset_weights = np.asarray([ds.num_frames for ds in self.datasets])
            self.dataset_weights = self.dataset_weights / self.dataset_weights.sum()
        else:
            if len(dataset_weights) != len(self.datasets):
                raise ValueError(
                    f"dataset_weights must have length {len(self.datasets)}, "
                    f"got {len(dataset_weights)}."
                )

            w = np.asarray(dataset_weights, dtype=np.float64)

            if (w < 0).any():
                raise ValueError("dataset_weights must be non-negative.")
            if w.sum() == 0:
                raise ValueError("At least one dataset weight must be positive.")

            # Normalize weights
            self.dataset_weights = (w / w.sum()).tolist()

        self.seed = seed

    # ----------------------------------------------------------
    # Properties (to match MultiLeRobotDataset)
    # ----------------------------------------------------------
    @property
    def num_frames(self) -> int:
        """Total frames across all datasets (approximate)."""
        return sum(ds.num_frames for ds in self.datasets)

    @property
    def num_episodes(self) -> int:
        """Total number of episodes across datasets."""
        return sum(ds.num_episodes for ds in self.datasets)

    @property
    def fps(self) -> Optional[int]:
        """FPS is assumed consistent across datasets."""
        return self.datasets[0].fps

    def __len__(self):
        """IterableDataset normally has no __len__, but we return num_frames for compatibility."""
        return self.num_frames

    # ----------------------------------------------------------
    # Main Streaming Iterator
    # ----------------------------------------------------------
    def __iter__(self):
        """
            Main streaming logic:

            Case 1 — No dataset_weights:
                Simple sequential concatenation.
                Each dataset is fully exhausted before moving to the next one:

                    ds0 → ds0 → ds0 → ... → ds0
                                            ↓
                    ds1 → ds1 → ds1 → ... → ds1
                                            ↓
                    ds2 → ds2 → ds2 → ... → ds2

            Case 2 — With dataset_weights:
                Weighted multi-stream sampling (with replacement).

                At each step, one dataset is sampled according to dataset_weights,
                and one sample is yielded from that dataset. When a dataset iterator
                is exhausted, it is automatically restarted (wrap-around).

                This produces an effectively infinite mixed stream whose long-run
                sampling frequency converges to the specified weights:

                    step:   1     2     3     4     5     6     ...
                    choice: ds0   ds0   ds1   ds0   ds0   ds1  ...
                            ↑           ↑
                        p≈0.9       p≈0.1

                (Example: dataset_weights = [0.9, 0.1])

                Epoch length is therefore not defined by dataset exhaustion and must
                be controlled externally (e.g. via max_steps or steps_per_epoch).

            Multi-worker support:
                - Each worker uses a different RNG seed for dataset selection.
                - Dataset iterators are independent across workers.
        """
        worker_info = get_worker_info()

        # Each worker uses a different RNG seed for sampling
        if worker_info is None:
            rng_seed = self.seed
        else:
            rng_seed = self.seed + worker_info.id

        rng = np.random.default_rng(rng_seed)

        # Create iterators for each dataset
        iterators = {i: iter(ds) for i, ds in enumerate(self.datasets)}
        active_ids = list(iterators.keys())

        # ------------------------------------------------------
        # 1) Simple concatenation mode
        # ------------------------------------------------------
        # Gate on `_weights_explicit`, not `dataset_weights is None`: the ctor
        # always populates dataset_weights (frame-proportional when None was
        # passed), so a None check here could never fire.
        if not self._weights_explicit:
            for ds_idx in active_ids:
                it = iterators[ds_idx]
                for sample in it:
                    if self.add_dataset_index and isinstance(sample, dict):
                        sample["dataset_index"] = torch.tensor(ds_idx, dtype=torch.long)
                    yield sample
            return

        # ------------------------------------------------------
        # 2) Weighted multi-stream merging
        # ------------------------------------------------------
        weights = np.asarray(self.dataset_weights, dtype=np.float64)

        while True:  
            cur_weights = weights / weights.sum()
            ds_idx = rng.choice(len(self.datasets), p=cur_weights)

            it = iterators[ds_idx]
            try:
                sample = next(it)
            except StopIteration:
                it = iter(self.datasets[ds_idx])
                iterators[ds_idx] = it
                sample = next(it) 

            if self.add_dataset_index and isinstance(sample, dict):
                sample["dataset_index"] = torch.tensor(ds_idx, dtype=torch.long)
            yield sample

    # ----------------------------------------------------------
    # Metadata concatenation
    # ----------------------------------------------------------
    def _build_combined_metadata(self) -> CombinedMeta:
        """
        Concatenate dataset episode boundaries into unified metadata.

        This replicates MultiLeRobotDataset's CombinedMeta but for streaming.
        It does NOT store heavy feature definitions—only lightweight indices.

        Behavior:
        - Episodes are concatenated in order:
              ds0 episodes → ds1 episodes → ...
        - Frame indexing becomes continuous across datasets.
        """
        episodes = {
            "dataset_from_index": [],
            "dataset_to_index": [],
        }

        running_frame = 0
        stats: Dict[str, Any] = {}
        schemas: Dict[str, Any] = {}

        for i, ds in enumerate(self.datasets):
            from_index = np.asarray(ds.meta.episodes["dataset_from_index"]) + running_frame
            to_index = np.asarray(ds.meta.episodes["dataset_to_index"]) + running_frame

            episodes["dataset_from_index"].extend(from_index.tolist())
            episodes["dataset_to_index"].extend(to_index.tolist())

            label = (
                getattr(ds, "repo_id", None)
                or getattr(ds.meta, "repo_id", None)
                or getattr(getattr(ds.meta, "root", None), "name", None)
                or f"dataset_{i}"
            )
            if getattr(ds.meta, "stats", None):
                stats[str(label)] = ds.meta.stats
            if getattr(ds.meta, "schema", None) is not None:
                schemas[str(label)] = ds.meta.schema

            running_frame += ds.num_frames

        if stats:
            stats = {"_schema": "multi_repo_v1", **stats}
        return CombinedMeta(episodes=episodes, stats=stats, schemas=schemas)

    def __repr__(self) -> str:
        return (
            f"MultiStreamingLeRobotDataset(\n"
            f"  Num datasets: {len(self.datasets)},\n"
            f"  Num frames (approx): {self.num_frames},\n"
            f"  Num episodes (approx): {self.num_episodes},\n"
            f"  Has weights: {self.dataset_weights is not None},\n"
            f")"
        )
