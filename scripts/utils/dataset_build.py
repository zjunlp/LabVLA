"""Dataset construction: LeRobot v2.1 adapter + Transform chain + multi-repo
PI0 mixture.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from scripts.utils.dataloader import SkipBadSamplesDataset
from scripts.utils.dataset_helpers import (
    TransformedAdapterDataset,
    _check_gripper_semantic_compat,
    _parse_csv_set,
)


def build_dataset(args):
    """Build dataset from LeRobot v2.1 repos using adapter + Transform chain.

    Flow: read fps from info.json -> create adapter -> hydrate transforms -> wrap.
    """
    from adapters import create_adapter
    from transforms.core import hydrate_all, compose
    from dataset.utils import load_json, cast_stats_to_numpy
    from dataset.transforms import ImageTransforms
    from policies.LabVLA.configuration_labvla import LabVLADatasetConfig

    # Filter empty strings so "a,b," / "a,,b" don't create a phantom repo_id=""
    # that poisons path construction later.
    repo_ids = [r.strip() for r in args.repo_ids.split(",") if r.strip()]
    if not repo_ids:
        raise ValueError(
            f"--repo_ids={args.repo_ids!r} parsed to an empty list. "
            f"Pass at least one non-empty repo name."
        )
    data_root = getattr(args, "data_root", None)
    action_supervision_skip_repos = _parse_csv_set(
        getattr(args, "action_supervision_skip_repos", None)
    )
    if action_supervision_skip_repos:
        unknown_skip_repos = action_supervision_skip_repos.difference(repo_ids)
        if unknown_skip_repos:
            raise ValueError(
                "--action_supervision_skip_repos contains repos not present in "
                f"--repo_ids: {sorted(unknown_skip_repos)}"
            )
        if getattr(args, "training_phase", "posttrain") != "vlm_pretrain":
            raise ValueError(
                "--action_supervision_skip_repos is only implemented for "
                "training_phase='vlm_pretrain', where action supervision is "
                "FAST CE. Posttrain still has MSE action loss and must use "
                "semantic-compatible action targets."
            )
        logging.warning(
            "[action-supervision] FAST action CE will be disabled for repos: %s. "
            "Their annotation/text losses remain active.",
            sorted(action_supervision_skip_repos),
        )

    # PI0MixtureDataset (the map-style multi-repo dataset) honors `weights`
    # natively in __getitem__, so there is no streaming-vs-weights guard here.
    dataset_weights = getattr(args, "dataset_weights", None)
    streaming = getattr(args, "streaming", False)  # noqa: F841 reserved for future StreamingLeRobotDataset wiring

    # Parse --dataset_schema into a per-repo map of resolved DatasetSchema.
    # Accepted forms:
    #   None / "" -> no override, fall through to manifest/auto-infer.
    #   "name"    -> same spec applied to all repos.
    #   "repoA=spec1,repoB=spec2" -> per-repo map (if len(repo_ids) > 1).
    schema_spec_arg = getattr(args, "dataset_schema", None)
    per_repo_schema: dict[str, object] = {}
    if schema_spec_arg:
        from schema import resolve
        if "=" in schema_spec_arg:
            for entry in schema_spec_arg.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if "=" not in entry:
                    raise ValueError(
                        f"--dataset_schema map entry {entry!r} missing '=' "
                        f"separator; expected 'repoA=spec'."
                    )
                repo, spec = entry.split("=", 1)
                per_repo_schema[repo.strip()] = resolve(spec.strip())
        else:
            resolved = resolve(schema_spec_arg.strip())
            for rid in repo_ids:
                per_repo_schema[rid] = resolved
        for rid in repo_ids:
            if rid in per_repo_schema:
                logging.info(
                    "Using --dataset_schema override for %s: schema_id=%s",
                    rid, per_repo_schema[rid].schema_id,
                )

    # Load external stats if provided. Fail loud rather than fall back: a typo'd
    # or unmounted path would otherwise silently regress to adapter-discovered
    # stats. _validate_stats_chunk_size is shared so both the global
    # --external_stats_path and the per-repo external_stats_map enforce
    # _chunk_size == args.chunk_size (a mismatch silently miscomputes
    # normalization for the action-chunk distribution).
    def _validate_stats_chunk_size(stats: dict, source: str) -> None:
        ext_cs = stats.get("_chunk_size") if isinstance(stats, dict) else None
        if ext_cs is None:
            logging.warning(
                "external_stats %s has no _chunk_size — likely legacy stats. "
                "Verify it was generated against chunk_size=%d.",
                source, args.chunk_size,
            )
            return
        if int(ext_cs) != int(args.chunk_size):
            raise ValueError(
                f"external_stats _chunk_size={int(ext_cs)} does not match "
                f"--chunk_size={int(args.chunk_size)} (source: {source}). "
                "Either regenerate stats with the matching chunk_size or "
                "fix the launch flag."
            )

    external_stats = None
    if getattr(args, "external_stats_path", None):
        stats_path = Path(args.external_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"--external_stats_path was set but the file does not exist: "
                f"{stats_path}. Refusing to silently fall back to "
                f"adapter-discovered stats."
            )
        external_stats = cast_stats_to_numpy(load_json(stats_path))
        logging.info(f"Loaded external stats from {stats_path}")
        _validate_stats_chunk_size(external_stats, str(stats_path))

    # Per-repo stats overrides; entries override the global
    # --external_stats_path for matching repo_ids.
    from data_process.stats.resolver import parse_external_stats_map
    external_stats_map = parse_external_stats_map(
        getattr(args, "external_stats_map", None)
    )
    if external_stats_map:
        logging.info(
            f"Parsed external_stats_map: {len(external_stats_map)} per-repo overrides"
        )

    # Image augmentation
    image_transforms = None
    if getattr(args, "image_augmentation", True):
        from dataset.transforms import ImageTransformsConfig, ImageTransformConfig
        cfg = ImageTransformsConfig(
            enable=True,
            max_num_transforms=3,
            random_order=True,
            tfs={
                "brightness": ImageTransformConfig(weight=0.3, type="ColorJitter", kwargs={"brightness": (0.7, 1.3)}),
                "contrast": ImageTransformConfig(weight=0.4, type="ColorJitter", kwargs={"contrast": (0.6, 1.4)}),
                "saturation": ImageTransformConfig(weight=0.5, type="ColorJitter", kwargs={"saturation": (0.5, 1.5)}),
                "sharpness": ImageTransformConfig(weight=0.3, type="SharpnessJitter", kwargs={"sharpness": (0.5, 1.5)}),
            },
        )
        image_transforms = ImageTransforms(cfg)
        image_transforms.set_base_seed(int(getattr(args, "seed", 42)))

    # repo_id is required by parent DatasetConfig but unused here.
    ds_cfg = LabVLADatasetConfig(
        repo_id=repo_ids[0],
        height=args.image_height,
        width=args.image_width,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        action_mode=getattr(args, "action_mode", "delta"),
        use_fast_tokenizer=getattr(args, "use_fast_tokenizer", False),
        fast_tokenizer_path=getattr(args, "fast_tokenizer_path", "/all-flash-data/Embodied_models/fast"),
        discrete_action_vocab_size=getattr(args, "discrete_action_vocab_size", 2048),
        # Fallback aligned to the CLI/dataclass default (240) so arg-less callers
        # (unit tests, notebooks building config from a partial Namespace) don't
        # silently downshift and re-truncate FAST/annotation targets.
        discrete_action_max_length=getattr(args, "discrete_action_max_length", 240),
        source_shape_convergence=getattr(args, "source_shape_convergence", False),
        # vlm_pretrain only: drives DiscretizeStateTransformFn insertion + Qwen3VL
        # processor max_length bump.
        training_phase=getattr(args, "training_phase", "posttrain"),
        discretize_state_in_vlm_pretrain=getattr(
            args, "discretize_state_in_vlm_pretrain", True,
        ),
        # Plumb the local VLM path so Qwen processor + AnnotationTokenize don't
        # fall back to the HF repo id under offline mode.
        vlm_pretrained_path=args.vlm_pretrained_path,
        # LabUtopia gripper canonicalize for OXE-pretrain warmstart.
        snap_gripper_to_binary=getattr(args, "snap_gripper_to_binary", False),
        gripper_max_width=getattr(args, "gripper_max_width", 0.04),
        gripper_canonical_dim=getattr(args, "gripper_canonical_dim", 7),
    )

    datasets = []
    all_stats = {}
    all_schemas = {}
    per_repo_frames: list[int] = []

    is_single_repo = len(repo_ids) == 1
    for repo_id in repo_ids:
        # Resolve root path. The "data_root IS the dataset root" branch is
        # single-repo only; multi-repo MUST use the per-repo
        # <data_root>/<repo_id>/ layout (else repos 2..N read repo-1's
        # info.json) and fail loud if any repo is missing.
        root = None
        if data_root:
            candidate = Path(data_root) / repo_id
            if candidate.exists():
                root = str(candidate)
            elif (is_single_repo
                    and Path(data_root).exists()
                    and (Path(data_root) / "meta" / "info.json").exists()):
                # Single-repo only: allow data_root itself to be the dataset
                # root (back-compat for legacy launch scripts).
                root = data_root
            else:
                # Multi-repo: a missing per-repo subdir is fatal (falling
                # through would load repo-1's info.json against repo-N's frames).
                raise FileNotFoundError(
                    f"build_dataset: data_root={data_root!r} does not contain "
                    f"a per-repo directory for {repo_id!r}. Multi-repo runs "
                    f"require <data_root>/<repo_id>/ for every repo (got "
                    f"{len(repo_ids)} repos)."
                )

        # Read fps from info.json first (avoids creating adapter twice).
        # VQA-only repos have meta/vqa_manifest.json instead of info.json — they
        # have no fps / no time-aligned action chunks, so skip the lookup and
        # pass delta_timestamps=None to the adapter (the VQA adapter ignores it).
        info_path = Path(root or data_root) / "meta" / "info.json"
        vqa_manifest = Path(root or data_root) / "meta" / "vqa_manifest.json"
        if info_path.exists():
            with open(info_path) as _info_f:
                _info = json.load(_info_f)
            fps = _info["fps"]
            # If data_process.cleanup invalidated stats but they were never
            # recomputed, training would use a wrong norm; reject at the entry
            # point. External stats (global or per-repo) override the on-disk
            # invalid stats. LABVLA_ALLOW_STATS_INVALIDATED=1 opts out.
            _has_external_for_this_repo = bool(external_stats) or (
                repo_id in external_stats_map
            )
            if _info.get("stats_invalidated") and not _has_external_for_this_repo and \
                    os.environ.get("LABVLA_ALLOW_STATS_INVALIDATED") != "1":
                raise ValueError(
                    f"{info_path}: stats_invalidated=true (set by "
                    f"data_process/cleanup/apply.py). Recompute stats via "
                    "`python -m data_process stats --dataset <root>` or "
                    "supply --external_stats_path / per-repo external_stats_map. "
                    "Set LABVLA_ALLOW_STATS_INVALIDATED=1 to bypass."
                )
            delta_timestamps = {"action": [i / fps for i in range(args.chunk_size)]}
        elif vqa_manifest.exists():
            delta_timestamps = None
            logging.info(
                f"[build_dataset] {repo_id!r}: VQA-only repo "
                f"(meta/vqa_manifest.json present); skipping fps lookup."
            )
        else:
            raise FileNotFoundError(
                f"build_dataset: neither {info_path} nor {vqa_manifest} found. "
                "Need either a LeRobot info.json or a VQA vqa_manifest.json."
            )

        # 1. Create adapter. A --dataset_schema match for this repo is passed
        #    through as override (Tier 0 of discover_schema). A per-repo
        #    external_stats_map entry overrides the global --external_stats_path.
        effective_external_stats = external_stats
        if repo_id in external_stats_map:
            map_path = Path(external_stats_map[repo_id])
            if map_path.exists():
                effective_external_stats = cast_stats_to_numpy(load_json(map_path))
                logging.info(
                    f"Loaded external stats for {repo_id!r} from {map_path} "
                    f"(via --external_stats_map)"
                )
                # Apply the same chunk_size validation as --external_stats_path:
                # a repo-specific stats file built for a different chunk_size
                # would silently miscompute action-chunk normalization.
                _validate_stats_chunk_size(
                    effective_external_stats,
                    f"{repo_id!r}@{map_path}",
                )
            else:
                raise FileNotFoundError(
                    f"--external_stats_map[{repo_id!r}] = {map_path} does not exist"
                )

        adapter = create_adapter(
            repo_id=repo_id,
            root=root,
            data_root=data_root,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            external_stats=effective_external_stats,
            override_schema=per_repo_schema.get(repo_id),
        )

        # 2. Resolve schema — populated by the adapter via discover_schema.
        schema = getattr(adapter.meta, "schema", None)
        if schema is None:
            raise RuntimeError(
                f"build_dataset: adapter for {repo_id!r} yielded no "
                f"DatasetSchema. Create meta/labvla_manifest.json."
            )

        # 3. Resolve stats (adapter stats + optional external override). Copy
        # before update to avoid mutating adapter.meta.stats in-place. Use
        # effective_external_stats (per-repo override) here, not the global
        # external_stats, so a map entry isn't clobbered after the adapter merge.
        stats = dict(adapter.meta.stats) if adapter.meta.stats else {}
        if effective_external_stats:
            stats.update(effective_external_stats)

        # Also validate chunk_size for adapter-discovered stats (meta/stats.json),
        # not just the external paths above: a mismatched chunk_size silently
        # drives delta-action normalization with the wrong distribution.
        # (warns when _chunk_size is absent in legacy stats; raises on mismatch.)
        if stats:
            _validate_stats_chunk_size(stats, f"{repo_id} (adapter.meta.stats)")

        # 4. Hydrate transforms.
        transforms = list(ds_cfg.data_transforms.inputs)
        gripper_norm_mode = getattr(args, "gripper_norm_mode", None)
        if gripper_norm_mode is None:
            gripper_norm_mode = "q01_q99" if getattr(args, "normalize_gripper", True) else "noop"
        transforms = hydrate_all(
            transforms, schema=schema, stats=stats,
            action_mode=getattr(args, "action_mode", "delta"),
            normalize_arm_joints=getattr(args, "normalize_arm_joints", True),
            normalize_gripper=gripper_norm_mode,
        )
        if repo_id in action_supervision_skip_repos:
            from transforms.fast_action import DropFastActionSupervisionTransformFn

            transforms.append(DropFastActionSupervisionTransformFn(
                reason=(
                    "repo excluded from FAST action CE by "
                    "--action_supervision_skip_repos"
                )
            ))
            logging.warning(
                "[action-supervision] %s: dropping FAST action tokens after "
                "transforms; annotation losses remain enabled.",
                repo_id,
            )
        transform_fn = compose(transforms)

        # 5. Wrap with transforms.
        transformed = TransformedAdapterDataset(adapter, transform_fn)
        if getattr(args, "data_error_skip", True):
            transformed = SkipBadSamplesDataset(
                transformed,
                enabled=True,
                max_attempts=getattr(args, "data_error_skip_max_attempts", 64),
                log_first=getattr(args, "data_error_log_first", 20),
                label=repo_id,
            )
        datasets.append(transformed)
        all_stats[repo_id] = stats
        all_schemas[repo_id] = getattr(adapter.meta, "schema", None)
        per_repo_frames.append(int(adapter.num_frames))

        logging.info(
            f"Loaded: {repo_id} ({adapter.num_episodes} eps, {adapter.num_frames} frames, "
            f"robot={adapter.meta.robot_type}, format={type(adapter).__name__})"
        )

    # Asset-binding summary: one line per repo logging which schema / stats /
    # gripper semantics actually got bound (independent of what the launcher
    # claimed), to diagnose silent mis-bindings that otherwise only surface as
    # eval-time anomalies.
    _action_mode = getattr(args, "action_mode", "delta")
    logging.info(
        f"[asset-binding] training_phase={getattr(args, 'training_phase', 'posttrain')!r} "
        f"action_mode={_action_mode!r} repos={len(repo_ids)}"
    )
    for _rid in repo_ids:
        _sch = all_schemas.get(_rid)
        _grip_sem = getattr(_sch, "gripper_semantic", None) if _sch else None
        _sch_id = getattr(_sch, "schema_id", "<none>") if _sch else "<none>"
        _stats_keys = sorted(all_stats.get(_rid, {}).keys())[:6] if all_stats.get(_rid) else []
        logging.info(
            f"[asset-binding]   {_rid}: schema_id={_sch_id!r} "
            f"gripper_semantic={_grip_sem!r} stats_keys={_stats_keys}"
        )

    # Cross-repo gripper-semantic guard: a mix of e.g. robointer (velocity) +
    # LabUtopia (metric width) would otherwise train on contradictory targets
    # (q01/q99 aligns scale, not semantics). Schemas with gripper_semantic=None
    # are skipped (legacy / opt-in).
    if len(datasets) >= 2:
        _check_gripper_semantic_compat(
            repo_ids,
            [all_schemas.get(r) for r in repo_ids],
            ignored_repo_ids=action_supervision_skip_repos,
        )

    if len(datasets) == 1:
        return datasets[0], all_stats, all_schemas

    # Multi-dataset: pi0 n^0.43 volume-weighted mixing (Section V-A of the pi0
    # paper). --dataset_weights, if present, takes precedence (manual tuning).
    from dataset.pi0_mixture import PI0MixtureDataset
    if getattr(args, "dataset_weights", None):
        raw_weights = [float(w) for w in args.dataset_weights.split(",")]
        if len(raw_weights) != len(datasets):
            raise ValueError(
                f"--dataset_weights has {len(raw_weights)} values but "
                f"{len(datasets)} datasets were loaded."
            )
        logging.info(f"[π0-mix] using manual --dataset_weights override")
    else:
        exponent = 0.43
        raw_weights = [float(n) ** exponent for n in per_repo_frames]
        logging.info(
            f"[π0-mix] computing n^{exponent} weights from frame counts: "
            f"{list(zip(repo_ids, per_repo_frames))}"
        )

    # Forward the global --seed so PI0Mixture sampling is deterministic across
    # ranks/restarts; otherwise two ranks asking the same idx get different
    # samples and resume can't reproduce the pre-restart trajectory.
    multi_ds = PI0MixtureDataset(
        datasets, weights=raw_weights, repo_ids=repo_ids,
        seed=int(getattr(args, "seed", 42)),
    )
    return multi_ds, all_stats, all_schemas
