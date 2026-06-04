#!/usr/bin/env python

# LabVLA Training Script (lerobot LabVLAPolicy + custom data pipeline).
# Architecture: VLM + Action Expert joint training, continuous action generation
# via Flow Matching.

import gc
import json
import logging
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# CUDA allocator config must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.multiprocessing as _mp
# DataLoader workers ship tensors via torch.multiprocessing. The default
# `file_descriptor` strategy fails at high worker x prefetch counts (worker bus
# errors / rebuild_storage_fd in _pin_memory_loop); `file_system` passes tensors
# through /dev/shm and is the robust option for high-throughput multi-worker.
_mp.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, send_to_device
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(PROJ_ROOT))

from policies.LabVLA.modeling_labvla import LabVLAPolicy
from scripts.utils.builders import build_config, build_optimizer
from scripts.utils.checkpoint import (
    _accelerate_state_dir,
    _checkpoint_incomplete_marker,
    _clear_checkpoint_incomplete_marker,
    _is_recoverable_checkpoint_save_error,
    _load_training_state,
    _rotate_checkpoints,
    _torch_load_with_retry,
    _torch_save_with_retry,
    _validate_full_resume_contract,
    _write_checkpoint_incomplete_marker,
    load_checkpoint,
    save_checkpoint,
)
from scripts.utils.cli_args import parse_args
from scripts.utils.collate import LabVLACollator
from scripts.utils.dataloader import (
    DataLoaderWorkerSeed,
    DeterministicBatchSampler,
    should_persist_workers,
)
from scripts.utils.dataset_build import build_dataset
from scripts.utils.dataset_helpers import _try_malloc_trim
from scripts.utils.runtime import (
    AverageMeter,
    BackgroundBatchBuffer,
    _set_epoch_recursive,
    _set_sampler_start_recursive,
    format_time,
    set_seed,
    setup_logging,
)
from utils.storage_retry import run_with_storage_retry


# Pre-train-loop scaffolding (parse_args, build_dataset, build_config /
# build_optimizer, checkpoint helpers) lives in scripts/utils/ (see imports
# above). This file keeps the training loop, the batch-to-device helper, and the
# __main__ entry point.


def convert_batch_for_policy(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train(args):
    # Lock each process to its GPU before any CUDA op, else every process
    # initializes a CUDA context on cuda:0 (~416MB each).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Only enable find_unused_parameters when partial freezing creates unused params
    has_unused = args.freeze_vision_encoder or args.train_expert_only or args.train_vlm_only
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=has_unused)
    # Override the default 10-min NCCL collective timeout: under IO jitter a
    # single slow sample can stall a rank for minutes and trigger false-positive
    # cluster aborts. Raise so storage hiccups don't cascade into crashes.
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    pg_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=max(1, int(args.dist_timeout_seconds)))
    )
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
    )
    is_main = accelerator.is_main_process
    device = accelerator.device

    setup_logging(accelerator)

    # Under DeepSpeed, accelerate's clip_grad_norm_ is a no-op for clipping (it
    # only returns engine.get_global_grad_norm(); max_norm is ignored). Clipping
    # is driven solely by the DeepSpeed config's `gradient_clipping` inside
    # engine.step(). So inject the requested clip value into the live DeepSpeed
    # config here, before accelerator.prepare builds the engine; otherwise
    # --grad_clip_norm is silently ineffective. Non-DeepSpeed backends are
    # unaffected (they clip via clip_grad_norm_ in the loop).
    _ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    _ds_config = getattr(_ds_plugin, "deepspeed_config", None)
    if isinstance(_ds_config, dict):
        _clip = float(getattr(args, "grad_clip_norm", 0.0) or 0.0)
        if _clip > 0:
            _prev = _ds_config.get("gradient_clipping", 0.0)
            _ds_config["gradient_clipping"] = _clip
            if is_main:
                logging.info(
                    "[grad-clip-fix] DeepSpeed gradient_clipping set to %.4g "
                    "(was %r). accelerate.clip_grad_norm_ does NOT clip under "
                    "DeepSpeed; clipping is enforced by engine.step() via this "
                    "config value.",
                    _clip, _prev,
                )
        elif is_main:
            logging.info(
                "[grad-clip-fix] --grad_clip_norm<=0; leaving DeepSpeed "
                "gradient_clipping disabled (no clipping requested)."
            )

    # InitProcessGroupKwargs only sets the default PG (PG 0) timeout. DeepSpeed
    # ZeRO creates extra PGs inside accelerator.prepare() via new_group; without
    # an explicit timeout those fall back to PyTorch's default 600s and AllReduce
    # on PG 1 triggers SIGABRT after 10min even when --dist_timeout_seconds is
    # large. There is no portable public API to set a default PG timeout across
    # torch 2.0..2.5, so wrap new_group itself: callers that don't pass timeout=
    # inherit our value (calls that already pass one are left untouched). Install
    # before accelerator.prepare(). The original is recorded and restored via an
    # atexit hook so the patch doesn't leak to later torch.distributed importers.
    try:
        import atexit as _atexit
        import torch.distributed as _tdist
        from datetime import timedelta as _td
        _orig_new_group = _tdist.new_group
        _ng_target_to = _td(seconds=int(args.dist_timeout_seconds))
        def _new_group_with_timeout(*ng_args, **ng_kwargs):
            if "timeout" not in ng_kwargs:
                ng_kwargs["timeout"] = _ng_target_to
            return _orig_new_group(*ng_args, **ng_kwargs)
        _tdist.new_group = _new_group_with_timeout

        def _restore_new_group():
            try:
                if _tdist.new_group is _new_group_with_timeout:
                    _tdist.new_group = _orig_new_group
            except Exception:
                pass
        _atexit.register(_restore_new_group)

        if is_main:
            logging.info(
                f"[PG-timeout] monkey-patched torch.distributed.new_group → default "
                f"timeout={args.dist_timeout_seconds}s for DeepSpeed/internal PGs "
                f"(workaround: accelerate InitProcessGroupKwargs only sets default PG; "
                f"atexit hook restores the original function on clean exit)"
            )
    except Exception as _e:
        logging.warning(f"[PG-timeout] new_group monkey-patch skipped: {_e}")

    # Graceful Ctrl+C handling. batch_buffer is constructed later; expose it via
    # this dict so the signal handler can reach it when it runs.
    _graceful = {"ctrl_c_count": 0, "batch_buffer": None}

    def _signal_handler(signum, frame):
        _graceful["ctrl_c_count"] += 1
        if _graceful["ctrl_c_count"] == 1:
            logging.warning("\n[LabVLA] Ctrl+C received, shutting down gracefully...")
            # Stop the background data producer FIRST so its thread releases the
            # CUDA pinned/stream buffers before os._exit; otherwise Ctrl+C takes
            # ~35s to free GPU memory on master while workers hang on NCCL.
            bb = _graceful.get("batch_buffer")
            if bb is not None:
                try:
                    bb.stop()
                except Exception as e:
                    logging.warning(f"batch_buffer.stop() in signal handler failed: {e}")
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception as e:
                    logging.warning(f"destroy_process_group in signal handler failed: {e}")
            os._exit(0)
        else:
            os._exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    # Also catch SIGTERM so a launcher's `trap ... TERM` triggers the same
    # graceful batch_buffer + CUDA cleanup; otherwise TERM falls back to Python's
    # default (immediate exit) and leaks pinned memory + ZMQ state.
    signal.signal(signal.SIGTERM, _signal_handler)

    if is_main:
        if args.training_phase == "vlm_pretrain":
            train_mode = "VLM pretrain"
        elif args.knowledge_isolation:
            train_mode = f"Flow Matching + KI (α={args.ki_mse_weight})"
        else:
            train_mode = "Flow Matching, No KI"
        logging.info("=" * 60)
        logging.info(f"LabVLA Training ({train_mode})")
        logging.info("=" * 60)
        logging.info(f"VLM backbone: {args.vlm_pretrained_path}")
        logging.info(f"Num processes: {accelerator.num_processes}")
        logging.info(f"Device: {device}")
        logging.info(f"Attention: {args.attn_implementation}")
        logging.info(f"Gradient checkpointing: {args.gradient_checkpointing}")

        # ki_mse_weight is meaningful only in posttrain+KI; vlm_pretrain loss is
        # pure FAST-CE (no MSE term), so warn if it was explicitly set.
        if args.training_phase == "vlm_pretrain" and args.ki_mse_weight != 10.0:
            logging.warning(
                "[LOW-V5-04] --ki_mse_weight=%s ignored: vlm_pretrain phase "
                "uses pure FAST-CE loss with no MSE term. Set "
                "training_phase=posttrain (with knowledge_isolation=true) "
                "if you want this α weight to take effect.",
                args.ki_mse_weight,
            )

    set_seed(args.seed)

    # cuDNN auto-tuning.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = build_config(args)

    if is_main:
        logging.info("Building dataset...")
    dataset, data_stats, data_schemas = build_dataset(args)
    # Schema for checkpoint serialization: single-repo uses its own; multi-repo
    # uses the first (deployment currently supports single-robot ckpts).
    _active_schema = None
    if data_schemas:
        _first_schema_key = next(iter(data_schemas))
        _active_schema = data_schemas[_first_schema_key]

        # The action head uses a single global action_dim from the first repo's
        # schema, so heterogeneous action_dims across repos would silently
        # miscompute MSE (wider repos drop tail dims; narrower repos re-include
        # padded dims). Fail loud. Scope the guard to POSTTRAIN flow-matching:
        # vlm_pretrain supervises via per-sample FAST/annotation CE, not the
        # global MSE. Repos in --action_supervision_skip_repos are excluded
        # (their action stream is dropped at the transform level).
        _is_posttrain = getattr(args, "training_phase", "posttrain") == "posttrain"
        if _is_posttrain and len(data_schemas) > 1:
            _skip_repos = set(
                (getattr(args, "action_supervision_skip_repos", "") or "").split(",")
            )
            _skip_repos = {r.strip() for r in _skip_repos if r.strip()}
            _action_dims = {
                rid: int(sum(getattr(sch, "action_dims", ()) or ()))
                for rid, sch in data_schemas.items()
                if sch is not None and rid not in _skip_repos
            }
            _unique = set(_action_dims.values())
            if len(_unique) > 1 and os.environ.get(
                "LABVLA_ALLOW_HETEROGENEOUS_ACTION_DIMS"
            ) != "1":
                raise ValueError(
                    "Multi-repo posttrain detected heterogeneous action dims "
                    f"across action-supervised repos: {_action_dims} "
                    f"(skipped: {sorted(_skip_repos) or 'none'}). The action "
                    "head currently uses a single global action_dim derived "
                    "from the first repo's schema, so heterogeneous mixing "
                    "would silently miscompute MSE. Either (a) unify the "
                    "schemas to a canonical action layout, (b) train each "
                    "repo separately, (c) add the divergent repo to "
                    "--action_supervision_skip_repos so its action stream is "
                    "excluded from MSE, or (d) implement per-sample "
                    "action_dim masking. To bypass this guard for ad-hoc "
                    "experimentation, set "
                    "LABVLA_ALLOW_HETEROGENEOUS_ACTION_DIMS=1."
                )
            elif len(_unique) > 1 and is_main:
                logging.warning(
                    "[multi-repo-action-dim] heterogeneous action dims allowed "
                    "via env: %s. Tail dims of wider repos will be ignored by "
                    "loss; padded dims of narrower repos will be re-included.",
                    _action_dims,
                )
        elif not _is_posttrain and len(data_schemas) > 1 and is_main:
            # Explicit log line: the posttrain guard is intentionally bypassed
            # in vlm_pretrain (per-sample FAST/annotation CE, not a global MSE).
            _all_dims = {
                rid: int(sum(getattr(sch, "action_dims", ()) or ()))
                for rid, sch in data_schemas.items() if sch is not None
            }
            logging.info(
                "[multi-repo-action-dim] training_phase=%r → skipping "
                "heterogeneous-action-dim guard (FAST CE / annotation CE "
                "are per-sample, not a global MSE). Repo dims: %s",
                getattr(args, "training_phase", "posttrain"),
                _all_dims,
            )

    # Populate config.output_features["action"] with the REAL action dim from the
    # active schema BEFORE building the policy, so original_action_dim resolves to
    # the schema's value rather than the max_action_dim=32 default from
    # validate_features(). Otherwise _forward_posttrain computes MSE over all 32
    # dims, and the ~24 zero-padded dims dominate the gradient (u_t = 0 - noise on
    # padded dims), starving the real arm/gripper dims (gripper alone ~3%).
    if _active_schema is not None:
        from dataset.utils import PolicyFeature, FeatureType
        real_action_dim = int(sum(_active_schema.action_dims))
        if real_action_dim > 0 and real_action_dim <= config.max_action_dim:
            config.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(real_action_dim,),
            )
            if is_main:
                logging.info(
                    "[action-dim-fix] config.output_features[action].shape set to "
                    "(%d,) from schema %r (was defaulting to max_action_dim=%d).",
                    real_action_dim,
                    _active_schema.schema_id,
                    config.max_action_dim,
                )
        else:
            # Refuse to start on an out-of-range real_action_dim rather than
            # fall through and train on padded dims; raise max_action_dim or fix
            # the schema.
            raise ValueError(
                f"Schema {_active_schema.schema_id!r} real action_dim={real_action_dim} "
                f"is outside [1, max_action_dim={config.max_action_dim}]. "
                "Loss would otherwise train on padded dims (real_action_dim "
                ">max_action_dim is unrepresentable; ==0 means no action "
                "supervision). Raise --max_action_dim or fix the schema."
            )

        # Propagate gripper_action_dims from the schema into the config so the
        # per-element loss weighting can find the gripper dims at loss time.
        # (gripper_loss_weight itself is mirrored in build_config; this runtime
        # override is needed because the CLI doesn't know the schema yet then.)
        _grip_dims = tuple(
            int(d) for d in getattr(_active_schema, "gripper_action_dims", ()) or ()
        )
        if _grip_dims:
            config.gripper_action_dims = _grip_dims

        # When gripper_loss_weight > 1.0 in a multi-repo posttrain, all
        # action-supervised schemas must share gripper_action_dims AND
        # gripper_semantic; otherwise the weighting applies the first repo's
        # gripper layout to every batch (wrong dims weighted, contradictory
        # physics). Fail loud unless LABVLA_ALLOW_HETEROGENEOUS_GRIPPER=1.
        _is_posttrain_again = (
            getattr(args, "training_phase", "posttrain") == "posttrain"
        )
        if (
            _is_posttrain_again
            and config.gripper_loss_weight != 1.0
            and len(data_schemas) > 1
        ):
            _skip = {
                r.strip()
                for r in (getattr(args, "action_supervision_skip_repos", "") or "").split(",")
                if r.strip()
            }
            _grip_layouts = {
                rid: (
                    tuple(int(d) for d in getattr(sch, "gripper_action_dims", ()) or ()),
                    getattr(sch, "gripper_semantic", None),
                )
                for rid, sch in data_schemas.items()
                if sch is not None and rid not in _skip
            }
            _unique_layouts = set(_grip_layouts.values())
            if len(_unique_layouts) > 1 and os.environ.get(
                "LABVLA_ALLOW_HETEROGENEOUS_GRIPPER"
            ) != "1":
                raise ValueError(
                    "Multi-repo posttrain with --gripper_loss_weight!=1.0 "
                    f"detected heterogeneous gripper layouts: {_grip_layouts}. "
                    "The current weighting code applies a SINGLE gripper-dim "
                    "mask (taken from the first repo's schema), so other "
                    "repos' gripper supervision would be either un-weighted "
                    "or wrongly weighted at non-gripper dims, and "
                    "incompatible gripper_semantic would mix contradictory "
                    "physics. Either (a) unify schemas to a canonical "
                    "gripper layout/semantic, (b) drop --gripper_loss_weight "
                    "to 1.0, (c) skip-list the divergent repo, or (d) "
                    "implement per-sample gripper-mask weighting. Bypass "
                    "with LABVLA_ALLOW_HETEROGENEOUS_GRIPPER=1."
                )
        if is_main and config.gripper_loss_weight != 1.0:
            logging.info(
                "[gripper-loss-weight] gripper-dim MSE scaled by %.3f at dims %s "
                "(from schema %r). Default 1.0 = no-op; >1.0 counteracts the "
                "structural dilution of a single gripper scalar against %d "
                "joint dims in uniform-average MSE.",
                config.gripper_loss_weight, _grip_dims,
                _active_schema.schema_id,
                max(real_action_dim - len(_grip_dims), 0),
            )

    # Detect whether any active schema declares annotation_losses; if not, skip
    # the annotation meter/logging to keep the MSE-only path quiet.
    _has_annotation_losses = False
    if data_schemas:
        for _sch in data_schemas.values():
            if _sch is not None and bool(getattr(_sch, "annotation_losses", ())):
                _has_annotation_losses = True
                break

    # Startup-time preflight for the runtime guard in _forward_posttrain's
    # non-KI+annotation branch (which raises when dit_layerwise_vlm_features=True
    # but only fires on the first annotation batch, minutes into the run). Reject
    # the bad combination here, before any GPU memory is touched.
    if (
        _has_annotation_losses
        and not bool(getattr(args, "knowledge_isolation", False))
        and bool(getattr(args, "dit_layerwise_vlm_features", False))
    ):
        raise ValueError(
            "[R9-F4] training_phase='posttrain' + knowledge_isolation=False + "
            "dit_layerwise_vlm_features=True is unsupported when any active "
            "schema declares annotation_losses. The non-KI+annotation branch "
            "of _forward_posttrain manually calls self.vlm.language_model "
            "WITHOUT output_hidden_states=True, so layerwise users would get "
            "last-layer hidden only on annotation batches — polluting any "
            "layerwise ablation. Fix one of: (a) disable "
            "dit_layerwise_vlm_features, (b) switch to "
            "knowledge_isolation=True, or (c) remove annotation_losses from "
            "the active schema(s)."
        )

    # Annotation CE in _compute_annotation_ces() lazily imports Liger's fused
    # CE kernel at the first annotation batch; a missing liger_kernel would
    # otherwise raise ImportError tens of minutes into the run. Probe the import
    # here so it fails fast when annotation_losses is in scope. The helper keeps
    # its own try/except as a second-line defense.
    if _has_annotation_losses:
        try:
            from liger_kernel.transformers import (  # noqa: F401
                LigerFusedLinearCrossEntropyLoss,
            )
        except ImportError as _e:
            raise RuntimeError(
                "[R10-M10-03] An active dataset schema declares "
                "`annotation_losses`, but `liger_kernel` cannot be imported. "
                "The annotation CE path in `_compute_annotation_ces` depends "
                "on Liger's fused linear+CE kernel; without it the first "
                "annotation batch (typically several minutes into training) "
                "would crash. Install liger-kernel (>=0.7) or remove "
                "annotation_losses from the active schema. Original import "
                f"error: {_e}"
            )

    # Extract norm_stats to save in checkpoints (deployment loads from the ckpt
    # dir, not a separate --norm_stats_path). Multi-repo saves a dict-of-repos
    # tagged so deployment can index by repo_id; single-repo keeps the flat
    # schema for back-compat with deployment/serve_labvla.py.
    _active_norm_stats = None
    if data_stats:
        from dataset.utils import serialize_dict
        if len(data_stats) == 1:
            first_key = next(iter(data_stats))
            _active_norm_stats = serialize_dict(data_stats[first_key])
        else:
            _active_norm_stats = {
                "_schema": "multi_repo_v1",
                **{k: serialize_dict(v) for k, v in data_stats.items()},
            }

    full_resume_requested = bool(args.resume) and not getattr(args, "load_weights_only", False)
    resume_accelerate_state_dir = _accelerate_state_dir(args.resume) if full_resume_requested else None
    resume_uses_accelerate_state = (
        bool(resume_accelerate_state_dir)
        and resume_accelerate_state_dir.exists()
    )
    pre_resume_train_state = {}
    pre_resume_start_step = 0
    pre_resume_start_epoch = 0
    if full_resume_requested:
        if _checkpoint_incomplete_marker(args.resume).exists():
            raise RuntimeError(
                f"Checkpoint {args.resume} is marked incomplete "
                f"({_checkpoint_incomplete_marker(args.resume)} exists). "
                "Use a completed checkpoint or inspect the prior save failure."
            )
        pre_resume_train_state = _load_training_state(args.resume)
        if pre_resume_train_state:
            _validate_full_resume_contract(pre_resume_train_state.get("args", {}) or {}, args)
            pre_resume_start_step = int(pre_resume_train_state.get("step", 0))
            pre_resume_start_epoch = int(pre_resume_train_state.get("epoch", 0))
        resume_requires_distributed_state = bool(
            pre_resume_train_state.get("full_resume_requires_distributed_state", True)
        )
        if (
            resume_requires_distributed_state
            and not resume_uses_accelerate_state
            and not getattr(args, "allow_legacy_partial_resume", False)
        ):
            raise RuntimeError(
                f"Checkpoint {args.resume} has no accelerate_state/ distributed "
                "checkpoint. Refusing legacy partial resume because DeepSpeed "
                "ZeRO optimizer shards cannot be restored correctly from the "
                "old rank0-only training_state.pt. Use --load_weights_only true "
                "for warmstart, or --allow_legacy_partial_resume true only for "
                "explicit debugging."
            )
        if (
            not resume_requires_distributed_state
            and not resume_uses_accelerate_state
            and is_main
        ):
            logging.warning(
                "Checkpoint %s has no accelerate_state/ and declares "
                "full_resume_requires_distributed_state=false; resuming from "
                "standalone training_state.pt only.",
                args.resume,
            )

    # Each DataLoader worker needs a deterministic but rank+worker-unique seed;
    # otherwise all workers share torch's post-fork RNG state and resume after
    # preemption drifts silently.
    worker_init_fn = DataLoaderWorkerSeed(
        base_seed=int(args.seed),
        rank=int(accelerator.process_index),
        num_workers=max(1, int(args.num_workers)),
    )

    # LabVLACollator lives in scripts/utils/collate.py so multiprocessing can
    # pickle it for DataLoader workers (nested closures silently fail to pickle,
    # falling back to default_collate).
    collator = LabVLACollator(
        trim_token_padding_to_batch=args.trim_token_padding_to_batch,
    )
    if is_main and args.trim_token_padding_to_batch:
        logging.info(
            "[phase-b] trim_token_padding_to_batch=true: "
            "collate removes all-pad prompt/FAST/annotation token tails per batch"
        )

    persistent_workers = should_persist_workers(
        num_workers=args.num_workers,
        dataset=dataset,
        homogeneous_mixture_batches=getattr(args, "homogeneous_mixture_batches", False),
    )
    if is_main and args.num_workers > 0 and not persistent_workers:
        logging.info(
            "[π0-mix] persistent_workers=false on non-homogeneous mixture path "
            "so worker-side PI0MixtureDataset sees epoch changes"
        )

    dataloader_prefetch_factor = (
        max(1, int(args.dataloader_prefetch_factor)) if args.num_workers > 0 else None
    )
    if is_main and args.num_workers > 0:
        logging.info(
            "DataLoader workers=%d prefetch_factor=%d persistent_workers=%s",
            args.num_workers,
            dataloader_prefetch_factor,
            persistent_workers,
        )

    dataloader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=dataloader_prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        collate_fn=collator,
    )

    if getattr(args, "homogeneous_mixture_batches", False):
        from dataset.pi0_mixture import PI0MixtureBatchSampler, PI0MixtureDataset

        if isinstance(dataset, PI0MixtureDataset):
            total_batches = max(1, len(dataset) // int(args.batch_size))
            start_batch_index = (
                int(pre_resume_start_step) % total_batches
                if full_resume_requested and pre_resume_start_step > 0
                else 0
            )
            batch_sampler = PI0MixtureBatchSampler(
                dataset,
                batch_size=args.batch_size,
                drop_last=True,
                seed=args.seed,
                start_batch_index=start_batch_index,
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                **dataloader_kwargs,
            )
            if is_main:
                logging.info(
                    "[π0-mix] homogeneous_mixture_batches=true: "
                    f"batch_size={args.batch_size}, batches/epoch={len(batch_sampler):,}, "
                    f"start_batch_index={start_batch_index}"
                )
        else:
            raise TypeError(
                "--homogeneous_mixture_batches true requires PI0MixtureDataset; "
                f"got {type(dataset).__name__}. Refusing to silently change batching."
            )
    elif getattr(args, "resumable_dataloader", True):
        total_batches = max(1, len(dataset) // int(args.batch_size))
        start_batch_index = (
            int(pre_resume_start_step) % total_batches
            if full_resume_requested and pre_resume_start_step > 0
            else 0
        )
        batch_sampler = DeterministicBatchSampler(
            len(dataset),
            args.batch_size,
            drop_last=True,
            seed=args.seed,
            epoch=pre_resume_start_epoch,
            start_batch_index=start_batch_index,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **dataloader_kwargs,
        )
        if is_main:
            logging.info(
                "[resume] deterministic batch sampler enabled: "
                f"batch_size={args.batch_size}, batches/epoch={len(batch_sampler):,}, "
                f"start_batch_index={start_batch_index}, epoch={pre_resume_start_epoch}"
            )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs,
        )

    if getattr(dataloader, "batch_size", None) is None:
        deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", None)
        if isinstance(deepspeed_config, dict):
            micro_batch = int(args.batch_size)
            accumulation_steps = int(args.gradient_accumulation_steps)
            global_batch = micro_batch * accumulation_steps * int(accelerator.num_processes)
            if deepspeed_config.get("train_micro_batch_size_per_gpu") == "auto":
                deepspeed_config["train_micro_batch_size_per_gpu"] = micro_batch
            if deepspeed_config.get("gradient_accumulation_steps") == "auto":
                deepspeed_config["gradient_accumulation_steps"] = accumulation_steps
            if deepspeed_config.get("train_batch_size") == "auto":
                deepspeed_config["train_batch_size"] = global_batch
            if is_main:
                logging.info(
                    "[deepspeed] explicit batch sizes for custom batch_sampler: "
                    f"micro={micro_batch}, grad_accum={accumulation_steps}, "
                    f"global={global_batch}"
                )

    if args.job_name is None:
        args.job_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_labvla_pretrain"
    output_dir = Path(args.output_dir) / args.job_name
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {output_dir}")
        try:
            setup_logging(accelerator, log_file=output_dir / "train.log")
            logging.info(f"train.log file handler added at {output_dir / 'train.log'}")
        except Exception as e:
            logging.warning(f"Could not attach train.log file handler: {e}")
    accelerator.wait_for_everyone()

    if is_main:
        logging.info("Building model...")
    policy = LabVLAPolicy(config)
    if is_main:
        logging.info(str(policy))

    # scheduler.step()/ema.update() advance only at gradient-sync boundaries
    # (once per optimizer update = once per GA micro-batches), so warmup/decay
    # steps and the cosine decay-vs-horizon check are in OPTIMIZER-step units.
    # The loop iterates args.steps MICRO-batches => args.steps // GA optimizer
    # updates. Passing the raw micro-step count would make the horizon check
    # unit-inconsistent (GA>1 could under-complete LR decay without tripping the
    # auto-scale guard). Pass the true optimizer-step horizon. GA=1 is unchanged.
    _ga = max(1, int(args.gradient_accumulation_steps))
    _optimizer_steps = args.steps // _ga
    if _ga > 1 and is_main:
        if args.steps % _ga != 0:
            logging.warning(
                "[T1] --steps=%d is not a multiple of gradient_accumulation_steps=%d; "
                "the final %d micro-batch(es) never reach a sync boundary, so their "
                "gradients are dropped. Set --steps to a multiple of GA.",
                args.steps, _ga, args.steps % _ga,
            )
        logging.info(
            "[T1] gradient_accumulation_steps=%d → %d optimizer updates over %d "
            "micro-batches. --warmup_steps/--decay_steps are in optimizer-step "
            "units (advance once per update); note --save_freq/--steps and the "
            "saved resume `step` remain micro-batch counts.",
            _ga, _optimizer_steps, args.steps,
        )
    optimizer, scheduler = build_optimizer(policy, config, num_training_steps=_optimizer_steps)

    start_step = 0
    start_epoch = 0  # nominal epoch counter; resumed below if available.
    # Load model weights BEFORE accelerator.prepare (prepare wraps the policy for
    # DS/DDP; post-prepare state_dict keys have a module. prefix that won't match
    # a freshly-built policy).
    if args.resume:
        load_weights_only = getattr(args, "load_weights_only", False)
        # Full resume applies strict=True (architecture must match); warmstart
        # keeps strict=False (intentional shape mismatches when transferring a
        # pretrain head).
        if load_weights_only:
            load_checkpoint(
                args.resume, policy, optimizer=None, scheduler=None,
                strict=False,
            )
            start_step = 0
            if is_main:
                logging.info(f"load_weights_only: loaded model weights from {args.resume}, starting from step 0")
        elif resume_uses_accelerate_state:
            if is_main:
                logging.info(
                    f"Full resume will load distributed accelerator state from "
                    f"{resume_accelerate_state_dir} after accelerator.prepare"
                )
        else:
            # Legacy partial-resume path: load only model weights here;
            # optimizer/scheduler/RNG are restored AFTER accelerator.prepare so a
            # DS ZeRO-formatted optimizer state can be accepted by the DS-wrapped
            # optimizer (avoids KeyError: 'param_groups'). strict=True so
            # architectural mismatch fails loudly.
            load_checkpoint(
                args.resume, policy, optimizer=None, scheduler=None,
                strict=True,
            )
            if is_main:
                logging.info(
                    f"Legacy partial resume loaded model weights from {args.resume}; "
                    f"will best-effort restore optimizer/scheduler/RNG after prepare"
                )

    policy, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, scheduler
    )

    if args.resume and not getattr(args, "load_weights_only", False):
        # Optimizer is now DS-wrapped, so reloading the DS-formatted
        # optimizer_state_dict from training_state.pt succeeds.
        if resume_uses_accelerate_state:
            if is_main:
                logging.info(f"Loading distributed accelerator state: {resume_accelerate_state_dir}")
            accelerator.load_state(str(resume_accelerate_state_dir))
            train_state = pre_resume_train_state
            start_step = int(train_state.get("step", 0))
            start_epoch = int(train_state.get("epoch", 0))
            if is_main:
                logging.info(
                    f"Full resume restored distributed model/optimizer/scheduler/RNG "
                    f"at step={start_step}, epoch={start_epoch}"
                )
        else:
            train_state_path = Path(args.resume) / "training_state" / "training_state.pt"
            train_state = _torch_load_with_retry(train_state_path) if train_state_path.exists() else {}
            if "optimizer_state_dict" in train_state:
                try:
                    optimizer.load_state_dict(train_state["optimizer_state_dict"])
                    if is_main:
                        logging.info("Restored optimizer state (post-prepare, DS-aware)")
                except Exception as e:
                    if is_main:
                        logging.warning(
                            f"Post-prepare optimizer restore failed: {e}. "
                            f"Training will continue but optimizer momentum is lost."
                        )
            if "scheduler_state_dict" in train_state:
                try:
                    scheduler.load_state_dict(train_state["scheduler_state_dict"])
                    if is_main:
                        logging.info("Restored scheduler state (post-prepare)")
                except Exception as e:
                    if is_main:
                        logging.warning(f"Post-prepare scheduler restore failed: {e}")
            if "rng_state" in train_state:
                rng = train_state["rng_state"]
                try:
                    import random as _random
                    import numpy as _np
                    if "python" in rng:
                        _random.setstate(rng["python"])
                    if "numpy" in rng:
                        _np.random.set_state(rng["numpy"])
                    if "torch_cpu" in rng:
                        torch.set_rng_state(rng["torch_cpu"])
                    if "torch_cuda_all" in rng and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
                    if is_main:
                        logging.info("Restored RNG state (post-prepare)")
                except Exception as e:
                    if is_main:
                        logging.warning(f"Post-prepare RNG restore failed: {e}")
            start_step = int(train_state.get("step", 0))
            # Restore the nominal epoch counter so PI0Mixture keeps advancing
            # per-epoch seeds across resume (defaults to 0 on older ckpts).
            # Restored on resume: optimizer / scheduler / RNG / epoch counter,
            # plus the inside-epoch sampler cursor for the batch-sampler paths.
            start_epoch = int(train_state.get("epoch", 0))
        if train_state:
            if is_main:
                logging.info(
                    f"Resumed at step={start_step}, epoch={start_epoch}; "
                    f"resumable DataLoader starts at batch offset derived from step."
                )
            _n_called = _set_epoch_recursive(dataset, start_epoch)
            if is_main:
                logging.info(
                    f"Forwarded set_epoch({start_epoch}) to {_n_called} "
                    f"dataset object(s) post-resume."
                )

    # accelerator.prepare with bf16 makes DeepSpeed call self.module.bfloat16(),
    # which recasts EVERY parameter including ki_head.classifier (intentionally
    # fp32 for CE-logit precision; no precision tradeoffs). Put the classifier
    # back in fp32 after prepare. Negligible overhead since it only runs in
    # ki_head.compute_ce_loss.
    def _unwrap_policy_module(m):
        return getattr(m, "module", m)  # unwrap DS/DDP

    def _recast_ki_classifier_fp32(*, log: bool = False):
        _policy = _unwrap_policy_module(policy)
        _model = getattr(_policy, "model", _policy)
        _ki_head = getattr(_model, "ki_head", None)
        if _ki_head is None or not hasattr(_ki_head, "classifier"):
            return None
        _ki_head.classifier.to(dtype=torch.float32)
        if log and is_main:
            logging.info("[ki_head] classifier re-cast to fp32 after accelerator.prepare")
        return _ki_head

    _ki_head = _recast_ki_classifier_fp32(log=True)
    if _ki_head is not None:
        # Defensive assert: if the cast above is ever removed or prepare
        # regresses, fail loudly here rather than silently degrade CE precision.
        if hasattr(_ki_head, "assert_classifier_fp32_invariant"):
            _ki_head.assert_classifier_fp32_invariant()

    # EMA setup (openpi-aligned). Instantiated AFTER accelerator.prepare so the
    # unwrapped model has its final parameter layout. EMA tracks the unwrapped
    # policy's named_parameters(); under ZeRO-2 the full parameter set is
    # replicated per rank, so each rank's EMA buffer is identical without
    # cross-rank reduction.
    ema = None
    ema_decay_arg = float(getattr(args, "ema_decay", 0.0) or 0.0)
    if ema_decay_arg > 0.0:
        from optim.ema import ModelEMA
        ema = ModelEMA(_unwrap_policy_module(policy), decay=ema_decay_arg)
        if is_main:
            logging.info(
                "EMA enabled: %r (openpi-aligned formula: "
                "ema = decay*ema + (1-decay)*params, updated every step)",
                ema,
            )
        # Resume: load EMA state if a previous checkpoint saved one.
        if args.resume:
            _ema_path = Path(args.resume) / "training_state" / "ema_state.pt"
            if _ema_path.exists():
                try:
                    _ema_state = _torch_load_with_retry(_ema_path)
                    ema.load_state_dict(_ema_state)
                    if is_main:
                        logging.info(
                            "Restored EMA state from %s (decay=%.4f)",
                            _ema_path, ema.decay,
                        )
                except Exception as e:
                    if is_main:
                        logging.warning(
                            "EMA resume failed (%s); EMA will continue from "
                            "freshly-initialized buffer (= current params).",
                            e,
                        )
            elif is_main:
                logging.info(
                    "No ema_state.pt at %s; starting EMA from current params.",
                    _ema_path,
                )

    # wandb
    wandb_logger = None
    if args.wandb_enable and is_main:
        try:
            import wandb
            # Stringify unserializable values so wandb.init doesn't crash when
            # args grows a new non-jsonable field; _safe_config never raises.
            def _is_jsonable(x) -> bool:
                import json as _json
                try:
                    _json.dumps(x)
                    return True
                except (TypeError, ValueError):
                    return False

            def _safe_config(a) -> dict:
                out = {}
                for k, v in vars(a).items():
                    out[k] = v if _is_jsonable(v) else str(v)
                return out

            # Resume-aware wandb run: if resuming with full state and a saved
            # wandb_run_id, reconnect to that run for continuous step indices;
            # otherwise start a new run.
            _resume_wandb_id = None
            if args.resume and not getattr(args, "load_weights_only", False):
                _ts_path = Path(args.resume) / "training_state" / "training_state.pt"
                if _ts_path.exists():
                    try:
                        _ts = _torch_load_with_retry(_ts_path)
                        _resume_wandb_id = _ts.get("wandb_run_id")
                    except Exception as e:
                        logging.warning(f"Could not read wandb_run_id from {_ts_path}: {e}")

            _init_kwargs = dict(
                project=args.wandb_project,
                name=args.job_name,
                config=_safe_config(args),
                mode=args.wandb_mode,
            )
            if _resume_wandb_id:
                _init_kwargs["id"] = _resume_wandb_id
                _init_kwargs["resume"] = "must"
                logging.info(f"WandB resuming run_id={_resume_wandb_id}")
            wandb.init(**_init_kwargs)
            # Define a custom step metric so out-of-order / resumed steps don't
            # silently drop (all wandb.log() calls pass step=current_step).
            try:
                wandb.define_metric("train_step")
                wandb.define_metric("*", step_metric="train_step")
            except Exception as e:
                logging.debug(f"wandb.define_metric skipped: {e}")
            wandb_logger = wandb
        except Exception as e:
            logging.warning(f"WandB init failed: {e}")

    loss_meter = AverageMeter("loss")
    loss_action_meter = AverageMeter("loss_action")
    loss_ce_meter = AverageMeter("loss_ce")
    loss_mse_meter = AverageMeter("loss_mse")
    # Init the annotation-loss meter only when a schema declares
    # annotation_losses; downstream callsites guard with `is not None`.
    loss_ann_meter = AverageMeter("loss_ann") if _has_annotation_losses else None
    grad_norm_meter = AverageMeter("grad_norm")
    update_time_meter = AverageMeter("update_s")
    data_time_meter = AverageMeter("data_s")

    train_start_time = time.time()
    if is_main:
        logging.info(f"Training started: {datetime.now()}, total steps: {args.steps}")

    # Background prefetch buffer (buffer_size batches per GPU, transferred to the
    # current process's GPU).
    use_prefetch_buffer = bool(args.prefetch_buffer)
    batch_buffer = None
    dl_iter = None
    # Nominal epoch counter for the sync (no-prefetch-buffer) path; the async
    # path's authoritative counter lives in batch_buffer._epoch and is mirrored
    # back into nominal_epoch at save time.
    nominal_epoch = int(start_epoch)

    def _on_epoch_advance(new_epoch: int) -> None:
        # Producer-thread callback; mutates dataset._epoch in-process. DataLoader
        # workers (separate processes) only see the fork-time snapshot, so
        # perfect epoch isolation needs persistent_workers=False or a custom
        # Sampler.
        try:
            n = _set_epoch_recursive(dataset, int(new_epoch))
            sampler_n = _set_sampler_start_recursive(
                dataloader,
                0,
                epoch=int(new_epoch),
            )
            if is_main and n > 0:
                logging.info(
                    f"[π0-mix] epoch advance → set_epoch({new_epoch}) "
                    f"forwarded to {n} dataset(s)"
                )
            if is_main and sampler_n > 0:
                logging.info(
                    f"[resume] epoch advance → reset sampler offset for "
                    f"{sampler_n} sampler(s)"
                )
        except Exception as e:
            logging.warning(f"epoch advance failed: {e}")

    if use_prefetch_buffer:
        batch_buffer = BackgroundBatchBuffer(
            dataloader, device,
            buffer_size=args.prefetch_buffer_size,
            convert_fn=convert_batch_for_policy,
            warmup_fraction=0.8,
            on_epoch_advance=_on_epoch_advance,
        )
        # Seed the producer-thread counter so resume continues from the
        # checkpointed epoch instead of starting at 0.
        if start_epoch:
            batch_buffer._epoch = int(start_epoch)
        # Register batch_buffer with the signal handler so Ctrl+C can stop the
        # producer thread cleanly (releases CUDA pinned/stream buffers).
        _graceful["batch_buffer"] = batch_buffer
        if is_main:
            logging.info(
                f"Prefetch buffer: ON (size={args.prefetch_buffer_size}, "
                f"separate CUDA stream for async H2D transfer)"
            )
            logging.info(
                f"Waiting for buffer warmup to {int(args.prefetch_buffer_size * 0.8)}/"
                f"{args.prefetch_buffer_size}..."
            )
        warmup_start = time.time()
        warmup_level = batch_buffer.wait_for_warmup(timeout=600)
        if is_main:
            logging.info(
                f"Buffer warmup done: {warmup_level}/{args.prefetch_buffer_size} "
                f"in {time.time() - warmup_start:.1f}s"
            )
    else:
        dl_iter = iter(dataloader)
        if is_main:
            logging.info("Prefetch buffer: OFF (synchronous data loading)")

    policy.train()

    # Disable Python automatic GC.
    gc.disable()
    gc_interval = 100
    # Trust PyTorch's caching allocator: periodic empty_cache showed up as ~10s
    # stalls (each call is a full-device sync at high reserved memory), and the
    # memory profile is stable with no fragmentation growth. The NaN-path
    # empty_cache below is kept to release the graph after a skipped backward.

    pbar = None
    if is_main:
        pbar = tqdm(total=args.steps, initial=start_step, desc="Training",
                    unit="step", dynamic_ncols=True, miniters=1, smoothing=0.1)

    # NaN corruption guard. Under DeepSpeed, engine.step() (clip + optimizer
    # update) runs INSIDE accelerator.backward(), so the post-backward grad-NaN
    # check below can only detect a poisoned step's aftermath, not prevent it. If
    # a step permanently corrupts params (rare DeepSpeed ZeRO-2 bf16 edge case,
    # seen with exactly 2 ranks), every later step is NaN and skipped forever
    # while the step counter advances. Detect that: count consecutive NaN-skips;
    # past a threshold, scan live params and ABORT if non-finite. If params are
    # finite it's just bad-data batches, so keep skipping (just log loudly).
    _nan_skip_streak = 0
    _NAN_SKIP_ABORT = int(getattr(args, "max_consecutive_nan_skips", 0) or 0) or 25

    # Under DeepSpeed, accelerator.backward(loss) runs engine.step() (optimizer +
    # grad-clip + scheduler) internally at the sync boundary, and the loop's
    # optimizer.step()/zero_grad() are no-ops on the DS wrapper. So the
    # post-backward grad-NaN branch can't actually "skip" a DS update (master
    # params already stepped; bf16 ZeRO-2 has no overflow-skip). The real DS
    # guard is the PRE-backward loss-finiteness gate. Under DDP the post-backward
    # branch is correct. Detect the backend so the branch reports honestly while
    # the streak-abort still halts a corrupting run.
    _is_deepspeed = getattr(accelerator.state, "deepspeed_plugin", None) is not None

    def _find_nonfinite_params(limit: int = 8):
        _m = accelerator.unwrap_model(policy)
        _bad = []
        for _n, _p in _m.named_parameters():
            if _p is None or _p.numel() == 0:
                continue
            if not torch.isfinite(_p).all():
                _bad.append(_n)
                if len(_bad) >= limit:
                    break
        return _bad

    def _abort_or_warn_on_nan_streak(streak: int, force: bool = False):
        """Called from a NaN-skip path on all ranks uniformly (the streak is
        driven by the all-reduced NaN flag, so every rank hits the threshold on
        the same step). Aborts iff params are corrupted, and the decision is made
        collectively (all-reduce MAX of the local 'corrupted' flag) so every rank
        raises together rather than one aborting while others hang on a later
        collective.

        force=True bypasses the streak threshold and checks param finiteness now.
        Used on the DeepSpeed grad-NaN path, where engine.step() already ran
        inside accelerator.backward() at the sync boundary, so a non-finite grad
        may have already corrupted the fp32 master params this step (bf16 ZeRO-2
        has no overflow-skip); waiting 25 steps there is wrong. If DS grad-clip
        neutralized the spike (params still finite), fall through to warn."""
        if not force and streak < _NAN_SKIP_ABORT:
            return
        bad = _find_nonfinite_params()
        local_bad = 1.0 if bad else 0.0
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            _bt = torch.tensor(local_bad, device=accelerator.device)
            torch.distributed.all_reduce(_bt, op=torch.distributed.ReduceOp.MAX)
            any_bad = bool(_bt.item() > 0)
        else:
            any_bad = bool(local_bad > 0)
        if any_bad:
            example = bad if bad else ["(non-finite params detected on another rank)"]
            _why = (
                "a non-finite gradient was applied POST-step under DeepSpeed "
                "(engine.step() ran inside accelerator.backward; bf16 ZeRO-2 has no "
                "overflow-skip), corrupting the fp32 master params"
                if force else
                f"{streak} consecutive NaN/Inf skips left params non-finite"
            )
            raise RuntimeError(
                f"[nan-guard] Aborting: {_why} (e.g. {example}). Training is permanently "
                f"corrupted — no real updates are happening. This is NOT transient bad data. "
                f"Likely causes: DeepSpeed ZeRO-2 bf16 instability (reproduced with exactly 2 "
                f"ranks on this stack — use >=4 GPUs or run small debug jobs without DeepSpeed), "
                f"or an unclipped gradient spike (ensure --grad_clip_norm>0). Fix the run config "
                f"and restart."
            )
        if is_main:
            if force:
                logging.error(
                    "[nan-guard] non-finite gradient detected POST-step under DeepSpeed, but "
                    "params are still finite (DeepSpeed grad-clipping likely neutralized the "
                    "spike this step). Ensure --grad_clip_norm>0; watch for recurrence."
                )
            else:
                logging.error(
                    "[nan-guard] %d consecutive NaN/Inf-loss skips but parameters are still "
                    "finite — likely a run of bad-data batches. Continuing to skip; inspect "
                    "data_error_skip / the offending dataset.", streak,
                )

    for step in range(start_step, args.steps):
        data_start = time.perf_counter()
        if batch_buffer is not None:
            batch = batch_buffer.get()
        else:
            try:
                batch = next(dl_iter)
            except StopIteration:
                # Sync-path epoch boundary: bump the counter and forward
                # set_epoch() to mixture datasets before recreating the iterator
                # so the new pass uses the next epoch's seed.
                nominal_epoch += 1
                _on_epoch_advance(nominal_epoch)
                dl_iter = iter(dataloader)
                batch = next(dl_iter)
            batch = convert_batch_for_policy(batch, device)
        data_time_meter.update(time.perf_counter() - data_start)

        update_start = time.perf_counter()

        skip_update = False
        # At gradient_accumulation_steps=1, skip accelerator.accumulate() to
        # avoid an AttributeError when active_dataloader becomes None on small
        # datasets across multiple epochs.
        _accumulate_ctx = (
            accelerator.accumulate(policy)
            if args.gradient_accumulation_steps > 1
            else nullcontext()
        )
        with _accumulate_ctx:
            with accelerator.autocast():
                loss, output_dict = policy(batch)

            # All-reduce the NaN detection BEFORE accelerator.backward(). The
            # reverse order deadlocks DS ZeRO-2: the rank that saw NaN skips
            # backward() and never posts reduce_scatter, so healthy ranks wait
            # 600s on PG_1 and the NCCL watchdog crashes the job. Correct
            # sequence: detect -> all-reduce(MAX) -> uniform branch.
            local_nan = bool(torch.isnan(loss).item() or torch.isinf(loss).item())
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                nan_tensor = torch.tensor(
                    1.0 if local_nan else 0.0, device=accelerator.device
                )
                torch.distributed.all_reduce(nan_tensor, op=torch.distributed.ReduceOp.MAX)
                any_rank_nan = bool(nan_tensor.item() > 0)
            else:
                any_rank_nan = local_nan

            if any_rank_nan:
                skip_update = True
                _nan_skip_streak += 1
                if is_main:
                    logging.warning(
                        f"Step {step+1}: NaN/Inf loss detected on at least one rank "
                        f"(local_nan={local_nan}), ALL ranks skipping update to keep "
                        f"collectives in sync. (consecutive NaN skips={_nan_skip_streak})"
                    )
                # Do NOT call backward() on any rank when skipping; this keeps DS
                # PG_1 empty on all ranks uniformly.
                optimizer.zero_grad(set_to_none=True)
                del loss, output_dict       # release graph so next forward doesn't hold two activation sets
                del batch
                torch.cuda.empty_cache()
                _abort_or_warn_on_nan_streak(_nan_skip_streak)
                continue

            accelerator.backward(loss)
            if config.optimizer_grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    policy.parameters(), config.optimizer_grad_clip_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )
            # NaN gradient detection: skip steps with non-finite gradients.
            # clip_grad_norm_ returns None on non-sync gradient-accumulation steps.
            if grad_norm is None:
                grad_nan = any(
                    p.grad is not None and not torch.isfinite(p.grad).all().item()
                    for p in policy.parameters()
                )
            elif isinstance(grad_norm, torch.Tensor):
                grad_nan = torch.isnan(grad_norm) or torch.isinf(grad_norm)
            else:
                grad_nan = math.isnan(grad_norm) or math.isinf(grad_norm)
            # Same all-reduce(MAX) guard as skip_update so optimizer.step() is
            # called by all ranks or none.
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                gn_tensor = torch.tensor(
                    1.0 if grad_nan else 0.0, device=accelerator.device
                )
                torch.distributed.all_reduce(gn_tensor, op=torch.distributed.ReduceOp.MAX)
                grad_nan = bool(gn_tensor.item() > 0)
            if grad_nan:
                optimizer.zero_grad(set_to_none=True)
                _nan_skip_streak += 1
                if is_main:
                    if _is_deepspeed:
                        # The DeepSpeed update already happened inside
                        # accelerator.backward() at this sync boundary; we can
                        # only skip the trailing scheduler/EMA advance.
                        logging.warning(
                            f"Step {step+1}: non-finite grad detected POST-step under "
                            f"DeepSpeed — engine.step() already applied this update; "
                            f"checking params for corruption NOW and aborting if non-finite "
                            f"(consecutive NaN skips={_nan_skip_streak})"
                        )
                    else:
                        logging.warning(
                            f"Step {step+1}: NaN grad_norm, skipping update "
                            f"(consecutive NaN skips={_nan_skip_streak})"
                        )
                del loss, output_dict
                del batch
                torch.cuda.empty_cache()
                # Under DeepSpeed the bad update already landed, so check param
                # finiteness immediately (force=True); under DDP the optimizer
                # step was genuinely skipped, so the streak gate applies.
                _abort_or_warn_on_nan_streak(_nan_skip_streak, force=_is_deepspeed)
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Reached a real update: reset the consecutive-NaN-skip streak.
            _nan_skip_streak = 0

        if accelerator.sync_gradients:
            scheduler.step()
            # Update EMA on the optimizer sync boundary so accumulation steps
            # aren't counted multiple times (openpi updates once per
            # optimizer.step / gradient-accumulation cycle).
            if ema is not None:
                ema.update(_unwrap_policy_module(policy))

        update_time_meter.update(time.perf_counter() - update_start)
        loss_meter.update(output_dict["loss"])
        loss_action_meter.update(output_dict["loss_action"])
        if "loss_ce" in output_dict:
            loss_ce_meter.update(output_dict["loss_ce"])
        if "loss_mse" in output_dict:
            loss_mse_meter.update(output_dict["loss_mse"])
        if loss_ann_meter is not None and "loss_annotation_total" in output_dict:
            loss_ann_meter.update(output_dict["loss_annotation_total"])
        if accelerator.sync_gradients:
            grad_norm_meter.update(
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )

        current_step = step + 1

        if current_step == 1 and is_main:
            _unwrapped = accelerator.unwrap_model(policy)
            _model = getattr(_unwrapped, "model", None)
            _ki = getattr(_model, "ki_head", None) if _model is not None else None
            if _ki is not None and hasattr(_ki, "classifier"):
                _cdt = _ki.classifier.weight.dtype
                _edt = _ki.embedding.weight.dtype if hasattr(_ki, "embedding") else None
                logging.info(
                    f"[verify] After step 1: ki_head.classifier.weight.dtype={_cdt}, "
                    f"ki_head.embedding.weight.dtype={_edt} "
                    f"(expected: classifier=torch.float32 / embedding=torch.bfloat16)"
                )
                if _cdt != torch.float32:
                    logging.warning(
                        f"[verify] KI classifier dtype regressed to {_cdt} after optimizer.step(). "
                        "compute_ce_loss still casts matmul operands to fp32, but parameter "
                        "storage was not restored as expected."
                    )
            try:
                _lr_now = optimizer.param_groups[0]["lr"]
                _sched_last = getattr(scheduler, "last_epoch", None)
                logging.info(
                    f"[verify] After step 1: scheduler.last_epoch={_sched_last}, "
                    f"optimizer.lr={_lr_now:.3e} (resume continuity probe)"
                )
            except Exception:
                pass

        if is_main and pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{output_dict['loss']:.4f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.2e}"},
                             refresh=False)

        if args.log_freq > 0 and current_step % args.log_freq == 0 and is_main:
            elapsed = time.time() - train_start_time
            avg_update = update_time_meter.avg
            iters_per_sec = 1.0 / avg_update if avg_update > 0 else 0

            buf_info = f" | buf: {batch_buffer.qsize}/{args.prefetch_buffer_size}" if batch_buffer is not None else ""
            mem_info = ""
            if torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                mem_info = f" | mem: {alloc_gb:.1f}/{reserved_gb:.1f}/{peak_gb:.1f}GB(alloc/rsv/peak)"
            # Extra columns only when KI/vlm_pretrain is active.
            extra_line = ""
            if loss_ce_meter.count > 0:
                extra_line += f" | loss_ce: {loss_ce_meter.avg:.4f}"
            if loss_mse_meter.count > 0:
                extra_line += f" | loss_mse: {loss_mse_meter.avg:.4f}"
            if loss_ann_meter is not None and loss_ann_meter.count > 0:
                extra_line += f" | loss_ann: {loss_ann_meter.avg:.4f}"

            log_msg = (
                f"Step {current_step}/{args.steps} | "
                f"{iters_per_sec:.2f} it/s | "
                f"loss: {loss_meter.avg:.4f} | "
                f"loss_action: {loss_action_meter.avg:.4f}"
                f"{extra_line} | "
                f"grad_norm: {grad_norm_meter.avg:.4f} | "
                f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                f"data_s: {data_time_meter.avg:.3f} | "
                f"elapsed: {format_time(elapsed)}"
                f"{buf_info}{mem_info}"
            )
            if pbar is not None:
                tqdm.write(log_msg)
            else:
                logging.info(log_msg)

            if wandb_logger:
                wandb_payload = {
                    "loss": loss_meter.avg,
                    "loss_action": loss_action_meter.avg,
                    "grad_norm": grad_norm_meter.avg,
                    "lr": optimizer.param_groups[0]["lr"],
                    "update_s": update_time_meter.avg,
                    "data_s": data_time_meter.avg,
                    # Custom step metric so resumed runs don't clobber.
                    "train_step": current_step,
                }
                if loss_ce_meter.count > 0:
                    wandb_payload["loss_ce"] = loss_ce_meter.avg
                if loss_mse_meter.count > 0:
                    wandb_payload["loss_mse"] = loss_mse_meter.avg
                if loss_ann_meter is not None and loss_ann_meter.count > 0:
                    wandb_payload["loss_annotation_total"] = loss_ann_meter.avg
                # Per-dim losses are a GPU tensor in output_dict; convert only
                # when actually logging (once per log_freq).
                pdl = output_dict.get("per_dim_action_losses", None)
                pdd = output_dict.get("per_dim_action_dim", None)
                if pdl is not None and pdd is not None:
                    try:
                        pdl_host = pdl.cpu().numpy().tolist()
                        for i in range(min(int(pdd), len(pdl_host))):
                            wandb_payload[f"loss_action_dim{i}"] = float(pdl_host[i])
                    except Exception as _e:
                        logging.debug(f"per_dim_action_losses convert skipped: {_e}")
                wandb_logger.log(wandb_payload, step=current_step)

            loss_meter.reset()
            loss_action_meter.reset()
            loss_ce_meter.reset()
            loss_mse_meter.reset()
            if loss_ann_meter is not None:
                loss_ann_meter.reset()
            grad_norm_meter.reset()
            update_time_meter.reset()
            data_time_meter.reset()

        if current_step % args.save_freq == 0 or current_step == args.steps:
            accelerator.wait_for_everyone()
            ckpt_dir = output_dir / f"checkpoint-{current_step}"
            if is_main:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                _write_checkpoint_incomplete_marker(ckpt_dir, current_step)
            accelerator.wait_for_everyone()
            distributed_state_dir = ckpt_dir / "accelerate_state"
            accelerator.save_state(str(distributed_state_dir))
            accelerator.wait_for_everyone()
            if is_main:
                try:
                    # Call-site pre-check so the reproducibility hazard
                    # (norm_stats without schema => deploy can't reconstruct the
                    # layout) surfaces here, not only inside save_checkpoint. Not
                    # swallowed by the transient-I/O catch below; it's a
                    # correctness bug, not a transient failure.
                    if _active_schema is None and _active_norm_stats is not None:
                        raise RuntimeError(
                            "save_checkpoint: norm_stats provided but no schema "
                            "— reproducibility hazard. A checkpoint with norm "
                            "stats but no labvla_schema.json cannot be redeployed "
                            "without manually re-attaching the schema. Check "
                            "build_dataset() — it must have populated "
                            "data_schemas for the primary repo."
                        )
                    # Snapshot the live nominal epoch (async path's truth is the
                    # producer thread; sync path's is the main-thread counter).
                    # Resume forwards it back via _set_epoch_recursive after RNG
                    # restore.
                    _epoch_for_ckpt = (
                        int(batch_buffer.epoch)
                        if batch_buffer is not None
                        else int(nominal_epoch)
                    )
                    save_checkpoint(
                        ckpt_dir, current_step,
                        accelerator.unwrap_model(policy),
                        optimizer, scheduler, args,
                        norm_stats=_active_norm_stats,
                        schema=_active_schema,
                        all_schemas=data_schemas,  # persist all per-repo schemas
                        epoch=_epoch_for_ckpt,
                        include_optimizer_state=False,
                        include_scheduler_state=False,
                        distributed_state_saved=True,
                    )
                    # Save EMA state alongside training_state; resume reads
                    # <ckpt>/training_state/ema_state.pt directly.
                    if ema is not None:
                        _ema_save_path = ckpt_dir / "training_state" / "ema_state.pt"
                        _ema_save_path.parent.mkdir(parents=True, exist_ok=True)
                        _torch_save_with_retry(ema.state_dict(), _ema_save_path)
                        logging.info(
                            "Saved EMA state to %s (%d tracked params, decay=%.4f)",
                            _ema_save_path, len(ema), ema.decay,
                        )
                    _clear_checkpoint_incomplete_marker(ckpt_dir)
                    # Update the `last` symlink -> newest checkpoint only after
                    # save_checkpoint succeeds, so `last` never points at a
                    # partially written rank0 artifact.
                    from utils.train_utils import update_last_checkpoint
                    run_with_storage_retry(
                        lambda: update_last_checkpoint(ckpt_dir),
                        path=ckpt_dir.parent / "last",
                        description="update last checkpoint symlink",
                    )
                    _rotate_checkpoints(
                        output_dir,
                        int(getattr(args, "max_keep_ckpts", 0) or 0),
                    )
                    # Surface video-fallback reason counts so silent I/O rot
                    # (e.g. MSE never dropping on a repo) becomes visible.
                    try:
                        _ds_list = getattr(dataset, "datasets", None) or [dataset]
                        for _i, _ds in enumerate(_ds_list):
                            _dataset_adapter = (
                                getattr(_ds, "_adapter", None)
                                or getattr(_ds, "adapter", None)
                                or _ds
                            )
                            if hasattr(_dataset_adapter, "video_fallback_summary"):
                                _sum = _dataset_adapter.video_fallback_summary()
                                if _sum:
                                    logging.info(f"[video-fallback] dataset[{_i}]: {_sum}")
                    except Exception as _e:
                        logging.debug(f"video_fallback_summary skipped: {_e}")
                except BaseException as _e:
                    if not _is_recoverable_checkpoint_save_error(_e):
                        raise
                    logging.exception(
                        "Checkpoint rank0 artifact save failed after retry budget "
                        "at step=%s for %s. Distributed state may exist, but `last` "
                        "was not advanced; training will continue and retry at the "
                        "next save point.",
                        current_step,
                        ckpt_dir,
                    )
            accelerator.wait_for_everyone()

        # Periodic Python GC only; the CUDA caching allocator self-manages (the
        # NaN-path empty_cache above is kept to release the graph after a skipped
        # backward).
        if current_step % gc_interval == 0:
            gc.collect()
            # Also force glibc to return idle heap pages to the OS on the rank
            # process. Workers self-trim via _maybe_worker_malloc_trim.
            _try_malloc_trim()

        # GPU memory growth tracing (every 50 steps for the first 600, off by default).
        if args.mem_trace and current_step <= 600 and current_step % 50 == 0 and is_main and torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / 1e6
            rsv_mb = torch.cuda.memory_reserved() / 1e6
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            mem_msg = (
                f"[MEM-TRACE] step={current_step} "
                f"alloc={alloc_mb:.0f}MB rsv={rsv_mb:.0f}MB peak={peak_mb:.0f}MB "
                f"cached={rsv_mb-alloc_mb:.0f}MB"
            )
            if pbar is not None:
                tqdm.write(mem_msg)
            else:
                logging.info(mem_msg)

    gc.enable()
    if is_main:
        if pbar:
            pbar.close()
        total_seconds = time.time() - train_start_time
        logging.info("=" * 60)
        logging.info("Training completed!")
        logging.info(f"Total training time: {format_time(total_seconds)} ({total_seconds/3600:.2f}h)")
        logging.info("=" * 60)

    if wandb_logger:
        wandb_logger.finish()

    if batch_buffer is not None:
        batch_buffer.stop()
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    accelerator.end_training()

    # Restore torch.distributed.new_group at the end of train() so same-process
    # callers (test harnesses, REPLs) see the original function as soon as
    # training returns. The atexit hook above covers paths that bypass this
    # clean return (signal-handler hard-exit, etc.).
    try:
        if "_tdist" in dir() and "_orig_new_group" in dir():
            if _tdist.new_group is _new_group_with_timeout:  # noqa: F821
                _tdist.new_group = _orig_new_group  # noqa: F821
    except Exception:
        pass


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
