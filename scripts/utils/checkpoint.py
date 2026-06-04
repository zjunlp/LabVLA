"""Checkpoint I/O, atomic-write helpers, resume contract validation."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import torch

from utils.storage_retry import is_transient_storage_error, run_with_storage_retry


# Keys whose checkpoint-vs-current divergence blocks a full resume. Anything
# touching architecture shape, optimizer state, dataset semantics, or RNG seed
# must appear here — letting a mismatched arg slip past defeats the contract
# (see `_validate_full_resume_contract`).
_FULL_RESUME_CONTRACT_KEYS = (
    "batch_size",
    "gradient_accumulation_steps",
    "training_phase",
    "knowledge_isolation",
    "use_fast_tokenizer",
    "action_mode",
    "dit_num_layers",
    "dit_num_heads",
    "dit_head_dim",
    "chunk_size",
    "max_state_dim",
    "max_action_dim",
    "lr",
    "vlm_lr",
    "dit_lr",
    "warmup_steps",
    "decay_steps",
    "decay_lr",
    "weight_decay",
    "grad_clip_norm",
    "gradient_checkpointing",
    "gc_visual_encoder",
    "gc_language_model",
    "gc_dit",
    "freeze_vision_encoder",
    "train_expert_only",
    "train_vlm_only",
    "seed",
    "repo_ids",
    "dataset_weights",
    "dataset_schema",
    "external_stats_map",
    "homogeneous_mixture_batches",
    "trim_token_padding_to_batch",
    "source_shape_convergence",
    "discrete_action_vocab_size",
    "discrete_action_max_length",
    # These training-semantic args change attention pattern, gripper
    # supervision, per-dim normalization, or the FAST CE skip set; surface any
    # ckpt-vs-current divergence rather than letting it drift silently.
    "pi05_block_attention_mask",       # R8 attention semantics
    "action_supervision_skip_repos",   # disables FAST CE per-repo
    "normalize_arm_joints",            # input normalization (per-joint stats)
    "normalize_gripper",               # input normalization (gripper channel)
    "gripper_norm_mode",               # raw/q01_q99/standard/noop
    "snap_gripper_to_binary",          # binarize gripper target
    "gripper_loss_weight",             # per-dim MSE weighting in action head
    # More keys that change training semantics, model topology, or the data
    # pipeline; resume must not succeed while their meaning has drifted.
    "attn_implementation",             # FA2 vs SDPA backend (drives pi05 mask compat)
    "dit_interleave_self_attention",   # DiT block layout (Spirit interleave on/off)
    "dit_layerwise_vlm_features",      # DiT cross-attn reads layerwise VLM features
    "dit_dropout",                     # DiT regularization
    "ki_mse_weight",                   # KI loss mixture (α; π0.5 uses 10)
    "discretize_state_in_vlm_pretrain",  # state representation in VLM pretrain
    "image_height",                    # image pipeline / token count
    "image_width",                     # image pipeline / token count
    "image_augmentation",              # train-time image aug on/off
    "external_stats_path",             # single-repo norm-stats override path
    "fast_tokenizer_path",             # FAST tokenizer source path
    "vlm_pretrained_path",             # VLM init weights source path
    "ema_decay",                       # EMA buffer existence + decay rate
    # Data-pipeline / RNG keys that change training semantics across resume:
    # gripper_max_width / gripper_canonical_dim drive the snap-gripper transform;
    # data_root flips between dataset copies; num_workers changes the per-worker
    # seed formula; resumable_dataloader switches sampler semantics;
    # data_error_skip{,_max_attempts} change the sample stream on bad samples.
    "gripper_max_width",
    "gripper_canonical_dim",
    "data_root",
    "num_workers",
    "resumable_dataloader",
    "data_error_skip",
    "data_error_skip_max_attempts",
)


def _torch_load_with_retry(path: Path):
    return run_with_storage_retry(
        lambda: torch.load(str(path), map_location="cpu", weights_only=False),
        path=path,
        description="torch.load",
    )


def _atomic_write_with_retry(path: Path, writer, *, description: str) -> None:
    """Write one checkpoint file via tmp-in-same-dir + atomic replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")

    def _write() -> None:
        if tmp_path.exists():
            tmp_path.unlink()
        writer(tmp_path)
        os.replace(str(tmp_path), str(path))

    try:
        run_with_storage_retry(_write, path=path, description=description)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logging.debug("Could not remove temporary checkpoint file %s", tmp_path)


def _torch_save_with_retry(obj, path: Path) -> None:
    _atomic_write_with_retry(
        Path(path),
        lambda tmp_path: torch.save(obj, str(tmp_path)),
        description="torch.save",
    )


def _json_dump_with_retry(obj, path: Path) -> None:
    def _write(tmp_path: Path) -> None:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)

    _atomic_write_with_retry(Path(path), _write, description="json checkpoint write")


def _json_safe_config_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_config_value(asdict(value))
    if isinstance(value, tuple):
        return [_json_safe_config_value(v) for v in value]
    if isinstance(value, list):
        return [_json_safe_config_value(v) for v in value]
    if isinstance(value, dict):
        return {
            str(_json_safe_config_value(k)): _json_safe_config_value(v)
            for k, v in value.items()
        }
    json.dumps(value)
    return value


def _normalize_resume_contract_value(key: str, value):
    if key == "repo_ids":
        if value is None:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(str(v).strip() for v in value if str(v).strip())
        return tuple(part.strip() for part in str(value).split(",") if part.strip())
    # Resolve path-like fields to absolute before comparison so cosmetic
    # differences ("./foo" vs "/abs/foo", trailing slash) don't trigger false
    # mismatches. Real divergence (different weights/stats file) still surfaces.
    if key in ("external_stats_path", "fast_tokenizer_path",
               "vlm_pretrained_path"):
        if value is None or value == "":
            return None
        try:
            return str(Path(str(value)).resolve())
        except Exception:
            return str(value).rstrip("/")
    return value


def _checkpoint_incomplete_marker(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / ".INCOMPLETE"


def _write_checkpoint_incomplete_marker(checkpoint_dir: str | Path, step: int) -> None:
    marker = _checkpoint_incomplete_marker(checkpoint_dir)
    _json_dump_with_retry(
        {"step": int(step), "status": "checkpoint write in progress"},
        marker,
    )


def _clear_checkpoint_incomplete_marker(checkpoint_dir: str | Path) -> None:
    marker = _checkpoint_incomplete_marker(checkpoint_dir)
    if marker.exists():
        marker.unlink()


def _rotate_checkpoints(output_dir: Path, keep_last: int, *, sticky_every: int = 50000) -> None:
    keep_last = int(keep_last)
    if keep_last <= 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    checkpoints.sort()
    protected = {path for _, path in checkpoints[-keep_last:]}
    for step, path in checkpoints:
        if path in protected:
            continue
        if sticky_every > 0 and step % int(sticky_every) == 0:
            continue
        shutil.rmtree(path, ignore_errors=True)
        logging.info("Removed old checkpoint %s (save_total_limit=%d)", path, keep_last)


def _clone_shared_tensors_for_safetensors(state_dict: dict) -> dict:
    """Prepare a state_dict for ``safetensors.save_file``.

    Qwen ties ``lm_head.weight`` and ``language_model.embed_tokens.weight``.
    Deployment strict-loads via ``safetensors.torch.load_model``; saving both
    aliases makes the tied input-embedding alias an unexpected key. Keep the
    canonical lm_head tensor and omit the duplicate embed_tokens alias. Other
    shared tensors are still cloned so safetensors can serialize them.
    """

    tied_lm_head_key = "model.vlm.lm_head.weight"
    tied_embed_key = "model.vlm.model.language_model.embed_tokens.weight"
    tied_group_ptr = None

    lm_head = state_dict.get(tied_lm_head_key)
    embed = state_dict.get(tied_embed_key)
    if (
        isinstance(lm_head, torch.Tensor)
        and isinstance(embed, torch.Tensor)
        and lm_head.data_ptr() == embed.data_ptr()
    ):
        tied_group_ptr = lm_head.data_ptr()

    seen_data_ptrs = {}
    deduped = {}
    for key, tensor in state_dict.items():
        if (
            key == tied_embed_key
            and tied_group_ptr is not None
            and isinstance(tensor, torch.Tensor)
            and tensor.data_ptr() == tied_group_ptr
        ):
            continue
        if isinstance(tensor, torch.Tensor):
            ptr = tensor.data_ptr()
            if ptr in seen_data_ptrs:
                deduped[key] = tensor.clone()
            else:
                seen_data_ptrs[ptr] = key
                deduped[key] = tensor
        else:
            deduped[key] = tensor
    return deduped


def _is_recoverable_checkpoint_save_error(exc: BaseException) -> bool:
    """Only shared-storage transient failures are safe to skip and retry later."""

    return is_transient_storage_error(exc)


def _load_training_state(checkpoint_path: str | Path) -> dict:
    train_state_path = Path(checkpoint_path) / "training_state" / "training_state.pt"
    if not train_state_path.exists():
        return {}
    return _torch_load_with_retry(train_state_path)


def _accelerate_state_dir(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path) / "accelerate_state"


def _validate_full_resume_contract(saved_args: dict, current_args) -> None:
    mismatches = []
    for key in _FULL_RESUME_CONTRACT_KEYS:
        old = _normalize_resume_contract_value(key, saved_args.get(key))
        new = _normalize_resume_contract_value(key, getattr(current_args, key, None))
        if old != new:
            mismatches.append((key, old, new))
    if mismatches:
        preview = "; ".join(
            f"{key}: ckpt={old!r} current={new!r}"
            for key, old, new in mismatches[:12]
        )
        extra = "" if len(mismatches) <= 12 else f"; ... +{len(mismatches) - 12} more"
        raise RuntimeError(
            "Full resume contract mismatch. Use --load_weights_only true for "
            "warmstart/finetune, or keep the original training config for exact "
            f"resume. Mismatches: {preview}{extra}"
        )


def save_checkpoint(checkpoint_dir, step, policy, optimizer, scheduler, args,
                    norm_stats=None, schema=None, all_schemas=None,
                    epoch: int = 0, include_optimizer_state: bool = True,
                    include_scheduler_state: bool = True,
                    distributed_state_saved: bool = False):
    """Persist a training checkpoint.

    Resume semantics:
      - Restored on full distributed resume: optimizer state, LR scheduler state, RNG state
        (Python / NumPy / PyTorch CPU+CUDA), wandb run id, nominal epoch counter
        (forwarded to PI0Mixture-style datasets via ``set_epoch``). Full
        distributed resume requires a sibling ``accelerate_state/`` written by
        ``accelerator.save_state()``; this helper records whether that
        distributed state was saved by its caller.
      - Restored for current deterministic paths: per-DataLoader sampler
        cursor *within* an epoch. Homogeneous PI0Mixture batches and the
        deterministic single-dataset batch sampler both derive the resume
        offset from the checkpoint step and reset it at the next epoch boundary.

    Backwards compat: older ckpts lack the ``epoch`` field; load/resume paths
    default to 0 when the key is absent.
    """
    # Deployment auto-discovery relies on BOTH norm_stats.json AND labvla_schema.json
    # being written alongside the model. If norm_stats is given but schema is
    # not, deployment falls back to --robot_type + RobotConfig (sub-optimal
    # reproducibility). Fail loudly so callers wire both.
    if norm_stats is not None and schema is None:
        raise ValueError(
            "save_checkpoint: norm_stats provided without schema. "
            "Pass schema=<DatasetSchema> so deployment can reproduce the layout. "
            "Typically: schema=data_schemas[first_repo_id] from build_dataset()."
        )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_dir = checkpoint_dir / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        state_dict = _clone_shared_tensors_for_safetensors(policy.state_dict())
        _atomic_write_with_retry(
            model_dir / "model.safetensors",
            lambda tmp_path: save_file(state_dict, str(tmp_path)),
            description="save model.safetensors",
        )
    except ImportError:
        _torch_save_with_retry(policy.state_dict(), model_dir / "pytorch_model.bin")

    import json
    config_dict = {}
    skip_config_fields = {"input_features", "output_features"}
    for k, v in vars(policy.config).items():
        if k.startswith("_") or k in skip_config_fields:
            continue
        try:
            config_dict[k] = _json_safe_config_value(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"save_checkpoint: LabVLAConfig field {k!r} is not JSON-serializable "
                "by the typed checkpoint serializer. Add an explicit conversion "
                "instead of falling back to repr strings."
            ) from exc
    if hasattr(args, "action_mode"):
        config_dict["action_mode"] = str(args.action_mode)
    _json_dump_with_retry(config_dict, model_dir / "config.json")

    # Save norm_stats alongside the checkpoint so deployment can load them
    # without needing a separate --norm_stats_path argument.
    if norm_stats is not None:
        norm_stats_path = checkpoint_dir / "norm_stats.json"
        _json_dump_with_retry(norm_stats, norm_stats_path)
        logging.info(f"Saved norm_stats to {norm_stats_path}")

    # Save DatasetSchema alongside the checkpoint so deployment can auto-discover
    # the dataset layout without relying on --robot_type CLI arg.
    if schema is not None:
        schema_path = checkpoint_dir / "labvla_schema.json"
        _json_dump_with_retry(
            schema.to_dict() if hasattr(schema, "to_dict") else schema,
            schema_path,
        )
        logging.info(f"Saved schema to {schema_path}")

    # Persist *all* per-repo schemas in multi-repo training so the operator can
    # pick the right one at deploy time. labvla_schema.json (above) keeps the
    # first repo so the existing auto-discover flow still works.
    if all_schemas and len(all_schemas) > 1:
        schemas_path = checkpoint_dir / "labvla_schemas.json"
        dumpable = {}
        for rid, sch in all_schemas.items():
            try:
                dumpable[rid] = sch.to_dict() if hasattr(sch, "to_dict") else sch
            except Exception as _e:
                logging.warning(f"Could not serialize schema for repo {rid!r}: {_e}")
        _json_dump_with_retry({"_schema": "multi_repo_v1", **dumpable}, schemas_path)
        logging.info(f"Saved all repo schemas to {schemas_path} (n={len(dumpable)})")

    train_state_dir = checkpoint_dir / "training_state"
    train_state_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: Under DeepSpeed ZeRO-2, optimizer.state_dict() only contains the
    # local shard. Restoring optimizer state from this checkpoint across different
    # world sizes will fail. For full optimizer state, use DeepSpeed's
    # optimizer.consolidate_state_dict() or accelerator.save_state().
    # Capture RNG state so resume with --load_weights_only false reproduces the
    # same data shuffling / augmentation sequence as an uninterrupted run.
    # Best-effort: missing keys on load are skipped.
    import random as _random
    import numpy as _np
    rng_state = {
        "python": _random.getstate(),
        "numpy": _np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    # Capture wandb run_id so full-resume can reconnect to the same run.
    _wandb_run_id = None
    try:
        import wandb as _wb
        if _wb.run is not None:
            _wandb_run_id = _wb.run.id
    except Exception:
        pass

    train_state = {
        "step": step,
        # Nominal epoch counter; lets resume re-prime PI0Mixture's per-epoch
        # RNG seed (see save_checkpoint docstring).
        "epoch": int(epoch),
        "args": vars(args),
        "rng_state": rng_state,
        "wandb_run_id": _wandb_run_id,
        "accelerate_state": "accelerate_state",
        "full_resume_requires_distributed_state": bool(distributed_state_saved),
        "distributed_state_saved": bool(distributed_state_saved),
    }
    if include_optimizer_state:
        train_state["optimizer_state_dict"] = optimizer.state_dict()
    if include_scheduler_state:
        train_state["scheduler_state_dict"] = scheduler.state_dict()
    _torch_save_with_retry(train_state, train_state_dir / "training_state.pt")

    logging.info(f"Checkpoint saved to {checkpoint_dir} at step {step}")


def _is_deepspeed_optimizer_state(state: dict) -> bool:
    """Detect DeepSpeed ZeRO optimizer state_dict format.

    Standard PyTorch optimizer state_dict has ['state', 'param_groups']. DS
    ZeRO wraps this under 'base_optimizer_state' and adds 'zero_stage',
    'partition_count', 'loss_scaler', etc. If we see those, the current
    optimizer must be a DS-wrapped optimizer (which happens only AFTER
    accelerator.prepare) for load_state_dict to succeed.
    """
    if not isinstance(state, dict):
        return False
    ds_markers = {"base_optimizer_state", "zero_stage", "partition_count",
                  "single_partition_of_fp32_groups", "ds_version"}
    has_ds_marker = any(k in state for k in ds_markers)
    # Two-factor check: DS markers present AND standard PyTorch optimizer keys
    # absent, to avoid false positives from third-party wrappers reusing names.
    has_pt_keys = ("state" in state and "param_groups" in state)
    return has_ds_marker and not has_pt_keys


def load_checkpoint(checkpoint_path, policy, optimizer=None, scheduler=None,
                    strict: bool | None = None):
    """Load model weights (and optionally optimizer/scheduler/RNG).

    When `optimizer` is passed but the saved optimizer state is in DeepSpeed
    ZeRO format, calling `optimizer.load_state_dict(...)` on a raw PyTorch
    optimizer raises `KeyError: 'param_groups'`. Callers wanting full resume
    with DeepSpeed must invoke this AFTER `accelerator.prepare`, when the
    optimizer is DS-wrapped and accepts DS format. We detect the format and
    either use the provided optimizer if compatible, or skip with a loud warning
    (still returning step so training resumes with weights + step).

    Full resume defaults to strict=True so architectural drift (changed
    dit_num_layers, missing ki_head, renamed projections, etc.) fails loudly
    instead of silently re-initializing missing parameters. Weights-only
    warm-start (optimizer is None) uses strict=False, since it routinely
    transfers a pretrain ckpt into a smaller downstream head with intentionally
    different shapes. Override via the ``strict`` arg for legacy behavior.
    """
    checkpoint_path = Path(checkpoint_path)

    weights_only_warmstart = optimizer is None and scheduler is None
    if strict is None:
        strict = not weights_only_warmstart

    model_dir = checkpoint_path / "pretrained_model"
    if model_dir.exists():
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_dir / "model.safetensors"))
        except (ImportError, FileNotFoundError):
            state_dict = _torch_load_with_retry(model_dir / "pytorch_model.bin")
        load_result = policy.load_state_dict(state_dict, strict=strict)
        if load_result.missing_keys:
            logging.warning(f"Missing keys in checkpoint: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logging.warning(f"Unexpected keys in checkpoint: {load_result.unexpected_keys}")

        train_state_path = checkpoint_path / "training_state" / "training_state.pt"
        if train_state_path.exists():
            train_state = _torch_load_with_retry(train_state_path)
            if optimizer and "optimizer_state_dict" in train_state:
                saved_opt = train_state["optimizer_state_dict"]
                # Check if saved state is DS-format vs optimizer is DS-wrapped.
                is_ds_saved = _is_deepspeed_optimizer_state(saved_opt)
                is_ds_opt = type(optimizer).__module__.startswith("deepspeed")
                if is_ds_saved and not is_ds_opt:
                    logging.warning(
                        "Optimizer state in checkpoint is DeepSpeed ZeRO format "
                        "(keys include 'base_optimizer_state'), but the target "
                        "optimizer is a raw PyTorch optimizer (not yet wrapped "
                        "by accelerator.prepare). Skipping optimizer restore to "
                        "avoid KeyError. To fully resume, call load_checkpoint "
                        "AFTER accelerator.prepare, or restart with "
                        "--load_weights_only true."
                    )
                else:
                    try:
                        optimizer.load_state_dict(saved_opt)
                    except (ValueError, RuntimeError, KeyError) as e:
                        logging.warning(
                            f"Skipping optimizer state restore: {e}. "
                            f"This is expected when resuming across a different "
                            f"world size / DeepSpeed toggle."
                        )
            if scheduler and "scheduler_state_dict" in train_state:
                try:
                    scheduler.load_state_dict(train_state["scheduler_state_dict"])
                except (ValueError, RuntimeError, KeyError) as e:
                    logging.warning(f"Skipping scheduler state restore: {e}")
            # Restore RNG state only on full resume (optimizer present).
            # Weights-only resume intentionally starts with fresh RNG because
            # it's usually a finetune on different data.
            if optimizer and "rng_state" in train_state:
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
                    logging.info("Restored RNG state from checkpoint")
                except Exception as e:
                    logging.warning(f"Skipping RNG state restore: {e}")
            return train_state.get("step", 0)
        return 0

    if checkpoint_path.is_file():
        ckpt = _torch_load_with_retry(checkpoint_path)
        if "model_state_dict" in ckpt:
            # Legacy bundle resume uses the same strict default as the model_dir
            # branch above (weights-only warmstart gets strict=False via the
            # optimizer-is-None path).
            policy.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if optimizer and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (KeyError, ValueError, RuntimeError) as e:
                logging.warning(f"Skipping optimizer state restore: {e}")
        if scheduler and "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (KeyError, ValueError, RuntimeError) as e:
                logging.warning(f"Skipping scheduler state restore: {e}")
        return ckpt.get("step", 0)

    raise FileNotFoundError(f"No valid checkpoint found at {checkpoint_path}")
