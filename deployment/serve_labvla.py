#!/usr/bin/env python
"""
LabVLA Deployment Server - Deploy LabVLA model for LabUtopia remote inference.

Architecture: Qwen3-VL VLM + Action Expert, Flow Matching, No Knowledge Insulation.
WebSocket protocol, compatible with openpi_client.WebsocketClientPolicy.

Usage:
    python serve_labvla.py \
        --pretrained_path /all-flash-data/LabVLA_final/outputs/.../checkpoint-30000/pretrained_model \
        --vlm_path /all-flash-data/vlm/Qwen3-VL-4B-Instruct \
        --port 8000 --device cuda
"""

import argparse
import json
import logging
import os
import sys
import socket
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

# Import LabVLA source. Default to the repo this file lives in rather than a
# hardcoded absolute path (which would reach into a stale checkout when run from
# another clone). LABVLA_ROOT still takes precedence for explicit targeting.
_DEFAULT_LABVLA_ROOT = str(Path(__file__).resolve().parents[1])
LABVLA_ROOT = os.environ.get("LABVLA_ROOT", _DEFAULT_LABVLA_ROOT)
LABVLA_SRC = os.path.join(LABVLA_ROOT, "src")
if LABVLA_SRC not in sys.path:
    sys.path.insert(0, LABVLA_SRC)
if LABVLA_ROOT not in sys.path:
    sys.path.insert(0, LABVLA_ROOT)

DEPLOYMENT_DIR = Path(__file__).resolve().parent
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from labsim_transforms import parse_image_to_uint8_hwc
from websocket_server import WebsocketPolicyServer

logger = logging.getLogger(__name__)


class LabVLALabUtopiaPolicy:
    """
    LabVLA deployment wrapper, adapted for LabUtopia evaluation.

    Inference pipeline:
    1. LabUtopia observation -> 3-camera images + state + language instruction
    2. Images converted to pixel_values via Qwen3-VL processor
    3. Language instruction tokenized to input_ids
    4. Build batch, call Policy.predict_action_chunk()
    5. Return action chunk
    """

    def __init__(
        self,
        pretrained_path: str,
        vlm_path: str = "/all-flash-data/vlm/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        default_prompt: str = "",
        chunk_size: int = 50,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        action_dim: int | None = None,
        num_inference_steps: int = 10,
        output_chunk_size: int | None = None,
        use_delta_action: bool | None = None,
        action_mode: str | None = None,
        norm_stats_path: str = None,
        robot_type: str = "franka",
        training_repo_id: str = None,
        allow_stats_fallback: bool = False,
        train_expert_only_override: bool | None = None,
        train_vlm_only_override: bool | None = None,
        normalize_arm_joints: bool = True,
        normalize_gripper: bool = True,
        gripper_norm_mode: str | None = None,
        snap_gripper_to_binary: bool = False,
        gripper_max_width: float = 0.04,
        gripper_canonical_dim: int = 7,
    ):
        self.device = device
        self.default_prompt = default_prompt
        self.pretrained_path = pretrained_path
        # Optional deploy-time overrides for the two freeze flags; None defers to
        # the checkpoint config.json.
        self._train_expert_only_override = train_expert_only_override
        self._train_vlm_only_override = train_vlm_only_override
        self.chunk_size = chunk_size
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        if output_chunk_size is not None and int(output_chunk_size) <= 0:
            raise ValueError(f"output_chunk_size must be positive or None, got {output_chunk_size!r}")
        self.output_chunk_size = int(output_chunk_size) if output_chunk_size is not None else None
        if action_mode not in (None, "delta", "abs"):
            raise ValueError(f"action_mode must be delta|abs|None, got {action_mode!r}")
        self._action_mode_override = action_mode
        self._use_delta_action_override = use_delta_action
        self.use_delta_action = True if use_delta_action is None else bool(use_delta_action)
        self.robot_type = robot_type
        # Explicit training repo_id for multi_repo_v1 norm_stats selection. None
        # falls back to the robot_type heuristic; if that's ambiguous,
        # _unwrap_multi_repo raises (no silent mis-selection).
        self.training_repo_id = training_repo_id
        # Fallback gate. By default require quantile (q01/q99) stats for every
        # action key; mean/std or min/max fallback is refused so a ckpt with
        # broken/half-migrated stats fails loudly at load instead of producing
        # physically-wrong actions. Set True only to explicitly trust fallback.
        self.allow_stats_fallback = bool(allow_stats_fallback)
        # Per-segment normalization toggles (mirror of training-side flags); MUST
        # match the values used to train the ckpt. False skips both state input
        # normalization and action output denormalization for that segment (model
        # expects/produces raw values).
        self.normalize_arm_joints = bool(normalize_arm_joints)
        self.normalize_gripper = bool(normalize_gripper)
        if gripper_norm_mode is None:
            gripper_norm_mode = "q01_q99" if self.normalize_gripper else "noop"
        self.gripper_norm_mode = str(gripper_norm_mode).strip().lower().replace("-", "_")
        if not self.normalize_gripper:
            self.gripper_norm_mode = "noop"
        if self.gripper_norm_mode not in ("q01_q99", "mean_std", "noop"):
            raise ValueError(
                "gripper_norm_mode must be one of q01_q99, mean_std, noop; "
                f"got {gripper_norm_mode!r}"
            )
        # LabUtopia gripper canonicalize: snap raw width [0, max] to {0, max}
        # BEFORE q01/q99 normalization, matching training-side
        # SnapGripperToEndpointsFn. When training set --snap_gripper_to_binary,
        # deploy MUST also snap to keep the input distribution unchanged.
        # Threshold = 0.5*max_width.
        self.snap_gripper_to_binary = bool(snap_gripper_to_binary)
        self.gripper_max_width = float(gripper_max_width)
        self.gripper_canonical_dim = int(gripper_canonical_dim)

        # Auto-load DatasetSchema from checkpoint (preferred source for dataset
        # layout: state_keys / action_keys / delta_mask / gripper_dims).
        # Multi-repo training writes both labvla_schema.json (first repo's, for
        # back-compat) and labvla_schemas.json (dict {repo_id: schema_dict}). If
        # a multi-repo bundle exists, select by training_repo_id; falling through
        # to the first repo's schema would mis-interpret another repo's layout.
        self._schema = None
        ckpt_dir = Path(pretrained_path)

        # 1) Look for the multi-repo bundle first.
        multi_schema = None
        for cand in [ckpt_dir / "labvla_schemas.json",
                     ckpt_dir.parent / "labvla_schemas.json"]:
            if cand.exists():
                with open(cand) as f:
                    multi_schema = json.load(f)
                # Drop "_"-prefixed metadata keys from the repo candidate map.
                # The bundle is written as {"_schema": "multi_repo_v1", **schemas},
                # so without filtering "_schema" would be advertised as a
                # selectable repo (and could bind a string to self._schema).
                if isinstance(multi_schema, dict):
                    multi_schema = {
                        k: v for k, v in multi_schema.items()
                        if not (isinstance(k, str) and k.startswith("_"))
                    }
                logger.info(
                    f"Detected multi-repo schema bundle at {cand} "
                    f"(repos={list(multi_schema.keys())})"
                )
                break

        if multi_schema is not None:
            if not isinstance(multi_schema, dict) or not multi_schema:
                raise RuntimeError(
                    f"serve_labvla: labvla_schemas.json present but malformed "
                    f"(expected non-empty dict after metadata filter, got "
                    f"{type(multi_schema).__name__}). Re-train to regenerate."
                )
            if self.training_repo_id is None:
                raise RuntimeError(
                    f"serve_labvla: checkpoint contains a multi-repo schema "
                    f"bundle (repos={list(multi_schema.keys())}) but no "
                    f"--training_repo_id was provided. Pass --training_repo_id "
                    f"so deployment can select the matching schema."
                )
            if self.training_repo_id not in multi_schema:
                raise RuntimeError(
                    f"serve_labvla: training_repo_id={self.training_repo_id!r} "
                    f"not found in multi-repo schema bundle "
                    f"(repos={list(multi_schema.keys())}). Pass a repo_id that "
                    f"matches one of the trained repos."
                )
            self._schema = multi_schema[self.training_repo_id]
            logger.info(
                f"Auto-loaded DatasetSchema for repo {self.training_repo_id!r} "
                f"from multi-repo bundle (schema_id={self._schema.get('schema_id')}, "
                f"source={self._schema.get('source')})"
            )
        else:
            # 2) Single-schema checkpoint: fall back to labvla_schema.json.
            for candidate in [ckpt_dir / "labvla_schema.json",
                              ckpt_dir.parent / "labvla_schema.json"]:
                if candidate.exists():
                    with open(candidate) as f:
                        schema_dict = json.load(f)
                    self._schema = schema_dict
                    logger.info(
                        f"Auto-loaded DatasetSchema from {candidate} "
                        f"(schema_id={schema_dict.get('schema_id')}, "
                        f"source={schema_dict.get('source')})"
                    )
                    break

        if self._schema is None:
            raise RuntimeError(
                f"serve_labvla: no labvla_schema.json or labvla_schemas.json "
                f"found in {ckpt_dir} or its parent. Training must save schema "
                f"alongside the model (scripts/train.py does this automatically). "
                f"Re-train with a recent train.py or manually drop a "
                f"DatasetSchema dump into the checkpoint dir."
            )
        # Derive real action_dim from the loaded schema's `action_dims` tuple.
        # action_dim defaults to None: when None, adopt the schema-derived value
        # (auto-derivation); when explicit, cross-check and fail loud on
        # mismatch.
        _cli_action_dim_explicit = action_dim is not None
        _schema_action_dims = self._schema.get("action_dims")
        if _schema_action_dims is not None:
            try:
                _schema_action_dim = int(sum(int(d) for d in _schema_action_dims))
            except Exception:
                logger.warning(
                    "[R7-F5] schema action_dims=%r is not summable; "
                    "skipping action_dim cross-check and keeping CLI "
                    "action_dim=%r.",
                    _schema_action_dims, action_dim,
                )
                _schema_action_dim = None
            if _schema_action_dim is None:
                if not _cli_action_dim_explicit:
                    raise ValueError(
                        "[R8-F4] action_dim auto-derivation requires a "
                        "summable schema action_dims field, but the loaded "
                        f"schema has action_dims={_schema_action_dims!r}. "
                        "Pass --action_dim=<N> explicitly, or re-emit the "
                        "schema with a valid action_dims tuple."
                    )
            else:
                if _cli_action_dim_explicit and _schema_action_dim != action_dim:
                    raise ValueError(
                        f"[R7-F5] CLI --action_dim={action_dim} disagrees with "
                        f"the checkpoint's schema action_dim={_schema_action_dim} "
                        f"(sum of {tuple(_schema_action_dims)}). The deploy "
                        f"action chunk would be silently truncated/padded — for "
                        f"non-8-dim checkpoints this drops entire arms/grippers. "
                        f"Pass --action_dim={_schema_action_dim} (or remove the "
                        f"flag and let serve_labvla auto-derive it from schema)."
                    )
                # Source-of-truth lock-in: regardless of CLI explicit-or-not,
                # use schema-derived value (guard above already ensured they
                # match when both are present).
                self.action_dim = _schema_action_dim
                action_dim = _schema_action_dim
                logger.info(
                    "[R7-F5/R8-F4] action_dim=%d %s schema action_dims=%r",
                    action_dim,
                    "resolved (CLI default None → auto-derive) from"
                    if not _cli_action_dim_explicit
                    else "matches CLI override against",
                    tuple(_schema_action_dims),
                )
        else:
            if not _cli_action_dim_explicit:
                raise ValueError(
                    "[R8-F4] action_dim auto-derivation requires the loaded "
                    "schema to expose an 'action_dims' field, but this "
                    "checkpoint's schema has none. Either pass "
                    "--action_dim=<N> explicitly, or re-train so "
                    "save_checkpoint emits action_dims into the schema dump."
                )
            logger.warning(
                "[R7-F5] schema has no 'action_dims' field; cannot cross-check "
                "CLI action_dim=%d. Re-train so save_checkpoint serializes "
                "action_dims, or pass --action_dim explicitly matching the "
                "trained model.",
                action_dim,
            )

        raw_mask = np.array(self._schema["delta_mask"], dtype=bool)
        # state_dim comes from the schema's state_dims, not the action-side
        # delta_mask: delta_mask is over ACTION dims, so len(delta_mask) only
        # equals state_dim when they happen to match (single arm 8/8) and is
        # wrong when state and action widths diverge. Fall back to len(raw_mask)
        # only when the schema lacks state_dims (older dumps) and warn.
        _state_dims = self._schema.get("state_dims")
        if _state_dims is not None:
            try:
                self.state_dim = int(sum(int(d) for d in _state_dims))
            except Exception:
                logger.warning(
                    "serve_labvla: schema state_dims=%r is not summable; "
                    "falling back to len(delta_mask)=%d (action-derived).",
                    _state_dims, len(raw_mask),
                )
                self.state_dim = len(raw_mask)
        else:
            logger.warning(
                "serve_labvla: schema has no 'state_dims' field; falling back to "
                "len(delta_mask)=%d. This was the legacy P0-02 path — re-train so "
                "save_checkpoint serializes state_dims.",
                len(raw_mask),
            )
            self.state_dim = len(raw_mask)
        # Per-robot gripper-binarize threshold (on q01/q99 [-1, 1]) from
        # schema.arm_layout.gripper_binarize_threshold; fallback 0.5
        # (starVLA/openpi convention) for older schemas without arm_layout. Read
        # at init rather than per-inference to keep the hot path allocation-free.
        self._gripper_binarize_threshold: float = 0.5
        _arm_layout_dict = self._schema.get("arm_layout")
        if isinstance(_arm_layout_dict, dict):
            _thr = _arm_layout_dict.get("gripper_binarize_threshold", 0.5)
            try:
                self._gripper_binarize_threshold = float(_thr)
            except (TypeError, ValueError):
                logger.warning(
                    "serve_labvla: arm_layout.gripper_binarize_threshold=%r "
                    "is not a float; falling back to 0.5.",
                    _thr,
                )
        # If raw_mask has True bits past action_dim those delta dims would be
        # silently dropped; fail loud instead.
        if len(raw_mask) > action_dim and raw_mask[action_dim:].any():
            lost = int(raw_mask[action_dim:].sum())
            raise ValueError(
                f"delta_mask length ({len(raw_mask)}) exceeds action_dim "
                f"({action_dim}) and tail has {lost} True bit(s). "
                f"Those delta dims would be silently dropped. Either raise "
                f"action_dim or fix the schema."
            )
        self._delta_mask = np.zeros(action_dim, dtype=bool)
        n = min(len(raw_mask), action_dim)
        self._delta_mask[:n] = raw_mask[:n]
        self._norm_stats = None
        if norm_stats_path:
            # Explicit override: use the provided norm_stats file.
            with open(norm_stats_path) as f:
                self._norm_stats = json.load(f)
            logger.info(f"Loaded norm stats from {norm_stats_path}")
        else:
            # Auto-discover norm_stats.json that save_checkpoint writes alongside
            # the model.
            ckpt_dir = Path(pretrained_path)
            # Handle both checkpoint-N/ and checkpoint-N/pretrained_model/ paths.
            for candidate in [ckpt_dir / "norm_stats.json",
                              ckpt_dir.parent / "norm_stats.json"]:
                if candidate.exists():
                    with open(candidate) as f:
                        self._norm_stats = json.load(f)
                    logger.info(f"Auto-loaded norm stats from checkpoint: {candidate}")
                    break
            if self._norm_stats is None:
                logger.info(
                    "No norm_stats found in checkpoint or via --norm_stats_path. "
                    "Assuming model was trained without processor-level normalization "
                    "(e.g. v3.0 finetune). Raw delta denormalization will be skipped."
                )

        # If norm_stats loaded, confirm the schema's state/action keys are all
        # present. A silent key-miss feeds raw unnormalized state to the model
        # and returns unnormalized delta -> physically wrong execution with no
        # error signal.
        if self._norm_stats is not None:
            probe = self._unwrap_multi_repo(self._norm_stats)
            if probe is None:
                raise RuntimeError(
                    "serve_labvla: norm_stats has multi_repo_v1 schema but no "
                    "candidate repo entry was selectable. Pass --norm_stats_path "
                    "to a flat single-repo stats file, or re-save the ckpt."
                )
            expected_keys = list(self._schema["state_keys"]) + list(self._schema["action_keys"])
            # When norm_stats was saved with the opposite LeRobot column-name
            # convention (v3 observation.state/action vs v2.1 state/actions),
            # fill missing schema-side keys from aliases so the strict check
            # passes. Contents are identical; only the key strings differ.
            _alias_map = {
                "state": ("observation.state",),
                "observation.state": ("state",),
                "actions": ("action",),
                "action": ("actions",),
                "actions_abs": ("action_abs",),
                "action_abs": ("actions_abs",),
            }
            remapped: list[tuple[str, str]] = []
            for _k in expected_keys:
                if _k in probe:
                    continue
                for _alias in _alias_map.get(_k, ()):
                    if _alias in probe:
                        probe[_k] = probe[_alias]
                        remapped.append((_k, _alias))
                        break
            if remapped:
                logger.info(
                    "[norm_stats] auto-remapped %d key(s) from LeRobot raw-column "
                    "aliases: %r. Numerical contents preserved; only the "
                    "dictionary keys differ between v2.1 (state/actions) and v3 "
                    "(observation.state/action) conventions.",
                    len(remapped), remapped,
                )
            missing = [k for k in expected_keys if k not in probe]
            if missing:
                raise RuntimeError(
                    f"serve_labvla: norm_stats is loaded but does not contain "
                    f"the schema's expected keys {missing!r}. Present keys: "
                    f"{[k for k in probe.keys() if not k.startswith('_')][:10]!r}. "
                    f"This would silently skip normalization -> wrong physical "
                    f"execution. Pass --norm_stats_path to the correct stats "
                    f"file, or re-save the ckpt with schema-aligned stats."
                )

            # Validate action normalization stats at init (not first inference).
            # The canonical deploy-time inverse is q01/q99 (matches training's
            # NormalizeTransformFn); mean/std and min/max are fallbacks accepted
            # only when allow_stats_fallback=True. Otherwise a ckpt lacking
            # q01/q99 is rejected loudly here.
            action_stat_keys = list(self._schema["action_keys"])
            missing_quantile: list[str] = []
            missing_any: list[str] = []
            for ak in action_stat_keys:
                entry = probe.get(ak, {}) or {}
                has_quantile = "q01" in entry and "q99" in entry
                has_meanstd = "mean" in entry and "std" in entry
                has_minmax = "min" in entry and "max" in entry
                if not (has_quantile or has_meanstd or has_minmax):
                    missing_any.append(ak)
                    continue
                if not has_quantile:
                    missing_quantile.append(ak)
            if missing_any:
                raise RuntimeError(
                    f"serve_labvla: action normalization stats for key(s) "
                    f"{missing_any!r} lack any of q01/q99, mean/std, or min/max. "
                    f"Deploy denormalization has nothing to invert — the model "
                    f"would emit raw [-1,1] model outputs as if they were joint "
                    f"angles, which is physically wrong. Re-run "
                    f"`python -m data_process stats` on the training dataset "
                    f"and re-save the ckpt with the regenerated norm_stats.json."
                )
            if missing_quantile and not self.allow_stats_fallback:
                raise RuntimeError(
                    f"serve_labvla: action normalization stats for key(s) "
                    f"{missing_quantile!r} lack q01/q99 quantile stats. Deploy "
                    f"would silently fall back to mean/std (or min/max), which "
                    f"does not match training-time NormalizeTransformFn and can "
                    f"produce physically-wrong actions. Either re-generate the "
                    f"ckpt's norm_stats.json with quantile stats "
                    f"(`python -m data_process stats ...`), or pass "
                    f"--allow_stats_fallback to explicitly trust the fallback "
                    f"path."
                )

            # Sanity check on the training source. The per-key stats in
            # norm_stats.json were sliced at train-time from stats["action"]
            # (delta) or stats["action_abs"] (abs); we can't tell which from the
            # sliced form, but we can verify the state-key stats are present
            # (both modes need raw state stats for observation normalization).
            state_stat_keys = list(self._schema["state_keys"])
            missing_state = [
                k for k in state_stat_keys
                if not (probe.get(k, {}).get("mean") is not None
                        or probe.get(k, {}).get("q01") is not None)
            ]
            if missing_state:
                raise RuntimeError(
                    f"serve_labvla: state normalization stats for key(s) "
                    f"{missing_state!r} are absent (no mean/std nor q01/q99). "
                    f"Observation state would be fed to the model raw, which "
                    f"does not match training. Re-run "
                    f"`python -m data_process stats ...` and re-save the ckpt."
                )
            # In abs mode we can't hard-detect from the sliced per-key stats
            # whether they came from 'action' (delta) or 'action_abs'; the
            # train-time guard in hydrate_all already raises if action_mode='abs'
            # but 'action_abs' is missing. Just log a marker for cross-checking.
            if not self.use_delta_action:
                logger.info(
                    "serve_labvla: use_delta_action=False (abs mode). "
                    "Assuming norm_stats.json was written from stats['action_abs']; "
                    "if the ckpt was produced by an older train.py that did not "
                    "slice from action_abs, the denormalization will be wrong. "
                    "Verify via `python scripts/train.py --action_mode abs` was "
                    "the actual training command."
                )

        logger.info(f"Loading LabVLA from {pretrained_path}...")
        self._load_policy(pretrained_path, vlm_path)
        logger.info("LabVLA loaded and ready for inference!")

    def _load_policy(self, pretrained_path: str, vlm_path: str):
        """Load LabVLA model from checkpoint."""
        from policies.LabVLA.configuration_labvla import LabVLAConfig
        from policies.LabVLA.modeling_labvla import LabVLAPolicy

        # train.py writes config.json under the pretrained_model/ subdir. Reading
        # only the top-level path would fall through to LabVLAConfig defaults
        # (strict=False absorbing every size mismatch), loading the wrong arch
        # for any non-default hyperparam. Probe the subdir first; fall back to
        # the top-level path only for legacy checkpoints.
        ckpt_root = Path(pretrained_path)
        config_path = None
        for _cand in (ckpt_root / "pretrained_model" / "config.json",
                      ckpt_root / "config.json"):
            if _cand.exists():
                config_path = _cand
                break
        if config_path is not None:
            logger.info(f"[C-V6-03] Reading checkpoint config from {config_path}")
            with open(config_path) as f:
                saved = json.load(f)

            # Forward every field the checkpoint saved so training_phase /
            # knowledge_isolation / use_fast_tokenizer / discrete_action_* /
            # ki_mse_weight survive into deploy. Filter by dataclass fields to
            # stay robust to unknown extra keys in older/newer configs.
            allowed_fields = set(LabVLAConfig.__dataclass_fields__.keys())
            saved_fwd = {k: v for k, v in saved.items() if k in allowed_fields}

            # Normalize image_resolution: JSON deserializes tuples as lists.
            if "image_resolution" in saved_fwd and isinstance(saved_fwd["image_resolution"], list):
                saved_fwd["image_resolution"] = tuple(saved_fwd["image_resolution"])
            if "optimizer_betas" in saved_fwd and isinstance(saved_fwd["optimizer_betas"], list):
                saved_fwd["optimizer_betas"] = tuple(saved_fwd["optimizer_betas"])

            # train.py falls back to str(v) for non-JSON-serializable config
            # fields. input_features / output_features are dict[str,
            # PolicyFeature] (not JSON-clean), so they persist as repr strings.
            # Forwarding such a string makes output_features a str, and
            # validate_features() short-circuits on a substring match, so the
            # broken str survives until output_features[ACTION].shape[0] raises
            # TypeError. Drop it so validate_features() rebuilds proper
            # PolicyFeature objects from max_state_dim / max_action_dim.
            for _f in ("input_features", "output_features"):
                if _f in saved_fwd and isinstance(saved_fwd[_f], str):
                    logger.info(
                        f"[C-V6-05] Dropping str-fallback {_f} from saved "
                        f"config; LabVLAConfig.validate_features will rebuild "
                        f"it from max_state_dim/max_action_dim."
                    )
                    saved_fwd.pop(_f)

            # Deploy-only overrides: never GC/compile/freeze_vision in serve.
            saved_fwd["vlm_pretrained_path"] = vlm_path
            saved_fwd["freeze_vision_encoder"] = False
            saved_fwd["gradient_checkpointing"] = False
            saved_fwd["compile_model"] = False

            # Optional deploy-time freeze-flag overrides.
            if self._train_expert_only_override is not None:
                saved_fwd["train_expert_only"] = bool(
                    self._train_expert_only_override
                )
                logger.info(
                    "[MED-V5-02] override train_expert_only="
                    f"{saved_fwd['train_expert_only']} from CLI."
                )
            if self._train_vlm_only_override is not None:
                saved_fwd["train_vlm_only"] = bool(
                    self._train_vlm_only_override
                )
                logger.info(
                    "[MED-V5-02] override train_vlm_only="
                    f"{saved_fwd['train_vlm_only']} from CLI."
                )

            config = LabVLAConfig(**saved_fwd)

            # Fail fast on non-deployable checkpoints. serve_labvla only exposes
            # predict_action_chunk -> sample_actions -> _denoise_step, which all
            # require a DiT action head; vlm_pretrain checkpoints have no DiT
            # (CE-only), so they'd load silently and crash deep in the
            # flow-matching loop on the first inference.
            if config.training_phase != "posttrain":
                raise ValueError(
                    f"Refusing to deploy a non-posttrain checkpoint "
                    f"(training_phase={config.training_phase!r}). "
                    f"serve_labvla.py only supports posttrain checkpoints "
                    f"that have a DiT action head; vlm_pretrain checkpoints "
                    f"are CE-only and cannot produce action chunks. "
                    f"Checkpoint path: {pretrained_path}"
                )

            saved_action_mode = saved.get("action_mode")
            if self._action_mode_override is not None:
                effective_action_mode = self._action_mode_override
            elif self._use_delta_action_override is not None:
                effective_action_mode = "delta" if self._use_delta_action_override else "abs"
            elif saved_action_mode in ("delta", "abs"):
                effective_action_mode = saved_action_mode
            else:
                effective_action_mode = "delta"
                logger.warning(
                    "Checkpoint config has no action_mode; using legacy "
                    "deploy default action_mode='delta'. Pass --action_mode abs "
                    "for abs-trained legacy checkpoints."
                )
            if (
                self._use_delta_action_override is not None
                and self._action_mode_override is not None
                and bool(self._use_delta_action_override) != (effective_action_mode == "delta")
            ):
                raise ValueError(
                    "--use_delta_action and --action_mode disagree. Use only "
                    "--action_mode=delta|abs for new deployments."
                )
            self.use_delta_action = (effective_action_mode == "delta")
            config.action_mode = effective_action_mode
            self.chunk_size = config.chunk_size
            self.max_state_dim = config.max_state_dim
            self.max_action_dim = config.max_action_dim
            self._tokenizer_max_length = config.tokenizer_max_length
            logger.info(
                f"Deploy config (M-06 fix): training_phase={config.training_phase}, "
                f"action_mode={config.action_mode}, "
                f"knowledge_isolation={config.knowledge_isolation}, "
                f"use_fast_tokenizer={config.use_fast_tokenizer}"
            )
        else:
            config = LabVLAConfig(
                vlm_pretrained_path=vlm_path,
                dtype="bfloat16",
                chunk_size=self.chunk_size,
                max_state_dim=self.max_state_dim,
                max_action_dim=self.max_action_dim,
                num_inference_steps=self.num_inference_steps,
                gradient_checkpointing=False,
                compile_model=False,
                action_mode=self._action_mode_override
                or ("delta" if self._use_delta_action_override is not False else "abs"),
            )
            self.use_delta_action = (config.action_mode == "delta")
            self._tokenizer_max_length = config.tokenizer_max_length

        # Load checkpoint
        checkpoint_dir = Path(pretrained_path)
        if (checkpoint_dir / "pretrained_model").exists():
            model_dir = str(checkpoint_dir / "pretrained_model")
        else:
            model_dir = str(checkpoint_dir)

        # Deploy loads strictly by default; otherwise a ckpt with mismatched DiT
        # layers / KI head / projection would leave those modules at random init
        # and serve garbage actions with no warning. Opt out only for
        # partial-load debugging.
        _deploy_strict = os.environ.get(
            "LABVLA_DEPLOY_ALLOW_PARTIAL_LOAD"
        ) != "1"
        self.policy = LabVLAPolicy.from_pretrained(
            pretrained_name_or_path=model_dir,
            config=config,
            strict=_deploy_strict,
        )
        if not _deploy_strict:
            logger.warning(
                "[deploy-load] LABVLA_DEPLOY_ALLOW_PARTIAL_LOAD=1 — loaded "
                "policy with strict=False; missing/unexpected keys (if any) "
                "leave modules at random init. Verify the ckpt matches the "
                "config before serving."
            )

        self.model_dtype = torch.bfloat16
        self.policy.to(device=self.device, dtype=self.model_dtype)
        # Restore KI classifier fp32 (policy.to(bf16) above clobbers it),
        # mirroring the training-side re-cast after accelerator.prepare.
        try:
            ki = getattr(getattr(self.policy, "model", self.policy), "ki_head", None)
            if ki is not None and hasattr(ki, "classifier"):
                ki.classifier.to(dtype=torch.float32)
                logger.info(
                    "[ki_head] classifier re-cast to fp32 after policy.to(bf16) "
                    "(deploy parity with training)"
                )
        except Exception as e:
            logger.warning(f"[ki_head] classifier fp32 restore failed: {e}")
        self.policy.eval()

        # VLM processor
        from transformers import Qwen3VLProcessor
        self.vlm_processor = Qwen3VLProcessor.from_pretrained(vlm_path)
        self.vision_start_token_id = self.vlm_processor.vision_start_token_id
        self.vision_end_token_id = self.vlm_processor.vision_end_token_id
        self.image_token_id = self.vlm_processor.image_token_id

        # Image patch info
        SPATIAL_MERGE_SIZE = 2
        h, w = config.image_resolution
        self.image_size = (h, w)
        dummy_img = torch.zeros(h, w, 3)
        dummy_out = self.vlm_processor.image_processor(
            [dummy_img], do_rescale=False, return_tensors="pt"
        )
        self._fixed_grid_thw = dummy_out["image_grid_thw"]
        self.num_patches = int(self._fixed_grid_thw[0, 0] * self._fixed_grid_thw[0, 1] * self._fixed_grid_thw[0, 2])
        self.num_image_tokens = self.num_patches // (SPATIAL_MERGE_SIZE ** 2)

        if self.output_chunk_size is not None and self.output_chunk_size > self.chunk_size:
            raise ValueError(
                f"output_chunk_size={self.output_chunk_size} exceeds model "
                f"chunk_size={self.chunk_size}; deploy can only shorten the "
                "returned action horizon."
            )

        logger.info(
            f"LabVLA loaded: chunk_size={self.chunk_size}, "
            f"output_chunk_size={self.output_chunk_size or self.chunk_size}, "
            f"max_action_dim={self.max_action_dim}, image_size={self.image_size}, "
            f"tokens_per_camera={self.num_image_tokens}"
        )

    def _process_image(self, image_hwc: np.ndarray) -> tuple:
        """Convert HWC uint8 image to Qwen3-VL pixel_values.

        Aligned with training pipeline:
        1. Aspect-ratio-preserving resize + zero padding (aligned with training-side and InternVLA-A1)
        2. Output in (C, H, W) format
        """
        target_h, target_w = self.image_size
        # HWC uint8 -> CHW float [0, 1]
        img_tensor = torch.from_numpy(image_hwc.copy()).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        # Aspect-ratio-preserving resize + zero padding (aligned with training-side base_processor.resize_image and InternVLA-A1)
        img_tensor = self._resize_with_pad(img_tensor, target_h, target_w)

        img_inputs = self.vlm_processor.image_processor(
            [img_tensor], do_rescale=False, return_tensors="pt"
        )
        pixel_values = img_inputs["pixel_values"]
        image_grid_thw = img_inputs["image_grid_thw"]
        return pixel_values, image_grid_thw

    @staticmethod
    def _resize_with_pad(
        image: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Resize keeping aspect ratio + zero-pad to target size.

        Aligned with training-side base_processor.resize_image() and InternVLA-A1's resize_with_pad.
        """
        import torch.nn.functional as F
        C, H, W = image.shape
        if H == target_h and W == target_w:
            return image
        scale = min(target_h / H, target_w / W)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        image_4d = image.unsqueeze(0)
        resized = F.interpolate(image_4d, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        return padded.squeeze(0).clamp(0.0, 1.0)

    # ---- norm_stats schema helpers ----
    # norm_stats.json written by training is keyed by the schema's state/action
    # keys (e.g. "observation.state", "action"). For multi-repo training it
    # wraps per-repo flat dicts under a "_schema": "multi_repo_v1" marker.

    def _unwrap_multi_repo(self, norm_stats):
        """If norm_stats is multi_repo_v1 format, pick the repo dictated by
        --repo_id (preferred), else by robot_type heuristic, else fail.

        Never silently pick the first repo (that would denormalize against the
        wrong dataset's q01/q99 range). Selection order:

          1. If training_repo_id was provided on CLI -> require exact match.
          2. Else if robot_type matches exactly one key -> use it.
          3. Else -> raise.
        """
        if not (isinstance(norm_stats, dict) and norm_stats.get("_schema") == "multi_repo_v1"):
            return norm_stats

        repo_keys = [k for k in norm_stats.keys() if k != "_schema"]
        if not repo_keys:
            return None

        # 1. Explicit --repo_id (authoritative).
        explicit = getattr(self, "training_repo_id", None)
        if explicit:
            if explicit not in norm_stats:
                raise ValueError(
                    f"serve_labvla: --repo_id={explicit!r} not found in "
                    f"multi_repo_v1 norm_stats. Available repos: {repo_keys}"
                )
            logger.info(
                "norm_stats multi_repo_v1 → using explicit --repo_id=%r", explicit,
            )
            return norm_stats[explicit]

        # 2. robot_type substring heuristic — require UNIQUE match.
        if self.robot_type:
            matches = [k for k in repo_keys if self.robot_type in k]
            if len(matches) == 1:
                logger.info(
                    "norm_stats multi_repo_v1 → robot_type=%r uniquely matched repo=%r",
                    self.robot_type, matches[0],
                )
                return norm_stats[matches[0]]
            if len(matches) > 1:
                raise ValueError(
                    f"serve_labvla: robot_type={self.robot_type!r} matches multiple "
                    f"repos {matches}; pass --repo_id=<one of {matches}> to disambiguate."
                )

        # 3. No basis for selection — refuse silent fallback.
        raise ValueError(
            f"serve_labvla: norm_stats has multi_repo_v1 format with repos "
            f"{repo_keys}, but neither --repo_id nor a matching --robot_type "
            f"was provided. Pass --repo_id=<one of {repo_keys}> to select "
            f"the stats for this deployment."
        )

    @staticmethod
    def _concat_keys(norm_stats: dict, keys):
        """Concat mean/std for each key from `keys` in order. Returns (None, None) on miss."""
        means, stds = [], []
        for k in keys:
            if k not in norm_stats:
                return None, None
            entry = norm_stats[k]
            if "mean" not in entry or "std" not in entry:
                return None, None
            means.append(np.asarray(entry["mean"], dtype=np.float32).reshape(-1))
            stds.append(np.asarray(entry["std"], dtype=np.float32).reshape(-1))
        return np.concatenate(means), np.concatenate(stds)

    @staticmethod
    def _concat_keys_minmax(norm_stats: dict, keys):
        """Concat min/max for each key. Used to set per-dim gripper binarization threshold."""
        mins, maxs = [], []
        for k in keys:
            if k not in norm_stats:
                return None, None
            entry = norm_stats[k]
            if "min" not in entry or "max" not in entry:
                return None, None
            mins.append(np.asarray(entry["min"], dtype=np.float32).reshape(-1))
            maxs.append(np.asarray(entry["max"], dtype=np.float32).reshape(-1))
        return np.concatenate(mins), np.concatenate(maxs)

    @staticmethod
    def _concat_keys_q01_q99(norm_stats: dict, keys):
        """Concat q01/q99 for each key in order. Returns (None, None) on miss.

        Mirrors `_concat_keys` but reads quantile fields, used by
        `_get_obs_q01_q99` to invert q01/q99 normalization that
        training applies to state via NormalizeTransformFn(mode='q01_q99').
        """
        q01s, q99s = [], []
        for k in keys:
            if k not in norm_stats:
                return None, None
            entry = norm_stats[k]
            if "q01" not in entry or "q99" not in entry:
                return None, None
            q01s.append(np.asarray(entry["q01"], dtype=np.float32).reshape(-1))
            q99s.append(np.asarray(entry["q99"], dtype=np.float32).reshape(-1))
        return np.concatenate(q01s), np.concatenate(q99s)

    def _get_obs_stats(self, norm_stats):
        """Return (mean, std) for observation state. None when training wasn't normalized."""
        if norm_stats is None:
            return None, None
        norm_stats = self._unwrap_multi_repo(norm_stats)
        if norm_stats is None:
            return None, None
        return self._concat_keys(norm_stats, self._schema["state_keys"])

    def _get_obs_q01_q99(self, norm_stats):
        """Return (q01, q99) for observation state. None when missing.

        Training uses ``mean_std`` for arm dims and ``q01_q99`` for the
        canonical gripper indices declared by ``schema.gripper_action_dims``
        (auto-injected by ``transforms/core.py::hydrate_all``); deploy must
        match that mixed policy, so the gripper inverse path needs q01/q99
        access here.
        """
        if norm_stats is None:
            return None, None
        norm_stats = self._unwrap_multi_repo(norm_stats)
        if norm_stats is None:
            return None, None
        return self._concat_keys_q01_q99(norm_stats, self._schema["state_keys"])

    def _get_delta_stats(self, norm_stats):
        """Return (mean, std) for delta action. None when training wasn't normalized."""
        if norm_stats is None:
            return None, None
        norm_stats = self._unwrap_multi_repo(norm_stats)
        if norm_stats is None:
            return None, None
        return self._concat_keys(norm_stats, self._schema["action_keys"])

    def _build_3camera_batch(
        self, images: list, prompt: str, state: np.ndarray
    ) -> dict:
        """Build input batch from 3 camera images, prompt, and state.

        Missing-camera handling: a None slot fills with a placeholder tensor (any
        valid camera's pixel_values) but sets attention_mask=0 for those vision
        tokens, so the VLM treats them as absent. Mirrors the training-side
        `<cam>_mask=0` that zeros the corresponding attention slice.
        """
        all_pixel_values = []
        all_image_grid_thw = []
        input_ids = []
        attention_mask = []

        # Pre-process the first valid image once to get a dtype/shape-correct
        # placeholder for missing slots. We rely on caller (infer) having
        # already returned early when valid_images is empty.
        first_valid_idx = next((i for i, img in enumerate(images) if img is not None), None)
        if first_valid_idx is None:
            raise RuntimeError(
                "_build_3camera_batch called with no valid images — caller "
                "must short-circuit empty valid_images before reaching here."
            )
        placeholder_pv, placeholder_grid = self._process_image(images[first_valid_idx])

        for i, img in enumerate(images):
            if img is not None:
                pv, grid = self._process_image(img)
                all_pixel_values.append(pv)
                all_image_grid_thw.append(grid)
                vision_ids = (
                    [self.vision_start_token_id]
                    + [self.image_token_id] * self.num_image_tokens
                    + [self.vision_end_token_id]
                )
                input_ids += vision_ids
                attention_mask += [1] * len(vision_ids)
            else:
                # Invalid camera: placeholder keeps the token sequence shape
                # fixed, mask=0 makes the VLM ignore these vision tokens.
                all_pixel_values.append(placeholder_pv)
                all_image_grid_thw.append(placeholder_grid)
                vision_ids = (
                    [self.vision_start_token_id]
                    + [self.image_token_id] * self.num_image_tokens
                    + [self.vision_end_token_id]
                )
                input_ids += vision_ids
                attention_mask += [0] * len(vision_ids)

        # Tokenize language instruction (max_length aligned with training-side tokenizer_max_length)
        lang_inputs = self.vlm_processor.tokenizer(
            prompt,
            max_length=self._tokenizer_max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids += lang_inputs.input_ids
        attention_mask += lang_inputs.attention_mask

        pixel_values = torch.cat(all_pixel_values, dim=0)
        image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

        # State normalization + padding (mixed normalization): arm joint dims use
        # mean_std (near-Gaussian; q01/q99 would compress the working region and
        # erode precision); gripper canonical dims use q01/q99 (bimodal width;
        # closed/open align with +/-1). Mirrors training-side hydrate_all
        # dim_overrides for schema.gripper_action_dims.
        state_normalized = state.copy()
        # Snap gripper width to {0, max_width} BEFORE normalization, mirroring
        # training-side SnapGripperToEndpointsFn. No-op when disabled or when the
        # canonical gripper dim is past the supplied state.
        if self.snap_gripper_to_binary and self.gripper_canonical_dim < len(state_normalized):
            threshold = 0.5 * self.gripper_max_width
            grip_idx = self.gripper_canonical_dim
            grip_val = state_normalized[grip_idx]
            state_normalized[grip_idx] = (
                self.gripper_max_width if grip_val > threshold else 0.0
            )
        oj_mean, oj_std = self._get_obs_stats(self._norm_stats)
        oj_q01, oj_q99 = self._get_obs_q01_q99(self._norm_stats)
        gripper_canonical = list(self._schema.get("gripper_action_dims") or ())
        if oj_mean is not None or oj_q01 is not None:
            ref_len = len(oj_mean) if oj_mean is not None else len(oj_q01)
            state_dim = min(len(state), ref_len)
            arm_idx = np.array(
                [d for d in range(state_dim) if d not in gripper_canonical],
                dtype=np.int64,
            )
            grip_idx = np.array(
                [d for d in range(state_dim) if d in gripper_canonical],
                dtype=np.int64,
            )
            # arm dims: mean_std preferred, q01/q99 fallback for legacy ckpts.
            # Skip entirely when --normalize_arm_joints=false (training noop).
            if arm_idx.size > 0 and self.normalize_arm_joints:
                if oj_mean is not None and oj_std is not None:
                    state_normalized[arm_idx] = (
                        (state[arm_idx] - oj_mean[arm_idx])
                        / (oj_std[arm_idx] + 1e-6)
                    )
                elif oj_q01 is not None and oj_q99 is not None:
                    state_normalized[arm_idx] = (
                        2.0 * (state[arm_idx] - oj_q01[arm_idx])
                        / (oj_q99[arm_idx] - oj_q01[arm_idx] + 1e-6) - 1.0
                    )
            # gripper dims: mode must mirror training-side gripper_norm_mode.
            # Normalize from state_normalized[grip_idx], not raw state[grip_idx]:
            # when snap_gripper_to_binary is on, the snap above already wrote the
            # snapped width into state_normalized, and reading raw state would
            # discard it. Identical to state on the no-snap path (state.copy()).
            grip_src = state_normalized[grip_idx]
            if grip_idx.size > 0 and self.gripper_norm_mode != "noop":
                if self.gripper_norm_mode == "q01_q99":
                    if oj_q01 is None or oj_q99 is None:
                        raise RuntimeError(
                            f"gripper_norm_mode='q01_q99' but observation.state stats "
                            f"lack q01/q99 — cannot normalize gripper dims. Ensure stats "
                            f"file contains q01/q99 keys or set gripper_norm_mode='mean_std'."
                        )
                    state_normalized[grip_idx] = (
                        2.0 * (grip_src - oj_q01[grip_idx])
                        / (oj_q99[grip_idx] - oj_q01[grip_idx] + 1e-6) - 1.0
                    )
                elif oj_mean is not None and oj_std is not None:
                    state_normalized[grip_idx] = (
                        (grip_src - oj_mean[grip_idx])
                        / (oj_std[grip_idx] + 1e-6)
                    )
                elif oj_q01 is not None and oj_q99 is not None:
                    state_normalized[grip_idx] = (
                        2.0 * (grip_src - oj_q01[grip_idx])
                        / (oj_q99[grip_idx] - oj_q01[grip_idx] + 1e-6) - 1.0
                    )
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[:min(len(state_normalized), self.max_state_dim)] = state_normalized[:self.max_state_dim]
        state_tensor = torch.from_numpy(state_padded).to(dtype=self.model_dtype)

        batch = {
            "observation.pixel_values": pixel_values.unsqueeze(0).to(device=self.device),
            "observation.image_grid_thw": image_grid_thw.to(self.device),
            "observation.input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device),
            "observation.attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(self.device),
            "observation.state": state_tensor.unsqueeze(0).to(self.device),
        }
        return batch

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on a LabUtopia observation."""
        start_time = time.monotonic()

        prompt = obs.get("prompt", obs.get("language_instruction", self.default_prompt))
        if not prompt:
            prompt = self.default_prompt

        # 3-camera images
        camera_keys = ["camera_1_rgb", "camera_2_rgb", "camera_3_rgb"]
        images = []
        for cam_key in camera_keys:
            if cam_key in obs and obs[cam_key] is not None:
                images.append(parse_image_to_uint8_hwc(obs[cam_key]))
            else:
                images.append(None)

        valid_images = [img for img in images if img is not None]
        if not valid_images:
            logger.error("No camera image found in observation!")
            output_len = self.output_chunk_size or self.chunk_size
            return {
                "actions": np.zeros((output_len, self.action_dim)),
            }

        # Pass None camera slots through (do NOT replace with first_valid):
        # _build_3camera_batch's mask=0 branch must fire for missing slots,
        # matching the training-side `<cam>_mask=0`.
        state = np.asarray(obs.get("state", np.zeros(self.state_dim)), dtype=np.float32)

        batch = self._build_3camera_batch(images, prompt, state)

        with torch.no_grad():
            action_chunk = self.policy.predict_action_chunk(batch)

        if isinstance(action_chunk, torch.Tensor):
            actions = action_chunk.detach().float().cpu().numpy()
        else:
            actions = np.asarray(action_chunk)

        if actions.ndim == 3:
            actions = actions[0]

        # Truncate to actual action dimension
        actions = actions[..., :self.action_dim]

        # Post-processing inverts training's NormalizeTransformFn +
        # DeltaActionTransformFn:
        #   Step 1: un-normalize action dims with the training mixed mode
        #           (arm=mean_std, gripper=q01_q99).
        #   Step 2: delta -> absolute for arm dims only (gripper is abs by
        #           delta_mask=False, already in raw units after Step 1).
        # Gripper outputs go through the q01/q99 inverse (NOT binarized in
        # normalized space): a normalized [-1, 1] maps to raw width in [q01, q99],
        # and out-of-range predictions extrapolate intentionally (no clamp). The
        # schema's gripper_binarize_threshold is unused at deploy (kept for
        # back-compat).
        # Do not clip normalized actions here: training doesn't clamp z-scores or
        # q01/q99, and legitimate arm deltas exceed [-1, 1]; clipping before the
        # inverse compresses reach motions and misplaces the end effector.
        arm_mask = self._delta_mask[:actions.shape[-1]]   # True = arm (delta-eligible)
        arm_idxs = np.where(arm_mask)[0]

        # Step 1: un-normalize action via mixed mode.
        #   - arm dims (delta_mask=True): mean_std inverse, x = norm*std + mean.
        #   - gripper dims (delta_mask=False): q01_q99 inverse,
        #     x = (norm + 1)/2 * (q99 - q01) + q01.
        # Each branch falls back to the other scheme when the preferred stats
        # fields are missing (legacy ckpts).
        gripper_canonical = list(self._schema.get("gripper_action_dims") or ())
        arm_idxs_act = np.array(
            [d for d in range(actions.shape[-1]) if d not in gripper_canonical],
            dtype=np.int64,
        )
        grip_idxs_act = np.array(
            [d for d in range(actions.shape[-1]) if d in gripper_canonical],
            dtype=np.int64,
        )
        if self._norm_stats is not None:
            nm = self._unwrap_multi_repo(self._norm_stats)
            aj_q01, aj_q99 = None, None
            aj_mean, aj_std = None, None
            if nm is not None:
                try:
                    vals = [nm[k] for k in self._schema["action_keys"]]
                    if all("q01" in v and "q99" in v for v in vals):
                        aj_q01 = np.concatenate(
                            [np.asarray(v["q01"], dtype=np.float32).reshape(-1) for v in vals]
                        )
                        aj_q99 = np.concatenate(
                            [np.asarray(v["q99"], dtype=np.float32).reshape(-1) for v in vals]
                        )
                    if all("mean" in v and "std" in v for v in vals):
                        aj_mean = np.concatenate(
                            [np.asarray(v["mean"], dtype=np.float32).reshape(-1) for v in vals]
                        )
                        aj_std = np.concatenate(
                            [np.asarray(v["std"], dtype=np.float32).reshape(-1) for v in vals]
                        )
                except KeyError:
                    pass

            # arm dims: prefer mean_std (matches training-side mixed norm).
            # 2026-04-28: skip arm denormalization when --normalize_arm_joints=false
            # (model was trained with arm noop → output is already raw delta).
            if arm_idxs_act.size > 0 and self.normalize_arm_joints:
                if aj_mean is not None and aj_std is not None:
                    actions[:, arm_idxs_act] = (
                        actions[:, arm_idxs_act] * (aj_std[arm_idxs_act] + 1e-6)
                        + aj_mean[arm_idxs_act]
                    )
                elif aj_q01 is not None and aj_q99 is not None:
                    actions[:, arm_idxs_act] = (
                        0.5 * (actions[:, arm_idxs_act] + 1.0)
                        * (aj_q99[arm_idxs_act] - aj_q01[arm_idxs_act] + 1e-6)
                        + aj_q01[arm_idxs_act]
                    )
            # gripper dims: mode must mirror training-side gripper_norm_mode.
            if grip_idxs_act.size > 0 and self.gripper_norm_mode != "noop":
                if self.gripper_norm_mode == "q01_q99" and aj_q01 is not None and aj_q99 is not None:
                    actions[:, grip_idxs_act] = (
                        0.5 * (actions[:, grip_idxs_act] + 1.0)
                        * (aj_q99[grip_idxs_act] - aj_q01[grip_idxs_act] + 1e-6)
                        + aj_q01[grip_idxs_act]
                    )
                elif aj_mean is not None and aj_std is not None:
                    actions[:, grip_idxs_act] = (
                        actions[:, grip_idxs_act] * (aj_std[grip_idxs_act] + 1e-6)
                        + aj_mean[grip_idxs_act]
                    )
                elif aj_q01 is not None and aj_q99 is not None:
                    actions[:, grip_idxs_act] = (
                        0.5 * (actions[:, grip_idxs_act] + 1.0)
                        * (aj_q99[grip_idxs_act] - aj_q01[grip_idxs_act] + 1e-6)
                        + aj_q01[grip_idxs_act]
                    )

            # Do not clamp gripper outputs here: the inverse q01/q99 above
            # intentionally extrapolates in raw-width space; deploy-time gripper
            # clipping caused LabUtopia accuracy drops.

        # Step 2: delta -> absolute for arm dims only (gripper has
        # delta_mask=False and is already absolute raw width after Step 1). In
        # abs mode the model predicts target poses, so skip the state add.
        if self.use_delta_action and arm_idxs.size > 0:
            # Mirror training's DeltaActionTransformFn: zero-pad the single-frame
            # state up to action width, then add it back on delta_mask dims. When
            # state_dim < action_dim, state[:len(arm_mask)] is shorter than
            # arm_mask and a boolean index would raise IndexError; zero-pad to
            # len(arm_mask) first so masked dims beyond the real state contribute
            # 0. No-op when state_dim >= action_dim.
            n_arm = len(arm_mask)
            state_for_add = state[:n_arm]
            if state_for_add.shape[0] < n_arm:
                state_for_add = np.concatenate([
                    state_for_add,
                    np.zeros(n_arm - state_for_add.shape[0], dtype=state_for_add.dtype),
                ])
            state_delta = state_for_add[arm_mask]
            actions[:, arm_mask] = actions[:, arm_mask] + state_delta[np.newaxis, :]

        # Pad to 8 dimensions
        if actions.shape[-1] < 8:
            padded = np.zeros((actions.shape[0], 8), dtype=np.float32)
            padded[:, :actions.shape[-1]] = actions
            actions = padded

        # Deploy-only horizon ablation: keep the model/checkpoint horizon intact
        # but expose only the first N actions, so shorter client-side execution
        # chunks can be tested without changing trained positional shapes.
        if self.output_chunk_size is not None:
            actions = actions[: self.output_chunk_size]

        result = {
            "actions": actions,
            "policy_timing": {"infer_ms": (time.monotonic() - start_time) * 1000},
        }

        logger.debug(f"Inference done in {(time.monotonic() - start_time)*1000:.1f}ms")
        return result

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "policy_type": "labvla",
            "action_dim": self.action_dim,
            "chunk_size": self.output_chunk_size or self.chunk_size,
            "model_chunk_size": self.chunk_size,
            "output_chunk_size": self.output_chunk_size or self.chunk_size,
            "state_dim": self.state_dim,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "num_inference_steps": self.num_inference_steps,
            "num_cameras": 3,
        }


def _strict_bool(value: str) -> bool:
    """Strict bool parser for the deploy CLI (matches the training CLI's
    _strict_bool). A typo in --normalize_arm_joints / --normalize_gripper /
    --snap_gripper_to_binary would otherwise silently flip deploy-time
    normalization and corrupt physical execution.
    """
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    raise argparse.ArgumentTypeError(
        f"expected 'true' or 'false' (case-insensitive), got {value!r}."
    )


# Data-semantics args that change the model's input/output distribution.
# Training persists these into training_state.pt["args"]. The deploy CLI
# defaults to current behavior, so a ckpt trained with a non-default value would
# silently use the wrong normalization unless every flag is re-specified by
# hand. Read them back from the ckpt and adopt the saved value when the operator
# didn't pass the flag, or warn loudly on mismatch when they did.
_DATA_SEMANTICS_ARGS = (
    "normalize_arm_joints",
    "normalize_gripper",
    "gripper_norm_mode",
    "snap_gripper_to_binary",
)


def _load_training_state_args(pretrained_path: str) -> Dict[str, Any] | None:
    """Best-effort read of training_state.pt["args"] from a checkpoint dir.

    Returns the saved argparse dict, or None when the sidecar is absent or
    unreadable (older ckpts, pretrained_model-only exports). Never raises:
    deployment must still work for checkpoints that predate this sidecar.
    """
    ckpt_dir = Path(pretrained_path)
    candidates = [
        ckpt_dir / "training_state" / "training_state.pt",
        ckpt_dir.parent / "training_state" / "training_state.pt",
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            # weights_only=False: trusted local training sidecar holding a plain
            # dict of argparse values, not just tensors.
            state = torch.load(str(cand), map_location="cpu", weights_only=False)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "[T6] found %s but could not load it (%s); deploy-time data "
                "semantics will fall back to CLI defaults.", cand, e,
            )
            return None
        saved_args = state.get("args") if isinstance(state, dict) else None
        if isinstance(saved_args, dict):
            logger.info("[T6] loaded training-time args from %s", cand)
            return saved_args
        return None
    return None


def _reconcile_data_semantics(args, explicit_flags: set) -> None:
    """Align deploy data-semantics with the ckpt's training_state.

    For each of `_DATA_SEMANTICS_ARGS`:
      - if the operator did NOT pass the flag, adopt the value saved at training
        time (so deploy reproduces the trained input/output distribution);
      - if the operator DID pass the flag and it disagrees with the saved value,
        keep the operator's value but emit a loud warning (explicit override).
    Mutates `args` in place. No-op when the ckpt has no readable training_state.
    """
    saved_args = _load_training_state_args(args.pretrained_path)
    if not saved_args:
        logger.info(
            "[T6] no training_state args available; deploy-time normalization "
            "semantics use CLI values/defaults (this is expected for older or "
            "pretrained_model-only checkpoints)."
        )
        return
    for key in _DATA_SEMANTICS_ARGS:
        if key not in saved_args:
            continue
        trained_val = saved_args[key]
        current_val = getattr(args, key, None)
        if key in explicit_flags:
            if trained_val != current_val:
                logger.warning(
                    "[T6] --%s=%r was set explicitly but the checkpoint was "
                    "trained with %s=%r. Mismatched data semantics produce a "
                    "train/deploy distribution gap; proceeding with your "
                    "explicit value. Drop the flag to auto-restore the "
                    "trained value.",
                    key, current_val, key, trained_val,
                )
            continue
        # Operator left it at the default: adopt the trained value.
        if trained_val != current_val:
            logger.warning(
                "[T6] restoring --%s=%r from checkpoint training_state "
                "(deploy default was %r). Pass --%s explicitly to override.",
                key, trained_val, current_val, key,
            )
        setattr(args, key, trained_val)


def parse_args():
    parser = argparse.ArgumentParser(description="LabVLA Deployment Server for LabUtopia")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to LabVLA checkpoint (e.g. checkpoint-30000 or checkpoint-30000/pretrained_model)")
    parser.add_argument("--vlm_path", type=str, default="/all-flash-data/vlm/Qwen3-VL-4B-Instruct")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--default_prompt", type=str, default="")
    # Default None (not 8) so serve_labvla auto-derives action_dim from the
    # schema's sum(action_dims); a fixed default would make the constructor
    # unable to distinguish "explicit override" from "use schema". Explicit
    # values are still cross-checked against the schema and fail loud on mismatch.
    parser.add_argument("--action_dim", type=int, default=None,
                        help="Action chunk dimensionality. Default None → "
                             "auto-derive from the loaded checkpoint's "
                             "schema action_dims. Explicit values are "
                             "cross-checked against schema and fail-loud "
                             "on mismatch (R7-F5 / R8-F4).")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--output_chunk_size", type=int, default=None,
                        help="Optional deploy-only output horizon. The model still "
                             "predicts checkpoint chunk_size actions; the server "
                             "returns only the first N actions and reports N in "
                             "metadata['chunk_size'].")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--action_mode", choices=["auto", "delta", "abs"], default="auto",
                        help="Deployment action semantics. auto reads checkpoint config; "
                             "delta adds current state back to arm deltas; abs returns "
                             "predicted target poses directly.")
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Normalization stats path (joint_norm_stats.json)")
    parser.add_argument("--robot_type", type=str, default="franka",
                        help="Robot type for MASK_MAPPING delta mask (e.g., franka, franka_robotiq, aloha)")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="CRIT-02: explicit training repo_id for multi_repo_v1 "
                             "norm_stats selection. Required when the ckpt was trained on "
                             "multiple repos AND robot_type is ambiguous across them; "
                             "e.g. --repo_id=robointer_droid_clean.")
    parser.add_argument("--allow_stats_fallback", action="store_true",
                        help="M-26: allow deploy to fall back from q01/q99 to mean/std "
                             "(or min/max) when quantile stats are missing from the ckpt. "
                             "Default false — a missing-quantile ckpt fails loudly at load. "
                             "Enable only when you have explicitly chosen to trust the "
                             "fallback denormalization path.")
    # Per-segment normalization toggles; MUST match the values used to train the
    # ckpt. Default true = mixed-norm behavior; set false for a ckpt trained with
    # --normalize_arm_joints=false / --normalize_gripper=false.
    parser.add_argument("--normalize_arm_joints", type=_strict_bool, default=True,
                        help="Apply mean_std denorm to arm joint dims at deploy. "
                             "Set false when ckpt was trained with arm noop.")
    parser.add_argument("--normalize_gripper", type=_strict_bool, default=True,
                        help="Apply q01_q99 denorm to gripper dims at deploy. "
                             "Set false when ckpt was trained with gripper noop.")
    parser.add_argument("--gripper_norm_mode", type=str, default=None,
                        choices=["q01_q99", "mean_std", "noop"],
                        help="Explicit gripper normalization mode used by training.")
    # Gripper canonicalize toggles; MUST match the training launcher. When
    # training set --snap_gripper_to_binary=true, deploy MUST also pass true so
    # the raw width is snapped to {0, max_width} before normalization.
    parser.add_argument("--snap_gripper_to_binary", type=_strict_bool, default=False,
                        help="Snap raw gripper width to {0, max_width} before "
                             "q01/q99 normalize. Set true ONLY for ckpts trained "
                             "with --snap_gripper_to_binary=true.")
    parser.add_argument("--gripper_max_width", type=float, default=0.04,
                        help="Max gripper width in meters (Franka Panda default 0.04).")
    parser.add_argument("--gripper_canonical_dim", type=int, default=7,
                        help="Canonical gripper index in the state/action vector (default 7).")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="M-27: size of the inference thread pool used by the "
                             "websocket server. Default 4 fits a single-GPU / 1-2 "
                             "client setup; raise only when you have spare GPU memory "
                             "for parallel in-flight inference.")
    parser.add_argument("--auth_token", type=str, default=os.environ.get("LABVLA_WS_AUTH_TOKEN"),
                        help="Bearer token required for websocket upgrades via "
                             "Authorization: Bearer <token>.")
    parser.add_argument("--use_delta_action", choices=["auto", "true", "false"], default="auto",
                        help="Legacy override; prefer --action_mode. auto reads checkpoint config.")
    parser.add_argument("--max_message_size", type=int, default=16 * 1024 * 1024,
                        help="Maximum websocket message size in bytes.")
    # Deploy-time overrides for train_expert_only / train_vlm_only. Default
    # 'auto' respects the checkpoint config.json; 'true'/'false' overrides at
    # inference load time only (ablation/debugging without re-saving the ckpt).
    parser.add_argument(
        "--train_expert_only", choices=["auto", "true", "false"], default="auto",
        help="MED-V5-02: override config.train_expert_only at deploy load time. "
             "'auto' = use the value saved in the checkpoint config.json (default).",
    )
    parser.add_argument(
        "--train_vlm_only", choices=["auto", "true", "false"], default="auto",
        help="MED-V5-02: override config.train_vlm_only at deploy load time. "
             "'auto' = use the value saved in the checkpoint config.json (default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    # Install DedupeFilter so schema / adapter warn_once calls dedupe.
    try:
        from utils.logging_utils import install_dedupe_filter
        install_dedupe_filter()
    except Exception:
        pass

    pretrained_path = Path(args.pretrained_path)
    if not pretrained_path.exists():
        logger.error(f"Pretrained path does not exist: {pretrained_path}")
        sys.exit(1)

    # Auto-restore (and cross-check) data-semantics from the checkpoint's
    # training_state. Detect which flags the operator passed explicitly (from the
    # raw argv tokens, since these defaults aren't None-sentinels) so only
    # untouched defaults are overridden; explicit values are kept but warned on
    # mismatch.
    _explicit_flags = {
        key for key in _DATA_SEMANTICS_ARGS
        if any(tok == f"--{key}" or tok.startswith(f"--{key}=") for tok in sys.argv[1:])
    }
    _reconcile_data_semantics(args, _explicit_flags)

    def _is_loopback_host(host: str) -> bool:
        return host in {"127.0.0.1", "localhost", "::1"}

    if not _is_loopback_host(args.host) and not args.auth_token:
        logger.error(
            "Refusing to serve on non-loopback host %r without --auth_token "
            "or LABVLA_WS_AUTH_TOKEN.",
            args.host,
        )
        sys.exit(2)

    # 'auto' -> None (no override); 'true'/'false' -> bool override.
    def _tristate(v: str) -> bool | None:
        v = v.lower()
        if v == "auto":
            return None
        return v == "true"

    policy = LabVLALabUtopiaPolicy(
        pretrained_path=str(pretrained_path),
        vlm_path=args.vlm_path,
        device=args.device,
        default_prompt=args.default_prompt,
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        num_inference_steps=args.num_inference_steps,
        output_chunk_size=args.output_chunk_size,
        action_mode=None if args.action_mode == "auto" else args.action_mode,
        use_delta_action=(
            None if args.use_delta_action == "auto"
            else args.use_delta_action == "true"
        ),
        norm_stats_path=args.norm_stats_path,
        robot_type=args.robot_type,
        training_repo_id=args.repo_id,
        allow_stats_fallback=args.allow_stats_fallback,
        train_expert_only_override=_tristate(args.train_expert_only),
        train_vlm_only_override=_tristate(args.train_vlm_only),
        normalize_arm_joints=args.normalize_arm_joints,
        normalize_gripper=args.normalize_gripper,
        gripper_norm_mode=args.gripper_norm_mode,
        snap_gripper_to_binary=args.snap_gripper_to_binary,
        gripper_max_width=args.gripper_max_width,
        gripper_canonical_dim=args.gripper_canonical_dim,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"

    logger.info("=" * 70)
    logger.info("LabVLA Deployment Server (3-Camera)")
    logger.info(f"  Checkpoint  : {args.pretrained_path}")
    logger.info(f"  VLM Path    : {args.vlm_path}")
    logger.info(f"  Device      : {args.device}")
    logger.info(f"  Host        : {args.host}:{args.port}")
    logger.info(f"  Hostname    : {hostname} ({local_ip})")
    logger.info(f"  Action dim  : {policy.action_dim}")
    logger.info(f"  Chunk size  : {policy.chunk_size}")
    logger.info(f"  Infer steps : {args.num_inference_steps}")
    logger.info("=" * 70)

    server = WebsocketPolicyServer(
        infer_fn=policy.infer,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
        max_workers=args.max_workers,
        auth_token=args.auth_token,
        max_message_size=args.max_message_size,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
