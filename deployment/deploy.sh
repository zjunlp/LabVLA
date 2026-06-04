#!/bin/bash
# ============================================================================
# LabVLA Deployment Script
# Deploy LabVLA model for LabUtopia remote inference.
#
# Usage:
#   bash deploy.sh                                          # default config
#   PORT=8001 CUDA_VISIBLE_DEVICES=1 bash deploy.sh        # custom GPU & port
#   DEFAULT_PROMPT="Press the button" bash deploy.sh       # with a default prompt
#   PRETRAINED_PATH=/path/to/ckpt bash deploy.sh           # custom checkpoint
# ============================================================================

set -e

# ---- Paths ----
# Derive repo root from this script's location (deployment/ sits at the repo
# root), matching serve_labvla.py's Path(__file__).parents[1]. Overridable via env.
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
LABVLA_ROOT="${LABVLA_ROOT:-$(cd "${DEPLOY_DIR}/.." && pwd)}"
VLM_PATH="/all-flash-data/vlm/Qwen3-VL-4B-Instruct"
# Default checkpoint is resolved relative to the derived repo root so the
# default works in whatever clone this script ships in.
PRETRAINED_PATH="${PRETRAINED_PATH:-${LABVLA_ROOT}/outputs/labvla_finetune_Level3_TransportBeaker_delta_fullparam_20260427_233322/checkpoint-5000}"

# ---- Server config ----
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31002}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
LABVLA_WS_AUTH_TOKEN="${LABVLA_WS_AUTH_TOKEN:-${AUTH_TOKEN:-}}"
MAX_MESSAGE_SIZE="${MAX_MESSAGE_SIZE:-16777216}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# ---- Model config ----
# Don't hardcode action_dim: serve_labvla.py auto-derives it from the checkpoint
# schema (and fail-louds on mismatch). Only forward --action_dim if ACTION_DIM is
# set explicitly, else a non-8-dim (e.g. dual-arm 14) ckpt would be truncated.
ACTION_DIM="${ACTION_DIM:-}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
OUTPUT_CHUNK_SIZE="${OUTPUT_CHUNK_SIZE:-}"
NUM_INFERENCE_STEPS=10
ACTION_MODE="${ACTION_MODE:-auto}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
# norm_stats are now auto-loaded from the checkpoint directory (norm_stats.json).
# Set NORM_STATS_PATH only if you need to override with a specific stats file.
NORM_STATS_PATH="${NORM_STATS_PATH:-}"
# Required for multi-repo checkpoints: selects the correct schema + norm_stats
# slice from the bundle. Single-repo checkpoints can leave this empty.
TRAINING_REPO_ID="${TRAINING_REPO_ID:-}"

# Per-segment normalization toggles. Default true → mixed-norm (arm=mean_std,
# grip=q01_q99). Set false when serving a ckpt trained with that segment OFF.
NORMALIZE_ARM_JOINTS="${NORMALIZE_ARM_JOINTS:-true}"
NORMALIZE_GRIPPER="${NORMALIZE_GRIPPER:-true}"
GRIPPER_NORM_MODE="${GRIPPER_NORM_MODE:-q01_q99}"

# 2026-04-30 gripper canonicalize. Set true when serving a ckpt trained with
# --snap_gripper_to_binary=true (e.g. labvla_finetune_transportbeaker_canonical).
# Default false preserves prior behaviour for legacy ckpts.
SNAP_GRIPPER_TO_BINARY="${SNAP_GRIPPER_TO_BINARY:-false}"
GRIPPER_MAX_WIDTH="${GRIPPER_MAX_WIDTH:-0.04}"
GRIPPER_CANONICAL_DIM="${GRIPPER_CANONICAL_DIM:-7}"

# ---- Conda environment ----
CONDA_ENV="${CONDA_ENV:-labvla}"

# ============================================================================

echo "============================================================"
echo " LabVLA Deployment Server"
echo "============================================================"
echo "  Checkpoint : ${PRETRAINED_PATH}"
echo "  VLM        : ${VLM_PATH}"
echo "  LABVLA_ROOT: ${LABVLA_ROOT}"
echo "  Device     : ${DEVICE} (CUDA_VISIBLE_DEVICES=${CUDA_DEVICE})"
echo "  Port       : ${PORT}"
echo "  Workers    : ${MAX_WORKERS}"
echo "  Action mode: ${ACTION_MODE}"
echo "  Output chunk: ${OUTPUT_CHUNK_SIZE:-${CHUNK_SIZE}}"
echo "  Conda env  : ${CONDA_ENV}"
echo "============================================================"

# Validate paths
if [ ! -d "${PRETRAINED_PATH}" ]; then
    echo "[ERROR] Checkpoint not found: ${PRETRAINED_PATH}"
    exit 1
fi

# locate model.safetensors
if [ -f "${PRETRAINED_PATH}/model.safetensors" ]; then
    MODEL_PATH="${PRETRAINED_PATH}"
elif [ -f "${PRETRAINED_PATH}/pretrained_model/model.safetensors" ]; then
    MODEL_PATH="${PRETRAINED_PATH}/pretrained_model"
else
    echo "[ERROR] model.safetensors not found in ${PRETRAINED_PATH} or ${PRETRAINED_PATH}/pretrained_model"
    exit 1
fi
echo "  Model file : ${MODEL_PATH}/model.safetensors"

if [ ! -d "${VLM_PATH}" ]; then
    echo "[ERROR] VLM path not found: ${VLM_PATH}"
    exit 1
fi

if [ ! -d "${LABVLA_ROOT}/src" ]; then
    echo "[ERROR] LABVLA_ROOT/src not found: ${LABVLA_ROOT}/src"
    exit 1
fi

case "${HOST}" in
    127.0.0.1|localhost|::1) ;;
    *)
        if [ -z "${LABVLA_WS_AUTH_TOKEN}" ]; then
            echo "[ERROR] Refusing HOST=${HOST} without LABVLA_WS_AUTH_TOKEN/AUTH_TOKEN."
            echo "        Use HOST=127.0.0.1 for local-only serving or set an auth token."
            exit 2
        fi
        ;;
esac

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "[INFO] Python: $(which python)"
echo "[INFO] PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# Set environment
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export LABVLA_ROOT="${LABVLA_ROOT}"

cmd=(
    python "${DEPLOY_DIR}/serve_labvla.py"
    --pretrained_path "${PRETRAINED_PATH}"
    --vlm_path "${VLM_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
    --chunk_size "${CHUNK_SIZE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --action_mode "${ACTION_MODE}"
    --normalize_arm_joints "${NORMALIZE_ARM_JOINTS}"
    --normalize_gripper "${NORMALIZE_GRIPPER}"
    --gripper_norm_mode "${GRIPPER_NORM_MODE}"
    --snap_gripper_to_binary "${SNAP_GRIPPER_TO_BINARY}"
    --gripper_max_width "${GRIPPER_MAX_WIDTH}"
    --gripper_canonical_dim "${GRIPPER_CANONICAL_DIM}"
    --max_message_size "${MAX_MESSAGE_SIZE}"
    --max_workers "${MAX_WORKERS}"
)

if [ -n "${ACTION_DIM}" ]; then
    cmd+=(--action_dim "${ACTION_DIM}")
fi

if [ -n "${OUTPUT_CHUNK_SIZE}" ]; then
    cmd+=(--output_chunk_size "${OUTPUT_CHUNK_SIZE}")
fi

if [ -n "${NORM_STATS_PATH}" ]; then
    cmd+=(--norm_stats_path "${NORM_STATS_PATH}")
fi

if [ -n "${TRAINING_REPO_ID}" ]; then
    cmd+=(--repo_id "${TRAINING_REPO_ID}")
fi

if [ -n "${DEFAULT_PROMPT}" ]; then
    cmd+=(--default_prompt "${DEFAULT_PROMPT}")
fi

if [ -n "${LABVLA_WS_AUTH_TOKEN}" ]; then
    cmd+=(--auth_token "${LABVLA_WS_AUTH_TOKEN}")
fi

echo "[INFO] Starting LabVLA server on ${HOST}:${PORT} ..."
"${cmd[@]}"
