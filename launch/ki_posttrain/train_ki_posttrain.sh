#!/usr/bin/env bash
# Clean KI posttrain entrypoint. Defaults to OXE + all LabEmbodied beta repos,
# including converted Level3 tasks when present. No resume is used by default.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-labvla}"
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-/all-flash-data/vlm/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJ_ROOT}/configs/deepspeed_zero2.json}"
DATA_ROOT="${DATA_ROOT:-/all-flash-data/Pretrain_Data}"
PYTHON_BIN="${PYTHON_BIN:-/data/rbc/miniconda3/envs/${CONDA_ENV}/bin/python}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_GPUS * NUM_MACHINES))}"
MASTER_ADDR="${MASTER_ADDR:-10.1.23.9}"
MASTER_PORT="${MASTER_PORT:-29653}"
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES_VAL:-0,1,2,3,4,5,6,7}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

if [ -z "${REPO_IDS:-}" ]; then
    LABEMBODIED_REPOS="$(
        DATA_ROOT="${DATA_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["DATA_ROOT"]) / "LabEmbodied_data_beta"
repos = []
if root.is_dir():
    for child in sorted(root.iterdir()):
        if (child / "meta" / "info.json").is_file():
            repos.append("LabEmbodied_data_beta/" + child.name)
print(",".join(repos))
PY
    )"
    REPO_IDS="oxe-auge_clean_v2"
    if [ -n "${LABEMBODIED_REPOS}" ]; then
        REPO_IDS="${REPO_IDS},${LABEMBODIED_REPOS}"
    fi
fi

DTYPE="${DTYPE:-bfloat16}"
DIT_NUM_LAYERS="${DIT_NUM_LAYERS:-18}"
DIT_NUM_HEADS="${DIT_NUM_HEADS:-8}"
DIT_HEAD_DIM="${DIT_HEAD_DIM:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
MAX_STATE_DIM="${MAX_STATE_DIM:-32}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-32}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-224}"
IMAGE_WIDTH="${IMAGE_WIDTH:-224}"

BATCH_SIZE="${BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
PREFETCH_BUFFER="${PREFETCH_BUFFER:-true}"
PREFETCH_BUFFER_SIZE="${PREFETCH_BUFFER_SIZE:-2}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-3600}"
TRIM_TOKEN_PADDING_TO_BATCH="${TRIM_TOKEN_PADDING_TO_BATCH:-true}"
SOURCE_SHAPE_CONVERGENCE="${SOURCE_SHAPE_CONVERGENCE:-true}"
DATA_ERROR_SKIP="${DATA_ERROR_SKIP:-true}"
DATA_ERROR_SKIP_MAX_ATTEMPTS="${DATA_ERROR_SKIP_MAX_ATTEMPTS:-64}"
DATA_ERROR_LOG_FIRST="${DATA_ERROR_LOG_FIRST:-20}"
TOTAL_STEPS="${TOTAL_STEPS:-280000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-8}"
LOG_FREQ="${LOG_FREQ:-1}"
SEED="${SEED:-42}"

LR="${LR:-1e-5}"
VLM_LR="${VLM_LR:-1e-5}"
DIT_LR="${DIT_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-8000}"
DECAY_STEPS="${DECAY_STEPS:-${TOTAL_STEPS}}"
DECAY_LR="${DECAY_LR:-5e-7}"

FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
TRAIN_VLM_ONLY="${TRAIN_VLM_ONLY:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
GC_VISUAL_ENCODER="${GC_VISUAL_ENCODER:-false}"
GC_LANGUAGE_MODEL="${GC_LANGUAGE_MODEL:-true}"
GC_DIT="${GC_DIT:-false}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ACTION_MODE="${ACTION_MODE:-delta}"
NORMALIZE_ARM_JOINTS="${NORMALIZE_ARM_JOINTS:-true}"
NORMALIZE_GRIPPER="${NORMALIZE_GRIPPER:-true}"

TRAINING_PHASE="posttrain"
KNOWLEDGE_ISOLATION="${KNOWLEDGE_ISOLATION:-true}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-true}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-/all-flash-data/Embodied_models/fast}"
KI_MSE_WEIGHT="${KI_MSE_WEIGHT:-10.0}"
DISCRETE_ACTION_VOCAB_SIZE="${DISCRETE_ACTION_VOCAB_SIZE:-2048}"
DISCRETE_ACTION_MAX_LENGTH="${DISCRETE_ACTION_MAX_LENGTH:-192}"
RESUME_CKPT="${RESUME_CKPT:-}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
JOB_NAME="${JOB_NAME:-labvla_ki_posttrain_$(date +%Y%m%d_%H%M%S)}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Keep the adapter fail-loud: overwide-dim truncation and tokenized-task string
# coercion are silent corruption paths. Hardcode safe values so `bash <script>`
# cannot inherit a dangerous env default; opt in per-repo only when known-safe.
export LABVLA_ALLOW_TOKENIZED_TASK_COERCION="0"
export LABVLA_ALLOW_TRUNCATE="0"
export LABVLA_V21_VALIDATE_PER_FILE="${LABVLA_V21_VALIDATE_PER_FILE:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_NCCL_DESYNC_DEBUG="${TORCH_NCCL_DESYNC_DEBUG:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
export LABVLA_STORAGE_RETRY_ENABLE="${LABVLA_STORAGE_RETRY_ENABLE:-1}"
export LABVLA_STORAGE_RETRY_TOTAL_SECONDS="${LABVLA_STORAGE_RETRY_TOTAL_SECONDS:-1800}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
export LABVLA_VIDEO_CACHE_MAX="${LABVLA_VIDEO_CACHE_MAX:-256}"
export LABVLA_WORKER_TRIM_EVERY="${LABVLA_WORKER_TRIM_EVERY:-5000}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-4}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"
export TORCH_CUDNN_V8_API_ENABLED="${TORCH_CUDNN_V8_API_ENABLED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

DS_ARGS=()
if [ -f "${DEEPSPEED_CONFIG}" ]; then
    DS_ARGS=(--use_deepspeed --deepspeed_multinode_launcher standard --deepspeed_config_file "${DEEPSPEED_CONFIG}")
fi
RESUME_ARGS=()
[ -n "${RESUME_CKPT}" ] && RESUME_ARGS+=(--resume "${RESUME_CKPT}" --load_weights_only true)

echo "=============================================="
echo "LabVLA final - KI posttrain"
echo "  Job:             ${JOB_NAME}"
echo "  Project:         ${PROJ_ROOT}"
echo "  Master:          ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Machines/rank:   ${NUM_MACHINES}/${MACHINE_RANK}"
echo "  Processes:       ${NUM_PROCESSES} (${NUM_GPUS} GPUs/node)"
echo "  Repos:           $(echo "${REPO_IDS}" | tr ',' '\n' | wc -l)"
echo "  Batch/GPU:       ${BATCH_SIZE} | workers=${NUM_WORKERS}"
echo "  Phase/action:    ${TRAINING_PHASE}/${ACTION_MODE}"
echo "  Resume:          ${RESUME_CKPT:-NONE}"
echo "=============================================="

cd "${PROJ_ROOT}"

exec accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision bf16 \
    "${DS_ARGS[@]}" \
    scripts/train.py \
    --vlm_pretrained_path "${VLM_PRETRAINED_PATH}" \
    --dtype "${DTYPE}" \
    --dit_num_layers "${DIT_NUM_LAYERS}" \
    --dit_num_heads "${DIT_NUM_HEADS}" \
    --dit_head_dim "${DIT_HEAD_DIM}" \
    --chunk_size "${CHUNK_SIZE}" \
    --max_state_dim "${MAX_STATE_DIM}" \
    --max_action_dim "${MAX_ACTION_DIM}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
    --prefetch_buffer "${PREFETCH_BUFFER}" \
    --prefetch_buffer_size "${PREFETCH_BUFFER_SIZE}" \
    --dist_timeout_seconds "${DIST_TIMEOUT_SECONDS}" \
    --trim_token_padding_to_batch "${TRIM_TOKEN_PADDING_TO_BATCH}" \
    --source_shape_convergence "${SOURCE_SHAPE_CONVERGENCE}" \
    --data_error_skip "${DATA_ERROR_SKIP}" \
    --data_error_skip_max_attempts "${DATA_ERROR_SKIP_MAX_ATTEMPTS}" \
    --data_error_log_first "${DATA_ERROR_LOG_FIRST}" \
    --steps "${TOTAL_STEPS}" \
    --save_freq "${SAVE_FREQ}" \
    --max_keep_ckpts "${MAX_KEEP_CKPTS}" \
    --log_freq "${LOG_FREQ}" \
    --seed "${SEED}" \
    --image_height "${IMAGE_HEIGHT}" \
    --image_width "${IMAGE_WIDTH}" \
    --lr "${LR}" \
    --vlm_lr "${VLM_LR}" \
    --dit_lr "${DIT_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip_norm "${GRAD_CLIP_NORM}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --decay_steps "${DECAY_STEPS}" \
    --decay_lr "${DECAY_LR}" \
    --freeze_vision_encoder "${FREEZE_VISION_ENCODER}" \
    --train_expert_only "${TRAIN_EXPERT_ONLY}" \
    --train_vlm_only "${TRAIN_VLM_ONLY}" \
    --compile_model "${COMPILE_MODEL}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --gc_visual_encoder "${GC_VISUAL_ENCODER}" \
    --gc_language_model "${GC_LANGUAGE_MODEL}" \
    --gc_dit "${GC_DIT}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --action_mode "${ACTION_MODE}" \
    --normalize_arm_joints "${NORMALIZE_ARM_JOINTS}" \
    --normalize_gripper "${NORMALIZE_GRIPPER}" \
    --training_phase "${TRAINING_PHASE}" \
    --knowledge_isolation "${KNOWLEDGE_ISOLATION}" \
    --use_fast_tokenizer "${USE_FAST_TOKENIZER}" \
    --fast_tokenizer_path "${FAST_TOKENIZER_PATH}" \
    --ki_mse_weight "${KI_MSE_WEIGHT}" \
    --repo_ids "${REPO_IDS}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --job_name "${JOB_NAME}" \
    --wandb_enable "${WANDB_ENABLE}" \
    --discrete_action_vocab_size "${DISCRETE_ACTION_VOCAB_SIZE}" \
    --discrete_action_max_length "${DISCRETE_ACTION_MAX_LENGTH}" \
    "${RESUME_ARGS[@]}"
