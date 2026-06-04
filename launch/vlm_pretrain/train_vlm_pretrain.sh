#!/usr/bin/env bash
# Clean VLM pretrain entrypoint. Run this script once per machine with the same
# JOB_NAME/MASTER_ADDR/MASTER_PORT and the correct MACHINE_RANK.

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
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29504}"
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES_VAL:-0,1,2,3,4,5,6,7}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

AGIBOT_TASKS="${AGIBOT_TASKS:-auto}"
AGIBOT_CAP="${AGIBOT_CAP:-20}"
if [ "${AGIBOT_TASKS}" = "auto" ]; then
    AGIBOT_TASKS="$(
        DATA_ROOT="${DATA_ROOT}" AGIBOT_CAP="${AGIBOT_CAP}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
cap = int(os.environ.get("AGIBOT_CAP", "20"))
root = data_root / "agibot_world_beta_lerobot" / "agibotworld"
items = []
if root.is_dir():
    for task in sorted(root.glob("task_*")):
        info = task / "meta" / "info.json"
        if not (task / "data").is_dir() or not info.is_file():
            continue
        try:
            frames = int(json.loads(info.read_text()).get("total_frames", 0))
        except Exception:
            frames = 0
        items.append((frames, task.relative_to(data_root).as_posix()))
items.sort(key=lambda item: (-item[0], item[1]))
print(",".join(path for _, path in items[:cap]))
PY
    )"
fi

if [ -z "${REPO_IDS:-}" ]; then
    REPO_IDS="robointer_droid_clean,oxe-auge_clean_v2,RoboInter-VQA"
    if [ -n "${AGIBOT_TASKS}" ]; then
        REPO_IDS="${REPO_IDS},${AGIBOT_TASKS}"
    fi
fi

if [ -z "${DATASET_SCHEMA:-}" ]; then
    DATASET_SCHEMA="robointer_droid_clean=robointer_droid_anno,oxe-auge_clean_v2=oxe_auge,RoboInter-VQA=robointer_vqa"
    for repo in ${AGIBOT_TASKS//,/ }; do
        [ -n "${repo}" ] && DATASET_SCHEMA="${DATASET_SCHEMA},${repo}=agibot_dual_arm"
    done
fi

if [ -z "${DATASET_WEIGHTS:-}" ] && [ -n "${AGIBOT_TASKS}" ]; then
    N_AGIBOT=$(echo "${AGIBOT_TASKS}" | tr ',' '\n' | grep -c . || true)
    if [ "${N_AGIBOT}" -gt 0 ]; then
        PER_AGIBOT_WEIGHT="$("${PYTHON_BIN}" -c "print(round(0.30 / ${N_AGIBOT}, 6))")"
        DATASET_WEIGHTS="0.20,0.20,0.30"
        for _ in $(seq 1 "${N_AGIBOT}"); do
            DATASET_WEIGHTS="${DATASET_WEIGHTS},${PER_AGIBOT_WEIGHT}"
        done
    fi
fi

AGIBOT_STATS_PATH="${AGIBOT_STATS_PATH:-${DATA_ROOT}/agibot_world_beta_lerobot/_canonical_stats.json}"
EXTERNAL_STATS_MAP="${EXTERNAL_STATS_MAP:-}"
if [ -z "${EXTERNAL_STATS_MAP}" ] && [ -n "${AGIBOT_TASKS}" ] && [ -f "${AGIBOT_STATS_PATH}" ]; then
    for repo in ${AGIBOT_TASKS//,/ }; do
        [ -n "${repo}" ] || continue
        if [ -z "${EXTERNAL_STATS_MAP}" ]; then
            EXTERNAL_STATS_MAP="${repo}=${AGIBOT_STATS_PATH}"
        else
            EXTERNAL_STATS_MAP="${EXTERNAL_STATS_MAP},${repo}=${AGIBOT_STATS_PATH}"
        fi
    done
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
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
PREFETCH_BUFFER="${PREFETCH_BUFFER:-true}"
PREFETCH_BUFFER_SIZE="${PREFETCH_BUFFER_SIZE:-2}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-900}"
HOMOGENEOUS_MIXTURE_BATCHES="${HOMOGENEOUS_MIXTURE_BATCHES:-true}"
TRIM_TOKEN_PADDING_TO_BATCH="${TRIM_TOKEN_PADDING_TO_BATCH:-true}"
SOURCE_SHAPE_CONVERGENCE="${SOURCE_SHAPE_CONVERGENCE:-true}"
DATA_ERROR_SKIP="${DATA_ERROR_SKIP:-true}"
DATA_ERROR_SKIP_MAX_ATTEMPTS="${DATA_ERROR_SKIP_MAX_ATTEMPTS:-64}"
DATA_ERROR_LOG_FIRST="${DATA_ERROR_LOG_FIRST:-20}"
ACTION_SUPERVISION_SKIP_REPOS="${ACTION_SUPERVISION_SKIP_REPOS:-robointer_droid_clean}"
TOTAL_STEPS="${TOTAL_STEPS:-280000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-8}"
LOG_FREQ="${LOG_FREQ:-1}"
SEED="${SEED:-42}"

LR="${LR:-5e-5}"
VLM_LR="${VLM_LR:-5e-5}"
DIT_LR="${DIT_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
DECAY_STEPS="${DECAY_STEPS:-${TOTAL_STEPS}}"
DECAY_LR="${DECAY_LR:-2.5e-6}"

FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
TRAIN_VLM_ONLY="${TRAIN_VLM_ONLY:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
GC_VISUAL_ENCODER="${GC_VISUAL_ENCODER:-false}"
GC_LANGUAGE_MODEL="${GC_LANGUAGE_MODEL:-true}"
GC_DIT="${GC_DIT:-false}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ACTION_MODE="${ACTION_MODE:-abs}"

TRAINING_PHASE="vlm_pretrain"
KNOWLEDGE_ISOLATION="${KNOWLEDGE_ISOLATION:-false}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-true}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-/all-flash-data/Embodied_models/fast}"
KI_MSE_WEIGHT="${KI_MSE_WEIGHT:-10.0}"
DISCRETE_ACTION_VOCAB_SIZE="${DISCRETE_ACTION_VOCAB_SIZE:-2048}"
DISCRETE_ACTION_MAX_LENGTH="${DISCRETE_ACTION_MAX_LENGTH:-224}"
DISCRETIZE_STATE_IN_VLM_PRETRAIN="${DISCRETIZE_STATE_IN_VLM_PRETRAIN:-true}"

RESUME_CKPT="${RESUME_CKPT:-}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
JOB_NAME="${JOB_NAME:-labvla_vlm_pretrain_$(date +%Y%m%d_%H%M%S)}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LABVLA_VQA_SKIP_BAD_RECORDS="${LABVLA_VQA_SKIP_BAD_RECORDS:-true}"
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

OPTIONAL_ARGS=()
[ -n "${DATASET_SCHEMA}" ] && OPTIONAL_ARGS+=(--dataset_schema "${DATASET_SCHEMA}")
[ -n "${DATASET_WEIGHTS:-}" ] && OPTIONAL_ARGS+=(--dataset_weights "${DATASET_WEIGHTS}")
[ -n "${EXTERNAL_STATS_MAP}" ] && OPTIONAL_ARGS+=(--external_stats_map "${EXTERNAL_STATS_MAP}")
OPTIONAL_ARGS+=(--discretize_state_in_vlm_pretrain "${DISCRETIZE_STATE_IN_VLM_PRETRAIN}")
OPTIONAL_ARGS+=(--discrete_action_vocab_size "${DISCRETE_ACTION_VOCAB_SIZE}")
OPTIONAL_ARGS+=(--discrete_action_max_length "${DISCRETE_ACTION_MAX_LENGTH}")

RESUME_ARGS=()
[ -n "${RESUME_CKPT}" ] && RESUME_ARGS+=(--resume "${RESUME_CKPT}")

DS_ARGS=()
if [ -f "${DEEPSPEED_CONFIG}" ]; then
    DS_ARGS=(--use_deepspeed --deepspeed_multinode_launcher standard --deepspeed_config_file "${DEEPSPEED_CONFIG}")
fi

echo "=============================================="
echo "LabVLA final - VLM pretrain"
echo "  Job:             ${JOB_NAME}"
echo "  Project:         ${PROJ_ROOT}"
echo "  Master:          ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Machines/rank:   ${NUM_MACHINES}/${MACHINE_RANK}"
echo "  Processes:       ${NUM_PROCESSES} (${NUM_GPUS} GPUs/node)"
echo "  Repos:           $(echo "${REPO_IDS}" | tr ',' '\n' | wc -l)"
echo "  Batch/GPU:       ${BATCH_SIZE} | workers=${NUM_WORKERS}"
echo "  Phase/action:    ${TRAINING_PHASE}/${ACTION_MODE}"
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
    --homogeneous_mixture_batches "${HOMOGENEOUS_MIXTURE_BATCHES}" \
    --trim_token_padding_to_batch "${TRIM_TOKEN_PADDING_TO_BATCH}" \
    --source_shape_convergence "${SOURCE_SHAPE_CONVERGENCE}" \
    --data_error_skip "${DATA_ERROR_SKIP}" \
    --data_error_skip_max_attempts "${DATA_ERROR_SKIP_MAX_ATTEMPTS}" \
    --data_error_log_first "${DATA_ERROR_LOG_FIRST}" \
    --action_supervision_skip_repos "${ACTION_SUPERVISION_SKIP_REPOS}" \
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
    "${OPTIONAL_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
