#!/usr/bin/env bash
# Clean LabUtopia Level3 finetune entrypoint. PRETRAINED_CKPT is required so
# this script cannot silently continue from an old bad finetune checkpoint.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-labvla}"
source /data/rbc/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-/all-flash-data/vlm/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJ_ROOT}/configs/deepspeed_zero2.json}"
DATA_ROOT="${DATA_ROOT:-/all-flash-data/lerobot/.cache}"

TASK="${TASK:-TransportBeaker}"
case "${TASK}" in
    HeatLiquid|heat|heatliquid|Level3_HeatLiquid)
        REPO_IDS="${REPO_IDS:-LabUtopia/Level3_HeatLiquid}"
        DATASET_SCHEMA="${DATASET_SCHEMA:-labutopia_level3_heat_liquid_v3}"
        TASK_SLUG="heatliquid"
        ;;
    PourLiquid|pour|pourliquid|Level3_PourLiquid)
        REPO_IDS="${REPO_IDS:-LabUtopia/Level3_PourLiquid}"
        DATASET_SCHEMA="${DATASET_SCHEMA:-labutopia_level3_pour_liquid_v3}"
        TASK_SLUG="pourliquid"
        ;;
    TransportBeaker|transport|transportbeaker|tb|Level3_TransportBeaker)
        REPO_IDS="${REPO_IDS:-LabUtopia/Level3_TransportBeaker}"
        DATASET_SCHEMA="${DATASET_SCHEMA:-labutopia_level3_transportbeaker_v3}"
        TASK_SLUG="transportbeaker"
        ;;
    open|Open|Level3_open)
        REPO_IDS="${REPO_IDS:-LabUtopia/Level3_open}"
        DATASET_SCHEMA="${DATASET_SCHEMA:-labutopia_level3_open_v3}"
        TASK_SLUG="open"
        ;;
    pick|Pick|Level3_pick)
        REPO_IDS="${REPO_IDS:-LabUtopia/Level3_pick}"
        DATASET_SCHEMA="${DATASET_SCHEMA:-labutopia_level3_pick_v3}"
        TASK_SLUG="pick"
        ;;
    press|Press|Level3_press)
        REPO_IDS="${REPO_IDS:-LabUtopia/Level3_press}"
        DATASET_SCHEMA="${DATASET_SCHEMA:-labutopia_level3_press_v3}"
        TASK_SLUG="press"
        ;;
    *)
        echo "[ERROR] unknown TASK=${TASK}" >&2
        exit 1
        ;;
esac

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
if [ -z "${PRETRAINED_CKPT}" ]; then
    echo "[ERROR] PRETRAINED_CKPT is required." >&2
    echo "Example: TASK=HeatLiquid PRETRAINED_CKPT=/path/to/checkpoint-50000 bash $0" >&2
    exit 1
fi
if [ ! -d "${PRETRAINED_CKPT}" ]; then
    echo "[ERROR] PRETRAINED_CKPT is not a directory: ${PRETRAINED_CKPT}" >&2
    exit 1
fi

EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${DATA_ROOT}/${REPO_IDS}/meta/stats_canonical_grip.json}"
if [ ! -d "${DATA_ROOT}/${REPO_IDS}" ]; then
    echo "[ERROR] dataset directory missing: ${DATA_ROOT}/${REPO_IDS}" >&2
    exit 1
fi
if [ ! -f "${EXTERNAL_STATS_PATH}" ]; then
    echo "[ERROR] stats file missing: ${EXTERNAL_STATS_PATH}" >&2
    exit 1
fi

NUM_GPUS="${NUM_GPUS:-7}"
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES_VAL:-1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29652}"

# DTYPE sets the model compute dtype (--dtype) only; accelerate mixed precision
# is hardcoded to bf16 below, so changing DTYPE does NOT switch off bf16.
DTYPE="${DTYPE:-bfloat16}"
DIT_NUM_LAYERS="${DIT_NUM_LAYERS:-18}"
DIT_NUM_HEADS="${DIT_NUM_HEADS:-8}"
DIT_HEAD_DIM="${DIT_HEAD_DIM:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
MAX_STATE_DIM="${MAX_STATE_DIM:-32}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-32}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-224}"
IMAGE_WIDTH="${IMAGE_WIDTH:-224}"

BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_BUFFER="${PREFETCH_BUFFER:-true}"
PREFETCH_BUFFER_SIZE="${PREFETCH_BUFFER_SIZE:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-8}"
LOG_FREQ="${LOG_FREQ:-50}"
SEED="${SEED:-42}"

LR="${LR:-5e-5}"
VLM_LR="${VLM_LR:-5e-5}"
DIT_LR="${DIT_LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DECAY_STEPS="${DECAY_STEPS:-${TOTAL_STEPS}}"
DECAY_LR="${DECAY_LR:-5e-5}"

FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
TRAIN_VLM_ONLY="${TRAIN_VLM_ONLY:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
GC_VISUAL_ENCODER="${GC_VISUAL_ENCODER:-true}"
GC_LANGUAGE_MODEL="${GC_LANGUAGE_MODEL:-false}"
GC_DIT="${GC_DIT:-false}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

TRAINING_PHASE="posttrain"
KNOWLEDGE_ISOLATION="${KNOWLEDGE_ISOLATION:-false}"
USE_FAST_TOKENIZER="${USE_FAST_TOKENIZER:-false}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-/all-flash-data/Embodied_models/fast}"
KI_MSE_WEIGHT="${KI_MSE_WEIGHT:-10.0}"

ACTION_MODE="${ACTION_MODE:-delta}"
NORMALIZE_ARM_JOINTS="${NORMALIZE_ARM_JOINTS:-true}"
NORMALIZE_GRIPPER="${NORMALIZE_GRIPPER:-true}"
GRIPPER_NORM_MODE="${GRIPPER_NORM_MODE:-q01_q99}"
SNAP_GRIPPER_TO_BINARY="${SNAP_GRIPPER_TO_BINARY:-false}"
GRIPPER_MAX_WIDTH="${GRIPPER_MAX_WIDTH:-0.04}"
GRIPPER_CANONICAL_DIM="${GRIPPER_CANONICAL_DIM:-7}"

# LabUtopia finetuning uses mixed action semantics: arm joints are delta,
# while the gripper channel stays absolute through schema.delta_mask.
if [ "${ACTION_MODE}" != "delta" ]; then
    echo "[ERROR] LabUtopia finetune requires ACTION_MODE=delta." >&2
    echo "        Arm joints must be delta; gripper remains absolute via schema.delta_mask." >&2
    echo "        Got ACTION_MODE=${ACTION_MODE}" >&2
    exit 1
fi

WANDB_ENABLE="${WANDB_ENABLE:-false}"
JOB_NAME="${JOB_NAME:-labvla_finetune_${TASK_SLUG}_$(date +%Y%m%d_%H%M%S)}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"
export TORCH_CUDNN_V8_API_ENABLED="${TORCH_CUDNN_V8_API_ENABLED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"

DS_ARGS=()
if [ -f "${DEEPSPEED_CONFIG}" ]; then
    DS_ARGS=(--use_deepspeed --deepspeed_config_file "${DEEPSPEED_CONFIG}")
fi

echo "=============================================="
echo "LabVLA final - LabUtopia Level3 finetune"
echo "  Job:             ${JOB_NAME}"
echo "  Project:         ${PROJ_ROOT}"
echo "  Task/repo:       ${TASK} / ${REPO_IDS}"
echo "  Schema:          ${DATASET_SCHEMA}"
echo "  CUDA devices:    ${CUDA_VISIBLE_DEVICES_VAL}"
echo "  Batch/GPU:       ${BATCH_SIZE} | workers=${NUM_WORKERS}"
echo "  Steps/save:      ${TOTAL_STEPS}/${SAVE_FREQ}"
echo "  LR recipe:       lr=${LR} vlm=${VLM_LR} dit=${DIT_LR} warmup=${WARMUP_STEPS} decay=${DECAY_LR}"
echo "  Prefetch:        buffer=${PREFETCH_BUFFER} size=${PREFETCH_BUFFER_SIZE}"
echo "  GC:              visual=${GC_VISUAL_ENCODER} lm=${GC_LANGUAGE_MODEL} dit=${GC_DIT}"
echo "  Action mode:     ${ACTION_MODE}"
echo "  Freeze vision:   ${FREEZE_VISION_ENCODER}"
echo "  Warmstart ckpt:  ${PRETRAINED_CKPT}"
echo "=============================================="

cd "${PROJ_ROOT}"

# Single-machine, bf16-mixed-precision only: no MACHINE_RANK/MASTER_ADDR plumbing,
# so --num_machines is hardcoded to 1. For non-bf16 or multi-node, use another entry.
exec accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
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
    --prefetch_buffer "${PREFETCH_BUFFER}" \
    --prefetch_buffer_size "${PREFETCH_BUFFER_SIZE}" \
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
    --gripper_norm_mode "${GRIPPER_NORM_MODE}" \
    --snap_gripper_to_binary "${SNAP_GRIPPER_TO_BINARY}" \
    --gripper_max_width "${GRIPPER_MAX_WIDTH}" \
    --gripper_canonical_dim "${GRIPPER_CANONICAL_DIM}" \
    --training_phase "${TRAINING_PHASE}" \
    --knowledge_isolation "${KNOWLEDGE_ISOLATION}" \
    --use_fast_tokenizer "${USE_FAST_TOKENIZER}" \
    --fast_tokenizer_path "${FAST_TOKENIZER_PATH}" \
    --ki_mse_weight "${KI_MSE_WEIGHT}" \
    --repo_ids "${REPO_IDS}" \
    --data_root "${DATA_ROOT}" \
    --dataset_schema "${DATASET_SCHEMA}" \
    --external_stats_path "${EXTERNAL_STATS_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --job_name "${JOB_NAME}" \
    --wandb_enable "${WANDB_ENABLE}" \
    --resume "${PRETRAINED_CKPT}" \
    --load_weights_only true
