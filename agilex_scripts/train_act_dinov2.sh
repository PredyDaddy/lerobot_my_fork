#!/usr/bin/env bash
# Train ACT + DINOv2 on the AgileX dataset using lerobot-train.

set -euo pipefail

SMOKE_TEST=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE_TEST=1
  shift
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-${HF_HUB_OFFLINE}}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

timestamp="$(date +'%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/train_act_dinov2_${timestamp}.log"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "ERROR: 'lerobot-train' not found in PATH. Try: conda activate lerobot_v4" >&2
  exit 127
fi

DATASET_REPO_ID="${DATASET_REPO_ID:-agilex_dataset1}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/agilex_dataset1}"
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root not found at ${DATASET_ROOT}" >&2
  exit 1
fi
for meta_file in "${DATASET_ROOT}/meta/info.json" "${DATASET_ROOT}/meta/stats.json"; do
  if [[ ! -f "${meta_file}" ]]; then
    echo "ERROR: missing dataset metadata ${meta_file}" >&2
    exit 1
  fi
done

DINOV2_PATH="${DINOV2_PATH:-${ROOT_DIR}/dinov2_base}"
if [[ ! -d "${DINOV2_PATH}" ]]; then
  echo "ERROR: DINOv2 directory not found at ${DINOV2_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DINOV2_PATH}/config.json" ]]; then
  echo "ERROR: ${DINOV2_PATH}/config.json not found (DINOv2 config missing)" >&2
  exit 1
fi
if ! compgen -G "${DINOV2_PATH}/*.safetensors" >/dev/null && \
  ! compgen -G "${DINOV2_PATH}/*.bin" >/dev/null; then
  echo "ERROR: DINOv2 weights (*.safetensors or *.bin) not found in ${DINOV2_PATH}" >&2
  exit 1
fi

if [[ "${SMOKE_TEST}" == "1" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
  STEPS="${STEPS:-1}"
  SAVE_FREQ="${SAVE_FREQ:-1}"
  EVAL_FREQ="${EVAL_FREQ:-1}"
  LOG_FREQ="${LOG_FREQ:-1}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/_smoke_act_dinov2}"
  JOB_NAME="${JOB_NAME:-smoke_act_dinov2}"
else
  BATCH_SIZE="${BATCH_SIZE:-32}"
  STEPS="${STEPS:-100000}"
  SAVE_FREQ="${SAVE_FREQ:-10000}"
  EVAL_FREQ="${EVAL_FREQ:-10000}"
  LOG_FREQ="${LOG_FREQ:-100}"
  NUM_WORKERS="${NUM_WORKERS:-4}"
  POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/act_dinov2_agilex}"
  JOB_NAME="${JOB_NAME:-act_dinov2_agilex}"
fi

DINOV2_IMAGE_SIZE="${DINOV2_IMAGE_SIZE:-224}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"

echo "[train_act_dinov2] dataset_root=${DATASET_ROOT}" >&2
echo "[train_act_dinov2] dinov2_path=${DINOV2_PATH}" >&2
echo "[train_act_dinov2] output_dir=${OUTPUT_DIR}" >&2
echo "[train_act_dinov2] policy_device=${POLICY_DEVICE} batch_size=${BATCH_SIZE} steps=${STEPS}" >&2

lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.use_imagenet_stats=true \
  --policy.type=act_dinov2 \
  --policy.device="${POLICY_DEVICE}" \
  --policy.push_to_hub=false \
  --policy.dinov2_model_name_or_path="${DINOV2_PATH}" \
  --policy.dinov2_local_files_only=true \
  --policy.dinov2_image_size="${DINOV2_IMAGE_SIZE}" \
  --policy.dinov2_output_mode=grid \
  --policy.freeze_backbone="${FREEZE_BACKBONE}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq="${EVAL_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --wandb.enable=false \
  "$@" 2>&1 | tee "${log_file}"
