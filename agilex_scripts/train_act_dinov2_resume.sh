#!/usr/bin/env bash
# Resume ACT + DINOv2 training from a saved checkpoint.

set -euo pipefail

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
log_file="${LOG_DIR}/train_act_dinov2_resume_${timestamp}.log"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "ERROR: 'lerobot-train' not found in PATH. Try: conda activate lerobot_v4" >&2
  exit 127
fi

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${ROOT_DIR}/outputs/train}"
TRAIN_RUN="${TRAIN_RUN:-act_dinov2_agilex}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}"
if [[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$(printf "%06d" "$((10#${CHECKPOINT_STEP}))")"
fi

CHECKPOINT_ROOT="${TRAIN_OUTPUT_ROOT}/${TRAIN_RUN}/checkpoints/${CHECKPOINT_STEP}/pretrained_model"
CONFIG_PATH="${CONFIG_PATH:-${CHECKPOINT_ROOT}/train_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_ROOT}/${TRAIN_RUN}}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: train_config.json not found at ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[train_act_dinov2_resume] config_path=${CONFIG_PATH}" >&2
echo "[train_act_dinov2_resume] output_dir=${OUTPUT_DIR}" >&2
echo "[train_act_dinov2_resume] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2

lerobot-train \
  --resume=true \
  --config_path="${CONFIG_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  "$@" 2>&1 | tee "${log_file}"
