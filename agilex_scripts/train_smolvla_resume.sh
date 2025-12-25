#!/usr/bin/env bash
# Resume SmolVLA training from an existing checkpoint (default: last checkpoint of smolvla_agilex_14d).
# Uses GPU 7 by default. Override with: GPU_ID=0 bash agilex_scripts/train_smolvla_resume.sh

set -euo pipefail

# Environment / cache setup (keeps HF caches local to the repo).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-${HF_HUB_OFFLINE}}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

# GPU / tokenizer settings
GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
timestamp="$(date +'%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/train_smolvla_resume_${timestamp}.log"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "ERROR: 'lerobot-train' not found in PATH. Activate your env (e.g., conda activate lerobot_v4)." >&2
  exit 127
fi

# Paths to resume from. Override if you want a different run or step.
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/smolvla_agilex_${ACTION_DIM:-14}d}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-last}"  # e.g., 010000 or last
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${CHECKPOINT_NAME}"
if [[ -L "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="$(readlink -f "${CHECKPOINT_DIR}")"
fi
TRAIN_CONFIG="${TRAIN_CONFIG:-${CHECKPOINT_DIR}/pretrained_model/train_config.json}"
TRAIN_STATE_DIR="${TRAIN_STATE_DIR:-${CHECKPOINT_DIR}/training_state}"

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "ERROR: output_dir not found: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: checkpoint dir not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_CONFIG}" ]]; then
  echo "ERROR: train_config.json not found at: ${TRAIN_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${TRAIN_STATE_DIR}" ]]; then
  echo "ERROR: training_state dir missing (optimizer/lr scheduler states needed): ${TRAIN_STATE_DIR}" >&2
  exit 1
fi

# Extract dataset root and VLM path from the saved config for quick validation.
DATASET_ROOT="$(python - <<'PY' "${TRAIN_CONFIG}" || exit 1
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["dataset"]["root"])
PY
)"
VLM_MODEL_PATH="$(python - <<'PY' "${TRAIN_CONFIG}" || exit 1
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["policy"].get("vlm_model_name",""))
PY
)"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root not found: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -n "${VLM_MODEL_PATH}" && ! -d "${VLM_MODEL_PATH}" ]]; then
  echo "ERROR: SmolVLM2 backbone path from config is missing: ${VLM_MODEL_PATH}" >&2
  exit 1
fi

START_STEP="$(python - <<'PY' "${TRAIN_STATE_DIR}/training_step.json" 2>/dev/null || true
import json,sys
print(json.load(open(sys.argv[1]))["step"])
PY
)"
echo "[resume] output_dir=${OUTPUT_DIR}" >&2
echo "[resume] checkpoint_dir=${CHECKPOINT_DIR} (start_step=${START_STEP:-unknown})" >&2
echo "[resume] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2

# Optional overrides
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"   # keep as 'cuda' because CUDA_VISIBLE_DEVICES masks to a single GPU
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-10000}"

# Build CLI (you can still append extra overrides at the end).
args=(
  --resume true
  --config_path="${TRAIN_CONFIG}"  # parser expects '=' style for config_path
  --output_dir "${OUTPUT_DIR}"
  --policy.device "${POLICY_DEVICE}"
  --log_freq "${LOG_FREQ}"
  --save_freq "${SAVE_FREQ}"
)

if [[ -n "${STEPS:-}" ]]; then
  args+=(--steps "${STEPS}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  args+=(--batch_size "${BATCH_SIZE}")
fi

lerobot-train "${args[@]}" "$@" 2>&1 | tee "${log_file}"
