#!/usr/bin/env bash
# Run a trained SmolVLA policy on the SO-101 follower arm using `lerobot-record`.
# This records an eval dataset while the policy controls the robot.

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

# Use GPU 0 by default.
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===== Paths (override these on the inference machine if needed) =====
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-${REPO_ROOT}/SmolVLM2-500M-Video-Instruct}"

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${REPO_ROOT}/outputs/train}"
TRAIN_RUN="${TRAIN_RUN:-smolvla_so101_6d}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}" # "last" or an integer like 30000
if [[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$(printf "%06d" "$((10#${CHECKPOINT_STEP}))")"
fi

# You can also directly set POLICY_DIR to a copied `pretrained_model` folder on another machine.
POLICY_DIR="${POLICY_DIR:-${TRAIN_OUTPUT_ROOT}/${TRAIN_RUN}/checkpoints/${CHECKPOINT_STEP}/pretrained_model}"

EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

# Avoid permission issues by keeping HF caches in the repo (override if needed).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

mkdir -p "${LOG_DIR}"
timestamp="$(date +'%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/infer_smolvla_so101_${TRAIN_RUN}_${CHECKPOINT_STEP}_${timestamp}.log"

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "ERROR: 'lerobot-record' not found in PATH." >&2
  echo "Activate your environment first, e.g.: conda activate lerobot_v4" >&2
  exit 127
fi

if [[ ! -d "${POLICY_DIR}" ]]; then
  echo "ERROR: policy directory not found: ${POLICY_DIR}" >&2
  exit 1
fi
if [[ ! -f "${POLICY_DIR}/config.json" || ! -f "${POLICY_DIR}/model.safetensors" ]]; then
  echo "ERROR: missing required files in policy dir: ${POLICY_DIR}" >&2
  echo "Expected at least: config.json + model.safetensors" >&2
  exit 1
fi
if [[ ! -f "${POLICY_DIR}/policy_preprocessor.json" || ! -f "${POLICY_DIR}/policy_postprocessor.json" ]]; then
  echo "ERROR: missing processor configs in policy dir: ${POLICY_DIR}" >&2
  echo "Expected: policy_preprocessor.json + policy_postprocessor.json" >&2
  exit 1
fi
if ! compgen -G "${POLICY_DIR}/policy_preprocessor_step_*_normalizer_processor.safetensors" >/dev/null; then
  echo "ERROR: missing normalizer state in policy dir: ${POLICY_DIR}" >&2
  exit 1
fi
if ! compgen -G "${POLICY_DIR}/policy_postprocessor_step_*_unnormalizer_processor.safetensors" >/dev/null; then
  echo "ERROR: missing unnormalizer state in policy dir: ${POLICY_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VLM_MODEL_PATH}" ]]; then
  echo "ERROR: SmolVLM2 backbone not found at: ${VLM_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${VLM_MODEL_PATH}/config.json" ]]; then
  echo "ERROR: Missing ${VLM_MODEL_PATH}/config.json (SmolVLM2 backbone looks incomplete)" >&2
  exit 1
fi
if [[ ! -f "${VLM_MODEL_PATH}/tokenizer.json" && ! -f "${VLM_MODEL_PATH}/tokenizer_config.json" ]]; then
  echo "ERROR: Missing tokenizer files in ${VLM_MODEL_PATH} (tokenizer.json / tokenizer_config.json)" >&2
  exit 1
fi
if ! compgen -G "${VLM_MODEL_PATH}/*.safetensors" >/dev/null && ! compgen -G "${VLM_MODEL_PATH}/*.bin" >/dev/null; then
  echo "ERROR: Missing model weights in ${VLM_MODEL_PATH} (*.safetensors / *.bin)" >&2
  exit 1
fi

# SO-101 action/state dims (6).
ACTION_DIM="${ACTION_DIM:-6}"
STATE_DIM="${STATE_DIM:-6}"
# Keep inference aligned with training: 2 real cameras + 1 empty camera.
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

python - <<'PY' "${POLICY_DIR}/config.json" "${ACTION_DIM}" "${STATE_DIM}"
import json
import sys

cfg = json.load(open(sys.argv[1]))
expected_action_dim = int(sys.argv[2])
expected_state_dim = int(sys.argv[3])

action_dim = int(cfg["output_features"]["action"]["shape"][0])
state_dim = int(cfg["input_features"]["observation.state"]["shape"][0])
if action_dim != expected_action_dim or state_dim != expected_state_dim:
    raise SystemExit(
        "ERROR: policy feature dims do not match expected SO-101 dims.\n"
        f"- expected action_dim={expected_action_dim}, got {action_dim}\n"
        f"- expected state_dim={expected_state_dim}, got {state_dim}\n"
        "Do NOT patch dims at inference time; re-train/export a correct 6D checkpoint."
    )

expected_imgs = {"observation.images.camera1", "observation.images.camera2"}
visual_keys = {
    k
    for k, v in cfg.get("input_features", {}).items()
    if isinstance(v, dict) and v.get("type") == "VISUAL"
}
missing = sorted(expected_imgs - visual_keys)
if missing:
    raise SystemExit(
        "ERROR: policy config is missing required image keys.\n"
        f"- missing: {missing}\n"
        "Expected at least camera1 and camera2. Re-train/export with matching keys."
    )

print(f"[infer_smolvla_so101] policy dims ok: action_dim={action_dim} state_dim={state_dim}")
print(f"[infer_smolvla_so101] policy visual keys: {sorted(visual_keys)}")
PY

POLICY_DEVICE_DEFAULT="cuda"
DISPLAY_DATA_DEFAULT="false"
PLAY_SOUNDS_DEFAULT="false"
EPISODE_TIME_S_DEFAULT="15"
RESET_TIME_S_DEFAULT="15"
NUM_EPISODES_DEFAULT="3"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  POLICY_DEVICE_DEFAULT="cpu"
  DISPLAY_DATA_DEFAULT="false"
  PLAY_SOUNDS_DEFAULT="false"
  EPISODE_TIME_S_DEFAULT="1"
  RESET_TIME_S_DEFAULT="0"
  NUM_EPISODES_DEFAULT="1"
  echo "ERROR: --smoke is not supported for so101_follower (no mock mode). Remove --smoke and set ROBOT_PORT." >&2
  exit 1
fi

POLICY_DEVICE="${POLICY_DEVICE:-${POLICY_DEVICE_DEFAULT}}"
DISPLAY_DATA="${DISPLAY_DATA:-${DISPLAY_DATA_DEFAULT}}"
PLAY_SOUNDS="${PLAY_SOUNDS:-${PLAY_SOUNDS_DEFAULT}}"

if [[ "${POLICY_DEVICE}" == cuda* ]]; then
  python - <<'PY' "${POLICY_DEVICE}"
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: POLICY_DEVICE is set to CUDA but torch.cuda.is_available() is False. "
        "Check your driver/CUDA install and CUDA_VISIBLE_DEVICES."
    )
device_count = torch.cuda.device_count()
policy_device = sys.argv[1]
if policy_device.startswith("cuda:"):
    try:
        idx = int(policy_device.split(":", 1)[1])
    except ValueError:
        raise SystemExit(f"ERROR: Invalid POLICY_DEVICE: {policy_device}")
    if idx < 0 or idx >= device_count:
        raise SystemExit(
            f"ERROR: POLICY_DEVICE={policy_device} but torch.cuda.device_count()={device_count}. "
            "Check CUDA_VISIBLE_DEVICES."
        )
print(f"[infer_smolvla_so101] torch.cuda.device_count={device_count}", file=sys.stderr)
PY
fi

# Robot port is required (SO-101 uses a serial port for the Feetech bus).
ROBOT_PORT="${ROBOT_PORT:-}"
if [[ -z "${ROBOT_PORT}" ]]; then
  echo "ERROR: ROBOT_PORT is not set. Example: ROBOT_PORT=/dev/ttyUSB0" >&2
  exit 1
fi

ROBOT_ID="${ROBOT_ID:-so101_eval}"

# Two cameras: top + wrist (OpenCV). Override indices/size/fps as needed.
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
TOP_CAM_INDEX_OR_PATH="${TOP_CAM_INDEX_OR_PATH:-0}"
WRIST_CAM_INDEX_OR_PATH="${WRIST_CAM_INDEX_OR_PATH:-1}"

CAMERA_CONFIG="${CAMERA_CONFIG:-$(cat <<EOF
{
  top: {"type": "opencv", "index_or_path": ${TOP_CAM_INDEX_OR_PATH}, "width": ${CAMERA_WIDTH}, "height": ${CAMERA_HEIGHT}, "fps": ${CAMERA_FPS}},
  wrist: {"type": "opencv", "index_or_path": ${WRIST_CAM_INDEX_OR_PATH}, "width": ${CAMERA_WIDTH}, "height": ${CAMERA_HEIGHT}, "fps": ${CAMERA_FPS}}
}
EOF
)}"

# Map dataset camera keys (top/wrist) -> policy camera keys (camera1/camera2).
DEFAULT_DATASET_RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
DATASET_RENAME_MAP="${DATASET_RENAME_MAP:-${DEFAULT_DATASET_RENAME_MAP}}"

# Dataset repo id must be in the form user/dataset_name.
HF_USER="${HF_USER:-cqy}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/eval_so101_${TRAIN_RUN}_${CHECKPOINT_STEP}}"

TASK="${TASK:-Place the block into the left box and release.}"

CONTINUOUS="${CONTINUOUS:-false}"
if [[ "${CONTINUOUS}" == "true" || "${CONTINUOUS}" == "1" ]]; then
  NUM_EPISODES_DEFAULT="1000000000"
  RESET_TIME_S_DEFAULT="0"
fi

NUM_EPISODES="${NUM_EPISODES:-${NUM_EPISODES_DEFAULT}}"
DATASET_FPS="${DATASET_FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-${EPISODE_TIME_S_DEFAULT}}"
RESET_TIME_S="${RESET_TIME_S:-${RESET_TIME_S_DEFAULT}}"

EVAL_BASE_DIR="${EVAL_OUTPUT_ROOT}/so101_eval_${TRAIN_RUN}_${CHECKPOINT_STEP}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_BASE_DIR}/${timestamp}}"
mkdir -p "${EVAL_BASE_DIR}"

if [[ -e "${EVAL_OUTPUT_DIR}" ]]; then
  echo "ERROR: EVAL_OUTPUT_DIR already exists: ${EVAL_OUTPUT_DIR}" >&2
  exit 1
fi

echo "[infer_smolvla_so101] policy_dir=${POLICY_DIR}" >&2
echo "[infer_smolvla_so101] vlm_model_path=${VLM_MODEL_PATH}" >&2
echo "[infer_smolvla_so101] output_dir=${EVAL_OUTPUT_DIR}" >&2
echo "[infer_smolvla_so101] robot_port=${ROBOT_PORT} robot_id=${ROBOT_ID}" >&2
echo "[infer_smolvla_so101] dataset.rename_map=${DATASET_RENAME_MAP}" >&2
echo "[infer_smolvla_so101] empty_cameras=${EMPTY_CAMERAS}" >&2
echo "[infer_smolvla_so101] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} policy_device=${POLICY_DEVICE}" >&2

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERA_CONFIG}" \
  --policy.path="${POLICY_DIR}" \
  --policy.vlm_model_name="${VLM_MODEL_PATH}" \
  --policy.device="${POLICY_DEVICE}" \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${EVAL_OUTPUT_DIR}" \
  --dataset.single_task="${TASK}" \
  --dataset.rename_map="${DATASET_RENAME_MAP}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false \
  --display_data="${DISPLAY_DATA}" \
  --play_sounds="${PLAY_SOUNDS}" \
  "$@" \
  2>&1 | tee "${log_file}"
