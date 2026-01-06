#!/usr/bin/env bash
# Run an ACT+DINOv2 policy on the AgileX robot using lerobot-record.

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

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT_DEFAULT="${SCRIPT_DIR}/.."
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
cd "${REPO_ROOT}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

# Keep caches under the repo by default (override if needed).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

timestamp="$(date +'%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/infer_act_dinov2_${timestamp}.log"

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "ERROR: 'lerobot-record' not found in PATH. Try: conda activate lerobot_v4" >&2
  exit 127
fi

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/train}"
TRAIN_RUN="${TRAIN_RUN:-act_dinov2_agilex}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-100000}"
if [[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$(printf "%06d" "$((10#${CHECKPOINT_STEP}))")"
fi

POLICY_DIR_DEFAULT="${TRAIN_OUTPUT_ROOT}/${TRAIN_RUN}/checkpoints/${CHECKPOINT_STEP}/pretrained_model"
POLICY_DIR="${POLICY_DIR:-${POLICY_DIR_DEFAULT}}"

if [[ ! -d "${POLICY_DIR}" ]]; then
  echo "ERROR: policy directory not found: ${POLICY_DIR}" >&2
  exit 1
fi
for required in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  if [[ ! -f "${POLICY_DIR}/${required}" ]]; then
    echo "ERROR: missing ${required} in ${POLICY_DIR}" >&2
    exit 1
  fi
done
PREPROC_PATTERN="${POLICY_DIR}/policy_preprocessor_step_*_normalizer_processor.safetensors"
if ! compgen -G "${PREPROC_PATTERN}" >/dev/null; then
  echo "ERROR: missing preprocessor normalizer state in ${POLICY_DIR}" >&2
  exit 1
fi
POSTPROC_PATTERN="${POLICY_DIR}/policy_postprocessor_step_*_unnormalizer_processor.safetensors"
if ! compgen -G "${POSTPROC_PATTERN}" >/dev/null; then
  echo "ERROR: missing postprocessor state in ${POLICY_DIR}" >&2
  exit 1
fi

ACTION_DIM="${ACTION_DIM:-14}"
STATE_DIM="${STATE_DIM:-14}"
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
        "ERROR: policy feature dims do not match expected AgileX dims.\n"
        f"- expected action_dim={expected_action_dim}, got {action_dim}\n"
        f"- expected state_dim={expected_state_dim}, got {state_dim}\n"
        "Re-train/export a checkpoint with matching dims before running inference."
    )

required_imgs = [
    "observation.images.camera_front",
    "observation.images.camera_left",
    "observation.images.camera_right",
]
missing = [k for k in required_imgs if k not in cfg.get("input_features", {})]
if missing:
    raise SystemExit(
        "ERROR: policy config is missing required image keys.\n"
        f"- missing: {missing}\n"
        "Update camera keys in this script to match your checkpoint (or re-export the checkpoint)."
    )
print(f"[infer_act_dinov2] policy dims ok: action_dim={action_dim} state_dim={state_dim}")
PY

POLICY_DEVICE_DEFAULT="cuda"
ROBOT_MOCK_DEFAULT="false"
CAMERA_MOCK_DEFAULT="false"
DISPLAY_DATA_DEFAULT="false"
PLAY_SOUNDS_DEFAULT="false"
EPISODE_TIME_S_DEFAULT="90"
RESET_TIME_S_DEFAULT="15"
NUM_EPISODES_DEFAULT="3"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  POLICY_DEVICE_DEFAULT="cpu"
  ROBOT_MOCK_DEFAULT="true"
  CAMERA_MOCK_DEFAULT="true"
  DISPLAY_DATA_DEFAULT="false"
  PLAY_SOUNDS_DEFAULT="false"
  EPISODE_TIME_S_DEFAULT="1"
  RESET_TIME_S_DEFAULT="0"
  NUM_EPISODES_DEFAULT="1"
fi

POLICY_DEVICE="${POLICY_DEVICE:-${POLICY_DEVICE_DEFAULT}}"
ROBOT_MOCK="${ROBOT_MOCK:-${ROBOT_MOCK_DEFAULT}}"
CAMERA_MOCK="${CAMERA_MOCK:-${CAMERA_MOCK_DEFAULT}}"
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
policy_device = sys.argv[1]
if policy_device.startswith("cuda:"):
    idx = int(policy_device.split(":", 1)[1])
    if idx < 0 or idx >= torch.cuda.device_count():
        raise SystemExit(
            f"ERROR: POLICY_DEVICE={policy_device} but torch.cuda.device_count()={torch.cuda.device_count()}."
        )
print(f"[infer_act_dinov2] torch.cuda.device_count={torch.cuda.device_count()}", file=sys.stderr)
PY
fi

HF_USER="${HF_USER:-cqy}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/eval_act_dinov2_${TRAIN_RUN}_${CHECKPOINT_STEP}}"
TASK="${TASK:-ACT DINOv2 AgileX eval (${TRAIN_RUN}, ckpt=${CHECKPOINT_STEP})}"
NUM_EPISODES="${NUM_EPISODES:-${NUM_EPISODES_DEFAULT}}"
DATASET_FPS="${DATASET_FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-${EPISODE_TIME_S_DEFAULT}}"
RESET_TIME_S="${RESET_TIME_S:-${RESET_TIME_S_DEFAULT}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/eval}"
EVAL_BASE_DIR="${EVAL_OUTPUT_ROOT}/act_dinov2_eval_${TRAIN_RUN}_${CHECKPOINT_STEP}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_BASE_DIR}/${timestamp}}"
mkdir -p "${EVAL_BASE_DIR}"

# Robot topics (update if your ROS setup differs).
ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
NODE_NAME="${NODE_NAME:-lerobot_agilex}"
PUPPET_LEFT_TOPIC="${PUPPET_LEFT_TOPIC:-/puppet/joint_left}"
PUPPET_RIGHT_TOPIC="${PUPPET_RIGHT_TOPIC:-/puppet/joint_right}"
MASTER_LEFT_TOPIC="${MASTER_LEFT_TOPIC:-/master/joint_left}"
MASTER_RIGHT_TOPIC="${MASTER_RIGHT_TOPIC:-/master/joint_right}"
ENABLE_FLAG_TOPIC="${ENABLE_FLAG_TOPIC:-/enable_flag}"
ROBOT_ID="${ROBOT_ID:-agilex_eval}"

CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_FRONT_TOPIC="${CAMERA_FRONT_TOPIC:-${CAMERA1_TOPIC:-/camera_f/color/image_raw}}"
CAMERA_LEFT_TOPIC="${CAMERA_LEFT_TOPIC:-${CAMERA2_TOPIC:-/camera_l/color/image_raw}}"
CAMERA_RIGHT_TOPIC="${CAMERA_RIGHT_TOPIC:-${CAMERA3_TOPIC:-/camera_r/color/image_raw}}"

CAMERA_CONFIG="$(cat <<EOF
{
  camera_front: {
    "type": "ros_camera",
    "topic_name": "${CAMERA_FRONT_TOPIC}",
    "width": ${CAMERA_WIDTH},
    "height": ${CAMERA_HEIGHT},
    "fps": ${CAMERA_FPS},
    "mock": ${CAMERA_MOCK}
  },
  camera_left: {
    "type": "ros_camera",
    "topic_name": "${CAMERA_LEFT_TOPIC}",
    "width": ${CAMERA_WIDTH},
    "height": ${CAMERA_HEIGHT},
    "fps": ${CAMERA_FPS},
    "mock": ${CAMERA_MOCK}
  },
  camera_right: {
    "type": "ros_camera",
    "topic_name": "${CAMERA_RIGHT_TOPIC}",
    "width": ${CAMERA_WIDTH},
    "height": ${CAMERA_HEIGHT},
    "fps": ${CAMERA_FPS},
    "mock": ${CAMERA_MOCK}
  }
}
EOF
)"

echo "[infer_act_dinov2] policy_dir=${POLICY_DIR}" >&2
echo "[infer_act_dinov2] output_dir=${EVAL_OUTPUT_DIR}" >&2
echo "[infer_act_dinov2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} policy_device=${POLICY_DEVICE}" >&2
echo "[infer_act_dinov2] robot_mock=${ROBOT_MOCK} camera_mock=${CAMERA_MOCK} display_data=${DISPLAY_DATA}" >&2

lerobot-record \
  --robot.type=agilex \
  --robot.mock="${ROBOT_MOCK}" \
  --robot.id="${ROBOT_ID}" \
  --robot.ros_master_uri="${ROS_MASTER_URI}" \
  --robot.node_name="${NODE_NAME}" \
  --robot.puppet_left_topic="${PUPPET_LEFT_TOPIC}" \
  --robot.puppet_right_topic="${PUPPET_RIGHT_TOPIC}" \
  --robot.master_left_topic="${MASTER_LEFT_TOPIC}" \
  --robot.master_right_topic="${MASTER_RIGHT_TOPIC}" \
  --robot.enable_flag_topic="${ENABLE_FLAG_TOPIC}" \
  --robot.cameras="${CAMERA_CONFIG}" \
  --policy.path="${POLICY_DIR}" \
  --policy.device="${POLICY_DEVICE}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${EVAL_OUTPUT_DIR}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false \
  --display_data="${DISPLAY_DATA}" \
  --play_sounds="${PLAY_SOUNDS}" \
  "$@" 2>&1 | tee "${log_file}"
