#!/usr/bin/env bash
# Inference script for running SmolVLA policy on the AgileX robot and recording eval episodes.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

# Optional: activate the project environment
# conda activate lerobot_v4
# If you want to see the live viewer, export DISPLAY=:0 (requires an X server).

ROOT_DIR="/mnt/data2/cqy/workspace/lerobot_my_fork"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +'%Y%m%d_%H%M%S')"

# Training run + checkpoint selector
# Example:
#   CHECKPOINT_STEP=100000 ./infer_smolvla.sh
#   CHECKPOINT_STEP=70000  ./infer_smolvla.sh  # auto-pads to 070000
#   CHECKPOINT_STEP=last   ./infer_smolvla.sh  # use the last checkpoint
TRAIN_RUN="smolvla_agilex_215"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}"
if [[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$(printf "%06d" "$((10#${CHECKPOINT_STEP}))")"
fi

log_file="${LOG_DIR}/infer_smolvla_${CHECKPOINT_STEP}_${timestamp}.log"

# Paths - using absolute paths
CHECKPOINT_DIR="${ROOT_DIR}/outputs/train/smolvla_agilex_215/checkpoints/${CHECKPOINT_STEP}/pretrained_model"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_DIR}" >&2
  echo "Available checkpoints:" >&2
  ls -1 "${ROOT_DIR}/outputs/train/smolvla_agilex_215/checkpoints/" 2>/dev/null || echo "  (none)"
  exit 1
fi

EVAL_BASE_DIR="${ROOT_DIR}/outputs/eval/${TRAIN_RUN}_${CHECKPOINT_STEP}"
EVAL_OUTPUT_DIR="${EVAL_BASE_DIR}/${timestamp}"  # unique per run to avoid FileExistsError
mkdir -p "${EVAL_BASE_DIR}"

# Dataset repo id must be in the form user/dataset_name; change `cqy` to your HF username if different.
HF_USER="${HF_USER:-cqy}"
DATASET_REPO_ID="${HF_USER}/eval_${TRAIN_RUN}_${CHECKPOINT_STEP}"

# Set to true only if you have a working display server
DISPLAY_DATA="${DISPLAY_DATA:-false}"

# Eval settings
CONTINUOUS="${CONTINUOUS:-false}"
if [[ "${CONTINUOUS}" == "true" || "${CONTINUOUS}" == "1" ]]; then
  DEFAULT_NUM_EPISODES="1000000000"
  DEFAULT_RESET_TIME_S="0"
else
  DEFAULT_NUM_EPISODES="3"
  DEFAULT_RESET_TIME_S="15"
fi

NUM_EPISODES="${NUM_EPISODES:-${DEFAULT_NUM_EPISODES}}"
DATASET_FPS="${DATASET_FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
RESET_TIME_S="${RESET_TIME_S:-${DEFAULT_RESET_TIME_S}}"

# Update the ROS image topics if your camera topic names differ.
CAMERA_CONFIG='{
  camera_left: {"type": "ros_camera", "topic_name": "/camera_l/color/image_raw",
    "width": 640, "height": 480, "fps": 30},
  camera_right: {"type": "ros_camera", "topic_name": "/camera_r/color/image_raw",
    "width": 640, "height": 480, "fps": 30},
  camera_front: {"type": "ros_camera", "topic_name": "/camera_f/color/image_raw",
    "width": 640, "height": 480, "fps": 30}
}'

echo "========================================"
echo "SmolVLA Inference"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output: ${EVAL_OUTPUT_DIR}"
echo "Log: ${log_file}"
echo "========================================"

lerobot-record \
  --robot.type=agilex \
  --robot.mock=false \
  --robot.id=agilex_eval \
  --robot.cameras="${CAMERA_CONFIG}" \
  --policy.path="${CHECKPOINT_DIR}" \
  --policy.device=cuda \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${EVAL_OUTPUT_DIR}" \
  --dataset.single_task="SmolVLA agilex eval checkpoint ${CHECKPOINT_STEP}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false \
  --display_data="${DISPLAY_DATA}" \
  2>&1 | tee "${log_file}"
