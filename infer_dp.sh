#!/usr/bin/env bash
# Example inference script for running Diffusion Policy on the AgileX robot and recording eval episodes.

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set -euo pipefail

# Optional: activate the project environment
# conda activate lerobot_v4
# If you want to see the live viewer, export DISPLAY=:0 (requires an X server).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +'%Y%m%d_%H%M%S')"

# Training run + checkpoint selector
# Example:
#   TRAIN_RUN=diffusion_agilex_215 CHECKPOINT_STEP=100000 ./infer_dp.sh
#   TRAIN_RUN=diffusion_agilex_215 CHECKPOINT_STEP=60000  ./infer_dp.sh  # auto-pads to 060000
TRAIN_RUN="${TRAIN_RUN:-diffusion_agilex_215}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}"
if [[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$(printf "%06d" "$((10#${CHECKPOINT_STEP}))")"
fi

log_file="${LOG_DIR}/infer_dp_${TRAIN_RUN}_${CHECKPOINT_STEP}_${timestamp}.log"

# Paths
CHECKPOINT_DIR="${ROOT_DIR}/outputs/train/${TRAIN_RUN}/checkpoints/${CHECKPOINT_STEP}/pretrained_model"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_DIR}" >&2
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

# Update the ROS image topics if your camera topic names differ.
CAMERA_CONFIG='{
  camera_left: {"type": "ros_camera", "topic_name": "/camera_l/color/image_raw",
    "width": 640, "height": 480, "fps": 30},
  camera_right: {"type": "ros_camera", "topic_name": "/camera_r/color/image_raw",
    "width": 640, "height": 480, "fps": 30},
  camera_front: {"type": "ros_camera", "topic_name": "/camera_f/color/image_raw",
    "width": 640, "height": 480, "fps": 30}
}'

lerobot-record \
  --robot.type=agilex \
  --robot.mock=false \
  --robot.id=agilex_eval \
  --robot.cameras="${CAMERA_CONFIG}" \
  --policy.path="${CHECKPOINT_DIR}" \
  --policy.device=cuda \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${EVAL_OUTPUT_DIR}" \
  --dataset.single_task="DP agilex eval ${TRAIN_RUN} checkpoint ${CHECKPOINT_STEP}" \
  --dataset.num_episodes=3 \
  --dataset.fps=30 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --display_data="${DISPLAY_DATA}" \
  2>&1 | tee "${log_file}"

