#!/usr/bin/env bash
# SmolVLA depends on a VLM backbone (SmolVLM2). If you want fully offline training,
# make sure the backbone is already available locally or in the HF cache.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

# Optional: activate the project environment
# conda activate lerobot_v4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +'%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/train_smolvla_${timestamp}.log"

# Local pretrained model directory (must contain at least `config.json` and `model.safetensors`).
POLICY_PATH="${ROOT_DIR}/smolvla_base"
if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "ERROR: Pretrained SmolVLA not found at: ${POLICY_PATH}" >&2
  echo "Expected files like: ${POLICY_PATH}/config.json and ${POLICY_PATH}/model.safetensors" >&2
  exit 1
fi

# SmolVLM2 backbone (local directory).
# If you need to download it:
#   HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com hf download \
#     HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir "${ROOT_DIR}/SmolVLM2-500M-Video-Instruct"
VLM_MODEL_PATH="${ROOT_DIR}/SmolVLM2-500M-Video-Instruct"
if [[ ! -d "${VLM_MODEL_PATH}" ]]; then
  echo "ERROR: SmolVLM2 backbone not found at: ${VLM_MODEL_PATH}" >&2
  exit 1
fi

# Dataset camera keys -> pretrained policy expected keys.
# Adjust this mapping if your dataset uses different camera names.
RENAME_MAP='{"observation.images.camera_front":"observation.images.camera1","observation.images.camera_left":"observation.images.camera2","observation.images.camera_right":"observation.images.camera3"}'

# AgileX dual-arm: 14 DOF (7 per arm: 6 joints + 1 gripper)
# Override the pretrained model's default 6-dim features to match actual robot dimensions.
INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[14]},"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera3":{"type":"VISUAL","shape":[3,256,256]}}'
OUTPUT_FEATURES='{"action":{"type":"ACTION","shape":[14]}}'

lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --policy.vlm_model_name="${VLM_MODEL_PATH}" \
  --policy.input_features="${INPUT_FEATURES}" \
  --policy.output_features="${OUTPUT_FEATURES}" \
  --dataset.repo_id=agilex_dataset1 \
  --dataset.root="${ROOT_DIR}/agilex_dataset1" \
  --rename_map="${RENAME_MAP}" \
  --output_dir="${ROOT_DIR}/outputs/train/smolvla_agilex_14dim" \
  --job_name=smolvla_agilex_14dim \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false \
  2>&1 | tee "${log_file}"
