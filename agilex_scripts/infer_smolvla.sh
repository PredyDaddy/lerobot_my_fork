#!/usr/bin/env bash
# Example inference script for running a SmolVLA policy on the AgileX robot and recording eval episodes.
#
# Notes:
# - This script uses absolute paths (resolved from this repo location).
# - SmolVLA checkpoints in this repo may keep the base config with action_dim=6; AgileX is 14-DoF.
#   We create a lightweight "patched" policy directory (symlinks + edited JSON) so lerobot-record can run.

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

set -euo pipefail

# Optional: activate the project environment
# conda activate lerobot_v4
# If you want to see the live viewer, export DISPLAY=:0 (requires an X server).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +'%Y%m%d_%H%M%S')"

# Training run + checkpoint selector
# Example:
#   TRAIN_RUN=smolvla_agilex_215 CHECKPOINT_STEP=100000 bash agilex_scripts/infer_smolvla.sh
#   TRAIN_RUN=smolvla_agilex_215 CHECKPOINT_STEP=60000  bash agilex_scripts/infer_smolvla.sh  # auto-pads to 060000
TRAIN_RUN="${TRAIN_RUN:-smolvla_agilex_215}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}"
if [[ "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT_STEP="$(printf "%06d" "$((10#${CHECKPOINT_STEP}))")"
fi

log_file="${LOG_DIR}/infer_smolvla_${TRAIN_RUN}_${CHECKPOINT_STEP}_${timestamp}.log"

# Absolute path to the original checkpoint directory.
ORIG_POLICY_DIR="${ROOT_DIR}/outputs/train/${TRAIN_RUN}/checkpoints/${CHECKPOINT_STEP}/pretrained_model"
if [[ ! -d "${ORIG_POLICY_DIR}" ]]; then
  echo "Checkpoint not found: ${ORIG_POLICY_DIR}" >&2
  exit 1
fi

# AgileX (dual-arm) action/state dims (14). Override if needed.
ACTION_DIM="${ACTION_DIM:-14}"
STATE_DIM="${STATE_DIM:-14}"

# Create a patched policy directory (symlinks to weights + patched JSON configs).
PATCH_ROOT="${ROOT_DIR}/outputs/eval/_patched_policies"
PATCHED_POLICY_DIR="${PATCH_ROOT}/${TRAIN_RUN}_${CHECKPOINT_STEP}/pretrained_model_action${ACTION_DIM}"
mkdir -p "${PATCHED_POLICY_DIR}"

export ORIG_POLICY_DIR PATCHED_POLICY_DIR ACTION_DIM STATE_DIM
python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


def _symlink_or_hardlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        # If it's already the right target, keep it.
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass

    try:
        os.link(src, dst)
        return
    except OSError:
        pass

    # Fallback copy (should rarely happen). Avoid copying huge weights.
    if src.name == "model.safetensors":
        raise RuntimeError(f"Failed to link model weights: {src} -> {dst}")
    dst.write_bytes(src.read_bytes())


orig = Path(os.environ["ORIG_POLICY_DIR"]).resolve()
patched = Path(os.environ["PATCHED_POLICY_DIR"]).resolve()
action_dim = int(os.environ.get("ACTION_DIM", "14"))
state_dim = int(os.environ.get("STATE_DIM", str(action_dim)))

patched.mkdir(parents=True, exist_ok=True)

# Link all state/weights files that the processor pipelines reference.
for src in orig.glob("*.safetensors"):
    _symlink_or_hardlink(src, patched / src.name)

for name in ["model.safetensors", "train_config.json"]:
    src = orig / name
    if src.exists():
        _symlink_or_hardlink(src, patched / name)

# Patch policy config so SmolVLA unpads actions to AgileX dimension.
cfg_path = orig / "config.json"
cfg = json.loads(cfg_path.read_text())
cfg.setdefault("output_features", {}).setdefault("action", {})
cfg["output_features"]["action"]["type"] = cfg["output_features"]["action"].get("type", "ACTION")
cfg["output_features"]["action"]["shape"] = [action_dim]

if "input_features" in cfg and "observation.state" in cfg["input_features"]:
    cfg["input_features"]["observation.state"]["shape"] = [state_dim]

(patched / "config.json").write_text(json.dumps(cfg, indent=4) + "\n")

# Patch processor JSONs for consistency (not strictly required but avoids confusion).
def _patch_processor_json(src_path: Path, dst_path: Path) -> None:
    data = json.loads(src_path.read_text())
    for step in data.get("steps", []):
        step_cfg = step.get("config", {})
        feats = step_cfg.get("features")
        if not isinstance(feats, dict):
            continue
        if isinstance(feats.get("action"), dict):
            feats["action"]["shape"] = [action_dim]
        if isinstance(feats.get("observation.state"), dict):
            feats["observation.state"]["shape"] = [state_dim]
    dst_path.write_text(json.dumps(data, indent=2) + "\n")


for fname in ["policy_preprocessor.json", "policy_postprocessor.json"]:
    src = orig / fname
    if src.exists():
        _patch_processor_json(src, patched / fname)
PY

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

lerobot-record \
  --robot.type=agilex \
  --robot.mock=false \
  --robot.id=agilex_eval \
  --robot.cameras="${CAMERA_CONFIG}" \
  --policy.path="${PATCHED_POLICY_DIR}" \
  --policy.device=cuda \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${EVAL_OUTPUT_DIR}" \
  --dataset.single_task="SmolVLA agilex eval ${TRAIN_RUN} checkpoint ${CHECKPOINT_STEP}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.fps="${DATASET_FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false \
  --display_data="${DISPLAY_DATA}" \
  2>&1 | tee "${log_file}"

