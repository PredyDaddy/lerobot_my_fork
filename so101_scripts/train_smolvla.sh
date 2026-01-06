#!/usr/bin/env bash
# SmolVLA depends on a VLM backbone (SmolVLM2). If you want fully offline training,
# make sure the backbone is already available locally or in the HF cache.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-${HF_HUB_OFFLINE}}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
# Use the last GPU (index 7) by default.
# Override at runtime with: GPU_ID=0 bash so101_scripts/train_smolvla.sh
GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

set -euo pipefail

# Optional: activate the project environment
# conda activate lerobot_v4

SMOKE_TEST=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE_TEST=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Avoid permission issues on clusters/containers by keeping HF caches in the repo.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

timestamp="$(date +'%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/train_smolvla_so101_${timestamp}.log"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "ERROR: 'lerobot-train' not found in PATH." >&2
  echo "Activate your environment first, e.g.: conda activate lerobot_v4" >&2
  exit 127
fi

# Local pretrained model directory (must contain at least `config.json` and `model.safetensors`).
POLICY_PATH="${POLICY_PATH:-${ROOT_DIR}/smolvla_base}"
if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "ERROR: Pretrained SmolVLA not found at: ${POLICY_PATH}" >&2
  echo "Expected files like: ${POLICY_PATH}/config.json and ${POLICY_PATH}/model.safetensors" >&2
  exit 1
fi
if [[ ! -f "${POLICY_PATH}/config.json" || ! -f "${POLICY_PATH}/model.safetensors" ]]; then
  echo "ERROR: Missing files in pretrained policy dir: ${POLICY_PATH}" >&2
  echo "Expected: config.json and model.safetensors" >&2
  exit 1
fi

# SmolVLM2 backbone (local directory).
# If you need to download it:
#   HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com hf download \
#     HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir "${ROOT_DIR}/SmolVLM2-500M-Video-Instruct"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-${ROOT_DIR}/SmolVLM2-500M-Video-Instruct}"
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

# Dataset (local).
DATASET_REPO_ID="${DATASET_REPO_ID:-so101_v3_dataset1_en}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/so101_v3_dataset1_en}"
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: Dataset root not found at: ${DATASET_ROOT}" >&2
  exit 1
fi
DATASET_INFO_JSON="${DATASET_ROOT}/meta/info.json"
if [[ ! -f "${DATASET_INFO_JSON}" ]]; then
  echo "ERROR: Dataset metadata not found at: ${DATASET_INFO_JSON}" >&2
  exit 1
fi
DATASET_STATS_JSON="${DATASET_ROOT}/meta/stats.json"
if [[ ! -f "${DATASET_STATS_JSON}" ]]; then
  echo "ERROR: Dataset stats not found at: ${DATASET_STATS_JSON}" >&2
  exit 1
fi
DATASET_TASKS_PARQUET="${DATASET_ROOT}/meta/tasks.parquet"
if [[ ! -f "${DATASET_TASKS_PARQUET}" ]]; then
  echo "ERROR: Dataset tasks not found at: ${DATASET_TASKS_PARQUET} (SmolVLA requires language tasks)" >&2
  exit 1
fi

# Validate dataset format and required keys early to avoid wasting training time.
DATASET_VERSION="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('codebase_version','unknown'))" "${DATASET_INFO_JSON}")"
if [[ "${DATASET_VERSION}" != "v3.0" ]]; then
  echo "ERROR: Expected LeRobot dataset v3.0 but got '${DATASET_VERSION}' in ${DATASET_INFO_JSON}" >&2
  exit 1
fi

# Read action/state dims from dataset metadata to avoid silent mismatches.
DATASET_ACTION_DIM="$(python -c "import json,sys; print(int(json.load(open(sys.argv[1]))['features']['action']['shape'][0]))" "${DATASET_INFO_JSON}")"
DATASET_STATE_DIM="$(python -c "import json,sys; print(int(json.load(open(sys.argv[1]))['features']['observation.state']['shape'][0]))" "${DATASET_INFO_JSON}")"
ACTION_DIM="${ACTION_DIM:-${DATASET_ACTION_DIM}}"
STATE_DIM="${STATE_DIM:-${DATASET_STATE_DIM}}"
if [[ "${ACTION_DIM}" != "${DATASET_ACTION_DIM}" || "${STATE_DIM}" != "${DATASET_STATE_DIM}" ]]; then
  echo "ERROR: Dimension mismatch between overrides and dataset meta." >&2
  echo "Dataset: action_dim=${DATASET_ACTION_DIM} state_dim=${DATASET_STATE_DIM}" >&2
  echo "Script:  action_dim=${ACTION_DIM} state_dim=${STATE_DIM}" >&2
  exit 1
fi

POLICY_MAX_ACTION_DIM="$(python -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('max_action_dim', 0)))" "${POLICY_PATH}/config.json")"
POLICY_MAX_STATE_DIM="$(python -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('max_state_dim', 0)))" "${POLICY_PATH}/config.json")"
if (( POLICY_MAX_ACTION_DIM <= 0 || POLICY_MAX_STATE_DIM <= 0 )); then
  echo "ERROR: Could not read max_action_dim/max_state_dim from ${POLICY_PATH}/config.json" >&2
  exit 1
fi
if (( ACTION_DIM > POLICY_MAX_ACTION_DIM )); then
  echo "ERROR: action_dim=${ACTION_DIM} exceeds policy.max_action_dim=${POLICY_MAX_ACTION_DIM}." >&2
  echo "Update smolvla config max_action_dim (and retrain) or reduce action dims." >&2
  exit 1
fi
if (( STATE_DIM > POLICY_MAX_STATE_DIM )); then
  echo "ERROR: state_dim=${STATE_DIM} exceeds policy.max_state_dim=${POLICY_MAX_STATE_DIM}." >&2
  echo "Update smolvla config max_state_dim (and retrain) or reduce state dims." >&2
  exit 1
fi

# Override the pretrained model's default 6-dim features to match dataset dimensions.
# Note: The pretrained SmolVLA base config expects 3 cameras. This dataset only has 2 cameras, so we keep
# two real cameras (camera1/2) and add ONE empty camera via `--policy.empty_cameras=1` (see EMPTY_CAMERAS).
# This avoids ending up with 4 visual keys (camera1/2/3 + empty_camera_0), which can be confusing.
INPUT_FEATURES="{\"observation.state\":{\"type\":\"STATE\",\"shape\":[${STATE_DIM}]},\"observation.images.camera1\":{\"type\":\"VISUAL\",\"shape\":[3,256,256]},\"observation.images.camera2\":{\"type\":\"VISUAL\",\"shape\":[3,256,256]}}"
OUTPUT_FEATURES="{\"action\":{\"type\":\"ACTION\",\"shape\":[${ACTION_DIM}]}}"

# Dataset camera keys -> pretrained policy expected keys.
DEFAULT_RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP:-${DEFAULT_RENAME_MAP}}"

# Create missing cameras as fully padded images (2 real + 1 empty = 3 cameras total).
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

# Ensure the dataset contains the keys used in the rename map, and that rename targets exist in policy features.
python - <<'PY' "${DATASET_INFO_JSON}" "${RENAME_MAP}" "${INPUT_FEATURES}" "${OUTPUT_FEATURES}"
import json
import sys

info = json.load(open(sys.argv[1]))
features = info.get("features", {})

rename_map = json.loads(sys.argv[2])
if not isinstance(rename_map, dict) or not all(
    isinstance(k, str) and isinstance(v, str) for k, v in rename_map.items()
):
    raise SystemExit("ERROR: --rename_map must be a JSON object mapping strings to strings.")

missing_src = [k for k in rename_map.keys() if k not in features]
if missing_src:
    raise SystemExit(f"ERROR: Dataset is missing required keys in meta/info.json: {missing_src}")

if len(set(rename_map.values())) != len(rename_map.values()):
    raise SystemExit("ERROR: --rename_map values must be unique (to avoid overwriting keys).")

input_features = json.loads(sys.argv[3])
output_features = json.loads(sys.argv[4])
allowed_targets = set(input_features.keys()) | set(output_features.keys())
unknown_targets = [v for v in rename_map.values() if v not in allowed_targets]
if unknown_targets:
    raise SystemExit(
        f"ERROR: --rename_map contains targets not present in policy features: {unknown_targets}. "
        f"Allowed targets: {sorted(allowed_targets)}"
    )
PY

python - <<'PY' "${DATASET_STATS_JSON}" "${ACTION_DIM}" "${STATE_DIM}"
import json
import sys

stats = json.load(open(sys.argv[1]))
action_dim = int(sys.argv[2])
state_dim = int(sys.argv[3])

def _len(key: str, stat: str) -> int:
    return len(stats[key][stat])

errors = []
for stat_name in ["mean", "std"]:
    if _len("action", stat_name) != action_dim:
        errors.append(f"stats[action][{stat_name}] length != action_dim ({_len('action', stat_name)} != {action_dim})")
    if _len("observation.state", stat_name) != state_dim:
        errors.append(
            f"stats[observation.state][{stat_name}] length != state_dim ({_len('observation.state', stat_name)} != {state_dim})"
        )

if errors:
    raise SystemExit("ERROR: Dataset stats dimension mismatch:\n- " + "\n- ".join(errors))
PY

if [[ "${SMOKE_TEST}" == "1" || "${SMOKE_TEST}" == "true" ]]; then
  BATCH_SIZE_DEFAULT="1"
  STEPS_DEFAULT="1"
  SAVE_FREQ_DEFAULT="1"
  LOG_FREQ_DEFAULT="1"
  NUM_WORKERS_DEFAULT="0"
  OUTPUT_DIR_DEFAULT="${ROOT_DIR}/outputs/train/_smoke_smolvla_so101_${ACTION_DIM}d_${timestamp}"
  JOB_NAME_DEFAULT="smoke_smolvla_so101_${ACTION_DIM}d"
else
  BATCH_SIZE_DEFAULT="32"
  STEPS_DEFAULT="50000"
  SAVE_FREQ_DEFAULT="5000"
  LOG_FREQ_DEFAULT="100"
  NUM_WORKERS_DEFAULT="4"
  OUTPUT_DIR_DEFAULT="${ROOT_DIR}/outputs/train/smolvla_so101_${ACTION_DIM}d"
  JOB_NAME_DEFAULT="smolvla_so101_${ACTION_DIM}d"
fi

BATCH_SIZE="${BATCH_SIZE:-${BATCH_SIZE_DEFAULT}}"
STEPS="${STEPS:-${STEPS_DEFAULT}}"
SAVE_FREQ="${SAVE_FREQ:-${SAVE_FREQ_DEFAULT}}"
LOG_FREQ="${LOG_FREQ:-${LOG_FREQ_DEFAULT}}"
NUM_WORKERS="${NUM_WORKERS:-${NUM_WORKERS_DEFAULT}}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_DEFAULT}}"

# Validate JSON CLI overrides before starting training.
python - <<PY
import json
json.loads("""${INPUT_FEATURES}""")
json.loads("""${OUTPUT_FEATURES}""")
PY

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
print(f"[train_smolvla] torch.cuda.device_count={device_count}", file=sys.stderr)
PY
fi

echo "[train_smolvla] dataset=${DATASET_REPO_ID} root=${DATASET_ROOT}" >&2
echo "[train_smolvla] action_dim=${ACTION_DIM} state_dim=${STATE_DIM}" >&2
echo "[train_smolvla] empty_cameras=${EMPTY_CAMERAS}" >&2
echo "[train_smolvla] output_dir=${OUTPUT_DIR} job_name=${JOB_NAME}" >&2
echo "[train_smolvla] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
echo "[train_smolvla] device=${POLICY_DEVICE} batch_size=${BATCH_SIZE} steps=${STEPS} num_workers=${NUM_WORKERS}" >&2

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: output_dir already exists: ${OUTPUT_DIR}" >&2
  echo "Pick a new OUTPUT_DIR (or remove the existing directory)." >&2
  exit 1
fi

lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --policy.vlm_model_name="${VLM_MODEL_PATH}" \
  --policy.input_features="${INPUT_FEATURES}" \
  --policy.output_features="${OUTPUT_FEATURES}" \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --rename_map="${RENAME_MAP}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --num_workers="${NUM_WORKERS}" \
  --policy.device="${POLICY_DEVICE}" \
  --policy.push_to_hub=false \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --wandb.enable=false \
  "$@" \
  2>&1 | tee "${log_file}"

if [[ "${SMOKE_TEST}" == "1" || "${SMOKE_TEST}" == "true" ]]; then
  cfg_path="${OUTPUT_DIR}/checkpoints/last/pretrained_model/config.json"
  if [[ -f "${cfg_path}" ]]; then
    python - <<PY
import json
from pathlib import Path

cfg = json.loads(Path("${cfg_path}").read_text())
print("[smoke] saved policy config:", "${cfg_path}")
print("[smoke] action_shape =", cfg["output_features"]["action"]["shape"])
print("[smoke] state_shape  =", cfg["input_features"]["observation.state"]["shape"])
PY
  else
    echo "WARNING: smoke test did not find saved config at: ${cfg_path}" >&2
  fi
fi
