#!/usr/bin/env bash
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=6

set -euo pipefail

# Optional: activate the project environment
# conda activate lerobot_v4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

timestamp="$(date +'%Y%m%d_%H%M%S')"
log_file="${LOG_DIR}/train_dp_${timestamp}.log"

lerobot-train \
  --dataset.repo_id=agilex_dataset1 \
  --dataset.root="${ROOT_DIR}/agilex_dataset1" \
  --policy.type=diffusion \
  --output_dir="${ROOT_DIR}/outputs/train/diffusion_agilex_215" \
  --job_name=diffusion_agilex_215 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=10000 \
  --eval_freq=10000 \
  --log_freq=100 \
  --wandb.enable=false \
  2>&1 | tee "${log_file}"
