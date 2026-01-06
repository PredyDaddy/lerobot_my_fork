# 使用本地数据集，避免连接 HuggingFace
export HF_HUB_OFFLINE=1

lerobot-dataset-viz \
    --repo-id cqy/agilex_both_side_blue_bottle \
    --root /home/agilex/.cache/huggingface/lerobot/cqy/agilex_both_side_blue_bottle \
    --episode-index 1