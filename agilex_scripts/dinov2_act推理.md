# ACT + DINOv2（`act_dinov2`）推理指南（换机器 / 上真机）

`agilex_scripts/infer_act_dinov2.sh` 用于在 **AgileX 机器人** 上调用 `lerobot-record` 跑 `act_dinov2` policy（在线推理 + 可选录制 eval 数据）。

---

## 1) 你需要从训练机拷贝什么

### A. 推荐（最稳）：只拷一个 checkpoint 的 `pretrained_model/`

从训练机拷贝下面这个目录到机器人机器任意位置（目录名随意）：
- `outputs/train/act_dinov2_agilex/checkpoints/100000/pretrained_model`
  - 或者用 `checkpoints/last/pretrained_model`（注意拷贝时尽量保留 `last` 软链接；不保留也没关系，直接用数字 step）

这个 `pretrained_model/` 目录里应至少包含：
- `config.json`
- `model.safetensors`
- `policy_preprocessor.json` / `policy_postprocessor.json`
- `policy_preprocessor_step_*_normalizer_processor.safetensors`
- `policy_postprocessor_step_*_unnormalizer_processor.safetensors`

### B. 也可以：整包拷贝训练 run 目录

如果你希望保留所有 checkpoint：
- 拷贝整个 `outputs/train/act_dinov2_agilex/` 到机器人机器上（推荐用 `rsync -a` 保留软链接 `checkpoints/last`）。

---

## 2) `dinov2_base/` 需要拷贝吗？

通常 **不需要**。

你的 checkpoint 在 `model.safetensors` 里已经包含 DINOv2 的权重；并且 `config.json` 里应保存了 `dinov2_config_json`，因此推理时不会再去读取训练机的 `dinov2_base` 路径。

在机器人机器上，你可以用下面命令确认（输出 `True` 表示不需要 `dinov2_base`）：

```bash
python - <<'PY'
import json
from pathlib import Path

cfg = json.loads(Path("PATH/TO/pretrained_model/config.json").read_text())
print(cfg.get("dinov2_config_json") is not None)
PY
```

只有在下面情况之一才需要把 `dinov2_base/` 也拷过去：
- 你要在机器人机器上“从头开始训练/微调”（不是推理）。
- 你的 `config.json` 里 `dinov2_config_json` 为空（非常少见；正常训练保存 checkpoint 后应为非空）。

---

## 3) 机器人机器需要准备的环境

1) 代码与环境：
- 有本仓库代码（建议与训练机同一版本/commit）。
- `conda activate lerobot_v4`
- `lerobot-record` 在 PATH 中可用：`which lerobot-record`
  - 如果不可用：在仓库根目录执行 `pip install -e .`

2) ROS / AgileX：
- 能连上 ROS master（`ROS_MASTER_URI`）
- topic 名称与脚本一致或通过环境变量覆盖（见下文）

---

## 4) 先跑 smoke（强烈建议）

不连真实 ROS，只验证“能加载 checkpoint + 推理循环能跑”：

```bash
POLICY_DIR=/abs/path/to/pretrained_model \
conda run -n lerobot_v4 bash agilex_scripts/infer_act_dinov2.sh --smoke
```

---

## 5) 真机推理（最常用）

### A. 最稳：直接指定 `POLICY_DIR`

```bash
GPU_ID=7 \
POLICY_DEVICE=cuda \
POLICY_DIR=/abs/path/to/pretrained_model \
ROS_MASTER_URI=http://192.168.1.234:11311 \
CAMERA_FRONT_TOPIC=/camera_f/color/image_raw \
CAMERA_LEFT_TOPIC=/camera_l/color/image_raw \
CAMERA_RIGHT_TOPIC=/camera_r/color/image_raw \
EVAL_OUTPUT_ROOT=/abs/path/to/outputs/eval \
HF_USER=cqy \
conda run -n lerobot_v4 bash agilex_scripts/infer_act_dinov2.sh
```

说明：
- 脚本会设置 `CUDA_VISIBLE_DEVICES=${GPU_ID}`，因此一般 `POLICY_DEVICE=cuda`（或 `cuda:0`）即可；不要写 `cuda:7`。
- 脚本默认 `push_to_hub=false`，数据会写到 `EVAL_OUTPUT_ROOT/act_dinov2_eval_${TRAIN_RUN}_${CHECKPOINT_STEP}/${timestamp}`。
- 如需固定输出目录，设置 `EVAL_OUTPUT_DIR=/abs/path/to/outputs/eval/my_run`。

### B. 如果你按原目录结构放到了仓库里

把训练产物放到机器人机器的仓库同路径：
- `${REPO_ROOT}/outputs/train/act_dinov2_agilex/...`

然后直接：
```bash
GPU_ID=7 POLICY_DEVICE=cuda conda run -n lerobot_v4 bash agilex_scripts/infer_act_dinov2.sh
```

---

## 6) 相机 key / 维度必须匹配（脚本会强校验）

脚本会读取 `POLICY_DIR/config.json` 并强校验：
- `action_dim == 14`
- `state_dim == 14`
- 必须包含 3 路图像 key：
  - `observation.images.camera_front`
  - `observation.images.camera_left`
  - `observation.images.camera_right`

如果你的 checkpoint 里相机 key 不是这三个（例如 `camera1/2/3`），你需要：
- 要么改 `infer_act_dinov2.sh` 里的相机 key（以及 camera 配置的 key）以匹配你的 checkpoint；
- 要么重新导出/训练一个 key 对齐的 checkpoint。

---

## 7) 常见问题

- 报错缺少 image keys：用 `cat ${POLICY_DIR}/config.json | python -m json.tool | rg \"observation.images\"` 看看 checkpoint 里到底用的相机 key 是什么，然后对齐脚本。
- GPU 不可用：脚本会在 `POLICY_DEVICE=cuda*` 时检查 `torch.cuda.is_available()`；如果失败，先解决驱动/CUDA/`CUDA_VISIBLE_DEVICES`。
- ROS topic 不对：用环境变量覆盖 `ROS_MASTER_URI` / `CAMERA_*_TOPIC` / 机械臂 topic。
