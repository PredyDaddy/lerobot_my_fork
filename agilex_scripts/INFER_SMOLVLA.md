# `infer_smolvla.sh` 修改指南（换机器推理）

`agilex_scripts/infer_smolvla.sh` 用于在 **AgileX 机器人** 上调用 `lerobot-record` 跑 SmolVLA policy（在线推理 + 录制 eval 数据）。

关键点：
- SmolVLA 结构里 **必须加载 VLM backbone（SmolVLM2）** 才能从图像+语言得到动作，所以换机器推理必须保证 `VLM_MODEL_PATH` 正确。
- **不要在推理阶段“把 6 维 patch 成 14 维”**。推理阶段硬改维度容易导致动作含义/归一化状态不一致；脚本现在会强校验维度，不匹配直接报错。

---

## 1) 你需要改哪些地方（全部是绝对路径/配置）

打开 `agilex_scripts/infer_smolvla.sh`，修改顶部 `# ===== Absolute paths ... =====` 那一段变量即可。

### A. 仓库与 VLM（必须）
- `REPO_ROOT`：推理机上本仓库绝对路径。
- `VLM_MODEL_PATH`：SmolVLM2 模型目录绝对路径（需包含 `config.json`、tokenizer 文件、权重 `*.safetensors`/`*.bin`）。

### B. Policy checkpoint（必须）
推荐直接指定（最稳）：
- `POLICY_DIR=/abs/path/to/.../checkpoints/XXXXXX/pretrained_model`

或者让脚本拼路径（确保路径存在）：
- `TRAIN_OUTPUT_ROOT`：训练输出根目录绝对路径
- `TRAIN_RUN`：训练 run 文件夹名
- `CHECKPOINT_STEP`：`last` 或数字（例如 `100000`）

### C. 输出与日志（建议改）
- `EVAL_OUTPUT_ROOT`：eval 数据输出根目录绝对路径
- `LOG_DIR`：日志目录绝对路径

### D. GPU / 推理设备
- `GPU_ID=7`：脚本会设置 `CUDA_VISIBLE_DEVICES=7`
- `POLICY_DEVICE=cuda`：一般保持默认即可（不要写 `cuda:7`；设置了 `CUDA_VISIBLE_DEVICES=7` 后，`cuda`/`cuda:0` 才是物理 7 号卡）

### E. 机器人 ROS topic（按你的环境确认）
- `ROS_MASTER_URI`
- `PUPPET_LEFT_TOPIC / PUPPET_RIGHT_TOPIC`
- `MASTER_LEFT_TOPIC / MASTER_RIGHT_TOPIC`
- `ENABLE_FLAG_TOPIC`
- `NODE_NAME / ROBOT_ID`

### F. 相机 topic（必须注意 key 名）
SmolVLA policy 期望输入的相机 key 是：
- `observation.images.camera1`
- `observation.images.camera2`
- `observation.images.camera3`

所以脚本里 camera 配置也必须是 `camera1/camera2/camera3`（而不是 `camera_front/camera_left/...`）。

你只需要把下面三个 topic 改成你真实相机对应的 ROS 图像 topic：
- `CAMERA1_TOPIC`（默认 front）
- `CAMERA2_TOPIC`（默认 left）
- `CAMERA3_TOPIC`（默认 right）

以及分辨率/FPS：
- `CAMERA_WIDTH / CAMERA_HEIGHT / CAMERA_FPS`

### G. HF 用户名 / eval 数据集命名
`lerobot-record` 在“带 policy 的录制”时要求数据集名以 `eval_` 开头；脚本默认已满足。
- 改 `HF_USER` 或直接改 `DATASET_REPO_ID`（必须是 `username/eval_...` 这种格式）

---

## 2) 必须满足的“正确 checkpoint 条件”

脚本启动前会强校验（不满足直接退出）：
- `config.json` 里必须是 `action_dim=14`、`state_dim=14`
- 必须包含 `observation.images.camera1/2/3`

如果你的 checkpoint 仍然是 6 维，说明训练/导出的 policy config 还是 6 维（需要用正确的 `--policy.input_features/--policy.output_features` 训练出来的 14D checkpoint），**推理脚本不会帮你强改维度**。

---

## 3) 典型运行方式

### A. 最稳（直接指定 `POLICY_DIR`）
```bash
GPU_ID=7 \
POLICY_DEVICE=cuda \
POLICY_DIR=/abs/path/to/outputs/train/<run>/checkpoints/100000/pretrained_model \
VLM_MODEL_PATH=/abs/path/to/SmolVLM2-500M-Video-Instruct \
EVAL_OUTPUT_ROOT=/abs/path/to/outputs/eval \
LOG_DIR=/abs/path/to/logs \
conda run -n lerobot_v4 bash agilex_scripts/infer_smolvla.sh
```

### B. 连续跑（不限制 episode 数、reset_time=0）
```bash
CONTINUOUS=true GPU_ID=7 POLICY_DEVICE=cuda ... bash agilex_scripts/infer_smolvla.sh
```

### C. 快速 smoke（只验证流程，不连真实 ROS）
```bash
POLICY_DEVICE=cpu \
ROBOT_MOCK=true \
CAMERA_MOCK=true \
POLICY_DIR=/abs/path/to/pretrained_model \
VLM_MODEL_PATH=/abs/path/to/SmolVLM2-500M-Video-Instruct \
conda run -n lerobot_v4 bash agilex_scripts/infer_smolvla.sh --smoke
```

---

## 4) 常见坑（换机器时）

- **`vlm_model_name` 路径不对**：checkpoint 保存的是训练机路径，新机器必须用 `VLM_MODEL_PATH` 覆盖（脚本已做 `--policy.vlm_model_name=...`）。
- **相机 key 不匹配**：policy 用 `camera1/2/3`，你写成 `camera_front/camera_left/right` 会直接 feature mismatch。
- **policy 还是 6 维**：推理会裁剪到 6 维，机器人动作就是错的；必须用 14D checkpoint。
- **`PLAY_SOUNDS=true` 但没装 `spd-say`**：会报错；脚本默认 `PLAY_SOUNDS=false`。

