# `agilex_infer_gr00t_n1d6.py` 使用说明（AgileX + GR00T N1.6）

`agilex_infer_gr00t_n1d6.py` 用于在 **AgileX 机器人** 上直接运行 **GR00T N1.6** 推理并发送关节目标位姿（不通过 `lerobot-record` 录制数据）。

脚本位置：`lerobot_my_fork/agilex_infer_gr00t_n1d6.py`

---

## 1) 你需要准备什么

### A. 一个可用的 GR00T N1.6 checkpoint 目录

`--checkpoint` 需要指向一个本地目录（推荐指向 Isaac-GR00T 的训练输出 `checkpoint-XXXX/`），通常应包含：
- `config.json`
- `model-*.safetensors` / `model.safetensors.index.json`
- `processor_config.json`
- `statistics.json`
- `embodiment_id.json`

示例（你之前训练产物）：
- `/mnt/data2/cqy/workspace/Isaac-GR00T-test/outputs/agilex_proj_diff_multi/checkpoint-2000`

### B. 运行环境（依赖）

脚本同时依赖：
- `lerobot_my_fork`（AgileXRobot、相机、ROS 相关）
- Isaac-GR00T 的 `gr00t` 包（用于加载 `Gr00tPolicy`、模型与 processor）

如果当前环境里不能 `import gr00t`，你可以：
- 用参数 `--isaac-groot-dir /path/to/Isaac-GR00T-repo` 临时把 Isaac-GR00T 仓库加入 `PYTHONPATH`

---

## 2) 最常用的运行方式

在 `lerobot_my_fork/` 根目录下执行：

### A. 真实机器人（GPU 推理）
```bash
python agilex_infer_gr00t_n1d6.py \
  --checkpoint /mnt/data2/cqy/workspace/Isaac-GR00T-test/outputs/agilex_proj_diff_multi/checkpoint-2000 \
  --task "Wipe the bottle with a sponge" \
  --device cuda \
  --fps 30
```

如果 `gr00t` 没安装到当前环境：
```bash
python agilex_infer_gr00t_n1d6.py \
  --isaac-groot-dir /mnt/data2/cqy/workspace/Isaac-GR00T-test \
  --checkpoint /mnt/data2/cqy/workspace/Isaac-GR00T-test/outputs/agilex_proj_diff_multi/checkpoint-2000 \
  --task "Wipe the bottle with a sponge" \
  --device cuda \
  --fps 30
```

### B. 固定运行时长
```bash
python agilex_infer_gr00t_n1d6.py \
  --checkpoint <path/to/checkpoint> \
  --task "..." \
  --duration 60
```

### C. 每次规划执行多步（减少重规划频率）
`--execution-horizon N` 表示一次推理得到一个 action chunk 后，连续执行 `N` 步再重规划。
```bash
python agilex_infer_gr00t_n1d6.py \
  --checkpoint <path/to/checkpoint> \
  --task "..." \
  --execution-horizon 4
```

### D. Mock 模式（不连接硬件）
```bash
python agilex_infer_gr00t_n1d6.py \
  --checkpoint <path/to/checkpoint> \
  --task "..." \
  --mock \
  --device cpu
```

---

## 3) 输入/输出约定（重要）

### A. 语言指令
- 通过 `--task` 传入（建议与训练数据 `tasks.jsonl` 的文本一致）。

### B. 相机 key 映射

GR00T checkpoint 的模态配置里视频 key 通常是 `front/left/right`，而 AgileXRobot 观测里相机 key 是 `camera_front/camera_left/camera_right`。

脚本内置映射：
- `front -> camera_front`
- `left  -> camera_left`
- `right -> camera_right`

如果你的 checkpoint 使用不同的 video key，需要修改 `agilex_infer_gr00t_n1d6.py` 里的 `DEFAULT_VIDEO_KEY_MAP`。

### C. 关节 state/action 切分

脚本目前只支持 AgileX 的 14 维关节顺序，并按下面方式切分成 GR00T 需要的 4 个 group：
- `left_arm`：6 维
- `left_gripper`：1 维
- `right_arm`：6 维
- `right_gripper`：1 维

如果你的 `processor_config.json` 里 `new_embodiment` 的 state/action keys 不是这四个，会直接报错（这是有意的强约束，避免维度/语义对不上导致“能跑但控制错”）。

---

## 4) 常见报错与排查

### A. `ModuleNotFoundError: No module named 'gr00t'`
- 用 `--isaac-groot-dir /path/to/Isaac-GR00T-test` 解决，或把 Isaac-GR00T 安装到当前环境。

### B. 相机缺失 / key 不匹配
报错类似：缺少 `camera_front` 或者 GR00T video key 找不到对应图像。
- 确认 ROS 相机 topic 是否正确（脚本里用的是 `/camera_f/color/image_raw` 等默认 topic）。
- 如果你的 checkpoint 只用 `front`，可以只保证前视相机存在；如果 checkpoint 需要 `left/right`，就必须提供对应相机。

### C. state keys 不符合
报错类似：`只支持 ... 但 checkpoint 需要: [...]`
- 说明你加载的 checkpoint 模态配置不是 AgileX/new_embodiment 这套（或被你训练时改过）。
- 解决思路：要么换对的 checkpoint，要么按你的模态配置修改脚本的 state/action 切分逻辑。

---

## 5) 建议的最小验证步骤

1. 先用 `--mock --device cpu` 跑通流程（至少验证能加载 checkpoint 并进入循环）。
2. 再在真实机器人上跑短时间 `--duration 10`，确认相机/关节 topic 正常、动作维度正确。
3. 最后再拉长运行时间、提高 `--fps` 或调 `--execution-horizon`。

