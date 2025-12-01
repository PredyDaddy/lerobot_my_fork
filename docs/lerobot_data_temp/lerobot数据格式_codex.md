# LeRobot 数据格式（v2.1）

## 总览
- LeRobotDataset 的当前代码版本为 `v2.1`（参见 `src/lerobot/datasets/lerobot_dataset.py` 中的 `CODEBASE_VERSION`）。数据集以 Hugging Face Hub 数据集仓库形式分发，也可完全离线使用。
- 核心思想：元数据在 `meta/`，帧级数据在 `data/`（Parquet），视觉模态在 `videos/`（或 `images/`），目录按 episode 分块（默认每 1000 个 episode 一个 chunk）。

## 文件结构与命名
```
<root>/
├── data/
│   └── chunk-000/episode_000000.parquet
├── videos/                       # use_videos=True 时存在
│   └── chunk-000/<video_key>/episode_000000.mp4
├── images/                       # use_videos=False 时存在
│   └── <image_key>/episode_000000/frame_000000.png
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl  # v2.1 推荐
    ├── stats.json            # v2.0 兼容
    └── tasks.jsonl
```
- 路径格式可在 `meta/info.json` 中覆盖：`data_path` 默认为 `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`，`video_path` 默认为 `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`，`DEFAULT_IMAGE_PATH` 为 `images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png`。
- chunk 号由 `episode_index // chunks_size` 计算，默认 `chunks_size=1000`。

## 元数据文件
- `meta/info.json`：数据集概览，字段包含：
  - `codebase_version`（如 `v2.1`）、`robot_type`、`fps`。
  - 计数：`total_episodes`、`total_frames`、`total_tasks`、`total_videos`、`total_chunks`。
  - 路径：`data_path`、`video_path`（无视频时为 `null`）、`splits`（形如 `"train": "0:<n>"`）。
  - `features`：特征字典（见下）。
- `meta/episodes.jsonl`：每行一个 episode，字段 `episode_index`、`tasks`（任务字符串列表）、`length`（帧数）。
- `meta/tasks.jsonl`：`{"task_index": int, "task": "<自然语言任务>"}` 映射，`task_index` 在帧数据中使用。
- `meta/episodes_stats.jsonl`（v2.1）：按 episode 存储统计量，`stats.json`（v2.0）存全局统计。加载时会聚合为全局 `stats`。

## 特征定义（features）
- `features` 是以 key 为名的字典，每个条目包含 `dtype`、`shape`、`names`。默认保留特征（不可省略）：
  - `timestamp`（`float32`, `(1,)`）：秒级时间戳，需满足相邻帧 `1/fps ± tolerance`。
  - `frame_index`、`episode_index`、`index`、`task_index`（均为 `int64`, `(1,)`）。
- 常见自定义特征示例：
  - 状态：`observation.state` → `{"dtype": "float32", "shape": (N,), "names": ["joint1", ...]}`。
  - 视觉：`observation.images.<cam>` → `{"dtype": "video"|"image", "shape": (C,H,W) 或 (H,W,C), "names": ["height","width","channels"]}`。
  - 动作：`action` 或 `action.state` → 同上，`dtype="float32"`。
- 约束与校验：
  - key 不能包含 `/`；`dtype` 支持数值、`image`、`video`、`string`。
  - 视频特征不会写入 Parquet，而是通过 `video_path` 对应的 mp4。
  - `validate_frame` 会检查 dtype/shape 一致性，帧写入前需满足。

## 帧级数据（Parquet）
- 每个 `episode_*.parquet` 存储该 episode 的所有帧。非视频特征以列存储；若使用 `use_videos=True`，视觉列包含 `VideoFrame` 引用（`path`、`timestamp`）。
- 通过 Hugging Face `datasets` 读取，加载时会 `hf_transform_to_torch` 将数值转为 `torch.Tensor`，图像转为 `float32`、通道优先 `[0,1]`。
- 数据集实例化时会验证时间戳同步（`check_timestamps_sync`），并可通过 `delta_timestamps` 请求相对帧（需为 `1/fps` 的整数倍，见 `check_delta_timestamps`）。

## 时间、采样与同步
- `fps` 定义采样频率；`tolerance_s` 默认 `1e-4`，用于校验相邻帧时间差及 `delta_timestamps`。
- `delta_timestamps`：以秒为单位的相对时间列表，按 key（如 `observation.images.cam_high`）指定；会转换为 `delta_indices = round(delta * fps)`，在数据访问时联动抓取邻近帧。

## 版本与兼容
- 创建/加载会检查 `info.json` 中的 `codebase_version`，与当前库主版本差异大时会抛出兼容性错误（`BackwardCompatibilityError`/`ForwardCompatibilityError`）。
- 若请求的 `revision` 不存在，会回退到同主版本的最新 tag，并给出警告。

## 创建、写入与推送
- 创建空数据集（示例）：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import hw_to_dataset_features

features = {
    **hw_to_dataset_features({"joint1": float, "joint2": float}, prefix="observation"),
    **hw_to_dataset_features({"cam_high": (3, 480, 640)}, prefix="observation"),
    "action": {"dtype": "float32", "shape": (2,), "names": ["joint1", "joint2"]},
}
meta = LeRobotDatasetMetadata.create(repo_id="user/robot", fps=30, features=features, use_videos=True)
```
- 采集时逐帧调用写入工具（`validate_frame` → `write_episode`/`write_episode_stats`），完成后可 `LeRobotDataset.push_to_hub()`，默认也上传 README（基于 `card_template.md`）并可选择是否上传视频。
- 读取时：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/pusht", episodes=[0,1], image_transforms=None, download_videos=True)
frame = ds[0]  # 包含各特征的 torch.Tensor / PIL.Image
```

## 实用提示
- 使用视频存储时，`update_video_info()` 会从首个 episode 的视频写入分辨率/编码信息，假设所有视频编码一致。
- 无 GPU/网络环境可通过设置 `root` 指向本地缓存或 `HF_LEROBOT_HOME` 环境变量离线运行。
- 新增配置字段或路径格式时，需同步更新 `info.json` 并确保 `get_hf_features_from_features` 能正确构建 `datasets.Features`。
