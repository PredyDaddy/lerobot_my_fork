# LeRobot 数据格式技术文档

> 本文档详细描述 LeRobot 数据集的完整格式规范，包括目录结构、元数据格式、数据存储方式等。

## 1. 概述

LeRobot 数据集是 Hugging Face 为机器人学习设计的标准化数据格式。该格式旨在：
- 支持多种数据模态（图像、视频、状态、动作等）
- 便于在 Hugging Face Hub 上共享
- 支持增量式数据收集
- 优化存储效率（使用视频压缩）

**当前版本**: `v2.1`

## 2. 目录结构

一个典型的 LeRobotDataset 目录结构如下：

```
dataset_root/
├── data/                           # Parquet 数据文件
│   ├── chunk-000/                  # 数据块 0 (episodes 0-999)
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── chunk-001/                  # 数据块 1 (episodes 1000-1999)
│   │   ├── episode_001000.parquet
│   │   └── ...
│   └── ...
├── meta/                           # 元数据文件
│   ├── info.json                   # 数据集基本信息
│   ├── episodes.jsonl              # Episode 信息列表
│   ├── episodes_stats.jsonl        # 每个 episode 的统计信息 (v2.1+)
│   ├── stats.json                  # 全局统计信息 (v2.0 兼容)
│   └── tasks.jsonl                 # 任务定义列表
├── videos/                         # 视频文件 (可选)
│   ├── chunk-000/
│   │   ├── observation.images.laptop/
│   │   │   ├── episode_000000.mp4
│   │   │   ├── episode_000001.mp4
│   │   │   └── ...
│   │   └── observation.images.phone/
│   │       └── ...
│   └── ...
└── images/                         # 图像文件 (临时，编码前)
    └── {image_key}/
        └── episode_{index:06d}/
            └── frame_{index:06d}.png
```

### 2.1 路径模板

| 文件类型 | 路径模板 |
|---------|---------|
| Parquet | `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` |
| Video | `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` |
| Image | `images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png` |

### 2.2 Chunk 机制

- 默认 `chunk_size = 1000`（每个 chunk 最多存储 1000 个 episodes）
- Episode 索引 `ep_idx` 对应的 chunk 编号: `ep_chunk = ep_idx // chunk_size`

## 3. 元数据文件详解

### 3.1 info.json

`meta/info.json` 是数据集的核心配置文件，包含数据集的完整定义：

```json
{
    "codebase_version": "v2.1",
    "robot_type": "so100",
    "total_episodes": 50,
    "total_frames": 15000,
    "total_tasks": 1,
    "total_videos": 100,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 30,
    "splits": {
        "train": "0:50"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        },
        "observation.images.laptop": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": 30,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [6],
            "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": null},
        "frame_index": {"dtype": "int64", "shape": [1], "names": null},
        "episode_index": {"dtype": "int64", "shape": [1], "names": null},
        "index": {"dtype": "int64", "shape": [1], "names": null},
        "task_index": {"dtype": "int64", "shape": [1], "names": null}
    }
}
```

#### info.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `codebase_version` | string | 数据集版本，当前为 "v2.1" |
| `robot_type` | string \| null | 机器人类型（如 "so100", "aloha", "koch"） |
| `total_episodes` | int | 总 episode 数量 |
| `total_frames` | int | 总帧数 |
| `total_tasks` | int | 任务数量 |
| `total_videos` | int | 视频文件总数 |
| `total_chunks` | int | chunk 数量 |
| `chunks_size` | int | 每个 chunk 的最大 episode 数，默认 1000 |
| `fps` | int | 数据采集帧率 |
| `splits` | dict | 数据集分割，如 `{"train": "0:50"}` |
| `data_path` | string | parquet 文件路径模板 |
| `video_path` | string \| null | 视频文件路径模板，无视频时为 null |
| `features` | dict | 特征定义字典 |

### 3.2 Features 特征定义

每个 feature 的定义格式：

```json
{
    "feature_name": {
        "dtype": "数据类型",
        "shape": [维度1, 维度2, ...],
        "names": ["维度名称1", ...] 或 null
    }
}
```

#### 支持的数据类型 (dtype)

| dtype | 说明 | 存储位置 |
|-------|------|---------|
| `float32` | 32位浮点数 | Parquet |
| `float64` | 64位浮点数 | Parquet |
| `int64` | 64位整数 | Parquet |
| `int32` | 32位整数 | Parquet |
| `bool` | 布尔值 | Parquet |
| `string` | 字符串 | Parquet |
| `image` | 图像数据 | Parquet (内嵌) |
| `video` | 视频帧 | 外部 .mp4 文件 |

#### 默认特征 (DEFAULT_FEATURES)

以下特征由系统自动添加，无需手动定义：

```python
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}
```

| 特征 | 说明 |
|------|------|
| `timestamp` | 当前帧在 episode 中的时间戳（秒） |
| `frame_index` | 当前帧在 episode 中的索引（从0开始） |
| `episode_index` | Episode 索引 |
| `index` | 全局帧索引（整个数据集） |
| `task_index` | 任务索引，对应 tasks.jsonl |

### 3.3 episodes.jsonl

每行一个 JSON 对象，记录每个 episode 的元信息：

```jsonl
{"episode_index": 0, "tasks": ["Pick up the red block"], "length": 300}
{"episode_index": 1, "tasks": ["Pick up the red block"], "length": 285}
{"episode_index": 2, "tasks": ["Place block in box"], "length": 320}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `episode_index` | int | Episode 索引 |
| `tasks` | list[str] | 该 episode 包含的任务（自然语言描述） |
| `length` | int | Episode 帧数 |

### 3.4 tasks.jsonl

任务定义文件，每行一个任务：

```jsonl
{"task_index": 0, "task": "Pick up the red block"}
{"task_index": 1, "task": "Place block in box"}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_index` | int | 任务索引 |
| `task` | str | 任务的自然语言描述 |

### 3.5 episodes_stats.jsonl (v2.1+)

每个 episode 的统计信息，用于归一化：

```jsonl
{"episode_index": 0, "stats": {"action": {"min": [...], "max": [...], "mean": [...], "std": [...], "count": [300]}}}
{"episode_index": 1, "stats": {"action": {"min": [...], "max": [...], "mean": [...], "std": [...], "count": [285]}}}
```

### 3.6 stats.json (v2.0 兼容)

全局统计信息，v2.1 版本会根据 `episodes_stats.jsonl` 动态聚合：

```json
{
    "observation.state": {
        "min": [0.0, -1.57, 0.0, -1.57, -3.14, 0.0],
        "max": [3.14, 1.57, 3.14, 1.57, 3.14, 1.0],
        "mean": [1.57, 0.0, 1.57, 0.0, 0.0, 0.5],
        "std": [0.5, 0.3, 0.5, 0.3, 0.8, 0.2],
        "count": [15000]
    },
    "action": {
        "min": [...],
        "max": [...],
        "mean": [...],
        "std": [...],
        "count": [...]
    }
}
```

统计字段说明：
- `min`: 最小值
- `max`: 最大值
- `mean`: 均值
- `std`: 标准差
- `count`: 样本数量

## 4. Parquet 数据文件

### 4.1 文件组织

每个 episode 对应一个独立的 parquet 文件，便于：
- 按需下载特定 episodes
- 增量式数据收集
- 并行处理

### 4.2 列结构

Parquet 文件包含以下列（具体根据 features 定义）：

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float32 | 帧时间戳 |
| `frame_index` | int64 | Episode 内帧索引 |
| `episode_index` | int64 | Episode 索引 |
| `index` | int64 | 全局帧索引 |
| `task_index` | int64 | 任务索引 |
| `observation.state` | Sequence[float32] | 机器人状态 |
| `action` | Sequence[float32] | 动作指令 |
| `observation.images.*` | Image (bytes) | 图像数据（dtype=image时） |

### 4.3 HuggingFace Features 映射

LeRobot features 到 HuggingFace datasets.Features 的映射规则：

```python
def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue  # 视频不存储在 parquet
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"]))
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        # ... 支持到 Array5D
    return datasets.Features(hf_features)
```

## 5. 视频文件

### 5.1 视频编码

LeRobot 使用 ffmpeg (通过 PyAV) 进行视频编码：

```python
# 默认编码参数
vcodec = "libsvtav1"  # AV1 编码器
pix_fmt = "yuv420p"   # 像素格式
g = 2                  # GOP 大小（关键帧间隔）
crf = 30              # 质量因子（越小质量越高）
```

支持的编码器：
- `libsvtav1` (推荐，AV1)
- `h264`
- `hevc` (H.265)

### 5.2 VideoFrame 类型

视频帧在数据集中以 `VideoFrame` 类型表示：

```python
@dataclass
class VideoFrame:
    pa_type: ClassVar[Any] = pa.struct({
        "path": pa.string(),      # 视频文件路径
        "timestamp": pa.float32()  # 帧时间戳
    })
```

### 5.3 视频解码

支持多种解码后端：
- `torchcodec` (推荐，性能最佳)
- `pyav` (默认回退)
- `video_reader` (torchvision)

```python
frames = decode_video_frames(
    video_path="videos/chunk-000/observation.images.laptop/episode_000000.mp4",
    timestamps=[0.0, 0.033, 0.066],  # 要提取的时间戳
    tolerance_s=1e-4,                 # 时间戳容差
    backend="torchcodec"
)
# 返回: torch.Tensor, shape [N, C, H, W], dtype float32, range [0, 1]
```

## 6. 数据集 Python API

### 6.1 加载数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 从 Hub 加载
dataset = LeRobotDataset(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    episodes=[0, 1, 2],  # 可选：只加载特定 episodes
    delta_timestamps={    # 可选：加载历史/未来帧
        "observation.state": [-0.1, 0, 0.1],
        "action": [0, 0.033, 0.066]
    }
)

# 访问数据
sample = dataset[0]
print(sample.keys())
# ['observation.state', 'observation.images.laptop', 'action',
#  'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'task']
```

### 6.2 创建新数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 定义特征
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["j1", "j2", "j3", "j4", "j5", "gripper"]
    },
    "observation.images.cam": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"]
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["j1", "j2", "j3", "j4", "j5", "gripper"]
    }
}

# 创建数据集
dataset = LeRobotDataset.create(
    repo_id="user/my_dataset",
    fps=30,
    features=features,
    robot_type="so100",
    use_videos=True
)

# 添加帧
dataset.add_frame(
    frame={
        "observation.state": np.array([...], dtype=np.float32),
        "observation.images.cam": image_array,  # HWC uint8 或 CHW float32
        "action": np.array([...], dtype=np.float32)
    },
    task="Pick up the object"
)

# 保存 episode
dataset.save_episode()

# 推送到 Hub
dataset.push_to_hub()
```

### 6.3 核心类结构

```
LeRobotDataset (torch.utils.data.Dataset)
├── meta: LeRobotDatasetMetadata
│   ├── info: dict (from info.json)
│   ├── tasks: dict
│   ├── episodes: dict
│   ├── stats: dict
│   └── episodes_stats: dict
├── hf_dataset: datasets.Dataset (from parquet files)
├── episode_data_index: dict
│   ├── from: Tensor  # 每个 episode 的起始帧索引
│   └── to: Tensor    # 每个 episode 的结束帧索引
└── episode_buffer: dict (录制时使用)
```

## 7. 数据验证

### 7.1 帧验证

LeRobot 在添加帧时会自动验证数据格式：

```python
def validate_frame(frame: dict, features: dict):
    # 检查特征是否完整
    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)

    # 验证每个特征的 dtype 和 shape
    for name in common_features:
        validate_feature_dtype_and_shape(name, features[name], frame[name])
```

### 7.2 时间戳同步检查

确保帧时间戳间隔符合 FPS 设定：

```python
check_timestamps_sync(
    timestamps=timestamps,
    episode_indices=episode_indices,
    episode_data_index=ep_data_index,
    fps=30,
    tolerance_s=1e-4  # 允许 0.1ms 的误差
)
```

### 7.3 版本兼容性

```python
# 检查数据集版本兼容性
check_version_compatibility(
    repo_id="lerobot/dataset",
    version_to_check="v2.0",
    current_version="v2.1"
)
```

## 8. 数据转换

### 8.1 从 v1.6 转换到 v2.0

```bash
python -m lerobot.datasets.v2.convert_dataset_v1_to_v2 \
    --repo-id lerobot/old_dataset \
    --local-dir ./converted_dataset
```

### 8.2 特征名称约定

| 前缀 | 类型 | 示例 |
|------|------|------|
| `observation.state` | 机器人状态 | 关节角度、末端位置 |
| `observation.images.*` | 相机图像 | `observation.images.cam_high` |
| `observation.environment_state` | 环境状态 | 物体位置 |
| `action` | 动作 | 目标关节角度 |

## 9. 多数据集支持

### 9.1 MultiLeRobotDataset

合并多个数据集：

```python
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

multi_dataset = MultiLeRobotDataset(
    repo_ids=["lerobot/dataset1", "lerobot/dataset2"],
    episodes={"lerobot/dataset1": [0, 1], "lerobot/dataset2": [0, 1, 2]},
)

# 只保留所有数据集共有的特征
print(multi_dataset.disabled_features)
```

## 10. 与 Hugging Face Hub 集成

### 10.1 上传数据集

```python
dataset.push_to_hub(
    branch="main",
    tags=["so100", "pick-and-place"],
    license="apache-2.0",
    tag_version=True,  # 自动创建版本标签
    push_videos=True,
    private=False
)
```

### 10.2 数据集卡片

自动生成 README.md 数据集卡片，包含：
- 数据集结构 (info.json)
- 特征定义
- 使用示例

## 11. 常见问题

### Q1: 视频 vs 图像存储

**视频存储 (`dtype: video`)**:
- 优点：文件小（帧间压缩）
- 缺点：随机访问较慢
- 适用：训练数据集

**图像存储 (`dtype: image`)**:
- 优点：随机访问快
- 缺点：占用空间大
- 适用：实时推理、调试

### Q2: delta_timestamps 用途

用于加载历史帧或预测未来帧：

```python
delta_timestamps = {
    "observation.state": [-0.1, 0],     # 当前帧和100ms前
    "action": [0, 0.033, 0.066, 0.1]    # 未来4帧动作
}
```

### Q3: episode_data_index 结构

```python
episode_data_index = {
    "from": tensor([0, 300, 585]),   # 每个 episode 起始帧
    "to": tensor([300, 585, 900])    # 每个 episode 结束帧
}
# Episode 0: frames 0-299
# Episode 1: frames 300-584
# Episode 2: frames 585-899
```

## 12. 参考资料

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [数据集示例](https://huggingface.co/lerobot)

---

*文档版本: v1.0 | 最后更新: 2024-12*
