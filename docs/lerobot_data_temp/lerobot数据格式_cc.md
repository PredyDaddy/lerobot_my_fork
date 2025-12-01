# LeRobot 数据集格式技术文档

## 目录
1. [概述](#概述)
2. [数据格式版本](#数据格式版本)
3. [数据集目录结构](#数据集目录结构)
4. [Parquet 文件格式](#parquet-文件格式)
5. [元数据结构](#元数据结构)
6. [视频数据存储](#视频数据存储)
7. [特征类型系统](#特征类型系统)
8. [数据加载流程](#数据加载流程)
9. [统计信息](#统计信息)
10. [版本兼容性](#版本兼容性)
11. [最佳实践](#最佳实践)

## 概述

LeRobot 数据集格式是为机器人学习专门设计的标准化数据存储格式，支持多模态观测数据（图像、状态）、动作序列、任务描述和元数据。该格式具有以下特点：

- **高效存储**：使用 Parquet 列式存储和 MP4 视频压缩
- **多模态支持**：同时支持数值、图像、视频等多种数据类型
- **时间序列**：内置时间戳和帧索引，支持精确的时间对齐
- **可扩展性**：模块化设计，易于添加新的传感器或任务类型
- **版本兼容**：支持向前兼容，提供版本转换工具

## 数据格式版本

### 当前版本
- **v2.1**: 最新版本，支持多任务、更灵活的 chunk 组织
- **v2.0**: 引入 chunk 结构，每个 episode 独立 parquet 文件
- **v1.6**: 旧版本，使用元数据目录和 safetensors 统计信息

### 版本对比

| 特性 | v1.6 | v2.0+ |
|------|------|-------|
| 统计信息格式 | safetensors | JSON |
| Parquet 组织 | 单文件 | 按 chunk 分文件 |
| 视频路径格式 | 扁平结构 | 按 chunk 分层结构 |
| 任务存储 | parquet 列 | 独立 tasks.jsonl |
| 分集信息 | parquet 列 | episodes.jsonl |
| 多任务支持 | 有限 | 完整支持 |

### 版本迁移

从 v1.6 迁移到 v2.0+：
```python
from lerobot.datasets.v2.convert_dataset_v1_to_v2 import convert_dataset
convert_dataset("path/to/v1/dataset", "path/to/v2/dataset")
```

## 数据集目录结构

### v2.x 完整结构

```
dataset_root/
├── data/                          # Parquet 数据文件
│   ├── chunk-000/                 # 数据块目录
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...                    # 每个 episode 一个文件
│   ├── chunk-001/
│   │   └── ...
│   └── ...
├── meta/                          # 元数据目录
│   ├── info.json                  # 数据集全局信息
│   ├── episodes.jsonl             # 分集描述
│   ├── stats.json                 # 全局统计信息
│   ├── episodes_stats.jsonl       # 分集统计（v2.1+）
│   └── tasks.jsonl                # 任务定义
├── videos/                        # 视频文件
│   ├── chunk-000/
│   │   ├── observation.images.cam1/
│   │   │   ├── episode_000000.mp4
│   │   │   └── ...
│   │   └── observation.images.cam2/
│   │       └── ...
│   └── ...
└── .gitignore                     # 忽略视频和 parquet 文件
```

### 路径模式

- **Parquet 路径模式**: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
- **视频路径模式**: `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`
- **Chunk 大小**: 可配置，通常为 1000 episodes

## Parquet 文件格式

### 文件组织

每个 episode 对应一个独立的 parquet 文件，文件名格式：`episode_{episode_index:06d}.parquet`

### 列结构

```python
{
    "index": int64,                  # 全局帧索引（跨所有 episodes 唯一）
    "episode_index": int64,          # 所属 episode 索引
    "frame_index": int64,            # Episode 内帧索引（从 0 开始）
    "timestamp": float32,            # 时间戳（秒，相对于 episode 开始）
    "task_index": int64,             # 任务索引
    "observation.state": float32[],  # 机器人状态（关节角度、速度等）
    "observation.images.{cam}": str, # 图像/视频文件路径（或视频元数据）
    "action": float32[],             # 动作指令
    "next.reward": float32,          # 奖励（可选）
    "next.done": bool,               # 终止标志（可选）
    "next.success": bool,            # 成功标志（可选）
}
```

### 示例数据

一个典型的 episode parquet 文件内容示例：

```python
import pandas as pd

df = pd.read_parquet("episode_000000.parquet")
print(df.head())

# 输出：
#    index  episode_index  frame_index  timestamp  observation.state  \
# 0      0              0            0     0.0000  [0.1, -0.2, 0.3, ...]
# 1      1              0            1     0.0333  [0.1, -0.2, 0.3, ...]
# 2      2              0            2     0.0667  [0.1, -0.2, 0.3, ...]
# 3      3              0            3     0.1000  [0.1, -0.2, 0.3, ...]
# 4      4              0            4     0.1333  [0.1, -0.2, 0.3, ...]
#
#   observation.images.top  action  \
# 0  videos/chunk-000/obs...  [0.0, 0.1, -0.1, ...]
# 1  videos/chunk-000/obs...  [0.0, 0.1, -0.1, ...]
# 2  videos/chunk-000/obs...  [0.0, 0.1, -0.1, ...]
# 3  videos/chunk-000/obs...  [0.0, 0.1, -0.1, ...]
# 4  videos/chunk-000/obs...  [0.0, 0.1, -0.1, ...]
```

### 时间戳规则

- **精度**: 微秒级（1e-6 秒）
- **起始**: Episode 第一帧 timestamp = 0.0
- **间隔**: 通常为 1/FPS（如 30 FPS 对应 0.0333 秒）
- **验证**: 加载时验证时间戳连续性，容差 1e-4 秒

## 元数据结构

### info.json

数据集的全局元数据文件。

```json
{
  "codebase_version": "v2.1",
  "robot_type": "koch",
  "env_type": "pusht",
  "total_episodes": 565,
  "total_frames": 169500,
  "total_tasks": 1,
  "total_videos": 565,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 30,
  "splits": {
    "train": "0:452",
    "val": "452:509",
    "test": "509:565"
  },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "encoding": {
    "video.codec": "h264",
    "video.pixel_format": "yuv420p",
    "video.width": 640,
    "video.height": 480,
    "video.fps": 30
  },
  "features": {
    "observation.state": {
      "dtype": "float32",
      "shape": [7],
      "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    },
    "observation.images.top": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "channels"],
      "info": {
        "video.height": 480,
        "video.width": 640,
        "video.fps": 30,
        "video.codec": "h264"
      }
    },
    "action": {
      "dtype": "float32",
      "shape": [2],
      "names": ["x", "y"]
    }
  }
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `codebase_version` | string | LeRobot 版本 |
| `robot_type` | string | 机器人类型（koch, aloha, xarm等） |
| `env_type` | string | 环境类型（pusht, aloha, xarm等） |
| `total_episodes` | int | 总分集数 |
| `total_frames` | int | 总帧数 |
| `total_tasks` | int | 任务数量 |
| `total_videos` | int | 视频文件数量 |
| `total_chunks` | int | 数据块数量 |
| `chunks_size` | int | 每个块包含的 episodes 数 |
| `fps` | int | 视频帧率 |
| `splits` | dict | 数据集划分（train:val:test） |
| `data_path` | string | Parquet 文件路径模板 |
| `video_path` | string | 视频文件路径模板 |
| `features` | dict | 特征定义 |

### episodes.jsonl

每个 episode 的描述信息，JSON Lines 格式。

```jsonl
{"episode_index": 0, "tasks": [0], "length": 300, "file": "data/chunk-000/episode_000000.parquet"}
{"episode_index": 1, "tasks": [1], "length": 250, "file": "data/chunk-000/episode_000001.parquet"}
{"episode_index": 2, "tasks": [0, 2], "length": 400, "file": "data/chunk-000/episode_000002.parquet"}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `episode_index` | int | Episode 唯一索引 |
| `tasks` | list[int] | 任务索引列表（多任务场景） |
| `length` | int | Episode 包含的帧数 |
| `file` | string | Parquet 文件路径 |

### tasks.jsonl

任务定义文件（v2.0+），JSON Lines 格式。

```jsonl
{"task_index": 0, "task": "Push the T-shaped block to the target"}
{"task_index": 1, "task": "Move the block to the green area"}
{"task_index": 2, "task": "Stack two blocks"}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_index` | int | 任务唯一索引 |
| `task` | string | 任务自然语言描述 |

### stats.json

全局统计信息文件。

```json
{
  "observation.state": {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "std": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "min": [-1.57, -1.57, -1.57, -1.57, -1.57, -1.57, -1.57],
    "max": [1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
    "count": 169500
  },
  "action": {
    "mean": [0.0, 0.0],
    "std": [0.1, 0.1],
    "min": [-0.5, -0.5],
    "max": [0.5, 0.5],
    "count": 169500
  }
}
```

### episodes_stats.jsonl

分集统计信息（v2.1+），JSON Lines 格式。

```jsonl
{"episode_index": 0, "observation.state": {"mean": [...], "std": [...], "min": [...], "max": [...]}}
{"episode_index": 1, "observation.state": {"mean": [...], "std": [...], "min": [...], "max": [...]}}
```

## 视频数据存储

### 目录结构

视频按 chunk 和摄像头组织：

```
videos/
├── chunk-000/
│   ├── observation.images.top/
│   │   ├── episode_000000.mp4
│   │   ├── episode_000001.mp4
│   │   └── ...
│   ├── observation.images.wrist/
│   │   └── episode_000000.mp4
│   └── observation.images.side/
│       └── episode_000000.mp4
└── ...
```

### 视频编码参数

默认编码设置：
```python
{
    "codec": "libx264",      # H.264 编码器
    "pixel_format": "yuv420p",  # YUV 4:2:0 色彩空间
    "quality": "medium",     # 质量预设
    "crf": 23,               # 恒定质量因子（0-51，18-28 推荐）
    "preset": "medium",      # 编码速度/质量权衡
    "fps": 30,               # 帧率
    "width": 640,            # 视频宽度
    "height": 480            # 视频高度
}
```

### 替代编码器

**H.265/HEVC**（更好的压缩）：
```python
{
    "codec": "libx265",
    "crf": 28,  # H.265 的 CRF 范围不同
    "preset": "medium"
}
```

**AV1**（最新标准）：
```python
{
    "codec": "libsvtav1",
    "crf": 35,
    "preset": "medium"
}
```

### 性能对比

| 编码器 | 压缩率 | 编码速度 | 解码速度 | CPU 占用 | 推荐使用 |
|--------|--------|----------|----------|----------|----------|
| H.264 | 1.0x | 快 | 快 | 低 | 通用 |
| H.265 | 1.5-2.0x | 中等 | 中等 | 中 | 存储优化 |
| AV1 | 2.0-3.0x | 慢 | 中等 | 高 |  archival |

## 特征类型系统

### 数值特征

**一维数组**（如关节角度）：
```json
{
  "dtype": "float32",
  "shape": [7],
  "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
}
```

**二维数组**（如图像特征）：
```json
{
  "dtype": "float32",
  "shape": [100, 256],
  "names": ["height", "width"]
}
```

### 图像/视频特征

**视频存储**：
```json
{
  "dtype": "video",
  "shape": [480, 640, 3],
  "names": ["height", "width", "channels"],
  "info": {
    "video.height": 480,
    "video.width": 640,
    "video.fps": 30,
    "video.codec": "h264"
  }
}
```

**图像存储**：
```json
{
  "dtype": "image",
  "shape": [480, 640, 3],
  "names": ["height", "width", "channels"]
}
```

### 字符串特征

```json
{
  "dtype": "string",
  "shape": [],
  "names": []
}
```

### 字典特征（嵌套）

```json
{
  "observation.state.arm": {
    "dtype": "float32",
    "shape": [6],
    "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
  },
  "observation.state.gripper": {
    "dtype": "float32",
    "shape": [1],
    "names": ["gripper_position"]
  }
}
```

## 数据加载流程

### 初始化流程 (`LeRobotDataset.__init__`)

```python
# 1. 元数据加载
metadata = LeRobotDatasetMetadata.from_hub(repo_id)  # 或 from_local(path)

# 2. 版本兼容性检查
if metadata.codebase_version not in SUPPORTED_VERSIONS:
    raise VersionError(f"Unsupported version: {metadata.codebase_version}")

# 3. 特征验证
features = metadata.features
for key, feature in features.items():
    validate_feature(feature)

# 4. Parquet 文件加载
df = load_parquet_files(metadata.data_path)

# 5. 视频验证（如果包含视频）
if has_video_features:
    validate_videos(metadata.video_path)

# 6. 时间戳验证
validate_timestamps_consecutives(df)
```

### 数据访问流程 (`__getitem__`)

```python
def __getitem__(self, idx):
    # 1. 从 parquet 获取行数据
    row = self.hf_dataset[idx]

    # 2. 初始化数据字典
    data = {}

    # 3. 加载非视觉数据
    for key in self.non_visual_keys:
        data[key] = row[key]

    # 4. 加载视觉数据（视频帧）
    for video_key in self.video_keys:
        # 根据 timestamps 解码视频帧
        frame = decode_video_frame(
            video_path=row[video_key],
            timestamp=row["timestamp"],
            tolerance=1e-4
        )
        data[video_key] = frame

    # 5. 应用变换
    if self.image_transforms:
        for video_key in self.video_keys:
            data[video_key] = self.image_transforms(data[video_key])

    return data
```

### 批处理加载

支持多种加载模式：

**顺序加载**（Sequential）：
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)
```

**随机加载**（Random）：
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

**Episodic Random**：按 episode 随机，保持时间连续性
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,  # Episode 级别
    collate_fn=episode_collate_fn
)
```

## 统计信息

### 全局统计

```json
{
  "observation.state": {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "std": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "q01": [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
    "q99": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    "min": [-1.57, -1.57, -1.57, -1.57, -1.57, -1.57, -1.57],
    "max": [1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
    "count": 169500
  },
  "action": {
    "mean": [0.0, 0.0],
    "std": [0.1, 0.1],
    "q01": [-0.4, -0.4],
    "q99": [0.4, 0.4],
    "min": [-0.5, -0.5],
    "max": [0.5, 0.5],
    "count": 169500
  }
}
```

### 分集统计

```jsonl
{"episode_index": 0, "observation.state": {"mean": [...], "std": [...]}, "action": {"mean": [...], "std": [...]}}
{"episode_index": 1, "observation.state": {"mean": [...], "std": [...]}, "action": {"mean": [...], "std": [...]}}
```

### 统计计算

**数值特征**：
- 完整统计：mean, std, min, max, q01, q99, count
- 增量更新：支持合并多个数据集的统计信息

**图像特征**：
- 智能采样：每 1000 帧采样 1 帧计算统计
- 内存优化：避免加载所有图像到内存

**计算示例**：
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    "my_dataset",
    fps=30,
    robot_type="koch"
)

# 添加数据后计算统计
dataset.stats = compute_stats(dataset)
dataset.save()
```

## 版本兼容性

### 支持的版本

当前支持的格式版本：
```python
SUPPORTED_VERSIONS = ["v1.6", "v2.0", "v2.1"]
```

### 版本检测

```python
from lerobot.datasets.lerobot_dataset_metadata import LeRobotDatasetMetadata

metadata = LeRobotDatasetMetadata.from_path("path/to/dataset")
print(f"Dataset version: {metadata.codebase_version}")
```

### 向前兼容

加载旧版本数据集时自动升级：
```python
# 内部处理流程
if metadata.codebase_version == "v1.6":
    # 透明转换 v1.6 -> v2.0
    metadata = transparent_upgrade(metadata)
elif metadata.codebase_version == "v2.0":
    # 直接使用
    pass
```

### 向后兼容错误

检测到不兼容版本时抛出详细错误：
```
ValueError: Dataset uses format v1.5, but this version of LeRobot
only supports v1.6+. Please upgrade your dataset using:
    from lerobot.datasets.v2.convert_dataset_v1_to_v2 import convert_dataset
    convert_dataset("old/path", "new/path")
```

## 最佳实践

### 创建数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="lerobot/my_dataset",
    fps=30,
    robot_type="koch",
    robot_type_to_config_name_mapping={},
    image_writer_processes=6,  # 视频编码进程数
    use_videos=True,           # 使用视频存储
    video=True
)

# 添加 episode
for episode_idx in range(num_episodes):
    episode_data = {
        "observation.state": states,  # shape: (T, state_dim)
        "observation.images.top": frames,  # shape: (T, H, W, 3)
        "action": actions,  # shape: (T, action_dim)
        "timestamp": timestamps
    }
    dataset.add_episode(episode_data, task=task)

# 计算统计信息并保存
dataset.consolidate()
dataset.stats = compute_stats(dataset)
dataset.save()
```

### 优化视频编码

```python
# 使用硬件加速（如果可用）
encoding = {
    "codec": "h264_nvenc",  # NVIDIA GPU 加速
    # 或 "h264_qsv"  # Intel Quick Sync
    # 或 "h264_videotoolbox"  # Apple Silicon
    "preset": "p4",
    "cq": 28
}

dataset = LeRobotDataset.create(
    repo_id="my_dataset",
    fps=30,
    robot_type="koch",
    video=True,
    encoding=encoding
)
```

### 数据验证

```python
# 验证数据集完整性
from lerobot.datasets.utils import verify_checksum

if not verify_checksum(dataset):
    raise ValueError("Dataset checksum mismatch!")

# 验证时间戳连续性
from lerobot.datasets.utils import validate_timestamps

issues = validate_timestamps(dataset)
if issues:
    print(f"Found {len(issues)} timestamp issues")
```

### 高效数据加载

```python
# 使用多进程加载
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # 根据 CPU 核心数调整
    pin_memory=True,  # 加速 GPU 传输
    prefetch_factor=2
)

# 使用 episode 采样
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=EpisodeSampler(dataset, batch_size=32),
    num_workers=4
)
```

### 迁移学习准备

```python
# 标准化数据（使用预计算统计）
class NormalizeTransform:
    def __init__(self, stats):
        self.stats = stats

    def __call__(self, data):
        for key in data:
            if key in self.stats:
                mean = self.stats[key]["mean"]
                std = self.stats[key]["std"]
                data[key] = (data[key] - mean) / std
        return data

# 应用到数据集
dataset.image_transforms = NormalizeTransform(dataset.stats)
```

### 故障排除

**视频加载慢**：
- 使用 TorchCodec 后端（更快）
- 减少 num_workers（避免 I/O 竞争）
- 检查磁盘性能（SSD 推荐）

**内存不足**：
- 使用视频存储而非图像存储
- 减少 batch_size
- 降低视频分辨率

**时间戳不匹配**：
- 检查摄像头同步
- 调整 tolerance 参数
- 验证视频帧率一致性

### 数据集发布

```python
# 推送到 HuggingFace Hub
dataset.push_to_hub(
    "lerobot/my_dataset",
    private=False,
    tags=["koch", "pusht", "v2.1"]
)

# 包含视频文件（默认为 lfs）
dataset.push_to_hub(
    "lerobot/my_dataset",
    patterns=["**/*.mp4", "**/*.parquet"]
)
```

## 相关文件和模块

### 核心实现
- `src/lerobot/common/datasets/lerobot_dataset.py` - 主数据集类
- `src/lerobot/common/datasets/lerobot_dataset_metadata.py` - 元数据管理
- `src/lerobot/common/datasets/video_utils.py` - 视频编解码工具
- `src/lerobot/common/datasets/utils.py` - 数据集工具函数
- `src/lerobot/common/datasets/compute_stats.py` - 统计计算

### 数据格式版本
- `src/lerobot/common/datasets/v1/`- 旧格式支持
- `src/lerobot/common/datasets/v2/`- 新格式实现
- `src/lerobot/datasets/v2/convert_dataset_v1_to_v2.py`- 版本转换

### 示例和测试
- `tests/test_datasets.py` - 数据集测试
- `tests/artifacts/` - 测试数据集
- `examples/` - 使用示例

### 配置和定义
- `src/lerobot/common/datasets/factory.py` - 数据集工厂
- `src/lerobot/common/policies/factory.py` - 策略工厂
- `src/lerobot/configs/` - 配置文件

## 技术细节补充

### Parquet 文件优势
- **列式存储**：只读取需要的列，节省 I/O
- **压缩**：Snappy 压缩，节省存储空间
- **向量化**：批量操作，加速处理
- **模式演进**：支持添加新列而不破坏旧代码

### 视频时间戳对齐
精确对齐算法：
```python
def find_nearest_frame(video_timestamps, target_timestamp, tolerance=1e-4):
    """
    找到最接近目标时间戳的帧索引
    """
    differences = np.abs(video_timestamps - target_timestamp)
    nearest_idx = np.argmin(differences)

    if differences[nearest_idx] > tolerance:
        raise TimestampMismatchError(
            f"No frame within tolerance {tolerance} of timestamp {target_timestamp}"
        )

    return nearest_idx
```

### Chunk 机制
Chunk 设计的目的：
- **性能**：每个目录文件数不过多（避免文件系统性能下降）
- **可管理性**：便于数据管理和迁移
- **并行化**：支持按 chunk 并行处理

默认 chunk 大小：1000 episodes
计算公式：`chunk_idx = episode_index // 1000`

### 内存映射（Memory Mapping）
对于大数据集：
- 使用 memory-mapped parquet 读取
- 延迟加载视频帧
- 避免一次性加载整个数据集到内存

### 数据完整性
- **Checksum**：计算重要文件的哈希值
- **验证**：加载时验证数据完整性
- **恢复**：损坏数据的恢复机制

---

**最后更新**: 2025-12-01
**版本**: v2.1
**作者**: LeRobot Team
**文档版本**: 1.0
