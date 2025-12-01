# CLAUDE.md

该文件为 Claude Code (claude.ai/code) 提供关于本仓库代码的操作指导。

## 项目概述

LeRobot 是 Hugging Face 开发的基于 PyTorch 的先进机器学习库，用于真实世界机器人应用。它提供模型、数据集和工具，专注于模仿学习和强化学习，旨在通过共享数据集和预训练模型降低机器人技术的入门门槛。

## 常用命令

### 开发环境搭建
```bash
# 创建虚拟环境
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# 以开发模式安装
pip install -e .

# 安装仿真环境（推荐）
pip install -e ".[aloha, pusht]"

# 安装所有扩展功能
pip install -e ".[all]"
```

### 测试
```bash
# 运行所有测试
python -m pytest -sv ./tests

# 运行指定测试
pytest tests/<TEST_TO_RUN>.py

# 运行端到端测试（通过 Makefile）
make test-end-to-end DEVICE=cpu
```

### 代码质量检查
```bash
# 运行预提交钩子
pre-commit run --all-files

# 类型检查（仅部分模块启用）
mypy src/lerobot/configs
mypy src/lerobot/model
mypy src/lerobot/envs
mypy src/lerobot/cameras
```

### 训练与评估
```bash
# 训练策略
lerobot-train --config_path=lerobot/diffusion_pusht

# 评估策略
lerobot-eval --policy.path=<模型路径> --env.type=<环境类型>

# 可视化数据集
lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0
```

## 架构概览

### 核心模块 (`src/lerobot/`)

**policies/**: 各种策略算法的实现
- `act.py`: Action Chunking with Transformers (ACT) 策略
- `diffusion.py`: 用于机器人操作的 Diffusion 策略
- `tdmpc.py`: 时序差分模型预测控制
- `smolvla.py`: 小型视觉-语言-动作模型
- `groot.py`: Gr00t 策略实现
- `vq_bet.py`: 向量量化行为Transformer

**datasets/**: 数据集处理和加载
- `lerobot_dataset.py`: 用于加载和管理机器人数据的主数据集类
- `compute_stats.py`: 数据集统计信息计算
- 支持通过 `delta_timestamps` 进行时序查询，获取多帧观测

**envs/**: 仿真环境接口
- 集成 gymnasium 环境（aloha、pusht、xarm）
- 提供标准化的训练和评估接口

**robots/**: 真实机器人实现
- `so100.py`, `so101.py`: SO-100/SO-101 机械臂
- `hopejr.py`: HopeJR 人形机械臂和手
- `reachy2.py`: Reachy2 机器人实现
- `lekiwi.py`: 移动机器人平台

**cameras/**: 相机接口和工具
- 支持 RealSense、OpenCV 等相机类型
- 图像预处理和变换工具

**motors/**: 电机控制接口
- `feetech.py`: Feetech 伺服电机
- `dynamixel.py`: Dynamixel 伺服电机
- `mock_motors.py`: 用于测试的模拟电机实现

**teleoperators/**: 遥操作接口
- 游戏手柄控制器
- 外骨骼接口
- 基于手机的遥操作

### 关键设计模式

1. **配置系统**: 使用 dataclass 结合 draccus 进行配置管理
2. **策略接口**: 所有策略继承自基础策略类，具有标准化的 `select_action` 方法
3. **数据集格式**: LeRobotDataset 格式，使用 parquet 存储元数据，MP4 存储视频
4. **模块化扩展**: 按功能组织的可选依赖（机器人、策略、仿真等）

### 入口点

库提供多个命令行脚本（在 `pyproject.toml` 中定义）：
- `lerobot-train`: 训练策略
- `lerobot-eval`: 评估训练好的策略
- `lerobot-record`: 录制演示数据
- `lerobot-teleoperate`: 遥操作机器人
- `lerobot-dataset-viz`: 可视化数据集
- `lerobot-calibrate`: 校准机器人系统

### 测试策略

- `tests/` 目录中的单元测试，按模块组织
- 硬件组件的模拟实现
- 通过 Makefile 进行策略训练/评估的端到端测试
- CI 使用所有扩展运行测试以确保兼容性

### 开发工作流程

1. 安装开发依赖: `pip install -e ".[dev]"`
2. 设置预提交钩子: `pre-commit install`
3. 提交前运行测试: `pytest tests -xvs`
4. 特定模块启用类型检查（configs、model、envs、cameras）
5. 使用 ruff 进行代码格式化（行长度: 110 字符）