# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在操作此代码仓库时提供指导。

## 项目概述

LeRobot 是 Hugging Face 开发的最先进的用于真实世界机器人技术的机器学习库，基于 PyTorch 构建。它通过提供模型、数据集和工具来降低机器人技术的入门门槛，使每个人都能从共享数据集和预训练模型中贡献和受益。

## 常用命令

### 安装
```bash
# 开发环境安装（包含所有额外功能）
uv sync --extra "dev" --extra "test" --extra "all"

# 基础安装
pip install -e "."

# 安装特定功能
pip install -e ".[aloha,pusht,xarm]"  # 模拟环境
pip install -e ".[dynamixel,feetech]"  # 电机控制器
pip install -e ".[intelrealsense]"    # 相机
```

### 测试
```bash
# 快速测试（提交代码前运行）
uv run pytest tests -vv --maxfail=10

# 运行特定测试文件
uv run pytest tests/test_policies.py -vv

# 端到端策略测试
make test-end-to-end DEVICE=cuda  # 或 DEVICE=cpu

# 单独的策略 E2E 测试
make test-act-ete-train
make test-diffusion-ete-eval
make test-tdmpc-ete-train
make test-smolvla-ete-train
```

### 训练和评估
```bash
# 训练策略（以 ACT 为例）
python -m lerobot.scripts.train \
    --policy.type=act \
    --policy.dim_model=512 \
    --env.type=aloha \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --output_dir=outputs/act/

# 评估策略
python -m lerobot.scripts.eval \
    --policy.path=outputs/act/checkpoints/000004/pretrained_model \
    --env.type=aloha

# 可视化数据集
python -m lerobot.scripts.visualize_dataset \
    --repo_id=lerobot/aloha_sim_transfer_cube_human
```

### 硬件工具
```bash
# 查找连接的相机
lerobot-find-cameras

# 校准机器人
lerobot-calibrate

# 录制演示数据
lerobot-record --robot.type=koch

# 遥操作机器人
lerobot-teleoperate --robot.type=aloha
```

### 代码质量
```bash
# 运行 pre-commit 钩子
pre-commit run --all-files

# 使用 ruff 格式化代码
ruff format .

# 代码检查
ruff check .

# 类型检查（目前在 CI 中禁用）
# mypy src/lerobot
```

## 架构概述

### 核心模块（`src/lerobot/`）

**策略** (`policies/`): 模仿学习算法
- `act.py`: 用于双手操作的 Action Chunking Transformers
- `diffusion.py`: 基于状态的扩散策略
- `tdmpc.py`: 轨迹分布式模型预测控制
- `vqbet.py`: 向量量化行为变换器
- `smolvla.py`: 小型视觉-语言-动作模型
- `pi0.py`: 视觉-语言-动作模型

**机器人** (`robots/`): 硬件抽象层
- `koch.py`: 单臂 Koch 机器人
- `aloha.py`: 双手 ALOHA 配置
- `so100.py`, `so101.py`: 经济实惠的 SO 系列机械臂
- `hope_jr.py`: 灵巧的手-臂系统
- `lekiwi.py`: 移动机械臂

**数据集** (`datasets/`): 机器人数据集处理
- 针对时序序列优化的自定义数据集格式
- 多模态观测（图像、状态、动作）
- 与 Hugging Face 数据集集成
- 图像增强流水线

**环境** (`envs/`): 模拟接口
- ALOHA: 双手操作任务
- PushT: 基于视觉的推动任务
- XArm: 单臂操作

**硬件** (`cameras/`, `motors/`)
- 相机接口：OpenCV、Intel RealSense
- 电机控制器：Dynamixel、Feetech
- 自动设备发现和校准

### 关键设计模式

1. **策略-环境接口**: 所有策略实现通用接口 `select_action()`，可与任何环境配合使用
2. **数据集格式**: 用于机器人数据集的标准化格式，支持时序序列
3. **硬件抽象**: 相机和电机的通用接口，便于更换
4. **配置管理**: 使用 Hydra 进行分层配置管理
5. **分布式训练**: 内置多 GPU 训练和检查点保存支持

### 测试策略

- **单元测试**: 使用模拟依赖的组件级测试
- **集成测试**: 环境和硬件集成测试
- **端到端测试**: 各策略的完整训练/评估流程
- **硬件测试**: 检测可用硬件的可选测试
- **性能测试**: 视频处理和推理的基准测试

### 重要约定

1. **设备处理**: 始终检查 `policy.device` 和 `env.device` 的一致性
2. **图像变换**: 使用数据集级变换以保证训练一致性
3. **动作空间**: 策略输出特定于环境格式的动作
4. **检查点**: 模型保存 config.json 以便可复现地恢复训练
5. **日志记录**: 使用 wandb 进行实验跟踪，rerun 进行可视化
6. **错误处理**: 对缺失的硬件组件提供优雅回退

## 常见开发任务

### 添加新策略
1. 在 `src/lerobot/policies/` 中创建策略文件
2. 实现必需的方法：`__init__`、`forward`、`select_action`、`reset`
3. 在 `src/lerobot/common/policies/` 中添加配置数据类
4. 在策略工厂中注册
5. 在 Makefile 中添加 E2E 测试

### 添加新机器人
1. 在 `src/lerobot/robots/` 中创建机器人文件
2. 使用校准和遥操作实现 `Robot` 接口
3. 在 `src/lerobot/common/robot_devices/motors/` 中添加电机配置
4. 创建校准和设置脚本
5. 添加示例配置

### 添加新环境
1. 在 `src/lerobot/envs/` 中创建环境包装器
2. 实现 gymnasium 接口
3. 添加到环境工厂
4. 如有需要，创数据集转换脚本
5. 为现有策略添加 E2E 测试

## 调试技巧

- **相机问题**: 使用 `lerobot-find-cameras` 检测连接的设备
- **电机问题**: 使用 `lerobot-find-port` 检查串口连接
- **训练问题**: 启用 wandb 日志记录并检查数据集可视化
- **模拟问题**: 设置 `MUJOCO_GL=egl` 用于无头渲染
- **硬件问题**: 操作前运行校准脚本

## 性能考虑

- **GPU 内存**: 对于大型模型，减小 batch_size 或使用梯度累积
- **数据加载**: 使用多个工作进程进行数据集加载
- **视频 处理**: torchcodec 提供更快的视频解码
- **分布式训练**: 使用 accelerate 进行多 GPU 设置
- **推理**: 评估时使用 `policy.eval()` 模式和 `torch.no_grad()`
