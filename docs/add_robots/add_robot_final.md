# LeRobot 机器人集成开发完整指南

> 本文档整合了多份开发指南的精华，详细介绍如何向 LeRobot 项目添加新的机器人支持，包括机器人类实现、电机控制器集成、相机系统集成、遥操作支持、测试与调试。

## 目录

1. [架构概述](#1-架构概述)
2. [前置准备](#2-前置准备)
3. [Robot 基类接口详解](#3-robot-基类接口详解)
4. [添加新机器人的详细步骤](#4-添加新机器人的详细步骤)
5. [电机控制详解](#5-电机控制详解)
6. [相机集成详解](#6-相机集成详解)
7. [遥操作设备集成](#7-遥操作设备集成)
8. [完整代码模板](#8-完整代码模板)
9. [配置与 CLI 示例](#9-配置与-cli-示例)
10. [测试指南](#10-测试指南)
11. [常见问题与调试技巧](#11-常见问题与调试技巧)
12. [最佳实践](#12-最佳实践)
13. [提交前检查清单](#13-提交前检查清单)

---

## 1. 架构概述

### 1.1 LeRobot 机器人架构设计

LeRobot 采用分层、模块化的架构设计，确保不同类型的机器人可以通过统一的接口集成：

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层（脚本）                           │
│         teleoperate / record / train / eval                 │
├─────────────────────────────────────────────────────────────┤
│                   Robot 基类接口                            │
│  connect / disconnect / get_observation / send_action       │
├─────────────────────────────────────────────────────────────┤
│           硬件抽象层（电机控制器 + 相机）                    │
│  MotorsBus (Dynamixel/Feetech) + Camera (OpenCV/Realsense) │
├─────────────────────────────────────────────────────────────┤
│                    物理机器人硬件                            │
│               机械臂 / 手 / 移动底盘                        │
└─────────────────────────────────────────────────────────────┘
```

**组件层次结构**：

```
Robot (机器人)
├── MotorsBus (电机总线)
│   └── Motor (电机) × N
├── Camera (相机) × N
└── Teleoperator (遥操作设备，可选)
```

### 1.2 核心组件说明

| 组件 | 路径 | 职责 |
|------|------|------|
| `Robot` 基类 | `src/lerobot/robots/robot.py` | 所有机器人的抽象基类，定义必须实现的接口 |
| `RobotConfig` | `src/lerobot/robots/config.py` | 配置抽象，使用装饰器注册 |
| `make_robot_from_config` | `src/lerobot/robots/utils.py` | 工厂函数，根据 `config.type` 构造具体机器人 |
| `MotorsBus` | `src/lerobot/motors/motors_bus.py` | 电机总线抽象 |
| `Camera` | `src/lerobot/cameras/camera.py` | 相机接口抽象 |
| `Teleoperator` | `src/lerobot/teleoperators/teleoperator.py` | 遥操作设备抽象 |

### 1.3 目录结构

```
src/lerobot/
├── robots/                    # 机器人实现
│   ├── robot.py              # Robot 基类
│   ├── config.py             # RobotConfig 基类
│   ├── utils.py              # make_robot_from_config 工厂函数
│   └── <robot_name>/         # 具体机器人目录
│       ├── __init__.py
│       ├── config_<robot_name>.py
│       └── <robot_name>.py
├── motors/                    # 电机控制
│   ├── motors_bus.py         # MotorsBus 基类
│   ├── feetech/              # Feetech 电机
│   └── dynamixel/            # Dynamixel 电机
├── cameras/                   # 相机系统
│   ├── camera.py             # Camera 基类
│   ├── opencv/               # OpenCV 相机
│   └── realsense/            # Intel RealSense
└── teleoperators/            # 遥操作设备
    ├── teleoperator.py       # Teleoperator 基类
    └── <teleop_name>/        # 具体遥操作器
```

### 1.4 设计原则

- **抽象基类约束**：所有机器人必须继承 `Robot` 基类并实现其抽象方法
- **配置驱动**：通过 `@dataclass` 配置类管理所有参数
- **注册机制**：使用 `@RobotConfig.register_subclass()` 装饰器自动注册
- **松耦合**：电机、相机、遥操作器可独立配置和替换
- **类型安全**：使用 Python 类型提示和 dataclass 配置，确保编译时检查

---

## 2. 前置准备

### 2.1 开发环境要求

```bash
# Python 版本要求
Python >= 3.10

# 推荐安装方式（使用 uv）
uv sync --extra dev --extra test

# 或使用 pip
pip install -e ".[dev,test]"
```

### 2.2 硬件依赖

**电机控制器**：

| 电机类型 | Python 包 | 安装命令 |
|---------|----------|---------|
| Feetech STS/SCS | `scservo-sdk` | `pip install scservo-sdk` |
| Dynamixel | `dynamixel-sdk` | `pip install dynamixel-sdk` |

**相机**：

| 相机类型 | 依赖 | 安装命令 |
|---------|------|---------|
| OpenCV 相机 | `opencv-python` | 已包含在基础依赖 |
| Intel RealSense | `pyrealsense2` | `pip install -e ".[intelrealsense]"` |

### 2.3 权限配置（Linux）

```bash
# 添加用户到 dialout 组（需要重新登录生效）
sudo usermod -aG dialout $USER

# 或临时授权
sudo chmod 666 /dev/ttyUSB0
```

### 2.4 硬件检测工具

```bash
# 查找连接的相机
lerobot-find-cameras

# 查找电机控制器端口
lerobot-find-port
```

### 2.5 准备信息清单

在开始开发前，请确认以下信息：

- [ ] 每个电机的 ID、型号、驱动模式
- [ ] 串口路径和波特率
- [ ] 行程限制（关节角度范围）
- [ ] 相机分辨率、帧率
- [ ] 是否需要力矩关闭、限幅参数
- [ ] URDF 文件（如需末端控制）

---

## 3. Robot 基类接口详解

### 3.1 必须实现的抽象属性和方法

| 方法/属性 | 类型 | 说明 | 参考实现 |
|-----------|------|------|----------|
| `config_class` | 类变量 | 配置类类型 | `config_class = MyArmConfig` |
| `name` | 类变量 | 机器人唯一标识 | `name = "my_arm"` |
| `observation_features` | 属性 | 观测特征字典 | `SO100Follower._motors_ft/_cameras_ft` |
| `action_features` | 属性 | 动作特征字典 | 通常与电机观测一致 |
| `is_connected` | 属性 | 连接状态 | 合并电机和相机状态 |
| `is_calibrated` | 属性 | 校准状态 | 委托给 `bus.is_calibrated` |
| `connect()` | 方法 | 建立连接 | 检查状态 → 连接电机 → 校准 → 连接相机 → 配置 |
| `calibrate()` | 方法 | 执行校准 | 半圈零点 + 行程扫描 |
| `configure()` | 方法 | 配置参数 | 设置电机模式、PID 等 |
| `get_observation()` | 方法 | 获取观测 | `sync_read` + `cam.async_read()` |
| `send_action()` | 方法 | 发送动作 | 限幅检查 + `sync_write` |
| `disconnect()` | 方法 | 断开连接 | 关闭力矩 + 断开相机 |

### 3.2 接口签名详解

```python
from lerobot.robots import Robot, RobotConfig

class MyRobot(Robot):
    config_class: type[RobotConfig]  # 配置类
    name: str                         # 机器人唯一标识

    # ========== 抽象属性 ==========

    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        """
        返回观测特征字典，描述 get_observation() 返回值的结构。

        格式: {"特征名": 类型或形状}
        示例: {
            "shoulder_pan.pos": float,
            "camera_top": (480, 640, 3)  # HWC 格式
        }
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        返回动作特征字典，描述 send_action() 期望的输入结构。

        格式: {"特征名": 类型}
        示例: {"shoulder_pan.pos": float, "gripper.pos": float}
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """返回机器人是否已连接（需同时检查电机和相机）"""
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """返回机器人是否已校准（如不需要校准则始终返回 True）"""
        pass

    # ========== 抽象方法 ==========

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        建立与机器人的连接。

        Args:
            calibrate: 是否在连接后自动校准

        Raises:
            DeviceAlreadyConnectedError: 如果已连接
        """
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """执行机器人校准（如不需要则为空实现）"""
        pass

    @abc.abstractmethod
    def configure(self) -> None:
        """配置机器人参数（电机模式、PID等）"""
        pass

    @abc.abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """
        获取当前观测值。

        Returns:
            dict: 键与 observation_features 一致

        Raises:
            DeviceNotConnectedError: 如果未连接
        """
        pass

    @abc.abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        发送动作命令到机器人。

        Args:
            action: 键与 action_features 一致

        Returns:
            dict: 实际发送的动作（可能被安全限制修改）

        Raises:
            DeviceNotConnectedError: 如果未连接
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """
        断开连接并释放资源。

        Raises:
            DeviceNotConnectedError: 如果未连接
        """
        pass
```

### 3.3 基类提供的辅助功能

`Robot` 基类自动处理校准文件管理：

```python
class Robot(abc.ABC):
    def __init__(self, config: RobotConfig):
        self.id = config.id  # 机器人实例 ID
        # 校准文件路径：~/.cache/huggingface/lerobot/calibration/robots/<name>/<id>.json
        self.calibration_dir = config.calibration_dir or HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()  # 自动加载已有校准

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """从 JSON 加载校准数据"""
        ...

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """保存校准数据到 JSON"""
        ...

---

## 4. 添加新机器人的详细步骤

### 4.1 步骤 1：创建目录结构

```bash
# 以添加 "my_arm" 机器人为例
mkdir -p src/lerobot/robots/my_arm
touch src/lerobot/robots/my_arm/__init__.py
touch src/lerobot/robots/my_arm/config_my_arm.py
touch src/lerobot/robots/my_arm/my_arm.py
```

目录结构示例：

```
my_arm/
├── __init__.py                 # 模块初始化，导出类和配置
├── config_my_arm.py            # 配置类定义
└── my_arm.py                   # 主机器人类实现
```

### 4.2 步骤 2：定义配置类

在 `config_my_arm.py` 中：

```python
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("my_arm")  # 注册名称，用于 CLI 和工厂函数
@dataclass
class MyArmConfig(RobotConfig):
    """MyArm 机器人配置"""

    # 必需参数
    port: str = "/dev/ttyUSB0"  # 电机串口

    # 可选参数
    id: str = "my_arm"
    calibration_dir: Path | None = None

    # 相机配置（可选）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 安全参数
    max_relative_target: float | None = None  # 最大相对目标位置限制
    disable_torque_on_disconnect: bool = True  # 断开时禁用扭矩

    # 机器人特定参数
    use_degrees: bool = False  # 使用度数而不是 RANGE_M100_100
    protocol_version: int = 0  # Feetech 协议版本

    def __post_init__(self):
        """参数验证"""
        if not self.port:
            raise ValueError("port 不能为空")
```

**配置字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `port` | `str` | 电机控制器串口路径 |
| `id` | `str` | 机器人实例 ID，用于区分多台同型号机器人 |
| `cameras` | `dict[str, CameraConfig]` | 相机配置字典 |
| `max_relative_target` | `float \| None` | 目标位置与当前位置的最大差值限制 |
| `disable_torque_on_disconnect` | `bool` | 断开时是否禁用扭矩（保护机器人） |

### 4.3 步骤 3：实现机器人类

在 `my_arm.py` 中创建机器人类（完整模板见第 8 节）：

```python
#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_my_arm import MyArmConfig

logger = logging.getLogger(__name__)


class MyArm(Robot):
    """MyArm 机器人实现示例"""

    config_class = MyArmConfig
    name = "my_arm"

    def __init__(self, config: MyArmConfig):
        super().__init__(config)
        self.config = config

        # 根据配置确定归一化模式
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # 初始化电机总线
        self.bus = FeetechMotorsBus(
            port=config.port,
            protocol_version=config.protocol_version,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode),
                "shoulder_lift": Motor(2, "sts3215", norm_mode),
                "elbow_flex": Motor(3, "sts3215", norm_mode),
                "wrist_flex": Motor(4, "sts3215", norm_mode),
                "wrist_roll": Motor(5, "sts3215", norm_mode),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # 初始化相机
        self.cameras = make_cameras_from_configs(config.cameras)

    # ... 实现其他抽象方法（见第 8 节完整模板）
```

### 4.4 步骤 4：设置 `__init__.py` 导出

在 `src/lerobot/robots/my_arm/__init__.py` 中：

```python
from .config_my_arm import MyArmConfig
from .my_arm import MyArm

__all__ = ["MyArm", "MyArmConfig"]
```

### 4.5 步骤 5：注册到工厂函数

编辑 `src/lerobot/robots/utils.py`，添加你的机器人：

```python
def make_robot_from_config(config: RobotConfig) -> Robot:
    # ... 现有代码 ...
    elif config.type == "my_arm":
        from .my_arm import MyArm
        return MyArm(config)
    # ... 其他代码 ...
```

---

## 5. 电机控制详解

### 5.1 Motor 数据类

```python
from lerobot.motors import Motor, MotorNormMode

# 电机定义
motor = Motor(
    id=1,                              # 电机 ID（硬件设定）
    model="sts3215",                   # 电机型号
    norm_mode=MotorNormMode.RANGE_M100_100  # 归一化模式
)
```

**归一化模式说明**：

| 模式 | 输出范围 | 用途 | 示例 |
|------|----------|------|------|
| `RANGE_0_100` | [0, 100] | 夹爪等单向运动 | 夹爪开合 |
| `RANGE_M100_100` | [-100, 100] | 关节旋转 | 肩、肘、腕关节 |
| `DEGREES` | 角度值 | 直接使用角度 | 需要精确角度控制时 |

### 5.2 MotorsBus 核心方法

```python
from lerobot.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors={...})

# 连接管理
bus.connect()
bus.disconnect(disable_torque=True)

# 力矩控制
bus.enable_torque()
bus.disable_torque()

# 同步读写（推荐，效率高）
positions = bus.sync_read("Present_Position")  # 返回 {motor_name: value}
bus.sync_write("Goal_Position", {motor_name: value})

# 单电机读写
value = bus.read("Present_Position", "shoulder_pan")
bus.write("Goal_Position", "shoulder_pan", 2048)

# 校准相关
bus.is_calibrated  # 校准状态
bus.write_calibration(calibration_dict)  # 写入校准数据
bus.set_half_turn_homings()  # 设置中位
bus.record_ranges_of_motion()  # 记录运动范围

# 上下文管理器（配置时使用）
with bus.torque_disabled():
    bus.configure_motors()
    bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
```

### 5.3 支持的电机型号

**Feetech 系列**（`src/lerobot/motors/feetech/`）：

| 型号 | 特点 |
|------|------|
| STS3215 | 常用伺服，支持位置/速度模式 |
| STS3250 | 高扭矩版本 |
| SCS0009 | 小型伺服 |

> ⚠️ **注意**：Protocol 1（如 `HopeJrHand`）不支持 `sync_read`，需逐个 `read`。

**Dynamixel 系列**（`src/lerobot/motors/dynamixel/`）：

| 型号 | 特点 |
|------|------|
| XL330-M288 | 轻量级，适合小型机械臂 |
| XL430-W250 | 中型伺服 |
| XM540-W270 | 高性能 |

### 5.4 校准流程

典型的校准流程：

```python
def calibrate(self) -> None:
    # 1. 检查是否有已存在的校准文件
    if self.calibration:
        user_input = input("发现已有校准，按 Enter 使用，或输入 'c' 重新校准: ")
        if user_input.strip().lower() != "c":
            self.bus.write_calibration(self.calibration)
            return

    # 2. 禁用力矩，允许手动移动
    self.bus.disable_torque()
    for motor in self.bus.motors:
        self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # 3. 记录中位
    input("将机械臂移动到运动范围中点，按 Enter 继续...")
    homing_offsets = self.bus.set_half_turn_homings()

    # 4. 记录运动范围
    print("依次移动各关节到极限位置...")
    range_mins, range_maxes = self.bus.record_ranges_of_motion()

    # 5. 构建校准数据
    self.calibration = {}
    for motor, m in self.bus.motors.items():
        self.calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,  # 0: 正常, 1: 反转
            homing_offset=homing_offsets[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )

    # 6. 保存校准
    self.bus.write_calibration(self.calibration)
    self._save_calibration()
```

---

## 6. 相机集成详解

### 6.1 相机配置

```python
from lerobot.cameras import OpenCVCameraConfig, IntelRealSenseCameraConfig

# OpenCV 相机（USB 相机、内置摄像头）
opencv_config = OpenCVCameraConfig(
    fps=30,
    width=640,
    height=480,
    index_or_path=0,  # 相机索引或设备路径
    rotation=None,    # 可选：Cv2Rotation.ROTATE_90_CLOCKWISE
    color_mode="rgb", # 可选：rgb, bgr, gray
)

# Intel RealSense
realsense_config = IntelRealSenseCameraConfig(
    fps=30,
    width=640,
    height=480,
    serial_number="123456789",  # 可选
)
```

> ⚠️ **重要**：`RobotConfig.__post_init__` 会校验 `width/height/fps` 非空，缺失会报错。

### 6.2 在机器人中使用相机

```python
from lerobot.cameras import make_cameras_from_configs

class MyRobot(Robot):
    def __init__(self, config):
        super().__init__(config)
        # 根据配置创建相机实例
        self.cameras = make_cameras_from_configs(config.cameras)

    def connect(self, calibrate=True):
        # 连接相机
        for cam in self.cameras.values():
            cam.connect(warmup=True)

    def get_observation(self):
        obs = {}
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()  # 返回 np.ndarray (H, W, C)
        return obs

    def disconnect(self):
        for cam in self.cameras.values():
            cam.disconnect()
```

### 6.3 查找可用相机

```bash
# CLI 命令
lerobot-find-cameras

# Python API
from lerobot.cameras.opencv import OpenCVCamera
cameras = OpenCVCamera.find_cameras()
print(cameras)  # [{"index": 0, "name": "USB Camera"}, ...]
```

---

## 7. 遥操作设备集成

如果你的机器人需要遥操作支持（如主从控制），需要实现 `Teleoperator` 类。

### 7.1 Teleoperator 基类接口

```python
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

@TeleoperatorConfig.register_subclass("my_teleop")
@dataclass
class MyTeleopConfig(TeleoperatorConfig):
    port: str = "/dev/ttyUSB1"


class MyTeleop(Teleoperator):
    config_class = MyTeleopConfig
    name = "my_teleop"

    @property
    def action_features(self) -> dict:
        """遥操作器输出的动作特征（必须与目标机器人的 action_features 对齐）"""
        return {"shoulder_pan.pos": float, "gripper.pos": float}

    @property
    def feedback_features(self) -> dict:
        """遥操作器接收的反馈特征（如力反馈）"""
        return {}  # 不支持反馈则返回空字典

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        # 校准逻辑
        pass

    def configure(self) -> None:
        # 配置遥操作器（如禁用力矩让用户自由移动）
        self.bus.disable_torque()

    def get_action(self) -> dict[str, float]:
        """读取当前位置作为目标动作"""
        positions = self.bus.sync_read("Present_Position")
        return {f"{motor}.pos": val for motor, val in positions.items()}

    def send_feedback(self, feedback: dict) -> None:
        """发送反馈（如力反馈）"""
        pass  # 如不支持

    def disconnect(self) -> None:
        self.bus.disconnect()
```

### 7.2 注册遥操作器

编辑 `src/lerobot/teleoperators/utils.py`：

```python
def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    # ... 现有代码 ...
    elif config.type == "my_teleop":
        from .my_teleop import MyTeleop
        return MyTeleop(config)
```

### 7.3 遥操作匹配要点

> ⚠️ **关键**：确保 teleop 的 `action_features` 与目标机器人的 `action_features` 键名完全一致，否则会出现 `KeyError` 或关节不动。

---

## 8. 完整代码模板

以下是一个完整的机器人实现模板，基于 Feetech 电机的 6 自由度机械臂：

### 8.1 配置类 (`config_my_arm.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("my_arm")
@dataclass
class MyArmConfig(RobotConfig):
    """MyArm 机器人配置"""

    port: str = "/dev/ttyUSB0"
    id: str = "my_arm"
    calibration_dir: Path | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    max_relative_target: float | None = 15.0
    disable_torque_on_disconnect: bool = True
    use_degrees: bool = False
    protocol_version: int = 0
```

### 8.2 机器人类 (`my_arm.py`)

```python
#!/usr/bin/env python
"""MyArm 机器人实现"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_my_arm import MyArmConfig

logger = logging.getLogger(__name__)


class MyArm(Robot):
    """
    MyArm 机器人实现

    硬件配置：
    - 6 个 Feetech STS3215 电机
    - 可选相机
    """

    config_class = MyArmConfig
    name = "my_arm"

    def __init__(self, config: MyArmConfig):
        super().__init__(config)
        self.config = config

        # 根据配置确定归一化模式
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # 初始化电机总线
        self.bus = FeetechMotorsBus(
            port=config.port,
            protocol_version=config.protocol_version,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode),
                "shoulder_lift": Motor(2, "sts3215", norm_mode),
                "elbow_flex": Motor(3, "sts3215", norm_mode),
                "wrist_flex": Motor(4, "sts3215", norm_mode),
                "wrist_roll": Motor(5, "sts3215", norm_mode),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # 初始化相机
        self.cameras = make_cameras_from_configs(config.cameras)

    # ========== 特征属性 ==========

    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征：{电机名.pos: float}"""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征：{相机名: (H, W, C)}"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ========== 连接管理 ==========

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. 连接电机总线
        self.bus.connect()

        # 2. 检查校准状态
        if not self.is_calibrated and calibrate:
            logger.info("Calibration mismatch or no calibration file found")
            self.calibrate()

        # 3. 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # 4. 配置机器人参数
        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 断开电机总线
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # 断开所有相机
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    # ========== 校准 ==========

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """交互式校准流程"""
        if self.calibration:
            user_input = input(
                f"Press ENTER to use existing calibration for {self.id}, "
                "or type 'c' to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"Running calibration of {self}")
        self.bus.disable_torque()

        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move robot to mid-range position and press ENTER...")
        homing_offsets = self.bus.set_half_turn_homings()

        print("Move all joints through their full range. Press ENTER to stop...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    # ========== 配置 ==========

    def configure(self) -> None:
        """配置电机运行参数"""
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # 可选：设置 PID 参数
                # self.bus.write("P_Coefficient", motor, 16)
                # self.bus.write("I_Coefficient", motor, 0)
                # self.bus.write("D_Coefficient", motor, 32)

    # ========== 观测与动作 ==========

    def get_observation(self) -> dict[str, Any]:
        """获取当前状态"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs = {}

        # 读取电机位置
        positions = self.bus.sync_read("Present_Position")
        for motor, val in positions.items():
            obs[f"{motor}.pos"] = val

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read motors: {dt_ms:.1f}ms")

        # 读取相机图像
        for cam_name, cam in self.cameras.items():
            start = time.perf_counter()
            obs[cam_name] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_name}: {dt_ms:.1f}ms")

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送目标位置"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # 提取目标位置
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # 安全检查：限制相对移动距离
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        # 写入电机
        self.bus.sync_write("Goal_Position", goal_pos)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} send_action: {dt_ms:.1f}ms")

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

### 8.3 模块导出 (`__init__.py`)

```python
from .config_my_arm import MyArmConfig
from .my_arm import MyArm

__all__ = ["MyArm", "MyArmConfig"]
```

---

## 9. 配置与 CLI 示例

### 9.1 Python 配置示例

```python
from lerobot.robots.my_arm import MyArm, MyArmConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# 创建配置
config = MyArmConfig(
    id="lab-arm-01",
    port="/dev/ttyUSB0",
    max_relative_target=12.0,
    cameras={
        "front_rgb": OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
        ),
    },
)

# 创建机器人实例
robot = MyArm(config)

# 使用
robot.connect(calibrate=True)
obs = robot.get_observation()
robot.send_action({"shoulder_pan.pos": 0.0, "gripper.pos": 50.0})
robot.disconnect()
```

### 9.2 CLI 使用示例

```bash
# 遥操作
python -m lerobot.teleoperate \
    --robot.type=my_arm \
    --robot.port=/dev/ttyUSB0 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyUSB1

# 录制数据集
python -m lerobot.record \
    --robot.type=my_arm \
    --robot.port=/dev/ttyUSB0 \
    --dataset.repo-id=my_username/my_dataset \
    --teleop.type=gamepad

# 回放数据集
python -m lerobot.replay \
    --robot.type=my_arm \
    --robot.port=/dev/ttyUSB0 \
    --dataset.repo-id=my_username/my_dataset \
    --dataset.episode=0
```

### 9.3 YAML 配置文件示例

```yaml
# examples/my_arm/config.yaml

robot:
  type: my_arm
  id: my_arm_001
  port: /dev/ttyUSB0
  disable_torque_on_disconnect: true
  max_relative_target: 15.0

  cameras:
    top_camera:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
      color_mode: rgb

teleop:
  type: so100_leader
  port: /dev/ttyUSB1
```

---

## 10. 测试指南

### 10.1 单元测试结构

在 `tests/robots/` 创建测试文件：

```python
# tests/robots/test_my_arm.py

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.my_arm import MyArm, MyArmConfig


def _make_bus_mock() -> MagicMock:
    """创建电机总线 Mock"""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False
    bus.is_calibrated = True

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def robot():
    """创建带 Mock 的机器人实例"""
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        bus_mock.sync_read.return_value = {m: i * 0.1 for i, m in enumerate(bus_mock.motors)}
        bus_mock.sync_write.return_value = None
        return bus_mock

    with (
        patch("lerobot.robots.my_arm.my_arm.FeetechMotorsBus", side_effect=_bus_side_effect),
        patch.object(MyArm, "configure", lambda self: None),
    ):
        cfg = MyArmConfig(port="/dev/null")
        robot = MyArm(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(robot):
    """测试连接和断开"""
    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected


def test_double_connect_raises(robot):
    """测试重复连接抛出异常"""
    from lerobot.errors import DeviceAlreadyConnectedError

    robot.connect()
    with pytest.raises(DeviceAlreadyConnectedError):
        robot.connect()


def test_get_observation(robot):
    """测试观测获取"""
    robot.connect()
    obs = robot.get_observation()
    expected_keys = {f"{m}.pos" for m in robot.bus.motors}
    assert set(obs.keys()) == expected_keys


def test_send_action(robot):
    """测试动作发送"""
    robot.connect()
    action = {f"{m}.pos": i * 10.0 for i, m in enumerate(robot.bus.motors)}
    returned = robot.send_action(action)
    assert returned == action
    robot.bus.sync_write.assert_called_once()


def test_observation_features(robot):
    """测试观测特征定义"""
    features = robot.observation_features
    for motor in robot.bus.motors:
        assert f"{motor}.pos" in features
        assert features[f"{motor}.pos"] == float


def test_action_features(robot):
    """测试动作特征定义"""
    features = robot.action_features
    for motor in robot.bus.motors:
        assert f"{motor}.pos" in features
```

### 10.2 运行测试

```bash
# 运行特定测试
pytest tests/robots/test_my_arm.py -v

# 运行所有机器人测试
pytest tests/robots/ -v

# 带覆盖率报告
pytest tests/robots/test_my_arm.py --cov=src/lerobot/robots/my_arm

# 代码检查
python -m ruff check src/lerobot/robots/my_arm/
python -m ruff format --check src/lerobot/robots/my_arm/
```

### 10.3 硬件集成测试

```python
# tests/robots/test_my_arm_hardware.py

import pytest

# 检测硬件是否可用
def hardware_available():
    import os
    return os.path.exists("/dev/ttyUSB0")


@pytest.mark.skipif(not hardware_available(), reason="Hardware not available")
class TestMyArmHardware:
    def test_real_connection(self):
        from lerobot.robots.my_arm import MyArm, MyArmConfig

        config = MyArmConfig(port="/dev/ttyUSB0")
        robot = MyArm(config)
        try:
            robot.connect()
            assert robot.is_connected
            obs = robot.get_observation()
            assert all(isinstance(v, float) for k, v in obs.items() if k.endswith(".pos"))
        finally:
            robot.disconnect()
```

---

## 11. 常见问题与调试技巧

### 11.1 问题速查表

| 问题 | 现象 | 排查/解决 |
|------|------|----------|
| 串口权限不足 | `Permission denied: '/dev/ttyUSB0'` | `sudo usermod -aG dialout $USER` 并重新登录 |
| 串口找不到 | `FileNotFoundError` | `ls /dev/ttyUSB*` 或 `lerobot-find-port` |
| 校准文件不匹配 | "Mismatch between calibration..." | 删除旧校准文件或重新校准 |
| Protocol 1 不支持 sync_read | `NotImplementedError` | 像 `HopeJrHand` 一样循环单独 `read` |
| 动作被截断 | 目标位置与预期不符 | 调整 `max_relative_target` 或设为 `None` |
| 相机参数缺失 | `ValueError` | 确保 `CameraConfig` 设置了 `width/height/fps` |
| 遥操作维度不匹配 | `KeyError` 或关节不动 | 确保 teleop 和 robot 的 `action_features` 键名一致 |
| 电机抖动 | 运行时抖动 | 降低 PID 的 P 系数 |
| 帧率低 | 相机读取慢 | 使用 `async_read()` 而非 `read()` |
| 重复连接 | `DeviceAlreadyConnectedError` | 检查是否有其他进程占用 |

### 11.2 串口连接调试

```bash
# 查看已连接的串口
ls -la /dev/ttyUSB* /dev/ttyACM*

# 查看设备详情
udevadm info /dev/ttyUSB0

# 检查是否被占用
lsof /dev/ttyUSB0

# 使用 lerobot 工具
lerobot-find-port
```

### 11.3 电机通信调试

```python
# 直接测试电机通信
from lerobot.motors.feetech import FeetechMotorsBus, Motor, MotorNormMode

bus = FeetechMotorsBus(
    port="/dev/ttyUSB0",
    motors={"test": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
    calibration=None,
)

bus.connect()

# 扫描电机
found = bus.broadcast_ping()
print(f"发现电机: {found}")

# 读取电机信息
print(f"Model: {bus.read('Model_Number', 'test')}")
print(f"Position: {bus.read('Present_Position', 'test')}")

bus.disconnect()
```

### 11.4 相机调试

```bash
# 检查相机是否被识别
v4l2-ctl --list-devices

# 测试相机
ffplay /dev/video0
```

```python
# Python 测试
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

config = {"cam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)}
cameras = make_cameras_from_configs(config)

camera = cameras["cam"]
camera.connect()

img = camera.async_read()
print(f"Image shape: {img.shape}")

camera.disconnect()
```

### 11.5 校准数据调试

```python
# 查看校准文件
import json
from pathlib import Path

cal_path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/my_arm/my_arm.json"

if cal_path.exists():
    with open(cal_path) as f:
        cal_data = json.load(f)
        for motor, data in cal_data.items():
            print(f"Motor: {motor}")
            print(f"  Homing offset: {data['homing_offset']}")
            print(f"  Range: [{data['range_min']}, {data['range_max']}]")
```

### 11.6 性能调试

```python
# 测试读取性能
import time

robot.connect(calibrate=False)

# 预热
for _ in range(10):
    robot.get_observation()

# Benchmark
n_iters = 100
start = time.perf_counter()

for _ in range(n_iters):
    robot.get_observation()

elapsed = time.perf_counter() - start
print(f"Average FPS: {n_iters / elapsed:.1f}")
print(f"Average latency: {elapsed / n_iters * 1000:.1f} ms")

robot.disconnect()
```

---

## 12. 最佳实践

### 12.1 代码风格

**命名规范**：

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块名 | snake_case | `my_arm.py` |
| 类名 | PascalCase | `MyArm`, `MyArmConfig` |
| 方法名 | snake_case | `get_observation` |
| 常量 | UPPER_CASE | `DEFAULT_BAUDRATE` |
| 配置注册名 | snake_case | `"my_arm"` |
| 电机名称 | snake_case | `shoulder_pan`, `gripper` |

**代码检查**：

```bash
# 格式化代码
ruff format src/lerobot/robots/my_arm/

# 代码检查
ruff check src/lerobot/robots/my_arm/
```

### 12.2 错误处理

使用 LeRobot 提供的异常类：

```python
from lerobot.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
    CalibrationError,
)

def connect(self):
    if self.is_connected:
        raise DeviceAlreadyConnectedError(f"{self} already connected")

def get_observation(self):
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")
```

### 12.3 日志记录

```python
import logging

logger = logging.getLogger(__name__)

class MyArm(Robot):
    def connect(self):
        logger.info(f"Connecting to {self}...")
        logger.debug(f"Opening port {self.config.port}")

    def get_observation(self):
        start = time.perf_counter()
        obs = self.bus.sync_read("Present_Position")
        elapsed = time.perf_counter() - start

        logger.debug(f"Motor read took {elapsed*1000:.1f}ms")

        if elapsed > 0.1:
            logger.warning(f"Slow motor read: {elapsed*1000:.1f}ms")

        return obs
```

### 12.4 性能优化

1. **使用批量操作**：`sync_read/sync_write` 比逐个读写快很多
2. **使用 `cached_property`**：避免重复计算特征字典
3. **异步相机读取**：使用 `async_read()` 减少阻塞
4. **合理的采样频率**：根据硬件能力设置 FPS

```python
from functools import cached_property

class MyArm(Robot):
    @cached_property  # 只计算一次
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}
```

### 12.5 安全性

**扭矩管理**：

```python
def disconnect(self):
    if self.config.disable_torque_on_disconnect:
        self.bus.disable_torque()  # 保护机器人
    self.bus.disconnect()
```

**位置限制**：

```python
def send_action(self, action):
    if self.config.max_relative_target is not None:
        present = self.bus.sync_read("Present_Position")
        goal = ensure_safe_goal_position(...)
    ...
```

**紧急停止**：

```python
def emergency_stop(self) -> None:
    """立即停止所有电机"""
    logger.warning("EMERGENCY STOP TRIGGERED!")
    self.bus.disable_torque()
```

### 12.6 配置管理

- 将硬件常量配置化：端口、波特率、PID 均放在 `RobotConfig`
- 使用 `config.id` 区分多台同型号机器人
- 校准文件复用：优先读已有校准，新增后调用 `_save_calibration()`

---

## 13. 提交前检查清单

在提交 PR 前，确保完成以下检查：

### 代码实现

- [ ] 继承 `Robot` 基类并实现所有抽象方法
- [ ] 创建配置类并使用 `@RobotConfig.register_subclass()` 注册
- [ ] 在 `robots/utils.py` 的 `make_robot_from_config` 中添加分支
- [ ] 设置正确的 `__init__.py` 导出
- [ ] 正确处理连接状态（抛出 `DeviceAlreadyConnectedError` / `DeviceNotConnectedError`）

### 测试

- [ ] 编写单元测试并确保通过
- [ ] 使用 Mock 测试避免硬件依赖
- [ ] 测试真实硬件（如有条件）

### 代码质量

- [ ] 运行 `ruff check src/lerobot/robots/my_arm/`
- [ ] 运行 `ruff format src/lerobot/robots/my_arm/`
- [ ] 添加必要的类型标注
- [ ] 添加 docstring 文档

### 验证命令

```bash
# 完整验证流程
python -m ruff check src/lerobot/robots/my_arm/
python -m ruff format --check src/lerobot/robots/my_arm/
pytest tests/robots/test_my_arm.py -v
```

---

## 参考资源

### 现有机器人实现参考

| 机器人 | 路径 | 特点 |
|--------|------|------|
| SO100 Follower | `src/lerobot/robots/so100_follower/` | 6 DOF 机械臂，Feetech 电机 |
| SO101 Follower | `src/lerobot/robots/so101_follower/` | SO100 改进版 |
| Koch Follower | `src/lerobot/robots/koch_follower/` | Dynamixel 电机 |
| LeKiwi | `src/lerobot/robots/lekiwi/` | 移动底盘 + 机械臂 |
| HopeJR | `src/lerobot/robots/hope_jr/` | 灵巧手，Protocol 1 |
| BiSO100 | `src/lerobot/robots/bi_so100_follower/` | 双臂配置 |

### 外部链接

- [LeRobot GitHub 仓库](https://github.com/huggingface/lerobot)
- [LeRobot 文档](https://huggingface.co/docs/lerobot)
- [Feetech 电机 SDK](https://gitee.com/ftservo/SCServoSDK)
- [Dynamixel SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

### 获取帮助

- 查看现有实现作为参考
- 在 GitHub 提交 Issue
- 加入 Hugging Face Discord 社区

---

*文档版本: 1.0 | 最后更新: 2025-12-01 | 整合自多份开发指南*







```



