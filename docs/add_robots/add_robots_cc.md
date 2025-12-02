# LeRobot 机器人集成开发指南

本文档详细介绍如何为 LeRobot 项目添加新的机器人支持，帮助开发者快速理解和实现机器人集成。

## 目录

1. [架构概述](#架构概述)
2. [前置准备](#前置准备)
3. [详细集成步骤](#详细集成步骤)
4. [代码示例](#代码示例)
5. [测试指南](#测试指南)
6. [调试技巧](#调试技巧)
7. [最佳实践](#最佳实践)

---

## 架构概述

### LeRobot 机器人集成架构

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

### 核心组件

1. **Robot 基类** (`src/lerobot/robots/robot.py`): 所有机器人的抽象基类，定义了必须实现的接口
2. **Motor 模块** (`src/lerobot/motors/`): 电机控制器抽象，支持 Dynamixel 和 Feetech 协议
3. **Camera 模块** (`src/lerobot/cameras/`): 相机接口，支持 OpenCV 和 Intel RealSense
4. **Teleoperator 模块** (`src/lerobot/teleoperators/`): 遥操作设备接口，用于人机交互

### 设计哲学

- **接口一致性**: 所有机器人都提供相同的 `get_observation()` 和 `send_action()` 接口
- **模块化**: 电机控制和相机是独立模块，可以单独使用或组合
- **可扩展性**: 通过工厂模式和配置系统，轻松添加新机器人类型
- **类型安全**: 使用 Python 类型提示和 dataclass 配置，确保编译时检查

---

## 前置准备

### 硬件要求

1. **机器人硬件**: 具备电机（伺服/步进）的机械臂、灵巧手或移动平台
2. **电机控制器**: 支持 Dynamixel（TTL/RS485）或 Feetech（UART）协议的控制器
3. **相机**（可选）: USB 摄像头、Intel RealSense 或其他兼容设备
4. **计算设备**: 运行 Linux 的 PC 或嵌入式设备（如 Raspberry Pi）

### 软件依赖

```bash
# 基础依赖
pip install -e "."

# 如果需要相机支持
pip install opencv-python

# 如果需要 RealSense 支持
pip install pyrealsense2

# 开发环境（推荐）
uv sync --extra "dev" --extra "test" --extra "all"
```

### 开发环境配置

```bash
# 克隆仓库
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 安装开发依赖
pip install pre-commit
pre-commit install

# 安装 LeRobot
pip install -e ".[dynamixel,feetech,intelrealsense]"
```

### 硬件检测工具

LeRobot 提供了硬件检测工具：

```bash
# 查找连接的相机
lerobot-find-cameras

# 查找电机控制器端口
lerobot-find-port

# 使用 dynamixel Wizard 或 Feetech 软件测试电机连接
```

---

## 详细集成步骤

### 步骤 1: 创建机器人目录结构

在 `src/lerobot/robots/` 下创建新的机器人目录：

```bash
mkdir -p src/lerobot/robots/my_robot_name
cd src/lerobot/robots/my_robot_name
touch __init__.py my_robot_name.py config_my_robot_name.py
```

目录结构示例：

```
my_robot_name/
├── __init__.py                 # 模块初始化，导出类和配置
├── my_robot_name.py            # 主机器人类实现
├── config_my_robot_name.py     # 配置类定义
└── my_robot_name.mdx -> docs/source/my_robot_name.mdx  # 符号链接到文档
```

### 步骤 2: 创建配置类

配置文件定义机器人的参数（端口、电机配置、相机配置等）：

```python
# src/lerobot/robots/my_robot_name/config_my_robot_name.py

from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("my_robot_name")
@dataclass
class MyRobotNameConfig(RobotConfig):
    """MyRobotName 机器人的配置类"""

    # 电机控制器端口（必填）
    port: str = "/dev/ttyUSB0"  # Linux 串口
    # port: str = "COM3"  # Windows 串口

    # 是否在断开连接时禁用扭矩（保护电机）
    disable_torque_on_disconnect: bool = True

    # 最大相对目标位置限制（安全保护）
    max_relative_target: int | None = None

    # 相机配置（可选）
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 角度单位设置
    use_degrees: bool = False  # 使用度数而不是 RANGE_M100_100
```

**关键说明**：
- `@RobotConfig.register_subclass("my_robot_name")` 装饰器将配置注册到配置系统
- `port`: 电机控制器的串口路径，可通过 `lerobot-find-port` 查找
- `cameras`: 相机配置字典，key 是相机名称，value 是 CameraConfig

### 步骤 3: 实现机器人类

机器人类是核心的集成点，必须实现所有抽象方法：

```python
# src/lerobot/robots/my_robot_name/my_robot_name.py

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_my_robot_name import MyRobotNameConfig

logger = logging.getLogger(__name__)


class MyRobotName(Robot):
    """
    MyRobotName 机器人实现

    这是一个示例机器人类，展示了如何集成新的机器人到 LeRobot。
    根据实际硬件修改电机配置和相机设置。
    """

    # 配置类映射
    config_class = MyRobotNameConfig

    # 机器人唯一名称（必须唯一）
    name = "my_robot_name"

    def __init__(self, config: MyRobotNameConfig):
        """初始化机器人"""
        super().__init__(config)
        self.config = config

        # 根据配置确定归一化模式
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # 初始化电机总线
        # 关键：配置每个电机的 ID、型号和归一化模式
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # 根据实际硬件配置电机
                # 格式: "motor_name": Motor(id, model, norm_mode)
                "joint_1": Motor(1, "sts3215", norm_mode_body),  # 肩关节
                "joint_2": Motor(2, "sts3215", norm_mode_body),  # 肩关节
                "joint_3": Motor(3, "sts3215", norm_mode_body),  # 肘关节
                "joint_4": Motor(4, "sts3215", norm_mode_body),  # 腕关节
                "joint_5": Motor(5, "sts3215", norm_mode_body),  # 腕关节
                "joint_6": Motor(6, "sts3215", norm_mode_body),  # 夹爪
            },
            calibration=self.calibration,  # 从基类加载的校准数据
        )

        # 初始化相机
        # make_cameras_from_configs 会根据配置自动创建相机实例
        self.cameras = make_cameras_from_configs(config.cameras)

    # ========== 特征定义 ==========

    @property
    def _motors_ft(self) -> dict[str, type]:
        """定义电机特征：每个电机输出 .pos（位置）"""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """定义相机特征：输出图像的形状 (height, width, channels)"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        观察特征：定义 get_observation() 返回的数据结构

        包括：
        - 电机位置：{motor_name}.pos: float
        - 相机图像：{camera_name}: (height, width, 3)
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        动作特征：定义 send_action() 期望的数据结构

        通常为电机目标位置：{motor_name}.pos: float
        """
        return self._motors_ft

    # ========== 连接管理 ==========

    @property
    def is_connected(self) -> bool:
        """
        检查机器人是否已连接

        需要同时检查：
        - 电机总线已连接
        - 所有相机已连接
        """
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        连接机器人

        步骤：
        1. 检查是否已连接
        2. 连接电机总线
        3. 如果需要，执行校准
        4. 连接所有相机
        5. 配置机器人参数

        Args:
            calibrate: 是否在校准缺失时自动校准
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. 连接电机总线
        self.bus.connect()

        # 2. 检查校准状态
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or "
                "no calibration file found"
            )
            self.calibrate()

        # 3. 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # 4. 配置机器人参数
        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """断开机器人连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 断开电机总线（根据配置决定是否禁用扭矩）
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # 断开所有相机
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    # ========== 校准管理 ==========

    @property
    def is_calibrated(self) -> bool:
        """检查机器人是否已校准"""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        校准机器人

        校准流程：
        1. 如果已有校准文件，询问用户使用现有还是重新校准
        2. 禁用扭矩，允许手动移动
        3. 记录每个电机的运动范围
        4. 计算零点偏移（homing_offset）
        5. 保存校准数据到文件

        校准数据包括：
        - homing_offset: 零点偏移
        - range_min: 最小位置
        - range_max: 最大位置
        - drive_mode: 驱动模式
        """
        if self.calibration:
            # 校准文件已存在，询问用户
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")

        # 禁用扭矩，允许手动移动
        self.bus.disable_torque()

        # 设置位置模式
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # 步骤 1: 将机器人移动到中间位置
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # 步骤 2: 记录每个电机的完整运动范围
        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        # 构建校准字典
        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        # 写入校准到电机
        self.bus.write_calibration(self.calibration)

        # 保存校准到文件
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    # ========== 配置管理 ==========

    def configure(self) -> None:
        """
        配置机器人参数

        在连接后调用一次，用于：
        - 设置电机控制参数（P/I/D 系数）
        - 配置最大加速度
        - 设置工作模式
        """
        # 使用扭矩禁用上下文管理器，确保配置期间扭矩关闭
        with self.bus.torque_disabled():
            # 配置电机参数（从校准数据加载）
            self.bus.configure_motors()

            # 设置控制模式为位置控制
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

                # 设置 PID 参数（避免抖动）
                # P 系数：减小可降低抖动（默认 32）
                self.bus.write("P_Coefficient", motor, 16)
                # I 系数：默认 0
                self.bus.write("I_Coefficient", motor, 0)
                # D 系数：默认 32
                self.bus.write("D_Coefficient", motor, 32)

    # ========== 数据交互 ==========

    def get_observation(self) -> dict[str, Any]:
        """
        获取当前观测

        读取：
        1. 所有电机的当前位置（Present_Position）
        2. 所有相机的图像数据

        Returns:
            字典包含：
            - {motor_name}.pos: float
            - {camera_name}: np.ndarray (H, W, 3)

        Raises:
            DeviceNotConnectedError: 如果机器人未连接
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 1. 读取电机位置（同步读取，提高效率）
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 2. 读取相机图像（异步读取，提高帧率）
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        发送动作指令

        将目标位置发送到电机。
        可选：如果设置了 max_relative_target，会进行安全限制，
        防止目标位置离当前位置太远。

        Args:
            action: 动作字典，格式：{motor_name}.pos: float

        Returns:
            实际发送到电机的动作（可能被裁剪）

        Raises:
            DeviceNotConnectedError: 如果机器人未连接
            ValueError: 如果动���格式不正确
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 解析动作：提取电机名称和位置
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # 安全检查：限制相对移动距离
        if self.config.max_relative_target is not None:
            # 读取当前位置
            present_pos = self.bus.sync_read("Present_Position")

            # 计算目标位置与当前位置
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }

            # 应用安全限制
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        # 发送目标位置到电机（同步写入）
        self.bus.sync_write("Goal_Position", goal_pos)

        # 返回实际发送的动作
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

**关键要点**：

1. **初始化** (`__init__`):
   - 调用 `super().__init__(config)` 加载基类功能（校准管理）
   - 创建 `MotorsBus` 实例，配置所有电机参数
   - 使用 `make_cameras_from_configs()` 自动创建相机实例

2. **特征定义**:
   - `observation_features`: 定义观测数据的结构，用于数据集创建和验证
   - `action_features`: 定义动作数据的结构，用于策略网络

3. **连接管理**:
   - 严格检查连接状态，防止重复连接
   - 先连接电机，再检查/执行校准，最后连接相机
   - 断开时根据配置决定是否禁用扭矩（保护机器人）

4. **校准流程**: 校准是机器人集成的关键步骤，确保：
   - 记录每个电机的运动范围
   - 计算零点偏移
   - 保存校准数据以便后续使用

### 步骤 4: 配置相机

如果机器人使用相机，需要在 Lerobot 的 config 系统中配置：

```python
# 在配置文件或实例化时添加
from lerobot.cameras import CameraConfig, ColorMode, Cv2Rotation

cameras = {
    "top_camera": CameraConfig(
        type="opencv",  # 或 "realsense"
        index=0,  # USB 索引或序列号
        width=640,
        height=480,
        fps=30,
        color_mode=ColorMode.RGB,  # RGB, BGR, GRAY
        rotation=Cv2Rotation.ROTATE_90_CLOCKWISE,  # 可选：图像旋转
    ),
    "wrist_camera": CameraConfig(
        type="opencv",
        index=1,
        width=640,
        height=480,
        fps=30,
    )
}
```

### 步骤 5: 注册到工厂函数

在 `src/lerobot/robots/robot.py` 的 `make_robot_from_config` 函数中注册：

```python
def make_robot_from_config(config: RobotConfig) -> Robot:
    # ... 现有代码 ...

    elif config.type == "my_robot_name":
        from .my_robot_name import MyRobotName
        return MyRobotName(config)

    else:
        raise ValueError(f"Unknown robot type: {config.type}")
```

也可使用动态导入：

```python
import importlib

def make_robot_from_config(config: RobotConfig) -> Robot:
    try:
        # 尝试动态导入
        module = importlib.import_module(f"lerobot.robots.{config.type}")
        robot_class = getattr(module, config.type.split("_")[0].capitalize())
        return robot_class(config)
    except (ImportError, AttributeError):
        raise ValueError(f"Unknown robot type: {config.type}")
```

### 步骤 6: 添加遥操作支持（可选）

如果需要远程操作，在 `src/lerobot/teleoperators/` 创建遥操作类：

```python
# src/lerobot/teleoperators/my_teleop/my_teleop.py

from ..teleoperator import Teleoperator
from .config_my_teleop import MyTeleopConfig


class MyTeleop(Teleoperator):
    """MyTeleop 遥操作设备实现"""

    config_class = MyTeleopConfig
    name = "my_teleop"

    def __init__(self, config: MyTeleopConfig):
        super().__init__(config)
        # 初始化遥操作硬件

    @property
    def action_features(self) -> dict:
        """遥操作输出的动作特征"""
        return {"joint_1.pos": float, "joint_2.pos": float, ...}

    @property
    def feedback_features(self) -> dict:
        """期望的反馈特征"""
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        """读取遥操作输入，返回动作"""
        # 从硬件读取输入
        return {"joint_1.pos": 0.0, ...}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """发送反馈到遥操作设备"""
        pass

    def disconnect(self) -> None:
        pass
```

### 步骤 7: 编写配置文件示例

创建 YAML 或 Python 配置文件供用户使用：

```yaml
# examples/my_robot_name/my_robot_config.yaml

# 机器人配置
robot:
  type: my_robot_name
  id: my_robot_001  # 机器人唯一标识
  port: /dev/ttyUSB0
  disable_torque_on_disconnect: true
  max_relative_target: null  # 或设置限制，如 50

  # 相机配置（可选）
  cameras:
    top_camera:
      type: opencv
      index: 0
      width: 640
      height: 480
      fps: 30
      color_mode: RGB

# 训练配置
data_dir: data
output_dir: outputs/my_robot_experiment

policy:
  type: act  # 或其他策略：diffusion, tdmpc, vqbet
  dim_model: 128
  n_action_steps: 10

# ... 其他配置 ...
```

---

## 代码示例

### 示例 1: 基本的机器人连接和测试

```python
#!/usr/bin/env python
"""测试 MyRobotName 机器人基本功能"""

from lerobot.robots.my_robot_name import MyRobotName, MyRobotNameConfig
from lerobot.cameras import CameraConfig

# 1. 创建配置
config = MyRobotNameConfig(
    id="test_robot_001",
    port="/dev/ttyUSB0",
    disable_torque_on_disconnect=True,
    cameras={
        "top": CameraConfig(
            type="opencv",
            index=0,
            width=640,
            height=480,
            fps=30,
        )
    }
)

# 2. 创建机器人实例
robot = MyRobotName(config)

# 3. 连接机器人（自动校准）
print("Connecting robot...")
robot.connect(calibrate=True)

# 4. 读取观测
print("\nReading observations...")
obs = robot.get_observation()
print(f"Observations: {list(obs.keys())}")
print(f"Joint positions: {[obs[k] for k in obs.keys() if k.endswith('.pos')]}")
print(f"Camera images: {[k for k in obs.keys() if not k.endswith('.pos')]}")

# 5. 发送动作
print("\nSending action...")
action = {
    "joint_1.pos": 0.0,
    "joint_2.pos": 0.0,
    "joint_3.pos": 0.0,
    "joint_4.pos": 0.0,
    "joint_5.pos": 0.0,
    "joint_6.pos": 0.0,
}
actual_action = robot.send_action(action)
print(f"Actual action sent: {actual_action}")

# 6. 断开连接
print("\nDisconnecting robot...")
robot.disconnect()
print("Done!")
```

### 示例 2: 手动校准流程

```python
#!/usr/bin/env python
"""手动校准 MyRobotName 机器人"""

from lerobot.robots.my_robot_name import MyRobotName, MyRobotNameConfig

def calibrate_robot():
    # 创建配置（无需相机）
    config = MyRobotNameConfig(
        id="my_robot_calibrated",
        port="/dev/ttyUSB0",
        cameras={},
    )

    robot = MyRobotName(config)

    # 清除现有校准（如果需要重新校准）
    # import os
    # if robot.calibration_fpath.exists():
    #     os.remove(robot.calibration_fpath)

    # 连接并强制校准
    print("=== Robot Calibration ===")
    print("This will calibrate your robot. Follow the instructions carefully.\n")

    robot.connect(calibrate=True)

    # 验证校准
    print("\n=== Verifying Calibration ===")
    print(f"Calibrated: {robot.is_calibrated}")
    print(f"Calibration saved to: {robot.calibration_fpath}")

    # 读取并打印校准数据
    for motor, cal in robot.calibration.items():
        print(f"\n{motor}:")
        print(f"  ID: {cal.id}")
        print(f"  Homing offset: {cal.homing_offset}")
        print(f"  Range: [{cal.range_min}, {cal.range_max}]")

    # 验证运动
    print("\n=== Testing Movement ===")
    obs = robot.get_observation()
    print(f"Current positions: {[obs[k] for k in obs.keys()]}")

    # 归零
    print("Moving to zero positions...")
    zero_action = {k: 0.0 for k in robot.action_features.keys()}
    robot.send_action(zero_action)

    # 断开
    robot.disconnect()

if __name__ == "__main__":
    calibrate_robot()
```

### 示例 3: 数据集录制和训练

```python
#!/usr/bin/env python
"""使用 MyRobotName 机器人录制数据集并训练"""

import torch
from pathlib import Path
from lerobot.scripts.control_robot import record, train
from lerobot.robots.my_robot_name import MyRobotNameConfig
from lerobot.cameras import CameraConfig
from lerobot.configs.train import TrainConfig
from lerobot.configs.policy.diffusion import DiffusionConfig

def record_dataset():
    """录制机器人演示数据"""

    # 机器人配置
    robot_config = MyRobotNameConfig(
        id="my_robot_001",
        port="/dev/ttyUSB0",
        cameras={
            "top": CameraConfig(type="opencv", index=0, width=640, height=480, fps=30)
        }
    )

    # 录制配置
    from lerobot.configs.control import RecordConfig, DatasetRecordConfig

    dataset_config = DatasetRecordConfig(
        repo_id="my_username/my_robot_dataset",
        fps=30,
        num_episodes=50,
        episode_time_s=30,
        root=Path("data"),
        push_to_hub=True,
    )

    record_config = RecordConfig(
        robot=robot_config,
        dataset=dataset_config,
        resume=True,
        play_sounds=True,
    )

    # 开始录制
    print("Starting data collection...")
    record(record_config)
    print("Data collection completed!")

def train_policy():
    """训练扩散策略"""

    # 策略配置（Diffusion Policy）
    policy_config = DiffusionConfig(
        type="diffusion",
        dim_model=128,
        n_action_steps=8,
        n_obs_steps=2,
    )

    # 训练配置
    train_config = TrainConfig(
        dataset_repo_id="my_username/my_robot_dataset",
        policy=policy_config,
        batch_size=64,
        num_epochs=3000,
        log_freq=100,
        save_freq=500,
        output_dir=Path("outputs/my_robot_policy"),
    )

    # 开始训练
    print("Starting training...")
    train(train_config)
    print("Training completed!")

if __name__ == "__main__":
    # 检查 GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        print("WARNING: GPU not available, training will be slow")

    # 录数据集
    # record_dataset()

    # 训练策略
    train_policy()
```

### 示例 4: 使用自定义电机驱动

如果需要支持新的电机协议：

```python
# src/lerobot/motors/my_custom_motor.py

from .motors_bus import MotorsBus, Motor, MotorCalibration


class MyCustomMotorsBus(MotorsBus):
    """自定义电机控制器总线实现"""

    def __init__(self, port: str, motors: dict[str, Motor], calibration: dict | None = None):
        super().__init__(port, motors, calibration)
        self.connection = None

    def connect(self) -> None:
        """连接到电机控制器"""
        # 实现连接逻辑
        import serial
        self.connection = serial.Serial(self.port, baudrate=1000000, timeout=0.1)
        self.is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        """断开连接"""
        if self.connection:
            if disable_torque:
                for motor in self.motors:
                    self.disable_torque(motor)
            self.connection.close()
            self.is_connected = False

    def read(self, data_name: str, motor_name: str) -> float:
        """读取单个电机的数据"""
        # 实现读取逻辑
        motor = self.motors[motor_name]
        # ... 发送指令并读取返回 ...
        return raw_value

    def write(self, data_name: str, motor_name: str, value: float) -> None:
        """写入单个电机的数据"""
        # 实现写入逻辑
        motor = self.motors[motor_name]
        # ... 发送指令 ...
        pass

    def sync_read(self, data_name: str, motor_names: list | None = None) -> dict[str, float]:
        """同步读取多个电机数据"""
        if motor_names is None:
            motor_names = list(self.motors.keys())

        results = {}
        for motor_name in motor_names:
            results[motor_name] = self.read(data_name, motor_name)

        return results

    def sync_write(self, data_name: str, values: dict[str, float]) -> None:
        """同步写入多个电机数据"""
        for motor_name, value in values.items():
            self.write(data_name, motor_name, value)

    # 实现其他必需的抽象方法...
    # enable_torque, disable_torque, set_homing, read_calibration, write_calibration, etc.
```

然后在机器人类中使用：

```python
from lerobot.motors.my_custom_motor import MyCustomMotorsBus

class MyRobotName(Robot):
    def __init__(self, config: MyRobotNameConfig):
        super().__init__(config)
        self.bus = MyCustomMotorsBus(
            port=config.port,
            motors={...},
            calibration=self.calibration,
        )
```

---

## 测试指南

### 单元测试

```python
# tests/test_my_robot.py

import pytest
from lerobot.robots.my_robot_name import MyRobotName, MyRobotNameConfig
from tests.mocks.mock_motors_bus import MockMotorsBus


class TestMyRobotName:
    def test_robot_initialization(self):
        """测试机器人初始化"""
        config = MyRobotNameConfig(
            id="test",
            port="/dev/ttyUSB0",
            cameras={},
        )
        robot = MyRobotName(config)

        assert robot.name == "my_robot_name"
        assert robot.config == config
        assert len(robot.bus.motors) == 6

    def test_observation_features(self):
        """测试观测特征定义"""
        config = MyRobotNameConfig(id="test", port="/dev/ttyUSB0", cameras={})
        robot = MyRobotName(config)

        features = robot.observation_features
        assert "joint_1.pos" in features
        assert features["joint_1.pos"] == float

    def test_action_features(self):
        """测试动作特征定义"""
        config = MyRobotNameConfig(id="test", port="/dev/ttyUSB0", cameras={})
        robot = MyRobotName(config)

        features = robot.action_features
        assert "joint_1.pos" in features
        assert features["joint_1.pos"] == float

    @pytest.fixture
    def mock_robot(self, monkeypatch):
        """创建带模拟电机的机器人"""
        def mock_connect(self):
            self.is_connected = True

        monkeypatch.setattr(MockMotorsBus, "connect", mock_connect)

        config = MyRobotNameConfig(id="test", port="/dev/ttyUSB0", cameras={})
        return MyRobotName(config)

    def test_connect_disconnect(self, mock_robot):
        """测试连接和断开"""
        robot = mock_robot

        # Connect
        robot.connect(calibrate=False)
        assert robot.is_connected

        # Check can't connect twice
        with pytest.raises(Exception):
            robot.connect()

        # Disconnect
        robot.disconnect()
        assert not robot.is_connected
```

### 集成测试

```python
# tests/integration/test_my_robot_hardware.py

import pytest
from lerobot.robots.my_robot_name import MyRobotName, MyRobotNameConfig


@pytest.mark.hardware  # 标记为硬件测试
@pytest.mark.skipif(not robot_available(), reason="Robot hardware not available")
class TestMyRobotHardware:
    def test_hardware_connection(self):
        """测试硬件连接"""
        config = MyRobotNameConfig(
            id="hardware_test",
            port="/dev/ttyUSB0",  # 替换成实际端口
            cameras={},  # 测试时可以不带相机
        )

        robot = MyRobotName(config)

        # Connect
        robot.connect(calibrate=True)
        assert robot.is_connected

        # Test observation
        obs = robot.get_observation()
        assert len(obs) == len(robot.bus.motors)
        assert all(k.endswith(".pos") for k in obs.keys())

        # Test action
        action = {k: 0.0 for k in robot.action_features.keys()}
        actual = robot.send_action(action)
        assert actual.keys() == action.keys()

        # Disconnect
        robot.disconnect()
        assert not robot.is_connected

    def test_calibration(self):
        """测试校准流程"""
        config = MyRobotNameConfig(id="cal_test", port="/dev/ttyUSB0", cameras={})
        robot = MyRobotName(config)

        # Remove existing calibration
        if robot.calibration_fpath.exists():
            robot.calibration_fpath.unlink()

        # Connect with calibration
        robot.connect(calibrate=True)
        assert robot.is_calibrated
        assert robot.calibration_fpath.exists()

        # Disconnect
        robot.disconnect()

    def test_camera_integration(self):
        """测试相机集成"""
        from lerobot.cameras import CameraConfig

        config = MyRobotNameConfig(
            id="camera_test",
            port="/dev/ttyUSB0",
            cameras={
                "cam": CameraConfig(type="opencv", index=0, width=640, height=480, fps=30)
            },
        )

        robot = MyRobotName(config)
        robot.connect(calibrate=False)

        # Test observation includes camera
        obs = robot.get_observation()
        assert "cam" in obs
        assert obs["cam"].shape == (480, 640, 3)

        robot.disconnect()
```

### 运行测试

```bash
# 运行单元测试
pytest tests/test_my_robot.py -v

# 运行集成测试（需要硬件）
pytest tests/integration/test_my_robot_hardware.py -v -m hardware

# 运行所有机器人测试
pytest tests/test_control_robot.py -v
```

---

## 调试技巧

### 1. 电机通信调试

```python
# 直接测试电机通信
from lerobot.motors.feetech import FeetechMotorsBus, Motor, MotorNormMode

# 创建电机总线
bus = FeetechMotorsBus(
    port="/dev/ttyUSB0",
    motors={
        "test_motor": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    },
    calibration=None,
)

# Connect
bus.connect()

# 读取电机信息
print(f"Model: {bus.read('Model_Number', 'test_motor')}")
print(f"ID: {bus.read('ID', 'test_motor')}")
print(f"Baud Rate: {bus.read('Baud_Rate', 'test_motor')}")

# 读取位置
pos = bus.read('Present_Position', 'test_motor')
print(f"Position: {pos}")

# 移动到目标位置
bus.write('Goal_Position', 'test_motor', 0.5)

# 禁用扭矩
bus.disable_torque()

# Disconnect
bus.disconnect()
```

### 2. 相机调试

```python
# 测试相机
from lerobot.cameras import CameraConfig, make_cameras_from_configs

config = CameraConfig(type="opencv", index=0, width=640, height=480, fps=30)
cameras = make_cameras_from_configs({"cam": config})

camera = cameras["cam"]
camera.connect()

# 读取图像
img = camera.read()
print(f"Image shape: {img.shape}")

# 使用异步读取（推荐）
img_async = camera.async_read()
print(f"Async image shape: {img_async.shape}")

camera.disconnect()
```

### 3. 校准数据调试

```python
# 检查校准文件
import json
from pathlib import Path

cal_path = Path.home() / ".cache" / "lerobot" / "calibration" / "my_robot_name" / "my_robot_id.json"

with open(cal_path) as f:
    cal_data = json.load(f)
    for motor, data in cal_data.items():
        print(f"Motor: {motor}")
        print(f"  Homing offset: {data['homing_offset']}")
        print(f"  Range: [{data['range_min']}, {data['range_max']}]")
        print()

# 修改校准（慎用！）
cal_data["joint_1"]["homing_offset"] += 100

with open(cal_path, "w") as f:
    json.dump(cal_data, f, indent=2)
```

### 4. 性能调试

```python
# 测试读取性能
import time

robot.connect(calibrate=False)

# 预热
for _ in range(10):
    robot.get_observation()

# Benchmark
n_iters = 100
total_time = 0
frames = 0

for i in range(n_iters):
    start = time.perf_counter()
    obs = robot.get_observation()
    elapsed = time.perf_counter() - start

    total_time += elapsed
    frames += 1

    if i % 10 == 0:
        print(f"FPS: {frames / total_time:.1f}")

print(f"\nAverage FPS: {frames / total_time:.1f}")
print(f"Average latency: {total_time / frames * 1000:.1f} ms")

robot.disconnect()
```

### 5. 常见问题解决

**问题 1: 连接超时**
```python
# 原因：波特率或端口错误
# 解决：检查端口和波特率
ls -la /dev/ttyUSB*  # 查找设备
lsof /dev/ttyUSB0    # 检查是否被占用

# 在 MotorsBus 中设置正确的波特率
self.connection = serial.Serial(self.port, baudrate=1000000, timeout=0.1)
```

**问题 2: 电机抖动**
```python
# 原因：PID 系数不当
# 解决：调整 PID 参数
for motor in self.bus.motors:
    self.bus.write("P_Coefficient", motor, 16)  # 降低 P 系数
    self.bus.write("I_Coefficient", motor, 0)
    self.bus.write("D_Coefficient", motor, 32)
```

**问题 3: 校准数据不准确**
```python
# 重新校准前清除旧数据
import os
from pathlib import Path

cal_path = Path("~/.cache/lerobot/calibration/my_robot_name").expanduser()
for file in cal_path.glob("*.json"):
    file.unlink()

# 重新连接以触发校准
robot.connect(calibrate=True)
```

**问题 4: 相机帧率低**
```python
# 使用异步读取提高性能
for cam_key, cam in self.cameras.items():
    obs_dict[cam_key] = cam.async_read()  # 非阻塞

# 检查相机实际支持的参数
print(f"Actual FPS: {cam.capture.get(cv2.CAP_PROP_FPS)}")
print(f"Actual width: {cam.capture.get(cv2.CAP_PROP_FRAME_WIDTH)}")
```

**问题 5: 内存泄漏**
```python
# 确保所有资源正确释放
try:
    robot.connect()
    # ... operations ...
finally:
    robot.disconnect()  # 确保断开连接

# 使用 with 语句（如果实现了 __enter__/__exit__）
# with robot:
#     robot.get_observation()
```

---

## 最佳实践

### 1. 代码风格

**命名规范**:
- 类名: `MyRobotName` (PascalCase)
- 文件名: `my_robot_name.py` (snake_case)
- 电机名称: `joint_1`, `shoulder_pitch`, `gripper` (descriptive_snake_case)
- 配置类: `MyRobotNameConfig` (PascalCase + Config 后缀)

**类型提示**: 始终使用类型提示
```python
def get_observation(self) -> dict[str, Any]:
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        pass
```

**文档字符串**: 为所有公共方法和类编写文档
```python
def connect(self, calibrate: bool = True) -> None:
    """
    Connect to the robot.

    Args:
        calibrate: Whether to calibrate if not already calibrated.

    Raises:
        DeviceAlreadyConnectedError: If already connected.

    Example:
        >>> robot.connect(calibrate=True)
        >>> robot.is_connected
        True
    """
```

### 2. 错误处理

**使用 LeRobot 的标准异常**:
```python
from lerobot.errors import (
    DeviceNotConnectedError,
    DeviceAlreadyConnectedError,
    MotorError,
    CalibrationError,
)

def get_observation(self) -> dict[str, Any]:
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    try:
        return self.bus.sync_read("Present_Position")
    except Exception as e:
        logger.error(f"Failed to read motor positions: {e}")
        raise MotorError(f"Failed to read motor positions: {e}") from e
```

**优雅降级**: 无相机时也能工作
```python
def __init__(self, config: MyRobotNameConfig):
    super().__init__(config)
    self.bus = MotorsBus(...)
    self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}

def get_observation(self) -> dict[str, Any]:
    obs = {}

    # 电机观测（必需）
    obs.update(self.bus.sync_read("Present_Position"))

    # 相机观测（可选）
    if self.cameras:
        for cam_key, cam in self.cameras.items():
            try:
                obs[cam_key] = cam.async_read()
            except Exception as e:
                logger.warning(f"Failed to read camera {cam_key}: {e}")

    return obs
```

### 3. 性能优化

**批量操作**: 始终使用 sync_read/sync_write
```python
# 不推荐：逐个读取（慢）
positions = {}
for motor in self.bus.motors:
    positions[motor] = self.bus.read("Present_Position", motor)  # N 次通信

# 推荐：批量读取（快）
positions = self.bus.sync_read("Present_Position")  # 1 次通信
```

**异步读取相机**: 提高整体帧率
```python
# 相机读取和电机读取并行
import threading

obs = {}

# 启动相机读取线程
cam_threads = {}
for cam_key, cam in self.cameras.items():
    def read_cam(key=cam_key, camera=cam):
        obs[key] = camera.async_read()
    cam_threads[cam_key] = threading.Thread(target=read_cam)
    cam_threads[cam_key].start()

# 同时读取电机（主线程）
motor_obs = self.bus.sync_read("Present_Position")
obs.update({f"{k}.pos": v for k, v in motor_obs.items()})

# 等待相机完成
for thread in cam_threads.values():
    thread.join()
```

**避免重复计算**: 使用缓存属性
```python
from functools import cached_property

@cached_property
def observation_features(self) -> dict:
    """只计算一次，后续从缓存读取"""
    return {**self._motors_ft, **self._cameras_ft}
```

### 4. 校准管理

**校准文件命名**:
```python
# 使用机器人 ID 作为文件名
self.calibration_fpath = self.calibration_dir / f"{self.id}.json"

# 示例:
# ~/.cache/lerobot/calibration/my_robot_name/my_robot_001.json
```

**校准数据版本控制**:
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class CalibrationData:
    motors: dict[str, MotorCalibration]
    version: str = "1.0"
    timestamp: float = None
    robot_type: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()
```

### 5. 日志和调试

**适当的日志级别**:
```python
import logging

logger = logging.getLogger(__name__)

def connect(self):
    logger.info(f"Connecting to {self}...")  # INFO: 重要状态变化
    logger.debug(f"Opening port {self.config.port}")  # DEBUG: 详细信息

    try:
        self.bus.connect()
    except Exception as e:
        logger.error(f"Failed to connect: {e}")  # ERROR: 错误
        logger.exception(e)  # 打印完整堆栈
        raise
```

**性能日志**:
```python
def get_observation(self) -> dict:
    start = time.perf_counter()
    obs = self.bus.sync_read("Present_Position")
    elapsed = time.perf_counter() - start

    logger.debug(f"Motor read took {elapsed*1000:.1f}ms")  # DEBUG 级别

    if elapsed > 0.1:
        logger.warning(f"Slow motor read: {elapsed*1000:.1f}ms")  # 慢操作警告

    return obs
```

### 6. 安全性

**扭矩管理**: 始终在断开时禁用扭矩（可选）
```python
def disconnect(self):
    if self.config.disable_torque_on_disconnect:
        self.bus.disable_torque()  # 保护机器人
    self.bus.disconnect()
```

**位置限制**: 实现软件限位
```python
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    # 检查位置是否安全
    goal_positions = {k.removesuffix(".pos"): v for k, v in action.items()}

    for motor, goal_pos in goal_positions.items():
        cal = self.calibration[motor]
        safe_range = (cal.range_min + 100, cal.range_max - 100)  # 安全边距

        if not (safe_range[0] <= goal_pos <= safe_range[1]):
            logger.error(f"Goal position {goal_pos} for {motor} exceeds safe range {safe_range}")
            raise ValueError(f"Unsafe goal position for {motor}")

    return self.bus.sync_write("Goal_Position", goal_positions)
```

**紧急停止**: 提供 E-Stop 机制
```python
def emergency_stop(self) -> None:
    """立即停止所有电机"""
    logger.warning("EMERGENCY STOP TRIGGERED!")
    self.bus.disable_torque()
    # 记录状态
    self.emergency_stopped = True
```

### 7. 文档和示例

**Docstring 示例**:
```python
class MyRobotName(Robot):
    """
    MyRobotName robot implementation.

    This robot has 6 degrees of freedom and supports both position and velocity control.

    Args:
        config: Robot configuration including motor IDs, camera settings, etc.

    Example:
        >>> config = MyRobotNameConfig(port="/dev/ttyUSB0")
        >>> robot = MyRobotName(config)
        >>> robot.connect(calibrate=True)
        >>> obs = robot.get_observation()
        >>> robot.disconnect()

    Note:
        Always calibrate before first use. Calibration data is saved to
        ~/.cache/lerobot/calibration/my_robot_name/{robot_id}.json
    """
```

**创建示例脚本**:
```bash
mkdir -p examples/my_robot_name
cat > examples/my_robot_name/demo.py << 'EOF'
#!/usr/bin/env python
"""Example demonstrating MyRobotName robot."""

from lerobot.robots.my_robot_name import MyRobotName, MyRobotNameConfig

def main():
    config = MyRobotNameConfig(port="/dev/ttyUSB0")
    robot = MyRobotName(config)

    robot.connect(calibrate=True)

    # Demo code here...

    robot.disconnect()

if __name__ == "__main__":
    main()
EOF
```

**README 模板**:
```markdown
# MyRobotName Integration

## Overview
Description of the robot...

## Features
- 6 DOF arm
- Position control
- Camera support
- etc.

## Installation

## Usage

## Calibration

## Troubleshooting

## Examples
```

### 8. 测试覆盖

**测试所有关键路径**:
```python
def test_robot_lifecycle(self):
    """测试完整的机器人生命周期"""
    config = MyRobotNameConfig(id="test", port="/dev/ttyUSB0", cameras={})
    robot = MyRobotName(config)

    # Connect
    robot.connect(calibrate=False)
    assert robot.is_connected
    assert robot.is_calibrated

    # Get observation
    obs = robot.get_observation()
    assert len(obs) == len(robot.bus.motors)

    # Send action
    action = {k: 0.0 for k in robot.action_features.keys()}
    actual = robot.send_action(action)
    assert actual == action

    # Disconnect
    robot.disconnect()
    assert not robot.is_connected
```

**模拟测试**: 为硬件依赖提供模拟实现
```python
# tests/mocks/mock_my_robot.py

from lerobot.robots.my_robot_name import MyRobotName, MyRobotNameConfig

class MockMyRobot(MyRobotName):
    """Mock robot for testing without hardware."""

    def __init__(self, config: MyRobotNameConfig):
        super().__init__(config)
        self.mock_position = {m: 0.0 for m in self.bus.motors}

    def connect(self, calibrate: bool = True) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_observation(self) -> dict[str, Any]:
        return {f"{m}.pos": pos for m, pos in self.mock_position.items()}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        for motor in self.bus.motors:
            self.mock_position[motor] = action[f"{motor}.pos"]
        return action
```

---

## 总结

本指南详细介绍了为 LeRobot 添加新机器人支持的完整流程，包括：

1. **架构理解**：掌握 LeRobot 的分层架构和核心组件
2. **代码实现**：创建配置类、机器人类，实现所有必需接口
3. **硬件集成**：集成电机控制器和相机系统
4. **测试验证**：编写单元测试和集成测试
5. **调试优化**：解决常见问题，优化性能
6. **最佳实践**：遵循代码风格、安全性、可维护性规范

### 后续步骤

1. **贡献代码**: 将新机器人实现提交到 LeRobot 仓库
   - Fork 仓库并创建新分支
   - 添加测试和文档
   - 提交 Pull Request

2. **分享经验**: 在 GitHub Discussions 或 Discord 分享集成经验

3. **持续改进**: 根据用户反馈优化实现

### 参考资源

- [LeRobot GitHub Repository](https://github.com/huggingface/lerobot)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [SO-100 Example Implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/robots/so100_follower)
- [Hope-JR Example Implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/robots/hope_jr)

### 获取帮助

如遇到问题：
- 查看现有实现作为参考
- 在 GitHub 提交 Issue
- 加入 Hugging Face Discord 社区

---

**文档版本**: 1.0
**最后更新**: 2025-12-01
**作者**: LeRobot 社区
