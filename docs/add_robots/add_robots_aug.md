# LeRobot 机器人集成开发指南

> 本文档详细描述如何向 LeRobot 项目添加新的机器人支持，包括机器人类实现、电机控制器集成、相机系统集成和遥操作支持。

## 1. 概述

### 1.1 LeRobot 机器人架构设计理念

LeRobot 采用模块化设计，将机器人系统分解为以下核心组件：

```
Robot (机器人)
├── MotorsBus (电机总线)
│   └── Motor (电机) × N
├── Camera (相机) × N
└── Teleoperator (遥操作设备)
```

**设计原则**：
- **抽象基类约束**：所有机器人必须继承 `Robot` 基类并实现其抽象方法
- **配置驱动**：通过 `@dataclass` 配置类管理所有参数
- **注册机制**：使用 `@RobotConfig.register_subclass()` 装饰器自动注册
- **松耦合**：电机、相机、遥操作器可独立配置和替换

### 1.2 目录结构

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

## 2. 前置准备

### 2.1 开发环境要求

```bash
# Python 版本要求
Python >= 3.10

# 安装开发依赖
pip install -e ".[dev,test]"

# 或使用 uv
uv sync --extra dev --extra test
```

### 2.2 硬件依赖

根据你的机器人，可能需要以下硬件驱动：

| 电机类型 | Python 包 | 安装命令 |
|---------|----------|---------|
| Feetech STS/SCS | `scservo-sdk` | `pip install scservo-sdk` |
| Dynamixel | `dynamixel-sdk` | `pip install dynamixel-sdk` |

| 相机类型 | 依赖 | 安装命令 |
|---------|------|---------|
| OpenCV 相机 | `opencv-python` | 已包含在基础依赖 |
| Intel RealSense | `pyrealsense2` | `pip install -e ".[intelrealsense]"` |

### 2.3 权限配置

Linux 系统需要配置串口权限：

```bash
# 添加用户到 dialout 组（需要重新登录生效）
sudo usermod -aG dialout $USER

# 或临时授权
sudo chmod 666 /dev/ttyUSB0
```

## 3. Robot 基类接口详解

### 3.1 必须实现的抽象属性和方法

```python
from lerobot.robots import Robot, RobotConfig

class MyRobot(Robot):
    # 类变量：必须设置
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
        """返回机器人是否已连接"""
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
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接并释放资源"""
        pass
```

### 3.2 基类提供的辅助功能

`Robot` 基类自动处理以下内容：

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
```

## 4. 添加新机器人的详细步骤

### 4.1 步骤 1：创建目录结构

```bash
# 以添加 "my_arm" 机器人为例
mkdir -p src/lerobot/robots/my_arm
touch src/lerobot/robots/my_arm/__init__.py
touch src/lerobot/robots/my_arm/config_my_arm.py
touch src/lerobot/robots/my_arm/my_arm.py
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

    # 机器人特定参数
    max_velocity: float = 100.0  # 最大速度限制

    def __post_init__(self):
        """参数验证"""
        if not self.port:
            raise ValueError("port 不能为空")
```

### 4.3 步骤 3：实现机器人类

在 `my_arm.py` 中：

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

from .config_my_arm import MyArmConfig

logger = logging.getLogger(__name__)


class MyArm(Robot):
    """
    MyArm 机器人实现示例。

    硬件配置：
    - 6 个 Feetech STS3215 电机
    - 可选相机
    """

    config_class = MyArmConfig
    name = "my_arm"

    def __init__(self, config: MyArmConfig):
        super().__init__(config)
        self.config = config

        # 初始化电机总线
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # 初始化相机（如果配置了）
        self.cameras = make_cameras_from_configs(config.cameras)

    # ========== 特征属性（使用 cached_property 优化性能）==========

    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征：{电机名.pos: float}"""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征：{相机名: (H, W, C)}"""
        return {
            cam: (cfg.height, cfg.width, 3)
            for cam, cfg in self.config.cameras.items()
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
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} 已连接")

        # 连接电机总线
        self.bus.connect()

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # 校准检查
        if not self.is_calibrated and calibrate:
            logger.info("需要校准，正在执行...")
            self.calibrate()

        self.configure()
        logger.info(f"{self} 连接成功")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        # 断开电机
        self.bus.disconnect()
        logger.info(f"{self} 已断开")

    # ========== 校准 ==========

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """交互式校准流程"""
        logger.info(f"开始校准 {self}")
        self.bus.disable_torque()

        # 设置电机为位置模式
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # 记录中位
        input("将机械臂移动到运动范围中点，按 Enter 继续...")
        homing_offsets = self.bus.set_half_turn_homings()

        # 记录运动范围
        print("依次移动各关节到极限位置，按 Enter 停止记录...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion(list(self.bus.motors))

        # 保存校准数据
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
        logger.info(f"校准已保存到 {self.calibration_fpath}")

    # ========== 配置 ==========

    def configure(self) -> None:
        """配置电机运行参数"""
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # ========== 观测与动作 ==========

    def get_observation(self) -> dict[str, Any]:
        """获取当前状态"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        start = time.perf_counter()
        obs = {}

        # 读取电机位置
        positions = self.bus.sync_read("Present_Position")
        for motor, val in positions.items():
            obs[f"{motor}.pos"] = val

        # 读取相机图像
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"get_observation: {dt_ms:.1f}ms")
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送目标位置"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} 未连接")

        start = time.perf_counter()

        # 提取目标位置
        goal_positions = {
            motor: action[f"{motor}.pos"]
            for motor in self.bus.motors
        }

        # 写入电机
        self.bus.sync_write("Goal_Position", goal_positions)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"send_action: {dt_ms:.1f}ms")
        return action
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

| 模式 | 范围 | 用途 |
|------|------|------|
| `RANGE_0_100` | [0, 100] | 夹爪等单向运动 |
| `RANGE_M100_100` | [-100, 100] | 关节旋转 |
| `DEGREES` | 角度值 | 直接使用角度 |

### 5.2 MotorsBus 核心方法

```python
from lerobot.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors={...})

# 连接管理
bus.connect()
bus.disconnect()

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
bus.record_ranges_of_motion(motors)  # 记录运动范围
```

### 5.3 支持的电机型号

**Feetech 系列**（`src/lerobot/motors/feetech/`）：
- STS3215, STS3250
- SCS0009, SCS1000 等

**Dynamixel 系列**（`src/lerobot/motors/dynamixel/`）：
- XL330-M288, XL430-W250
- XM540-W270, XC330-M288 等

## 6. 相机集成详解

### 6.1 相机配置

```python
from lerobot.cameras import CameraConfig, OpenCVCameraConfig, IntelRealSenseCameraConfig

# OpenCV 相机（USB 相机、内置摄像头）
opencv_config = OpenCVCameraConfig(
    fps=30,
    width=640,
    height=480,
    index=0,  # 相机索引或设备路径
    rotation=None,  # 可选：Cv2Rotation.ROTATE_90
)

# Intel RealSense
realsense_config = IntelRealSenseCameraConfig(
    fps=30,
    width=640,
    height=480,
    serial_number="123456789",  # 可选
)
```

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

## 7. 遥操作设备集成（可选）

如果你的机器人需要遥操作支持（如主从控制），需要实现 `Teleoperator` 类。

### 7.1 Teleoperator 基类接口

```python
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

class MyTeleop(Teleoperator):
    config_class = MyTeleopConfig
    name = "my_teleop"

    @property
    def action_features(self) -> dict:
        """遥操作器输出的动作特征"""
        return {"shoulder_pan.pos": float, ...}

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
        raise NotImplementedError  # 如不支持

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

## 8. 测试指南

### 8.1 单元测试结构

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
        bus_mock.sync_read.return_value = {m: i for i, m in enumerate(bus_mock.motors, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.is_calibrated = True
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


def test_get_observation(robot):
    """测试观测获取"""
    robot.connect()
    obs = robot.get_observation()
    expected_keys = {f"{m}.pos" for m in robot.bus.motors}
    assert set(obs.keys()) == expected_keys


def test_send_action(robot):
    """测试动作发送"""
    robot.connect()
    action = {f"{m}.pos": i * 10 for i, m in enumerate(robot.bus.motors, 1)}
    returned = robot.send_action(action)
    assert returned == action
    robot.bus.sync_write.assert_called_once()
```

### 8.2 运行测试

```bash
# 运行特定测试
pytest tests/robots/test_my_arm.py -v

# 运行所有机器人测试
pytest tests/robots/ -v

# 带覆盖率报告
pytest tests/robots/test_my_arm.py --cov=src/lerobot/robots/my_arm
```

### 8.3 硬件集成测试（需要真实硬件）

```python
# tests/robots/test_my_arm_hardware.py

import pytest

from lerobot.robots.my_arm import MyArm, MyArmConfig


@pytest.mark.skipif(not HARDWARE_AVAILABLE, reason="需要真实硬件")
class TestMyArmHardware:
    def test_real_connection(self):
        config = MyArmConfig(port="/dev/ttyUSB0")
        robot = MyArm(config)
        try:
            robot.connect()
            assert robot.is_connected
            obs = robot.get_observation()
            assert all(isinstance(v, float) for v in obs.values())
        finally:
            robot.disconnect()
```

## 9. 常见问题与调试技巧

### 9.1 串口连接问题

**问题**：`Permission denied: '/dev/ttyUSB0'`

**解决方案**：
```bash
# 方法 1：临时授权
sudo chmod 666 /dev/ttyUSB0

# 方法 2：添加用户到 dialout 组（永久）
sudo usermod -aG dialout $USER
# 然后重新登录
```

**问题**：找不到串口设备

**解决方案**：
```bash
# 查看已连接的串口
ls /dev/ttyUSB* /dev/ttyACM*

# 查看设备详情
udevadm info /dev/ttyUSB0

# 使用 lerobot 工具
lerobot-find-port
```

### 9.2 电机通信问题

**问题**：电机不响应 / 通信超时

**排查步骤**：
1. 检查波特率是否正确
2. 检查电机 ID 是否冲突
3. 检查供电是否充足
4. 使用官方工具（如 Feetech Debug Tool）测试

```python
# 调试代码：扫描电机
from lerobot.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors={})
bus.connect()
found = bus.broadcast_ping()  # 返回 {id: model_number}
print(f"发现电机: {found}")
```

**问题**：电机位置值异常

**可能原因**：
- 未校准或校准数据错误
- `MotorNormMode` 设置错误
- 电机 `drive_mode` 方向相反

### 9.3 相机问题

**问题**：相机打开失败

**排查步骤**：
```bash
# 检查相机是否被识别
v4l2-ctl --list-devices

# 测试相机
ffplay /dev/video0
```

**问题**：帧率低于预期

**解决方案**：
- 降低分辨率
- 使用 `async_read()` 而非 `read()`
- 检查 USB 带宽（避免多个高分辨率相机共用 USB Hub）

### 9.4 校准问题

**问题**：校准后位置不准确

**检查项**：
1. `homing_offset` 是否正确记录
2. `range_min` 和 `range_max` 是否覆盖实际运动范围
3. 电机是否有机械限位

```python
# 查看校准数据
import json
with open("~/.cache/huggingface/lerobot/calibration/robots/my_arm/my_arm.json") as f:
    print(json.dumps(json.load(f), indent=2))
```

## 10. 最佳实践

### 10.1 代码风格

- 使用 `ruff` 进行代码检查和格式化
- 遵循 Google 风格的 docstring
- 类型标注所有公共接口

```bash
# 格式化代码
ruff format src/lerobot/robots/my_arm/

# 代码检查
ruff check src/lerobot/robots/my_arm/
```

### 10.2 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块名 | snake_case | `my_arm.py` |
| 类名 | PascalCase | `MyArm`, `MyArmConfig` |
| 方法名 | snake_case | `get_observation` |
| 常量 | UPPER_CASE | `DEFAULT_BAUDRATE` |
| 配置注册名 | snake_case | `"my_arm"` |

### 10.3 错误处理

使用 LeRobot 提供的异常类：

```python
from lerobot.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
    CalibrationError,
)

def connect(self):
    if self.is_connected:
        raise DeviceAlreadyConnectedError(f"{self} 已连接")

def get_observation(self):
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} 未连接")
```

### 10.4 日志记录

使用标准 `logging` 模块：

```python
import logging

logger = logging.getLogger(__name__)

class MyArm(Robot):
    def connect(self):
        logger.info(f"正在连接 {self}")
        # ...
        logger.debug(f"电机配置: {self.bus.motors}")
```

### 10.5 性能优化

1. **使用 `sync_read/sync_write`**：批量操作比逐个读写快很多
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

## 11. 完整示例参考

以下是现有机器人实现的参考：

| 机器人 | 路径 | 特点 |
|--------|------|------|
| SO100 Follower | `src/lerobot/robots/so100_follower/` | 6 DOF 机械臂，Feetech 电机 |
| Koch Follower | `src/lerobot/robots/koch_follower/` | Dynamixel 电机 |
| LeKiwi | `src/lerobot/robots/lekiwi/` | 移动底盘 + 机械臂 |
| HopeJR | `src/lerobot/robots/hope_jr/` | 灵巧手 |
| BiSO100 | `src/lerobot/robots/bi_so100_follower/` | 双臂配置 |

## 12. 检查清单

在提交 PR 前，确保完成以下检查：

- [ ] 继承 `Robot` 基类并实现所有抽象方法
- [ ] 创建配置类并使用 `@RobotConfig.register_subclass()` 注册
- [ ] 在 `utils.py` 的工厂函数中添加机器人
- [ ] 设置正确的 `__init__.py` 导出
- [ ] 编写单元测试并确保通过
- [ ] 运行 `ruff check` 和 `ruff format`
- [ ] 更新相关文档（如 README）
- [ ] 测试真实硬件（如有条件）

## 13. 参考资料

- [LeRobot GitHub 仓库](https://github.com/huggingface/lerobot)
- [Feetech 电机 SDK](https://gitee.com/ftservo/SCServoSDK)
- [Dynamixel SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

---

*本文档由 Augment 自动生成，如有问题请提交 Issue。*

