# LeRobot 新机器人支持开发手册（Codex 版）

> 面向需要向 LeRobot 贡献新机器人支持的开发者，结合当前代码（例如 `src/lerobot/robots/so100_follower/so100_follower.py`、`koch_follower/koch_follower.py`、`lekiwi/lekiwi.py`、`hope_jr/hope_jr_arm.py` 等）整理出的可操作步骤与代码模板。

## 1. 架构概览

- **核心基类**  
  - `robots/robot.py:Robot`：统一的机器人抽象，强制实现连接、校准、配置、读取观测、下发动作、断开等接口，并声明 `observation_features` 与 `action_features`。  
  - `robots/config.py:RobotConfig`：配置抽象，使用 `@RobotConfig.register_subclass("<type>")` 注册，`type` 决定 CLI/工厂的反射加载。
- **硬件抽象**  
  - 电机：`motors/motors_bus.py:MotorsBus` 抽象，具体实现 `FeetechMotorsBus`（`motors/feetech/feetech.py`）、`DynamixelMotorsBus`（`motors/dynamixel/dynamixel.py`）。电机由 `Motor(id, model, norm_mode)` 描述，校准数据用 `MotorCalibration`。
  - 相机：`cameras/camera.py:Camera` 抽象，OpenCV/RealSense 等实现通过 `cameras/utils.py:make_cameras_from_configs` 工厂加载。
- **遥操作**  
  - `teleoperators/teleoperator.py:Teleoperator` 抽象，与机器人类似，提供 `get_action/send_feedback`。工厂在 `teleoperators/utils.py`。
- **工厂与 CLI**  
  - `robots/utils.py:make_robot_from_config` 使用 `config.type` 选择具体类，是训练/遥操作/记录脚本的入口（如 `python -m lerobot.teleoperate ...`）。
  - 校准文件默认写入 `~/.cache/huggingface/lerobot/calibration/robots/<name>/<id>.json`（`HF_LEROBOT_CALIBRATION`）。

## 2. 前置准备

- **环境**：Python 3.10+，推荐在仓库根目录执行：  
  ```bash
  uv sync --extra dev --extra test
  # 或
  pip install -e ".[dev,test]"
  ```
- **硬件/驱动**：根据所用总线安装依赖  
  - Feetech：`pip install scservo-sdk`  
  - Dynamixel：`pip install dynamixel-sdk`  
  - RealSense：`pip install -e ".[intelrealsense]"`（如使用）
- **权限**：Linux 串口需加入 dialout 组或调整设备权限；相机需确保 udev/驱动正常。
- **准备信息**：电机 ID/型号、波特率、控制模式；相机分辨率/帧率；是否需要力矩关闭、限幅参数等。

## 3. 添加新机器人：步骤拆解

### 3.1 创建目录与配置类
1. 在 `src/lerobot/robots/<robot_name>/` 下创建模块（可参考 `koch_follower/`、`hope_jr/`）。  
2. 定义配置：继承 `RobotConfig`，使用 `@RobotConfig.register_subclass("<type>")` 注册。例如 `so100_follower/config_so100_follower.py`：  
   - 声明必填字段（如 `port`），可选字段（如 `max_relative_target`、`use_degrees`）。  
   - 相机字段使用 `CameraConfig`，`RobotConfig.__post_init__` 会校验 width/height/fps 非空。  
   - 需要额外验证时覆写 `__post_init__`（见 `HopeJrHandConfig` 对左右手合法性的检查）。

### 3.2 实现 Robot 子类骨架
- 关键类变量：`config_class`、`name` 必填。构造函数中通常：  
  - 调用 `super().__init__(config)` 以初始化校准路径/加载校准。  
  - 构造电机总线（`FeetechMotorsBus`/`DynamixelMotorsBus`），传入 `motors` 字典与 `calibration`。  
  - 通过 `make_cameras_from_configs(config.cameras)` 创建相机。
- 必须实现的方法/属性（见 `Robot` 抽象）：`observation_features`、`action_features`、`is_connected`、`connect`、`is_calibrated`、`calibrate`、`configure`、`get_observation`、`send_action`、`disconnect`。
- 观测/动作描述：  
  - 多数机械臂使用 `{ "<joint>.pos": float }` 形式（参考 `SO100Follower._motors_ft`）。  
  - 相机观测形状定义为 `(height, width, 3)`；整合时 `observation_features = {**motors, **cameras}`。  
  - 行为空间通常与电机观测一致；如 `SO100FollowerEndEffector` 则定义了 `dtype/shape/names` 风格。

### 3.3 连接、校准、配置模式
- `connect` 通常流程：检查 `is_connected`，调用 `bus.connect()`，若未校准则 `calibrate()`，连接相机，最后 `configure()`。重复连接要抛出 `DeviceAlreadyConnectedError`。  
- `calibrate` 模式：  
  - Feetech 示例：`SO100Follower.calibrate` 将力矩关闭、设置位置模式、获取半圈零点、扫描行程并写入 `MotorCalibration`，随后保存到文件。  
  - 支持已有校准文件复用（询问用户）或 GUI（`HopeJrArm` 使用 `RangeFinderGUI`）。  
  - 双臂场景可复用单臂配置并派生 ID（`BiSO100Follower` 左右臂使用 `id_suffix`）。
- `configure`：在力矩关闭上下文中设置控制模式/ PID/ 电流限制等（见 `koch_follower.configure`、`lekiwi.configure`）。
- 安全限幅：如需限制目标变化量，使用 `ensure_safe_goal_position` 处理 `max_relative_target`。

### 3.4 读取观测与下发动作
- 读：  
  - 位置同步读取：`bus.sync_read("Present_Position")`，并统一后缀 `.pos`。  
  - 特殊读取：`HopeJrArm` 为避免总线阻塞，单独读取肩关节。  
  - 相机：循环 `cam.async_read()`，记录耗时日志便于调试。
- 写：  
  - `goal_pos = {key.removesuffix(".pos"): val for key,val in action.items()}`；必要时读取当前位姿后做限幅。  
  - 速度/全向底盘：`LeKiwi` 将 `x/y/theta` 速度转换为轮速原始值再调用 `sync_write("Goal_Velocity", ...)`。
- 断开：关闭力矩（可配置）、断开相机，抛出 `DeviceNotConnectedError` 处理重复断开。

### 3.5 集成电机控制器
- **Feetech**：  
  - 导入 `FeetechMotorsBus`, `OperatingMode`, `MotorNormMode`。  
  - 若使用协议 1（见 `HopeJrHand`），注意 `sync_read` 不可用；校准流程需兼容。  
  - 可启用 `apply_drive_mode`，通过 `MotorCalibration.drive_mode` 标记反转（`HopeJrHand` 对左右手取反）。  
- **Dynamixel**：  
  - 导入 `DynamixelMotorsBus`, `OperatingMode`，使用 `EXTENDED_POSITION` / `CURRENT_POSITION` 等模式（参考 `koch_follower.configure`）。  
  - 支持影子 ID（如 `ViperX.configure` 的 `Secondary_ID`）。  
- **自定义电机**：  
  - 继承 `MotorsBus`，实现 `_handshake/read/write/sync_read/sync_write/disable_torque/enable_torque/configure_motors` 等；确保 `normalized_data` 支持角度/百分比映射。  
  - `MotorNormMode` 枚举提供常见归一化（`RANGE_M100_100`、`RANGE_0_100`、`DEGREES`）。

### 3.6 相机系统集成
- 使用 `CameraConfig` 子类（如 `OpenCVCameraConfig` 或 `RealSenseCameraConfig`）描述参数。  
- 通过 `make_cameras_from_configs` 构造；`RobotConfig.__post_init__` 会拒绝缺失 width/height/fps 的相机配置。  
- 自定义相机：继承 `Camera` 抽象实现 `find_cameras/connect/read/async_read/disconnect`，并在 `cameras/utils.py:make_cameras_from_configs` 中添加类型分支。

### 3.7 配置与 CLI 示例
- 在 `lerobot/configs`（或外部 CLI 参数）中提供配置对象：  
  ```python
  # 假设新增 robot.type=my_arm
  from lerobot.robots.my_arm.config_my_arm import MyArmConfig
  robot = MyArmConfig(
      id="lab01",
      port="/dev/ttyUSB0",
      max_relative_target=15,
      cameras={"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
  )
  ```
- 运行遥操作冒烟：  
  ```bash
  python -m lerobot.teleoperate --robot.type=my_arm --robot.port=/dev/ttyUSB0 --teleop.type=so100_leader
  ```
- 训练/记录脚本同样依赖 `make_robot_from_config`，确保 `config.type` 已注册并在工厂中添加分支。

### 3.8 遥操作支持（可选）
- 若需要新遥操作设备，仿照机器人流程：  
  - 在 `teleoperators/<name>/` 下实现 `Teleoperator` 子类与配置（参考 `so100_leader/so100_leader.py`、`bi_so100_leader/bi_so100_leader.py`）。  
  - 在 `teleoperators/utils.py:make_teleoperator_from_config` 注册新 `config.type`。  
  - 确保 `action_features` 与目标机器人的 `action_features` 对齐，必要时在 `get_action` 内做映射或缩放。

## 4. 代码模板

```python
# src/lerobot/robots/my_arm/config_my_arm.py
from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("my_arm")
@dataclass
class MyArmConfig(RobotConfig):
    port: str
    max_relative_target: float | None = 10.0
    disable_torque_on_disconnect: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
```

```python
# src/lerobot/robots/my_arm/my_arm.py
import logging, time
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
    config_class = MyArmConfig
    name = "my_arm"

    def __init__(self, config: MyArmConfig):
        super().__init__(config)
        self.config = config
        norm = MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={"shoulder": Motor(1, "sts3215", norm), "elbow": Motor(2, "sts3215", norm)},
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self):  # 可提前缓存
        cam_ft = {k: (cfg.height, cfg.width, 3) for k, cfg in self.config.cameras.items()}
        mot_ft = {f"{m}.pos": float for m in self.bus.motors}
        return {**mot_ft, **cam_ft}

    @cached_property
    def action_features(self):
        return {f"{m}.pos": float for m in self.bus.motors}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("connected")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self):
        # 简化示例：若已有校准文件则直接写入，否则使用半圈零点 + 行程扫描
        if self.calibration:
            self.bus.write_calibration(self.calibration)
            return
        self.bus.disable_torque()
        for m in self.bus.motors:
            self.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
        homing = self.bus.set_half_turn_homings()
        ranges = self.bus.record_ranges_of_motion()
        self.calibration = {
            name: MotorCalibration(id=mot.id, drive_mode=0, homing_offset=homing[name],
                                   range_min=ranges[0][name], range_max=ranges[1][name])
            for name, mot in self.bus.motors.items()
        }
        self.bus.write_calibration(self.calibration)
        self._save_calibration()

    def configure(self):
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for m in self.bus.motors:
                self.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs = {f"{m}.pos": v for m, v in self.bus.sync_read("Present_Position").items()}
        for key, cam in self.cameras.items():
            obs[key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        goal = {k.removesuffix(".pos"): v for k, v in action.items()}
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal = ensure_safe_goal_position(
                {k: (v, present[k]) for k, v in goal.items()},
                self.config.max_relative_target,
            )
        self.bus.sync_write("Goal_Position", goal)
        return {f"{k}.pos": v for k, v in goal.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("disconnected")
```

> 添加工厂映射：在 `robots/utils.py` 中为新的 `config.type` 增加分支 `elif config.type == "my_arm": from .my_arm import MyArm; return MyArm(config)`。

## 5. 测试指南

- **单元测试优先模拟硬件**：参考 `tests/robots/test_so100_follower.py`，通过 `unittest.mock.patch` 替换总线类、构造 `MagicMock` 模拟 `is_connected/sync_read/sync_write`，避免依赖真实设备。  
- **建议覆盖**：  
  - 连接/断开状态切换与异常抛出。  
  - `get_observation` 键名与数值格式正确（含相机观测时可模拟 numpy 数组）。  
  - `send_action` 对限幅、写入命令的调用。  
  - 校准逻辑可在无硬件时通过注入假数据或拆分出纯函数。
- **运行命令**：`pytest tests/robots/test_my_arm.py`；新增电机/相机工具类时同步补充 `tests/motors/*` 或 `tests/cameras/*`。

## 6. 常见问题与调试技巧

- **串口找不到或占用**：使用 `python -m lerobot.find_port` 查找；若提示 `DeviceAlreadyConnectedError`，确认无其他进程占用。  
- **校准失效/抖动**：确认 `MotorCalibration` 写回成功（Feetech/Dynamixel 均有 `write_calibration`），并检查 `Operating_Mode` 与 PID 设置。  
- **动作范围异常**：校验 `MotorNormMode` 与硬件角度范围是否匹配；必要时通过 `max_relative_target` 加限幅。  
- **相机尺寸错误**：`RobotConfig.__post_init__` 会拒绝缺省的 width/height/fps；确认旋转/色彩模式（`CameraConfig` 的 `rotation/color_mode`）。  
- **双臂或多设备延迟**：避免在循环中重复初始化相机/总线；使用 `cached_property` 缓存特征；对大循环记录 `perf_counter` 日志（见各机器人 `logger.debug`）。

## 7. 最佳实践

- 遵循仓库风格：行长 110，双引号，尽量补全类型标注与必要的 Google 风格 docstring。  
- 抛出语义化错误：连接态使用 `DeviceAlreadyConnectedError` / `DeviceNotConnectedError`，避免静默失败。  
- 校准文件复用：优先读已有校准；新增校准后调用 `_save_calibration()`，确保以 `config.id` 区分多机。  
- 将硬件常量配置化：端口、波特率、PID、限速均放在 `RobotConfig`，默认值保持安全；不要硬编码到逻辑里。  
- 工厂注册与文档同步：新增机器人后更新 `robots/utils.py` 工厂映射、在 `docs/add_robots/` 记录使用说明，并提供最小可运行示例/测试。  
- 遥操作匹配：确保 teleop 输出键与机器人 `action_features` 对齐，如键名不一致需在 teleop 层做映射或转换。

借助以上流程，你可以快速把新的机器人总线、相机、遥操作设备集成进 LeRobot，并通过现有的记录/遥操作/训练脚本完成端到端验证。***
