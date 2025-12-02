# LeRobot 新机器人支持技术说明（Codex AAA 版）

> 本文基于当前仓库的实现（如 `src/lerobot/robots/so100_follower/so100_follower.py`、`src/lerobot/robots/so101_follower/so101_follower.py`、`src/lerobot/robots/hope_jr/hope_jr_arm.py`、`src/lerobot/motors/feetech/feetech.py`、`src/lerobot/cameras/camera.py` 等）总结，完整阐述如何向 LeRobot 添加新的机器人硬件。阅读完本指南后，开发者应能独立完成配置、驱动、遥操作、测试与调试。

## 1. 架构概述

- **统一抽象**
  - `src/lerobot/robots/robot.py:Robot` 为所有机器人提供标准接口（连接、断开、校准、配置、读取观测、发送动作），并管理校准文件目录（`HF_LEROBOT_CALIBRATION/robots/<name>/<id>.json`）。
  - `src/lerobot/robots/config.py:RobotConfig` 负责配置注册，借助 `@RobotConfig.register_subclass("<type>")` 将类型暴露给 CLI/工厂。
  - `src/lerobot/robots/utils.py:make_robot_from_config` 根据 `config.type` 构造具体机器人，因此新类型必须在此添加分支。
- **硬件抽象层**
  - 电机/总线：`src/lerobot/motors/motors_bus.py` 定义 `MotorsBus`、`Motor`、`MotorCalibration`、`MotorNormMode` 等通用概念。Feetech、Dynamixel 等具体实现位于 `src/lerobot/motors/feetech/`、`src/lerobot/motors/dynamixel/`。
  - 相机：`src/lerobot/cameras/camera.py` 规定 `Camera` 接口，`src/lerobot/cameras/utils.py:make_cameras_from_configs` 会自动根据 `CameraConfig` 创建 OpenCV/RealSense 等设备。
- **遥操作**
  - `src/lerobot/teleoperators/teleoperator.py` 与机器人抽象类似，提供 `get_action/send_feedback`。新机器人若需遥操作，应确保与现有 teleop 的 `action_features` 对齐，必要时新增 teleop 并在 `src/lerobot/teleoperators/utils.py` 注册。
- **测试与工具链**
  - `tests/robots/test_so100_follower.py` 展示了如何通过 mock bus 验证连接、观测、动作逻辑；`tests/test_control_robot.py` 使用 `tests/mocks/mock_robot.py`、`tests/mocks/mock_teleop.py` 走通校准/遥操作/录制/回放管线。

## 2. 前置准备

1. **环境与依赖**
   - Python ≥ 3.10，推荐在仓库根目录执行：
     ```bash
     uv sync --extra dev --extra test
     # 或
     pip install -e .[dev,test]
     ```
   - 电机 SDK：Feetech 需 `scservo-sdk`，Dynamixel 需 `dynamixel-sdk`；相机若用 RealSense 则启用 `.[intelrealsense]` 额外依赖。
   - Ruff/pytest：遵循 `AGENTS.md` 中的 `python -m ruff check src tests`、`pytest tests/` 标准流程。
2. **硬件准备**
   - 串口权限：Linux 加入 `dialout` 组或设置 udev 规则；Windows/macOS 需确认 COM/tty 端口号。
   - 明确每个关节的电机 ID、型号、驱动模式、行程限制，准备 URDF（若需要末端控制，如 `src/lerobot/robots/so100_follower/so100_follower_end_effector.py`）。
   - 相机的 `width/height/fps` 必须在配置中声明，否则 `RobotConfig.__post_init__` 会抛错。
3. **目录规划**
   - 机器人代码置于 `src/lerobot/robots/<robot_name>/`；配置单独文件（如 `config_<robot_name>.py`）。
   - 若涉及遥操作，创建 `src/lerobot/teleoperators/<teleop_name>/`。
   - 文档、示例、配置需同步更新（`docs/add_robots/`、`lerobot/configs/` 等）。

## 3. 实现步骤

### 3.1 新建配置类

- 在 `<robot_dir>/config_<robot>.py` 中继承 `RobotConfig`：
  ```python
  @RobotConfig.register_subclass("my_robot")
  @dataclass
  class MyRobotConfig(RobotConfig):
      port: str
      max_relative_target: float | None = None
      cameras: dict[str, CameraConfig] = field(default_factory=dict)
      disable_torque_on_disconnect: bool = True
  ```
- 常见字段示例：
  - **通信**：`port`、`baudrate`、`protocol_version`（参考 `HopeJrHandConfig`）。
  - **安全**：`max_relative_target` 用于 `ensure_safe_goal_position`（`src/lerobot/robots/utils.py`）。
  - **相机**：`dict[str, CameraConfig]`，使用 `OpenCVCameraConfig` 或 `RealSenseCameraConfig`。
  - **校准**：`id` 与 `calibration_dir` 由基类管理，可用于区分多台同型号机器人。

### 3.2 继承 `Robot`

1. 设置 `config_class` 与 `name`（字符串与 `config.type` 保持一致）。
2. 在 `__init__` 中：
   - 调用 `super().__init__(config)`，以加载/创建 `calibration_fpath`。
   - 实例化电机总线，例如 `FeetechMotorsBus(port=config.port, motors={...}, calibration=self.calibration)`。
     - 每个 `Motor` 指定 `id`、`model`、`MotorNormMode`。选择参考 `SO100Follower`（角度/百分比）或 `HopeJrHand`（Protocol 1 + `MotorNormMode.RANGE_0_100`）。
   - 调用 `make_cameras_from_configs(config.cameras)` 获取字典。
3. 可根据需要缓存 `observation_features`、`action_features`（见 `HopeJrArm` 使用 `@cached_property`）。

### 3.3 必需接口实现

| 方法/属性 | 关键点 | 参考 |
| --- | --- | --- |
| `observation_features` | 描述观测结构。关节位置通常为 `{f"{motor}.pos": float}`，相机返回 `(H, W, 3)` | `SO100Follower._motors_ft/_cameras_ft` |
| `action_features` | 通常与关节观测一致；如末端控制可返回 `dtype+shape+names` 字典 | `SO100FollowerEndEffector.action_features` |
| `is_connected` | 合并电机和相机状态，或维护布尔标志 | `HopeJrHand.is_connected` |
| `connect` | 检查 `DeviceAlreadyConnectedError`，`bus.connect()` → `calibrate()` → `cam.connect()` → `configure()` | `SO101Follower.connect` |
| `is_calibrated` | 委托给总线（`bus.is_calibrated`）或内部状态 | `SO100Follower.is_calibrated` |
| `calibrate` | 结合校准文件、GUI 或手动流程；结束后调用 `_save_calibration()` | `HopeJrArm.calibrate`（`RangeFinderGUI`） |
| `configure` | 在 `bus.torque_disabled()` 上下文中写寄存器（模式、PID、限速） | `SO101Follower.configure` |
| `get_observation` | 先 `bus.sync_read("Present_Position")`，再 `cam.async_read()` | `SO100Follower.get_observation` |
| `send_action` | 把 `.pos` 键转换成电机名，必要时调用 `ensure_safe_goal_position`，再 `sync_write("Goal_Position", goal_pos)` | `HopeJrArm.send_action` |
| `disconnect` | 关闭电机扭矩（可配置）、断开相机、记录日志 | `SO100Follower.disconnect` |

### 3.4 电机与校准集成

1. **Feetech（`src/lerobot/motors/feetech/feetech.py`）**
   - `OperatingMode.POSITION/VELOCITY` 控制模式；`HopeJrHand` 在配置时仅调用 `bus.configure_motors()`。
   - 当 `protocol_version=1`（手部）时不支持 `sync_read`，需逐个 `read`。
   - 校准策略：
     - 复用已有文件（提示用户，见 `SO101Follower.calibrate`）。
     - 半圈零点 + 行程扫描，生成 `MotorCalibration` 并 `bus.write_calibration()`。
     - GUI 测量：`RangeFinderGUI`（`HopeJrArm`、`HopeJrHand`）。
   - 反转驱动：为左右手互换，`HopeJrHand.calibrate` 将特定关节 `drive_mode` 设为 1。
2. **Dynamixel（`src/lerobot/motors/dynamixel/`）**
   - 常用模式 `OperatingMode.EXTENDED_POSITION`、`CURRENT_POSITION` 等，`ViperX.configure` 展示了如何设置影子 ID。
   - 需要在 `bus.configure_motors()` 后写附加寄存器。
3. **自定义总线**
   - 继承 `MotorsBus` 并实现 `_handshake/read/write/sync_read/sync_write/configure_motors` 等；遵循 `MotorNormMode` 归一化逻辑。
   - 在机器人类中像现有总线一样使用，unit test 可通过 mock bus 验证。

### 3.5 相机集成

- `CameraConfig` 子类位于 `src/lerobot/cameras/opencv/configuration_opencv.py`、`src/lerobot/cameras/realsense/`。
- `make_cameras_from_configs` 会根据 `config.cameras` 创建对应实例并存于 `self.cameras` 字典。
- 必须在 `connect()` 中遍历 `cam.connect()` 并在 `disconnect()` 中关闭。
- 观测中通常将 key 设为命名空间（如 `"front_rgb"`），值为 `np.ndarray`。在 `observation_features` 中同步声明 `(height, width, 3)`。
- 若需自定义相机：继承 `Camera`，实现 `find_cameras/connect/read/async_read/disconnect`，并在 `make_cameras_from_configs` 添加新 `cfg.type` 分支。

### 3.6 配置文件与 CLI

- `lerobot/configs/` 下的配置（如 `default.py`, `train.py`）会嵌入 `RobotConfig`，CLI 通过 `draccus` 解析。添加新机器人时：
  1. 在配置文件中导入新的 `MyRobotConfig`，提供默认字段。
  2. 或在命令行直接传 `--robot.type=my_robot --robot.port=/dev/ttyUSB0`。
- 若新增 teleop，需要在 `lerobot/teleoperate.py` 对应 `TeleoperateConfig` 中暴露参数。
- 为了让 `make_robot_from_config` 识别新类型，别忘了修改 `src/lerobot/robots/utils.py`。

### 3.7 遥操作支持

- 若现有 teleop（`teleoperators/so100_leader`、`teleoperators/gamepad` 等）已满足需求，只需在 `python -m lerobot.teleoperate` 时指定 `--teleop.type=...` 并确保维度匹配。
- 若需新遥操作端：
  1. 在 `src/lerobot/teleoperators/<name>/` 下创建配置和类，继承 `Teleoperator`。
  2. 实现 `action_features`（teleop → robot 指令）和 `feedback_features`（robot → teleop 反馈）。
  3. 在 `teleoperators/utils.py:make_teleoperator_from_config` 添加类型分支。
  4. 参照 `tests/mocks/mock_teleop.py` 为其编写单元测试。

## 4. 代码示例模板

以下模板演示了一个基于 Feetech 总线的 4 自由度手臂，覆盖配置、机器人类、`robots/utils.py` 注册以及遥操作映射。请根据实际硬件调整 ID、型号和校准策略。

```python
# src/lerobot/robots/my_arm/config_my_arm.py
from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("my_arm")
@dataclass
class MyArmConfig(RobotConfig):
    port: str
    max_relative_target: float | None = 15.0
    disable_torque_on_disconnect: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    protocol_version: int = 0
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
            protocol_version=config.protocol_version,
            motors={
                "shoulder": Motor(1, "sts3215", norm),
                "elbow": Motor(2, "sts3215", norm),
                "wrist": Motor(3, "sts3215", norm),
                "gripper": Motor(4, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
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

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            self.bus.write_calibration(self.calibration)
            return
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        input("Move robot to mid-range and press ENTER…")
        homing = self.bus.set_half_turn_homings()
        range_mins, range_maxes = self.bus.record_ranges_of_motion()
        self.calibration = {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
            for motor, m in self.bus.motors.items()
        }
        self.bus.write_calibration(self.calibration)
        self._save_calibration()

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs = {f"{m}.pos": v for m, v in self.bus.sync_read("Present_Position").items()}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        goal = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal = ensure_safe_goal_position(
                {m: (g, present[m]) for m, g in goal.items()},
                self.config.max_relative_target,
            )
        self.bus.sync_write("Goal_Position", goal)
        return {f"{m}.pos": v for m, v in goal.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
```

```python
# src/lerobot/robots/utils.py （新增分支）
elif config.type == "my_arm":
    from .my_arm.my_arm import MyArm
    return MyArm(config)
```

```python
# 可选：src/lerobot/teleoperators/my_arm_leader/my_arm_leader.py
# 继承 Teleoperator，实现 get_action，将遥操作输入映射到 {"shoulder.pos": float, ...}
```

## 5. 配置与运行示例

1. **配置文件示例（`lerobot/configs/default.py` 片段）**
   ```python
   from lerobot.robots.my_arm.config_my_arm import MyArmConfig
   from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

   default_robot = MyArmConfig(
       id="lab-arm-01",
       port="/dev/ttyUSB0",
       max_relative_target=12.0,
       cameras={
           "front_rgb": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
       },
   )
   ```
2. **遥操作冒烟**
   ```bash
   python -m lerobot.teleoperate --robot.type=my_arm --robot.port=/dev/ttyUSB0 \
       --teleop.type=so100_leader --teleop.port=/dev/ttyUSB1 --teleop.side=right
   ```
3. **录制/回放**（参考 `tests/test_control_robot.py` 流程）
   ```bash
   python -m lerobot.record --robot.type=my_arm --dataset.repo-id <your_repo> --teleop.type=gamepad
   python -m lerobot.replay --robot.type=my_arm --dataset.repo-id <your_repo> --dataset.episode=0
   ```

## 6. 测试指南

- **单元测试**
  - 仿照 `tests/robots/test_so100_follower.py`，对 `connect/get_observation/send_action` 编写 mock 测试。关键做法：
    - 用 `unittest.mock.patch` 替换 `FeetechMotorsBus`，返回具备 `connect/sync_read/sync_write` 的 MagicMock。
    - 验证 `is_connected` 状态机、观测字段、动作写入。
  - 若新增 teleop，同样参照 `tests/mocks/mock_teleop.py`。
- **端到端脚本**
  - 使用 `tests/test_control_robot.py` 中 `calibrate/teleoperate/record/replay` 的 Mock 流程确保 CLI 参数路径正确。
- **硬件回归**
  - 上位机：实机测试时记录日志（`logger.debug` 量测 `get_observation` 耗时，参考 `SO100Follower.get_observation`）。
  - 提交 PR 前至少运行：
    ```bash
    python -m ruff check src tests
    python -m ruff format --check src tests
    pytest tests/robots/test_my_arm.py
    ```
  - 需要硬件的长测试可放在 `tests/robots/` 并使用 `pytest.mark.hardware`，在 CI 中跳过。

## 7. 常见问题与调试技巧

| 问题 | 现象 | 排查/解决 |
| --- | --- | --- |
| 校准文件缺失或与电机不符 | `connect` 日志提示 “Mismatch between calibration...” | 复用 `self.calibration` 或重新运行 `calibrate()`；确保 `self.id` 与校准文件对应，必要时删除旧文件。
| Feetech Protocol 1 不支持 `sync_read` | `NotImplementedError` | 像 `HopeJrHand.get_observation` 一样循环单独 `read`。
| `max_relative_target` 限制太严 | 动作被截断并打印警告 | 调整配置或提供每个关节的 dict；日志由 `ensure_safe_goal_position` 输出。
| 相机参数缺失 | 初始化报 `ValueError` | 在 `CameraConfig` 中设置 `width/height/fps`；若使用视频文件需确认路径正确。
| 遥操作与机器人维度不匹配 | `KeyError` 或关节不动 | 确保 teleop `action_features` 与 robot `action_features` 相同；必要时在 teleop 中重命名键。
| 设备连接状态错乱 | 多次调用 `connect()` 不报错却行为异常 | 按照基类约定抛出 `DeviceAlreadyConnectedError`/`DeviceNotConnectedError`，避免半连接状态。
| Dynamixel 双电机关节不同步 | 一侧输出过载 | 类似 `ViperX.configure` 使用 `Secondary_ID` 同步影子电机。
| 末端控制震荡 | 逆解不收敛或越界 | 将末端坐标限制在 `end_effector_bounds`（参考 `SO100FollowerEndEffector.send_action`），并在 `config` 中调小 `end_effector_step_sizes`。

## 8. 最佳实践

- **代码风格**：遵循仓库约束（110 列、双引号、`snake_case`），复杂逻辑适当添加注释。
- **错误处理**：
  - 使用 `DeviceAlreadyConnectedError`、`DeviceNotConnectedError` 明确状态。
  1-1. 对用户交互（如校准提示）提供清晰提示文字。
- **安全性**：
  - 在 `configure` 中借助 `bus.torque_disabled()` 写寄存器，防止意外运动。
  - 在 `send_action` 中默认限幅、校验键名，必要时加入 `np.clip`（参考 `SO100FollowerEndEffector` 对末端坐标的 `np.clip`）。
- **扩展性**：
  - 将硬件常量集中在配置或类常量中，方便日后通过 CLI 覆盖。
  - 若类需要局部差异（如 `HopeJr` 手/臂），优先复用基类或组合。
- **调试日志**：用 `logger.debug` 记录 `get_observation`/`send_action` 耗时（见 `SO101Follower`），方便排查串口/USB 延迟。
- **文档/示例**：向 `docs/add_robots/`、`examples/` 添加说明，必要时提供 `*.mdx` 硬件安装指南（参见 `src/lerobot/robots/so100_follower/so100.mdx`）。

---

通过以上流程，从配置注册、电机/相机接入、遥操作联调到测试/调试的完整闭环已经建立。建议在新硬件上线前，先使用 mock 测试验证接口，再在安全条件下进行慢速校准，最后运行遥操作与录制脚本完成端到端验证。
