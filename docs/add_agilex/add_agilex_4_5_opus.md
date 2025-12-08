# Agilex 双臂机器人接入 LeRobot 实现路径

> 本文档由 Claude Opus 4.5 生成，描述如何将 Agilex Piper 双臂机器人接入 LeRobot 框架。

## 1. 目标概述

将 Agilex Piper 双臂机器人接入 LeRobot 框架，实现以下功能：

- `lerobot-record`: 录制双臂机器人的演示数据（关节角度 + ROS 相机图像）
- `lerobot-replay`: 回放录制的数据
- `lerobot-dataset-viz`: 可视化录制的数据集

## 2. 现有代码资源分析

### 2.1 agilex_script - 关节录制回放

**路径**: `agilex_script/`

**核心文件**:
| 文件 | 功能 |
|------|------|
| `record_dual_arm.py` | 双臂关节角度录制（被动监听 CAN） |
| `playback_dual_arm.py` | 双臂关节角度回放 |
| `agilex_infer.py` | Piper SDK 封装接口 |
| `get_state.py` | 关节状态查询 |

**关键特性**:
- 基于 Piper SDK (`C_PiperInterface_V2`)
- 支持 100Hz 采样频率
- 数据格式：JSON（关节角度单位：度）
- 单位转换：度 ↔ 毫度 (×1000)

**核心接口**:
```python
# 连接机械臂（被动监听模式）
arm = C_PiperInterface_V2(port)
arm.ConnectPort(piper_init=False, start_thread=True)

# 读取关节反馈
joint_msg = arm.GetArmJointMsgs()
joint_deg = [joint_msg.joint_state.joint_1 / 1000.0, ...]  # 毫度转度

# 发送关节命令
arm.JointCtrl(j1, j2, j3, j4, j5, j6)  # 单位：毫度

# 夹爪控制
arm.GripperCtrl(angle, effort, status_code, set_zero)  # angle 单位：万分之一 mm
```

### 2.2 aiglex_origin_code - ROS 相机

**路径**: `aiglex_origin_code/`

**核心文件**:
| 文件 | 功能 |
|------|------|
| `collect_data.py` | ROS 相机数据采集 |
| `replay_data.py` | 数据重播到 ROS 话题 |

**相机话题配置**:
```python
# RGB 相机
'/camera_f/color/image_raw'   # 前置相机 (cam_high)
'/camera_l/color/image_raw'   # 左腕部相机 (cam_left_wrist)
'/camera_r/color/image_raw'   # 右腕部相机 (cam_right_wrist)

# 深度相机（可选）
'/camera_f/depth/image_raw'
'/camera_l/depth/image_raw'
'/camera_r/depth/image_raw'
```

**关键特性**:
- 基于 ROS + CvBridge
- 多相机时间同步（基于时间戳最小值）
- 图像分辨率：480×640
- 采样频率：30Hz

**核心接口**:
```python
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

bridge = CvBridge()

# 订阅相机话题
rospy.Subscriber(topic, Image, callback, queue_size=1000, tcp_nodelay=True)

# ROS Image → OpenCV
img = bridge.imgmsg_to_cv2(msg, 'passthrough')

# OpenCV → ROS Image
msg = bridge.cv2_to_imgmsg(img, "bgr8")
```

## 3. LeRobot 机器人注册机制

### 3.1 核心架构（按当前仓库结构对齐）

```
src/lerobot/robots/
├── config.py              # RobotConfig 基类（ChoiceRegistry）
├── robot.py               # Robot 抽象基类
├── utils.py               # make_robot_from_config 工厂函数
├── __init__.py            # 导出所有机器人
├── so100_follower/        # 参考实现
│   ├── __init__.py
│   ├── config_so100_follower.py
│   └── so100_follower.py
└── agilex/                # 新增目录（当前为空，需要补齐）
    ├── __init__.py
    ├── config_agilex.py
    └── agilex.py

src/lerobot/cameras/
└── ros_camera/            # 新增目录
    ├── __init__.py
    ├── config_ros.py      # Config 派生自 CameraConfig
    └── ros_camera.py      # Camera 实现
```

### 3.2 注册流程

1. **配置类注册** - 使用 `@RobotConfig.register_subclass("robot_type")` 装饰器
2. **工厂函数** - 在 `robots/utils.py` 的 `make_robot_from_config()` 中添加分支；相机在 `cameras/utils.py` 添加分支
3. **导出** - 在各自 `__init__.py` 中导入配置类和实现类（当前仓库的 Agilex 目录为空，导入会报错，需要补齐）

### 3.3 必须实现的接口

```python
class Robot(abc.ABC):
    # 必须设置的类属性
    config_class: type[RobotConfig]
    name: str

    # 必须实现的属性
    @property
    def observation_features(self) -> dict: ...
    @property
    def action_features(self) -> dict: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_calibrated(self) -> bool: ...

    # 必须实现的方法
    def connect(self, calibrate: bool = True) -> None: ...
    def calibrate(self) -> None: ...
    def configure(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: ...
    def disconnect(self) -> None: ...
```

## 4. 实现路径

### 阶段一：创建 ROS 相机模块

**目标**: 封装 ROS 相机为 LeRobot 兼容的相机类

**文件**: `src/lerobot/cameras/ros_camera/config_ros.py`、`src/lerobot/cameras/ros_camera/ros_camera.py`

```python
from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig

@CameraConfig.register_subclass("ros")
@dataclass
class RosCameraConfig(CameraConfig):
    """ROS 相机配置"""
    topic: str                          # ROS 话题名
    width: int = 640
    height: int = 480
    fps: int = 30
    use_depth: bool = False
    depth_topic: str | None = None


class RosCamera:
    """ROS 相机封装"""
    config_class = RosCameraConfig
    name = "ros"

    def __init__(self, config: RosCameraConfig):
        self.config = config
        self._is_connected = False
        self._bridge = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def connect(self):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        if not rospy.core.is_initialized():
            rospy.init_node('lerobot_ros_camera', anonymous=True)

        self._bridge = CvBridge()
        rospy.Subscriber(
            self.config.topic,
            Image,
            self._image_callback,
            queue_size=1,
            tcp_nodelay=True
        )
        self._is_connected = True

    def _image_callback(self, msg):
        with self._frame_lock:
            self._latest_frame = self._bridge.imgmsg_to_cv2(msg, 'rgb8')

    def async_read(self) -> np.ndarray:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def read(self, color_mode=None):
        # 与 Camera 接口对齐的同步读取，必要时直接复用 async_read
        return self.async_read()

    def disconnect(self):
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected
```

**注册相机**:

在 `src/lerobot/cameras/ros_camera/__init__.py` 中导出 `RosCamera`/`RosCameraConfig`，并在 `src/lerobot/cameras/__init__.py` 中添加：
```python
from .ros_camera import RosCamera, RosCameraConfig
```

在 `src/lerobot/cameras/utils.py` 的 `make_cameras_from_configs()` 中添加：
```python
elif config.type == "ros":
    from .ros_camera import RosCamera
    cameras[name] = RosCamera(config)
```

---

### 阶段二：创建 Agilex 机器人模块

**目标**: 实现 Agilex 双臂机器人类

#### 2.1 配置类

**文件**: `src/lerobot/robots/agilex/config_agilex.py`

```python
from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import CameraConfig

@RobotConfig.register_subclass("agilex_bimanual")
@dataclass
class AgilexBimanualConfig(RobotConfig):
    """Agilex 双臂机器人配置"""
    # CAN 端口
    left_arm_port: str = "can_left"
    right_arm_port: str = "can_right"

    # 使能超时
    enable_timeout: float = 5.0

    # 运动速度 (10-100%)
    motion_speed: int = 50

    # 断开时禁用力矩
    disable_torque_on_disconnect: bool = True

    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 是否使用度数（True）或归一化值（False）
    use_degrees: bool = True

    # 关节数量
    num_joints_per_arm: int = 6

    # 夹爪配置
    use_gripper: bool = True
```

#### 2.2 机器人类

**文件**: `src/lerobot/robots/agilex/agilex.py`

```python
from functools import cached_property
from typing import Any
from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from .config_agilex import AgilexBimanualConfig

# Piper SDK 导入
from piper_sdk import C_PiperInterface_V2


class AgilexBimanual(Robot):
    """Agilex Piper 双臂机器人"""

    config_class = AgilexBimanualConfig
    name = "agilex_bimanual"

    # 关节名称
    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "wrist_rotate"
    ]

    def __init__(self, config: AgilexBimanualConfig):
        super().__init__(config)
        self.config = config

        self._left_arm: C_PiperInterface_V2 | None = None
        self._right_arm: C_PiperInterface_V2 | None = None
        self._is_connected = False

        # 初始化相机
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征定义"""
        features = {}

        # 左臂关节位置
        for joint in self.JOINT_NAMES:
            features[f"left_{joint}.pos"] = float

        # 右臂关节位置
        for joint in self.JOINT_NAMES:
            features[f"right_{joint}.pos"] = float

        # 夹爪位置
        if self.config.use_gripper:
            features["left_gripper.pos"] = float
            features["right_gripper.pos"] = float

        # 相机图像
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)

        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征定义"""
        features = {}

        # 左臂关节位置
        for joint in self.JOINT_NAMES:
            features[f"left_{joint}.pos"] = float

        # 右臂关节位置
        for joint in self.JOINT_NAMES:
            features[f"right_{joint}.pos"] = float

        # 夹爪位置
        if self.config.use_gripper:
            features["left_gripper.pos"] = float
            features["right_gripper.pos"] = float

        return features

    @property
    def is_connected(self) -> bool:
        arm_connected = self._is_connected
        cam_connected = all(cam.is_connected for cam in self.cameras.values())
        return arm_connected and cam_connected

    @property
    def is_calibrated(self) -> bool:
        # Agilex 机器人使用绝对编码器，无需校准
        return True

    def connect(self, calibrate: bool = True) -> None:
        """连接机器人"""
        if self.is_connected:
            raise RuntimeError("Robot already connected")

        # 连接左臂
        self._left_arm = self._connect_arm(self.config.left_arm_port)

        # 连接右臂
        self._right_arm = self._connect_arm(self.config.right_arm_port)

        self._is_connected = True

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def _connect_arm(self, port: str) -> C_PiperInterface_V2:
        """连接单个机械臂"""
        import time

        arm = C_PiperInterface_V2(port)
        arm.ConnectPort()

        # 检查并恢复状态
        status = arm.GetArmStatus()
        if status.arm_status.motion_status != 0:
            arm.EmergencyStop(0x02)  # 恢复
            time.sleep(0.5)

        # 使能机械臂
        start = time.time()
        while time.time() - start < self.config.enable_timeout:
            if arm.EnablePiper():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Failed to enable arm on {port}")

        # 切换到 Joint 模式
        arm.MotionCtrl_2(0x01, 0x01, self.config.motion_speed, 0x00)

        return arm

    def calibrate(self) -> None:
        """校准（Agilex 使用绝对编码器，无需校准）"""
        pass

    def configure(self) -> None:
        """配置机器人"""
        pass

    def get_observation(self) -> dict[str, Any]:
        """获取观测"""
        if not self.is_connected:
            raise RuntimeError("Robot not connected")

        obs = {}

        # 读取左臂关节
        left_joints = self._read_joint_deg(self._left_arm)
        for i, joint in enumerate(self.JOINT_NAMES):
            obs[f"left_{joint}.pos"] = left_joints[i]

        # 读取右臂关节
        right_joints = self._read_joint_deg(self._right_arm)
        for i, joint in enumerate(self.JOINT_NAMES):
            obs[f"right_{joint}.pos"] = right_joints[i]

        # 读取夹爪
        if self.config.use_gripper:
            obs["left_gripper.pos"] = self._read_gripper_mm(self._left_arm)
            obs["right_gripper.pos"] = self._read_gripper_mm(self._right_arm)

        # 读取相机
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()

        return obs

    def _read_joint_deg(self, arm: C_PiperInterface_V2) -> list[float]:
        """读取关节角度（度）"""
        msg = arm.GetArmJointMsgs()
        js = msg.joint_state
        return [
            js.joint_1 / 1000.0,
            js.joint_2 / 1000.0,
            js.joint_3 / 1000.0,
            js.joint_4 / 1000.0,
            js.joint_5 / 1000.0,
            js.joint_6 / 1000.0,
        ]

    def _read_gripper_mm(self, arm: C_PiperInterface_V2) -> float:
        """读取夹爪位置（mm）"""
        msg = arm.GetArmGripperMsgs()
        return msg.gripper_state.grippers_angle / 10000.0

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作"""
        if not self.is_connected:
            raise RuntimeError("Robot not connected")

        # 提取左臂关节
        left_joints = [
            action[f"left_{joint}.pos"] for joint in self.JOINT_NAMES
        ]

        # 提取右臂关节
        right_joints = [
            action[f"right_{joint}.pos"] for joint in self.JOINT_NAMES
        ]

        # 发送关节命令
        self._send_joint_deg(self._left_arm, left_joints)
        self._send_joint_deg(self._right_arm, right_joints)

        # 发送夹爪命令
        if self.config.use_gripper:
            if "left_gripper.pos" in action:
                self._send_gripper_mm(self._left_arm, action["left_gripper.pos"])
            if "right_gripper.pos" in action:
                self._send_gripper_mm(self._right_arm, action["right_gripper.pos"])

        return action

    def _send_joint_deg(self, arm: C_PiperInterface_V2, joints_deg: list[float]) -> None:
        """发送关节角度（度）"""
        # 转换为毫度
        joints_mdeg = [int(round(deg * 1000.0)) for deg in joints_deg]
        arm.JointCtrl(*joints_mdeg)

    def _send_gripper_mm(self, arm: C_PiperInterface_V2, gripper_mm: float) -> None:
        """发送夹爪位置（mm）"""
        angle_int = int(round(gripper_mm * 10000.0))
        arm.GripperCtrl(angle_int, 1000, 0x01, 0x00)

    def disconnect(self) -> None:
        """断开连接"""
        if not self.is_connected:
            return

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        # 断开机械臂
        if self._left_arm:
            if self.config.disable_torque_on_disconnect:
                self._left_arm.DisablePiper()
            self._left_arm.DisconnectPort()
            self._left_arm = None

        if self._right_arm:
            if self.config.disable_torque_on_disconnect:
                self._right_arm.DisablePiper()
            self._right_arm.DisconnectPort()
            self._right_arm = None

        self._is_connected = False
```

> 与基类约定保持一致：`connect`/`disconnect` 要可重入、出错时能回滚；`is_connected` 同时检查双臂与全部相机。

#### 2.3 模块导出

**文件**: `src/lerobot/robots/agilex/__init__.py`

```python
from .config_agilex import AgilexBimanualConfig
from .agilex import AgilexBimanual

__all__ = ["AgilexBimanualConfig", "AgilexBimanual"]
```

---

### 阶段三：注册到 LeRobot

#### 3.1 更新工厂函数

**文件**: `src/lerobot/robots/utils.py`

在 `make_robot_from_config()` 函数中添加：

```python
elif config.type == "agilex_bimanual":
    from .agilex import AgilexBimanual
    return AgilexBimanual(config)
```

#### 3.2 更新主导出

**文件**: `src/lerobot/robots/__init__.py`

添加导入：

```python
from .agilex import AgilexBimanual, AgilexBimanualConfig
```

> 仓库当前 `robots/__init__.py` 已提前导入不存在的 Agilex 符号，会导致 `import lerobot.robots` 直接失败。补齐上述实现后应同步修正导出顺序，确保导入正常。

---

### 阶段四：创建遥操作器（可选）

如果需要使用主臂进行遥操作，需要创建对应的 Teleoperator。

**文件**: `src/lerobot/teleoperators/agilex_leader/`

```python
@TeleoperatorConfig.register_subclass("agilex_bimanual_leader")
@dataclass
class AgilexBimanualLeaderConfig(TeleoperatorConfig):
    left_arm_port: str = "can_left"
    right_arm_port: str = "can_right"


class AgilexBimanualLeader(Teleoperator):
    """Agilex 双臂主臂遥操作器"""

    config_class = AgilexBimanualLeaderConfig
    name = "agilex_bimanual_leader"

    def get_action(self) -> dict[str, Any]:
        """读取主臂位置作为动作"""
        # 被动监听模式连接
        # 读取关节反馈和夹爪状态
        ...
```

---

## 5. 使用示例

### 5.1 录制数据

```bash
lerobot-record \
    --robot.type=agilex_bimanual \
    --robot.left_arm_port=can_left \
    --robot.right_arm_port=can_right \
    --robot.id=agilex_dual \
    --robot.cameras='{
        cam_high: {type: ros, topic: /camera_f/color/image_raw, width: 640, height: 480, fps: 30},
        cam_left_wrist: {type: ros, topic: /camera_l/color/image_raw, width: 640, height: 480, fps: 30},
        cam_right_wrist: {type: ros, topic: /camera_r/color/image_raw, width: 640, height: 480, fps: 30}
    }' \
    --dataset.repo_id=${HF_USER}/agilex-bimanual-demo \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick and place object" \
    --teleop.type=agilex_bimanual_leader \
    --teleop.left_arm_port=can_left \
    --teleop.right_arm_port=can_right
```

### 5.2 回放数据

```bash
lerobot-replay \
    --robot.type=agilex_bimanual \
    --robot.left_arm_port=can_left \
    --robot.right_arm_port=can_right \
    --robot.id=agilex_dual \
    --dataset.repo_id=${HF_USER}/agilex-bimanual-demo \
    --dataset.episode=0
```

### 5.3 可视化数据

```bash
lerobot-dataset-viz \
    --repo-id ${HF_USER}/agilex-bimanual-demo \
    --episode-index 0
```

---

## 6. 文件清单

### 需要新建的文件

| 文件路径 | 说明 |
|---------|------|
| `src/lerobot/cameras/ros_camera/config_ros.py` | ROS 相机配置 |
| `src/lerobot/cameras/ros_camera/ros_camera.py` | ROS 相机封装 |
| `src/lerobot/robots/agilex/__init__.py` | 模块导出 |
| `src/lerobot/robots/agilex/config_agilex.py` | 配置类 |
| `src/lerobot/robots/agilex/agilex.py` | 机器人实现 |
| `src/lerobot/teleoperators/agilex_leader/__init__.py` | 遥操作器导出（可选） |
| `src/lerobot/teleoperators/agilex_leader/config_agilex_leader.py` | 遥操作器配置（可选） |
| `src/lerobot/teleoperators/agilex_leader/agilex_leader.py` | 遥操作器实现（可选） |

### 需要修改的文件

| 文件路径 | 修改内容 |
|---------|---------|
| `src/lerobot/cameras/__init__.py` | 添加 RosCamera 导出 |
| `src/lerobot/cameras/utils.py` | 添加 ROS 相机工厂分支 |
| `src/lerobot/robots/__init__.py` | 添加 Agilex 导出 |
| `src/lerobot/robots/utils.py` | 添加 Agilex 工厂分支 |
| `src/lerobot/teleoperators/__init__.py` | 添加遥操作器导出（可选） |
| `src/lerobot/teleoperators/utils.py` | 添加遥操作器工厂分支（可选） |

---

## 4.5 数据格式对齐（参考 `demo_data_meta`）

`demo_data_meta/info.json` 展示了 `so101_follower` 的 v3.0 数据模式（30Hz）：动作与状态都是 6 个标量，图像键为 `observation.images.top/wrist`，其余列为系统自动添加的 `timestamp/frame_index/episode_index/task_index`。Agilex 双臂接入时请对齐同样的列名和分组，避免无关字段进入 `action/observation.state`：

- 必选列
  - `action`：浮点数组。建议命名为 `left_<joint>.pos`、`right_<joint>.pos`，以及可选 `left_gripper.pos`、`right_gripper.pos`（与 `observation.state` 同名同序）。记录时保持度/毫米单位，与 `send_action` 的接口一致。
  - `observation.state`：同一批标量键，值来自 `_read_joint_deg` 和 `_read_gripper_mm`。
  - `observation.images.<cam>`：ROS 相机帧，键名与摄像头配置一致，例如 `observation.images.cam_high`、`observation.images.cam_left_wrist`、`observation.images.cam_right_wrist`，shape 统一为 `(480, 640, 3)`，fps 30。
  - 系统列由 `lerobot-record` 自动填充：`timestamp`（秒）、`frame_index`、`episode_index`、`task_index`、`index`。
- 建议丢弃或放入 `info` 的字段
  - `record_dual_arm.py` 中的 `*_ctrl_deg`、电机温度/电流等高频诊断字段、`gripper_ctrl` 指令记录，如果不是回放必须的，避免写入 `action/observation.state`；必要时可单独写到自定义 `info` 字段或单独 JSON 供调试。
  - 如果需要保留主从控制命令与反馈，可在处理流水线里通过自定义 Processor 将其重命名为 `complementary.*`，避免与核心动作冲突。
- 采样频率
  - 关节采样可保持 100Hz，但写入帧时应按 `dataset.fps=30` 下采样（或在 `get_observation` 内按最近一次读数返回），与相机帧对齐。
  - `lerobot-replay` 按数据集 FPS 节拍回放，不会自动插值，若需要更高控制频率请在机器人实现中自行插值。

最终希望生成的 `info.json` 与示例类似，只是 `action/observation.state` 的 `names/shape` 变为双臂/多相机版本，从而可直接被训练与可视化脚本消费。

---

## 7. 依赖要求

### Python 依赖

```toml
# pyproject.toml [project.optional-dependencies]
agilex = [
    "piper_sdk",          # Agilex Piper SDK
    "rospy",              # ROS Python 客户端
    "cv_bridge",          # ROS-OpenCV 转换
    "sensor_msgs",        # ROS 消息类型
    "geometry_msgs",      # 如需基座或主从里程信息
]
```

### 系统依赖

- ROS Noetic (Ubuntu 20.04) 或 ROS2 Humble (Ubuntu 22.04)
- CAN 接口配置（`can_left`, `can_right`）
- RealSense 相机驱动（如使用 RealSense）

---

## 8. 注意事项

### 8.1 单位转换

| 数据类型 | LeRobot 单位 | Piper SDK 单位 | 转换系数 |
|---------|-------------|---------------|---------|
| 关节角度 | 度 (deg) | 毫度 (mdeg) | ×1000 |
| 夹爪位置 | 毫米 (mm) | 万分之一毫米 | ×10000 |
| 电机速度 | rad/s | mrad/s | ×1000 |

> 建议在 `_send_joint_deg` / `_read_joint_deg` 内集中做转换，保持动作/观测单位一致（deg）。

### 8.2 时间同步

- 相机采样频率：30Hz
- 关节采样频率：可达 100Hz
- 建议统一使用 30Hz 进行数据录制

### 8.3 安全考虑

1. 连接前检查机械臂状态
2. 异常时自动禁用力矩
3. 设置合理的运动速度限制
4. 回放前确认机械臂处于安全位置

---

## 9. 后续扩展

### 9.1 支持单臂模式

可以创建 `AgilexSingleArm` 类，只控制单个机械臂。

### 9.2 支持深度相机

在 `RosCameraConfig` 中已预留 `use_depth` 和 `depth_topic` 配置。

### 9.3 支持移动底盘

如果机器人有移动底盘，可以扩展 `observation_features` 和 `action_features` 添加底盘速度控制。

### 9.4 与现有抽象对齐

- Piper SDK 可包装成 `MotorsBus` 子类，复用 `Motor`/`MotorNormMode`/`ensure_safe_goal_position`，使关节命名和归一化方式与 `so100_follower` / `bi_so100_follower` 等保持一致。
- ROS 相机可借鉴 `aiglex_origin_code` 的 deque 同步逻辑，并明确 `async_read` 返回 `None` 时上层的处理策略（丢帧/沿用上一帧）。

---

## 10. 与现有 `agilex_script` 的关系

### 10.1 功能对比

`agilex_script/` 目录包含独立的录制/回放脚本，与 LeRobot 集成后的功能对应关系如下：

| 功能 | agilex_script（现有） | LeRobot 集成后 |
|------|----------------------|----------------|
| 录制 | `record_dual_arm.py` | `lerobot-record --robot.type=agilex_bimanual` |
| 回放 | `playback_dual_arm.py` | `lerobot-replay --robot.type=agilex_bimanual` |
| 状态查询 | `get_state.py` | `robot.get_observation()` |
| 零位发送 | `send_zero_pose.py` | 可通过 `robot.send_action()` 实现 |
| 数据格式 | JSON（关节角度 + 电机状态） | LeRobot Dataset（Parquet + MP4） |
| 相机支持 | 无 | ROS 相机集成 |
| 采样频率 | 100Hz | 30Hz（与相机对齐） |

### 10.2 代码复用

LeRobot 集成将复用 `agilex_script/` 中的以下核心逻辑：

| 来源文件 | 复用内容 | 目标位置 |
|---------|---------|---------|
| `agilex_infer.py` | `PiperSDKInterface` 封装、单位转换逻辑 | `AgilexBimanual._connect_arm()` |
| `record_dual_arm.py` | 被动监听模式连接 (`piper_init=False`) | `AgilexBimanualLeader.connect()` |
| `playback_dual_arm.py` | 主动控制模式、使能流程 | `AgilexBimanual._connect_arm()` |

### 10.3 关键差异

| 方面 | agilex_script | LeRobot 集成 |
|------|---------------|--------------|
| **连接模式** | 录制时被动监听，回放时主动控制 | 遥操作器（Leader）被动监听，机器人（Follower）主动控制 |
| **数据内容** | 包含 `*_ctrl_deg`、电机温度/电流等诊断字段 | 仅保留 `action` 和 `observation.state` 核心字段 |
| **回放结束** | 自动回到预设 HOME 位置 | 由用户控制，不强制回位 |
| **时间同步** | 基于系统时间戳 | 基于 `dataset.fps` 节拍控制 |

### 10.4 迁移建议

集成完成后，`agilex_script/` 目录可作为以下用途保留：

1. **调试工具** - 快速验证 CAN 连接和机械臂状态
2. **独立测试** - 不依赖 LeRobot 框架的简单录制/回放
3. **参考实现** - 查阅 Piper SDK 的原始用法

主要工作流应迁移到 LeRobot CLI：

```bash
# 旧方式
python agilex_script/record_dual_arm.py --duration 30 --output records/demo.json
python agilex_script/playback_dual_arm.py records/demo.json

# 新方式（集成后）
lerobot-record --robot.type=agilex_bimanual --dataset.repo_id=user/demo --dataset.num_episodes=1
lerobot-replay --robot.type=agilex_bimanual --dataset.repo_id=user/demo --dataset.episode=0
```

---

## 11. `aiglex_origin_code` 目录状态说明

### 11.1 当前状态

`aiglex_origin_code/` 目录在当前仓库中 **不存在**（可能未提交或已删除）。

文档第 2.2 节中描述的 ROS 相机代码（`collect_data.py`、`replay_data.py`）来源于外部参考实现，需要根据实际情况处理。

### 11.2 ROS 相机代码来源

ROS 相机的核心接口已在文档中提供，实现时可参考以下来源：

| 来源 | 说明 |
|------|------|
| 文档第 2.2 节 | 提供了 ROS 话题配置和 CvBridge 用法示例 |
| 文档第 4.1 节 | 提供了完整的 `RosCamera` 类实现 |
| ROS 官方文档 | http://wiki.ros.org/cv_bridge/Tutorials |

### 11.3 处理方案

根据实际需求，可选择以下方案之一：

**方案 A：使用文档中的实现（推荐）**

直接使用文档第 4.1 节提供的 `RosCamera` 类实现，该实现已包含：
- ROS 话题订阅
- CvBridge 图像转换
- 线程安全的帧缓存
- LeRobot Camera 接口兼容

**方案 B：补充原始代码**

如果需要参考原始实现，可从以下位置获取：
```bash
# 如果原始代码在其他位置，复制到仓库
mkdir -p aiglex_origin_code
cp /path/to/original/collect_data.py aiglex_origin_code/
cp /path/to/original/replay_data.py aiglex_origin_code/
```

**方案 C：创建占位说明**

如果原始代码不可用，创建说明文件：
```bash
mkdir -p aiglex_origin_code
echo "# ROS 相机原始代码\n\n该目录用于存放 Agilex 原始 ROS 相机代码。\n\n当前实现已迁移至 src/lerobot/cameras/ros_camera/。" > aiglex_origin_code/README.md
```

### 11.4 相机话题配置参考

无论采用哪种方案，ROS 相机话题配置保持一致：

```python
# RGB 相机话题
CAMERA_TOPICS = {
    "cam_high": "/camera_f/color/image_raw",        # 前置相机
    "cam_left_wrist": "/camera_l/color/image_raw",  # 左腕部相机
    "cam_right_wrist": "/camera_r/color/image_raw", # 右腕部相机
}

# 深度相机话题（可选）
DEPTH_TOPICS = {
    "cam_high_depth": "/camera_f/depth/image_raw",
    "cam_left_wrist_depth": "/camera_l/depth/image_raw",
    "cam_right_wrist_depth": "/camera_r/depth/image_raw",
}

# 图像参数
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_FPS = 30
```

---

## 12. 参考资料

- LeRobot 官方文档: https://github.com/huggingface/lerobot
- Piper SDK 文档: (内部文档)
- ROS 相机驱动: http://wiki.ros.org/realsense2_camera
- ROS CvBridge 教程: http://wiki.ros.org/cv_bridge/Tutorials
