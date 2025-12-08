# Agilex 双臂机器人 LeRobot 集成 - 实现计划

> 本文档由 Claude Code 生成，详细列出需要创建/修改的所有脚本及其功能。

---

## 1. 实现目标

将 Agilex Piper 双臂机器人集成到 LeRobot 框架，实现三个核心功能：

| 功能 | LeRobot 命令 | 说明 |
|------|-------------|------|
| 录制 | `lerobot-record` | 通过主臂遥操作录制双臂关节 + ROS 相机数据 |
| 回放 | `lerobot-replay` | 回放录制的数据到从臂 |
| 可视化 | `lerobot-dataset-viz` | 可视化录制的数据集（无需额外开发） |

---

## 2. 需要新建的文件

### 2.1 ROS 相机模块（3 个文件）

#### 文件 1: `src/lerobot/cameras/ros_camera/__init__.py`

**功能**: 模块导出

```python
from .config_ros import RosCameraConfig
from .ros_camera import RosCamera

__all__ = ["RosCamera", "RosCameraConfig"]
```

---

#### 文件 2: `src/lerobot/cameras/ros_camera/config_ros.py`

**功能**: ROS 相机配置类

**实现内容**:
- 继承 `CameraConfig` 基类
- 使用 `@CameraConfig.register_subclass("ros")` 注册
- 定义 ROS 话题名、分辨率、帧率等参数
- 预留深度相机配置

**关键属性**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `topic` | `str` | 必填 | ROS 图像话题名 |
| `width` | `int` | 640 | 图像宽度 |
| `height` | `int` | 480 | 图像高度 |
| `fps` | `int` | 30 | 帧率 |
| `use_depth` | `bool` | False | 是否使用深度 |
| `depth_topic` | `str \| None` | None | 深度话题名 |

---

#### 文件 3: `src/lerobot/cameras/ros_camera/ros_camera.py`

**功能**: ROS 相机封装类

**实现内容**:
- 继承 `Camera` 基类
- 实现 ROS 话题订阅
- 使用 CvBridge 转换图像格式
- 线程安全的帧缓存

**必须实现的方法**:
| 方法 | 功能 |
|------|------|
| `__init__(config)` | 初始化配置和内部状态 |
| `connect(warmup=True)` | 初始化 ROS 节点，订阅话题 |
| `read(color_mode=None)` | 同步读取最新帧 |
| `async_read(timeout_ms=...)` | 异步读取最新帧 |
| `disconnect()` | 断开连接 |
| `is_connected` (property) | 返回连接状态 |
| `find_cameras()` (static) | 返回空列表（ROS 相机通过话题发现） |

**核心逻辑**:
```python
# 订阅 ROS 话题
rospy.Subscriber(topic, Image, callback, queue_size=1, tcp_nodelay=True)

# 回调中转换图像
def _image_callback(self, msg):
    with self._frame_lock:
        self._latest_frame = self._bridge.imgmsg_to_cv2(msg, 'rgb8')
```

---

### 2.2 Agilex 机器人模块（3 个文件）

#### 文件 4: `src/lerobot/robots/agilex/__init__.py`

**功能**: 模块导出

```python
from .config_agilex import AgilexBimanualConfig
from .agilex import AgilexBimanual

__all__ = ["AgilexBimanualConfig", "AgilexBimanual"]
```

---

#### 文件 5: `src/lerobot/robots/agilex/config_agilex.py`

**功能**: Agilex 双臂机器人配置类

**实现内容**:
- 继承 `RobotConfig` 基类
- 使用 `@RobotConfig.register_subclass("agilex_bimanual")` 注册
- 定义 CAN 端口、运动参数、相机配置等

**关键属性**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `left_arm_port` | `str` | "can_left" | 左臂 CAN 端口 |
| `right_arm_port` | `str` | "can_right" | 右臂 CAN 端口 |
| `enable_timeout` | `float` | 5.0 | 使能超时（秒） |
| `motion_speed` | `int` | 50 | 运动速度 (10-100%) |
| `disable_torque_on_disconnect` | `bool` | True | 断开时禁用力矩 |
| `cameras` | `dict[str, CameraConfig]` | {} | 相机配置字典 |
| `use_degrees` | `bool` | True | 使用度数单位 |
| `use_gripper` | `bool` | True | 是否使用夹爪 |

---

#### 文件 6: `src/lerobot/robots/agilex/agilex.py`

**功能**: Agilex 双臂机器人实现类（从臂/Follower）

**实现内容**:
- 继承 `Robot` 基类
- 封装 Piper SDK 的主动控制模式
- 管理双臂和相机的连接/断开
- 实现关节读取和命令发送

**必须实现的方法**:
| 方法 | 功能 |
|------|------|
| `__init__(config)` | 初始化配置、创建相机实例 |
| `observation_features` (property) | 返回观测特征定义（关节 + 夹爪 + 相机） |
| `action_features` (property) | 返回动作特征定义（关节 + 夹爪） |
| `is_connected` (property) | 检查双臂和相机连接状态 |
| `is_calibrated` (property) | 返回 True（绝对编码器无需校准） |
| `connect(calibrate=True)` | 连接双臂（主动控制模式）和相机 |
| `calibrate()` | 空实现（无需校准） |
| `configure()` | 配置机器人（可选） |
| `get_observation()` | 读取关节角度、夹爪位置、相机图像 |
| `send_action(action)` | 发送关节命令和夹爪命令 |
| `disconnect()` | 断开双臂和相机 |

**内部辅助方法**:
| 方法 | 功能 |
|------|------|
| `_connect_arm(port)` | 连接单个机械臂（EnablePiper + MotionCtrl_2） |
| `_read_joint_deg(arm)` | 读取关节角度（毫度→度） |
| `_read_gripper_mm(arm)` | 读取夹爪位置（万分之一mm→mm） |
| `_send_joint_deg(arm, joints)` | 发送关节命令（度→毫度） |
| `_send_gripper_mm(arm, gripper)` | 发送夹爪命令（mm→万分之一mm） |

**连接流程**:
```
ConnectPort() → GetArmStatus() → EmergencyStop(0x02) → EnablePiper() → MotionCtrl_2(0x01, 0x01, speed, 0x00)
```

---

### 2.3 Agilex 遥操作器模块（3 个文件）

#### 文件 7: `src/lerobot/teleoperators/agilex_leader/__init__.py`

**功能**: 模块导出

```python
from .config_agilex_leader import AgilexBimanualLeaderConfig
from .agilex_leader import AgilexBimanualLeader

__all__ = ["AgilexBimanualLeaderConfig", "AgilexBimanualLeader"]
```

---

#### 文件 8: `src/lerobot/teleoperators/agilex_leader/config_agilex_leader.py`

**功能**: Agilex 双臂主臂配置类

**实现内容**:
- 继承 `TeleoperatorConfig` 基类
- 使用 `@TeleoperatorConfig.register_subclass("agilex_bimanual_leader")` 注册

**关键属性**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `left_arm_port` | `str` | "can_left" | 左主臂 CAN 端口 |
| `right_arm_port` | `str` | "can_right" | 右主臂 CAN 端口 |

---

#### 文件 9: `src/lerobot/teleoperators/agilex_leader/agilex_leader.py`

**功能**: Agilex 双臂主臂遥操作器实现类（主臂/Leader）

**实现内容**:
- 继承 `Teleoperator` 基类
- 封装 Piper SDK 的被动监听模式
- 读取主臂关节位置作为动作输出

**必须实现的方法**:
| 方法 | 功能 |
|------|------|
| `__init__(config)` | 初始化配置 |
| `action_features` (property) | 返回动作特征定义（与 Robot 一致） |
| `feedback_features` (property) | 返回空字典（主臂不需要反馈） |
| `is_connected` (property) | 检查双臂连接状态 |
| `is_calibrated` (property) | 返回 True |
| `connect(calibrate=True)` | 连接双臂（被动监听模式） |
| `calibrate()` | 空实现 |
| `configure()` | 空实现 |
| `get_action()` | 读取主臂关节位置和夹爪状态 |
| `send_feedback(feedback)` | 空实现（主臂不需要反馈） |
| `disconnect()` | 断开双臂 |

**连接流程（被动监听模式）**:
```python
arm = C_PiperInterface_V2(port)
arm.ConnectPort(piper_init=False, start_thread=True)  # 不初始化，只监听
```

**关键差异**:
| 方面 | Robot (Follower) | Teleoperator (Leader) |
|------|------------------|----------------------|
| 连接模式 | 主动控制 (`ConnectPort()`) | 被动监听 (`ConnectPort(piper_init=False)`) |
| EnablePiper | 需要 | 不需要 |
| send_action | 发送命令 | 不适用 |
| get_action | 不适用 | 读取位置 |

---

## 3. 需要修改的文件

### 3.1 相机模块注册（2 个文件）

#### 修改 1: `src/lerobot/cameras/__init__.py`

**修改内容**: 添加 ROS 相机导出

```python
# 在文件末尾添加
from .ros_camera import RosCamera, RosCameraConfig
```

---

#### 修改 2: `src/lerobot/cameras/utils.py`

**修改内容**: 在 `make_cameras_from_configs()` 函数中添加 ROS 相机分支

```python
# 在 elif 链中添加
elif cfg.type == "ros":
    from .ros_camera import RosCamera
    cameras[key] = RosCamera(cfg)
```

---

### 3.2 机器人模块注册（2 个文件）

#### 修改 3: `src/lerobot/robots/__init__.py`

**修改内容**: 添加 Agilex 机器人导出（可选，取决于是否需要在脚本中显式导入）

```python
# 在文件末尾添加（如果需要）
from .agilex import AgilexBimanual, AgilexBimanualConfig
```

---

#### 修改 4: `src/lerobot/robots/utils.py`

**修改内容**: 在 `make_robot_from_config()` 函数中添加 Agilex 分支

```python
# 在 elif 链中添加
elif config.type == "agilex_bimanual":
    from .agilex import AgilexBimanual
    return AgilexBimanual(config)
```

---

### 3.3 遥操作器模块注册（2 个文件）

#### 修改 5: `src/lerobot/teleoperators/__init__.py`

**修改内容**: 添加 Agilex 遥操作器导出（可选）

```python
# 在文件末尾添加（如果需要）
from .agilex_leader import AgilexBimanualLeader, AgilexBimanualLeaderConfig
```

---

#### 修改 6: `src/lerobot/teleoperators/utils.py`

**修改内容**: 在 `make_teleoperator_from_config()` 函数中添加 Agilex 分支

```python
# 在 elif 链中添加
elif config.type == "agilex_bimanual_leader":
    from .agilex_leader import AgilexBimanualLeader
    return AgilexBimanualLeader(config)
```

---

## 4. 文件清单汇总

### 4.1 新建文件（9 个）

| 序号 | 文件路径 | 功能 | 代码行数估计 |
|------|---------|------|-------------|
| 1 | `src/lerobot/cameras/ros_camera/__init__.py` | ROS 相机模块导出 | ~5 |
| 2 | `src/lerobot/cameras/ros_camera/config_ros.py` | ROS 相机配置类 | ~20 |
| 3 | `src/lerobot/cameras/ros_camera/ros_camera.py` | ROS 相机实现类 | ~80 |
| 4 | `src/lerobot/robots/agilex/__init__.py` | Agilex 机器人模块导出 | ~5 |
| 5 | `src/lerobot/robots/agilex/config_agilex.py` | Agilex 机器人配置类 | ~30 |
| 6 | `src/lerobot/robots/agilex/agilex.py` | Agilex 机器人实现类 | ~200 |
| 7 | `src/lerobot/teleoperators/agilex_leader/__init__.py` | Agilex 遥操作器模块导出 | ~5 |
| 8 | `src/lerobot/teleoperators/agilex_leader/config_agilex_leader.py` | Agilex 遥操作器配置类 | ~15 |
| 9 | `src/lerobot/teleoperators/agilex_leader/agilex_leader.py` | Agilex 遥操作器实现类 | ~120 |

**总计**: ~480 行代码

### 4.2 修改文件（6 个）

| 序号 | 文件路径 | 修改内容 | 修改行数 |
|------|---------|---------|---------|
| 1 | `src/lerobot/cameras/__init__.py` | 添加导入 | +1 |
| 2 | `src/lerobot/cameras/utils.py` | 添加工厂分支 | +3 |
| 3 | `src/lerobot/robots/__init__.py` | 添加导入（可选） | +1 |
| 4 | `src/lerobot/robots/utils.py` | 添加工厂分支 | +3 |
| 5 | `src/lerobot/teleoperators/__init__.py` | 添加导入（可选） | +1 |
| 6 | `src/lerobot/teleoperators/utils.py` | 添加工厂分支 | +3 |

**总计**: ~12 行修改

---

## 5. 数据与接口约束（补充）

- **键名与单位**：`action`/`observation.state` 使用 `left|right_<joint>.pos`（6 关节，度）与可选 `left|right_gripper.pos`（毫米），顺序一致；相机键 `observation.images.<cam_name>`（480×640×3，30Hz）。`lerobot-replay` 不会插值，100Hz 采样需在 robot 内下采样或取最新值。
- **剔除诊断字段**：`*_ctrl_deg`、温度/电流等高频诊断不要写入 `action`/`observation.state`，必要时放入自定义 info 或单独日志。
- **ROS 相机接口**：实现类必须继承 `Camera`，补全 `find_cameras`、`connect(warmup=True)`、`async_read(timeout_ms=...)`、`read`、`disconnect`，并处理 RGB→BGR（`ColorMode`），初始化已存在的 `rospy` 节点时不应报错。
- **导出可选**：`robots/__init__.py`、`teleoperators/__init__.py` 导入硬件依赖可能在无设备环境失败，可只在 factory 分支注册；遥操作器为可选，录制可用 `mock_teleop` 或政策替代。

---

## 5. 实现顺序

建议按以下顺序实现，确保每一步都可以独立测试：

```
阶段 1: ROS 相机模块
├── 1.1 创建 config_ros.py
├── 1.2 创建 ros_camera.py
├── 1.3 创建 __init__.py
├── 1.4 修改 cameras/utils.py
├── 1.5 修改 cameras/__init__.py
└── 1.6 测试: 独立运行 ROS 相机读取

阶段 2: Agilex 机器人模块
├── 2.1 创建 config_agilex.py
├── 2.2 创建 agilex.py
├── 2.3 创建 __init__.py
├── 2.4 修改 robots/utils.py
├── 2.5 修改 robots/__init__.py（可选）
└── 2.6 测试: lerobot-replay 回放已有数据

阶段 3: Agilex 遥操作器模块
├── 3.1 创建 config_agilex_leader.py
├── 3.2 创建 agilex_leader.py
├── 3.3 创建 __init__.py
├── 3.4 修改 teleoperators/utils.py
├── 3.5 修改 teleoperators/__init__.py（可选）
└── 3.6 测试: lerobot-record 录制新数据

阶段 4: 端到端测试
├── 4.1 录制完整 episode
├── 4.2 回放录制的 episode
├── 4.3 可视化数据集
└── 4.4 验证数据格式兼容性
```

---

## 6. 测试命令

### 6.1 单元测试

```bash
# 测试模块导入
python -c "from lerobot.cameras.ros_camera import RosCamera, RosCameraConfig; print('ROS Camera OK')"
python -c "from lerobot.robots.agilex import AgilexBimanual, AgilexBimanualConfig; print('Agilex Robot OK')"
python -c "from lerobot.teleoperators.agilex_leader import AgilexBimanualLeader; print('Agilex Leader OK')"
```

### 6.2 集成测试

```bash
# 录制测试
lerobot-record \
    --robot.type=agilex_bimanual \
    --robot.left_arm_port=can_left \
    --robot.right_arm_port=can_right \
    --robot.cameras='{
        cam_high: {type: ros, topic: /camera_f/color/image_raw, width: 640, height: 480, fps: 30}
    }' \
    --teleop.type=agilex_bimanual_leader \
    --teleop.left_arm_port=can_left \
    --teleop.right_arm_port=can_right \
    --dataset.repo_id=test/agilex-demo \
    --dataset.num_episodes=1 \
    --dataset.single_task="Test recording"

# 回放测试
lerobot-replay \
    --robot.type=agilex_bimanual \
    --robot.left_arm_port=can_left \
    --robot.right_arm_port=can_right \
    --dataset.repo_id=test/agilex-demo \
    --dataset.episode=0

# 可视化测试
lerobot-dataset-viz \
    --repo-id test/agilex-demo \
    --episode-index 0
```

---

## 7. 依赖检查

### 7.1 Python 依赖

```bash
# 检查 Piper SDK
python -c "from piper_sdk import C_PiperInterface_V2; print('Piper SDK OK')"

# 检查 ROS 依赖（需要在 ROS 环境中）
python -c "import rospy; print('rospy OK')"
python -c "from cv_bridge import CvBridge; print('cv_bridge OK')"
python -c "from sensor_msgs.msg import Image; print('sensor_msgs OK')"
```

### 7.2 系统依赖

```bash
# 检查 CAN 接口
ip link show can_left
ip link show can_right

# 检查 ROS 话题
rostopic list | grep camera
```

---

## 8. 与 agilex_script 的代码复用

| agilex_script 文件 | 复用内容 | 目标文件 |
|-------------------|---------|---------|
| `agilex_infer.py` | 单位转换逻辑 | `agilex.py` |
| `record_dual_arm.py` | 被动监听连接 | `agilex_leader.py` |
| `playback_dual_arm.py` | 主动控制连接 | `agilex.py` |
| `get_state.py` | 关节读取逻辑 | `agilex.py`, `agilex_leader.py` |

---

## 9. 注意事项

1. **单位转换**: 所有单位转换在 `_read_*` 和 `_send_*` 方法中集中处理
2. **连接模式**: Robot 使用主动控制，Teleoperator 使用被动监听
3. **线程安全**: ROS 相机回调需要使用锁保护帧缓存
4. **错误处理**: 连接失败时应清理已连接的资源
5. **帧丢失**: `async_read()` 返回 `None` 时上层应处理（沿用上一帧或跳过）
