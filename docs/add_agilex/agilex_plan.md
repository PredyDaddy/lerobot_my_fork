# Agilex 接入执行计划

目标：在 LeRobot 中完成 Agilex Piper 双臂 + ROS 三相机的录制/回放闭环，重用现有 `agilex_script` 的关节采集与 ROS 相机管线，产出 v3.0 规范数据集。

## 必需脚本与职责

- `src/lerobot/cameras/ros_camera/config_ros.py`  
  定义 ROS 相机配置，注册 `ros` 类型，包含话题名、分辨率、fps、可选深度话题。

- `src/lerobot/cameras/ros_camera/ros_camera.py`  
  ROS 相机实现，继承 `Camera`，提供 `find_cameras`/`connect`/`async_read`/`disconnect`，订阅 `/camera_f|l|r/color/image_raw` 等话题，返回 30Hz 480×640×3 帧。

- `src/lerobot/cameras/utils.py` & `src/lerobot/cameras/__init__.py`  
  工厂分支和导出，确保 `make_cameras_from_configs` 能实例化 `ros` 相机。

- `src/lerobot/robots/agilex/config_agilex.py`  
  Agilex 双臂配置，含左右臂 CAN 端口、速度/超时/力矩释放开关、是否用度数、夹爪开关、相机配置字典。

- `src/lerobot/robots/agilex/agilex.py`  
  Agilex 双臂实现：连接两只 Piper SDK 接口、读写关节/夹爪（度/毫米与 SDK 毫度/万分之一毫米转换）、聚合相机观测；`observation_features`/`action_features` 键名：`left|right_<joint>.pos`、`left|right_gripper.pos`，相机键 `observation.images.cam_*`。

- `src/lerobot/robots/agilex/__init__.py`，`src/lerobot/robots/utils.py`，`src/lerobot/robots/__init__.py`  
  导出并在 `make_robot_from_config` 加入 `agilex_bimanual` 分支。

- （可选）`src/lerobot/teleoperators/agilex_leader/{config_agilex_leader.py, agilex_leader.py, __init__.py}`  
  主臂遥操作器，被动监听主臂关节/夹爪，产出与 follower 相同键名的动作。

- 复用 CLI：`lerobot-record` / `lerobot-replay`（无需修改）。通过配置 `--robot.type=agilex_bimanual`、相机为 `ros`，即可录制/回放。

- 参考/迁移脚本：`agilex_script/record_dual_arm.py`（100Hz 监听）与 `playback_dual_arm.py`（回放）。可用于对比 SDK 通信与安全流程；必要时做数据转换工具，将旧 JSON 转为 v3.0 Parquet/MP4 结构。

## 数据与特征对齐（与 `demo_data_meta` 一致）

- `action` 与 `observation.state`：键序保持一致（左臂 6 关节、右臂 6 关节，可选双夹爪），单位为度/毫米；数据集写入顺序与 `info.json` names 对齐。
- `observation.images.*`：ROS 相机帧键名 `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist`，480×640×3，30Hz。
- 系统列由框架自动写入：`timestamp`、`frame_index`、`episode_index`、`task_index`、`index`。
- 诊断/高频字段（ctrl、温度、电流）如需保留，放入自定义 `info` 或独立调试文件，避免污染核心动作/观测。

## 验证清单

1) 安装依赖：`pip install -e ".[agilex]"` 确保 `piper_sdk`、`rospy`、`cv_bridge` 可用。  
2) 硬件连通：运行 `RosCamera.find_cameras()` 或 `rostopic list` 确认话题；`agilex_script/get_state.py` 读取一次关节。  
3) 录制烟囱测试：`lerobot-record` 录 1 个 episode，检查 `meta/info.json` 是否包含预期键名/shape。  
4) 回放：`lerobot-replay` 同一 episode，确认双臂与夹爪按录制动作复现，必要时在 `agilex.py` 内插值或限幅。  
5) 可视化：`lerobot-dataset-viz` 读取数据集，检查三路相机视频与关节轨迹。  
