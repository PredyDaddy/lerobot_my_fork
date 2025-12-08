# Agilex 集成现状与使用指南

本轮改动已完成的内容与已知注意事项整理如下，便于后续验证与排障。

## 已实现内容
- ROS 相机支持：`src/lerobot/cameras/ros_camera/`（config + 实现 + 导出），在 `cameras/utils.py` 注册 `type=ros`。
- Agilex 双臂从臂：`src/lerobot/robots/agilex/`（config + 实现 + 导出），在 `robots/utils.py` 注册 `type=agilex_bimanual`。
- Agilex 主臂遥操作器（可选）：`src/lerobot/teleoperators/agilex_leader/`，在 `teleoperators/utils.py` 注册 `type=agilex_bimanual_leader`。
- 观测/动作键名对齐 LeRobot v3.0：`left|right_<joint>.pos`（度）、可选 `left|right_gripper.pos`（毫米），相机键为配置名，落盘为 `observation.images.<cam>`。
- ROS 相机 topic 校验：topic 含空格会在解析阶段报错，避免录制中途才失败。

## 基本使用（示例命令）
确保 CAN 口已 up 且名称对应（示例：左臂 `can2`，右臂 `can1`），ROS 相机话题为 `/camera_l/color/image_raw`、`/camera_r/color/image_raw`。

```bash
lerobot-record \
  --robot.type=agilex_bimanual \
  --robot.left_arm_port=can2 \
  --robot.right_arm_port=can1 \
  --robot.id=agilex_dual \
  --robot.cameras='{cam_left: {type: ros, topic: /camera_l/color/image_raw, width: 640, height: 480, fps: 30}, cam_right: {type: ros, topic: /camera_r/color/image_raw, width: 640, height: 480, fps: 30}}' \
  --teleop.type=agilex_bimanual_leader \
  --teleop.left_arm_port=can2 \
  --teleop.right_arm_port=can1 \
  --dataset.repo_id=your_username/agilex_dataset1 \
  --dataset.single_task="Your task description" \
  --dataset.num_episodes=2 \
  --dataset.fps=30 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false
```

回放时同理，使用 `lerobot-replay` 指向录制好的数据集和 episode。

## 已知风险/注意事项
- ROS topic 不能含空格，命令换行时避免在 topic 处断行；解析阶段会直接报错。
- 未对相机帧缺失做额外处理，`async_read` 返回 `None` 将导致数据校验报错；确保话题有稳定帧率。
- Piper SDK 连接需要正确的 CAN 名称，若提示 “CAN socket <name> does not exist”，先 `ip link show | grep can` 校验接口名或运行 bring-up 脚本。
- 当前 `agilex_bimanual` 未做关节插值/限幅，录制/回放都按 30Hz 调用；如需 100Hz 控制，需要在机器人实现内自行插值或限速。

## 建议的后续验证
1) 成功录制 ≥1 个 episode，检查 `.cache/huggingface/lerobot/<repo>/meta/info.json` 键名与预期一致。  
2) 用 `lerobot-replay` 回放同一 episode，确认双臂与夹爪动作正常。  
3) 运行 `lerobot-dataset-viz`，检查左右相机视频与关节轨迹是否齐全。  
4) 若后续增加夹爪/安全逻辑，保持动作/观测键名不变以兼容已有数据。
