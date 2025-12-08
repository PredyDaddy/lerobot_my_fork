测试下来 can2是左臂，can1是右臂
```bash
  # 配置并启动 can2
  sudo ip link set can2 type can bitrate 1000000
  sudo ip link set can2 up

  # 配置并启动 can1
  sudo ip link set can1 type can bitrate 1000000
  sudo ip link set can1 up

  然后验证状态：

  ip link show type can

```

roslaunch astra_camera multi_camera.launch



rm -rf /home/agilex/.cache/huggingface/lerobot/your_username/agilex_dataset1


export DISPLAY=:0

  lerobot-record \
    --robot.type=agilex_bimanual \
    --robot.left_arm_port=can2 \
    --robot.right_arm_port=can1 \
    --robot.id=agilex_dual \
    --robot.cameras='{cam_left: {type: ros, topic: /camera_l/color/image_raw, width: 640, height: 480, fps: 30}, cam_right: {type: ros, topic: 
  /camera_r/color/image_raw, width: 640, height: 480, fps: 30}}' \
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