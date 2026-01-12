export DISPLAY=:0

lerobot-record \
    --robot.type=agilex \
    --robot.mock=false \
    --robot.cameras='{
      camera_left:  {"type": "ros_camera", "topic_name": "/camera_l/color/image_raw",     "width": 640, "height":
  480, "fps": 30},
      camera_right: {"type": "ros_camera", "topic_name": "/camera_r/color/image_raw",     "width": 640, "height":
  480, "fps": 30},
      camera_front: {"type": "ros_camera", "topic_name": "/camera_f/color/image_raw", "width": 640, "height":
  480, "fps": 30}
    }' \
    --teleop.type=agilex_teleop \
    --teleop.mock=false \
    --dataset.repo_id=cqy/agilex_vla_demo \
    --dataset.single_task="Pick up the black cups and place them in the orange box." \
    --dataset.num_episodes=5 \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=8 \
    --dataset.reset_time_s=8 \
    # --resume=true