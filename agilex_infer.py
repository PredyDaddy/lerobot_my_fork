#!/usr/bin/env python3
"""
Agilex ACT 策略推理脚本

直接加载策略进行持续推理，不使用 lerobot-record。
使用新版 lerobot 的 preprocessor/postprocessor 机制。

用法:
    python agilex_infer.py --checkpoint outputs/train/act_agilex_215_multi/checkpoints/100000/pretrained_model

    # 指定 FPS 和设备
    python agilex_infer.py --checkpoint <path> --fps 30 --device cuda

    # Mock 模式测试
    python agilex_infer.py --checkpoint <path> --mock
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.agilex import AgileXRobot, AgileXConfig
from lerobot.robots.agilex.config_agilex import JOINT_NAMES
from lerobot.cameras.ros_camera import RosCameraConfig
from lerobot.utils.control_utils import predict_action
from lerobot.policies.utils import make_robot_action

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局运行标志
_running = True


def signal_handler(signum, frame):
    """处理 Ctrl+C 信号"""
    global _running
    logger.info("收到停止信号，正在退出...")
    _running = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Agilex ACT 策略推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="策略 checkpoint 路径 (包含 config.json 和 model.safetensors)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="推理频率",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="推理持续时间(秒)，不指定则无限循环",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 mock 模式（不连接真实硬件）",
    )
    return parser.parse_args()


def format_observation(obs: dict) -> dict:
    """
    转换机器人观测格式为策略期望的格式

    机器人返回:
        - camera_left, camera_right, camera_front (图像, numpy array HWC)
        - left_shoulder_pan.pos, right_gripper.pos 等 (关节状态)

    策略期望:
        - observation.images.camera_left (图像)
        - observation.state (14维向量)
    """
    formatted = {}

    # 转换图像 key
    for cam in ["camera_left", "camera_right", "camera_front"]:
        if cam in obs:
            formatted[f"observation.images.{cam}"] = obs[cam]

    # 组装 state 向量 (14维: 左臂7关节 + 右臂7关节)
    state = []
    for side in ["left", "right"]:
        for joint in JOINT_NAMES:
            key = f"{side}_{joint}.pos"
            if key in obs:
                state.append(obs[key])
            else:
                logger.warning(f"缺少关节状态: {key}")
                state.append(0.0)

    formatted["observation.state"] = np.array(state, dtype=np.float32)

    return formatted


def validate_observation(obs: dict) -> bool:
    """
    验证观测数据有效性

    检查图像是否存在且不全为零
    """
    # 检查是否有图像
    image_keys = [k for k in obs.keys() if k in ["camera_left", "camera_right", "camera_front"]]
    if not image_keys:
        return False

    # 检查图像是否有效（不全为零）
    for key in image_keys:
        img = obs[key]
        if img is not None and np.sum(img) > 0:
            return True

    return False


def build_ds_features() -> dict:
    """
    构建 dataset features 结构用于 make_robot_action

    action 名称顺序: 左臂7关节 + 右臂7关节
    """
    action_names = []
    for side in ["left", "right"]:
        for joint in JOINT_NAMES:
            action_names.append(f"{side}_{joint}.pos")

    return {
        "action": {
            "names": action_names,
        }
    }


def main():
    global _running

    args = parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 检查 checkpoint 路径
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        logger.error(f"Checkpoint 路径不存在: {checkpoint_path}")
        sys.exit(1)

    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # ========== 加载策略 ==========
    logger.info(f"加载策略: {checkpoint_path}")
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.to(device)
    policy.eval()
    logger.info("策略加载完成")

    # ========== 加载 preprocessor/postprocessor ==========
    logger.info("加载 preprocessor/postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "device_processor": {"device": args.device}
        },
    )
    logger.info("处理器加载完成")

    # ========== 初始化机器人 ==========
    logger.info("初始化机器人...")
    camera_configs = {
        "camera_left": RosCameraConfig(
            topic_name="/camera_l/color/image_raw",
            width=640,
            height=480,
            fps=30,
        ),
        "camera_right": RosCameraConfig(
            topic_name="/camera_r/color/image_raw",
            width=640,
            height=480,
            fps=30,
        ),
        "camera_front": RosCameraConfig(
            topic_name="/camera_f/color/image_raw",
            width=640,
            height=480,
            fps=30,
        ),
    }

    robot_config = AgileXConfig(
        id="agilex_infer",
        mock=args.mock,
        cameras=camera_configs,
    )
    robot = AgileXRobot(robot_config)

    try:
        robot.connect()
        logger.info("机器人连接成功")
    except Exception as e:
        logger.error(f"机器人连接失败: {e}")
        sys.exit(1)

    # 构建 ds_features 用于 action 转换
    ds_features = build_ds_features()

    # ========== 推理循环 ==========
    logger.info(f"开始推理循环 (FPS={args.fps})")
    policy.reset()  # 清空 action queue

    step_count = 0
    start_time = time.time()
    frame_duration = 1.0 / args.fps

    try:
        while _running:
            loop_start = time.perf_counter()

            # 检查持续时间
            if args.duration is not None:
                elapsed = time.time() - start_time
                if elapsed >= args.duration:
                    logger.info(f"达到设定时间 {args.duration}s，停止推理")
                    break

            # 获取观测
            obs = robot.get_observation()

            # 验证观测有效性
            if not validate_observation(obs):
                logger.warning("当前帧观测无效，跳过")
                time.sleep(frame_duration)
                continue

            # 转换观测格式
            obs_formatted = format_observation(obs)

            # 策略推理
            action = predict_action(
                observation=obs_formatted,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
            )

            # 转换 action 格式
            action_dict = make_robot_action(action, ds_features)

            # 发送 action
            robot.send_action(action_dict)

            step_count += 1
            if step_count % 100 == 0:
                fps_actual = step_count / (time.time() - start_time)
                logger.info(f"步数: {step_count}, 实际 FPS: {fps_actual:.1f}")

            # 控制频率
            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = frame_duration - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        raise
    finally:
        # 断开机器人连接
        logger.info("断开机器人连接...")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")

    total_time = time.time() - start_time
    logger.info(f"推理结束，共 {step_count} 步，耗时 {total_time:.1f}s，平均 FPS: {step_count / total_time:.1f}")


if __name__ == "__main__":
    main()
