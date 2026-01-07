#!/usr/bin/env python3
"""
Agilex 单臂 ACT 策略推理脚本

用于推理单独训练的左臂或右臂 ACT 模型。
支持 2 个相机（camera_front + camera_left/right）和 7 个关节。

用法:
    # 左臂推理
    python agilex_infer_single_cc.py \
        --checkpoint outputs/act_agilex_left_yellow_bottle/checkpoints/last/pretrained_model \
        --arm left --fps 30 --binary-gripper

    # 右臂推理
    python agilex_infer_single_cc.py \
        --checkpoint outputs/act_agilex_right_yellow_bottle/checkpoints/last/pretrained_model \
        --arm right --fps 30

    # Mock 模式测试
    python agilex_infer_single_cc.py --checkpoint <path> --arm left --mock
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from lerobot.cameras.ros_camera import RosCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.agilex import AgileXConfig, AgileXRobot
from lerobot.robots.agilex.config_agilex import JOINT_NAMES
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============== 单臂配置 ==============

def get_arm_config(arm: str) -> dict[str, Any]:
    """获取指定臂的配置信息。

    Args:
        arm: 'left' 或 'right'

    Returns:
        包含相机名称、状态键、前缀等配置的字典
    """
    if arm == "left":
        return {
            "cameras": ("camera_left", "camera_front"),
            "state_keys": [f"left_{j}.pos" for j in JOINT_NAMES],
            "prefix": "left_",
        }
    elif arm == "right":
        return {
            "cameras": ("camera_right", "camera_front"),
            "state_keys": [f"right_{j}.pos" for j in JOINT_NAMES],
            "prefix": "right_",
        }
    else:
        raise ValueError(f"Invalid arm: {arm}, must be 'left' or 'right'")


@dataclass
class StopSignal:
    stopped: bool = False

    def __call__(self, _signum: int, _frame: object | None = None) -> None:
        if not self.stopped:
            logger.info("收到停止信号，正在退出...")
        self.stopped = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agilex 单臂 ACT 策略推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="策略 checkpoint 路径 (包含 config.json 和 model.safetensors)",
    )
    parser.add_argument(
        "--arm",
        type=str,
        required=True,
        choices=["left", "right"],
        help="选择控制的臂: left 或 right",
    )
    parser.add_argument("--fps", type=int, default=30, help="推理频率")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--duration", type=float, default=None, help="推理持续时间(秒)，不指定则无限循环")
    parser.add_argument("--mock", action="store_true", help="使用 mock 模式（不连接真实硬件）")
    parser.add_argument("--binary-gripper", action="store_true", help="启用夹爪二值化（增加抓取力度）")
    parser.add_argument("--gripper-threshold", type=float, default=0.7,
                        help="夹爪二值化阈值 (0-1)，低于此值设为0，高于设为最大值")

    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps 必须为正整数")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须为正数")
    return args


def has_valid_images(obs: dict[str, Any], camera_names: tuple[str, ...], *, allow_blank: bool) -> bool:
    """检查观测中是否有有效的图像。"""
    image_keys = [k for k in camera_names if k in obs]
    if not image_keys:
        return False
    if allow_blank:
        return True

    for key in image_keys:
        img = obs.get(key)
        if img is not None and np.any(img):
            return True
    return False


def format_observation(obs: dict[str, Any], arm_config: dict[str, Any]) -> dict[str, Any]:
    """格式化观测数据为策略输入格式。

    Args:
        obs: 原始观测字典
        arm_config: 臂配置信息

    Returns:
        格式化后的观测字典
    """
    formatted: dict[str, Any] = {}

    # 处理相机图像（2个相机）
    for cam in arm_config["cameras"]:
        if cam in obs:
            formatted[f"observation.images.{cam}"] = obs[cam]

    # 处理关节状态（7个关节）
    formatted["observation.state"] = np.fromiter(
        (float(obs.get(k, 0.0)) for k in arm_config["state_keys"]),
        dtype=np.float32,
        count=len(arm_config["state_keys"]),
    )
    return formatted


def binarize_gripper(action: dict[str, Any], arm: str, threshold: float) -> dict[str, Any]:
    """对夹爪值进行二值化处理。

    Args:
        action: 动作字典
        arm: 'left' 或 'right'
        threshold: 阈值 (0-1)

    Returns:
        处理后的动作字典
    """
    GRIPPER_MAX = 0.085
    threshold_value = GRIPPER_MAX * threshold
    key = f"{arm}_gripper.pos"

    if key in action:
        if action[key] < threshold_value:
            action[key] = 0.0
        else:
            action[key] = GRIPPER_MAX
    return action


def resolve_device(device_str: str) -> tuple[str, torch.device]:
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("请求使用 cuda，但当前环境不可用；已自动回退到 cpu。")
        device_str = "cpu"
    return device_str, torch.device(device_str)


def load_policy_stack(
    checkpoint_path: Path, *, device: torch.device, device_str: str
) -> tuple[ACTPolicy, Any, Any]:
    logger.info(f"加载策略: {checkpoint_path}")
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.to(device)
    policy.eval()

    logger.info("加载 preprocessor/postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": device_str}},
    )
    return policy, preprocessor, postprocessor


def make_robot(mock: bool, arm: str) -> AgileXRobot:
    """创建机器人实例，只配置需要的相机。

    Args:
        mock: 是否使用 mock 模式
        arm: 'left' 或 'right'

    Returns:
        AgileXRobot 实例
    """
    # 始终配置前置相机
    camera_configs = {
        "camera_front": RosCameraConfig(
            topic_name="/camera_f/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        ),
    }

    # 根据臂选择配置对应的侧面相机
    if arm == "left":
        camera_configs["camera_left"] = RosCameraConfig(
            topic_name="/camera_l/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        )
    else:  # right
        camera_configs["camera_right"] = RosCameraConfig(
            topic_name="/camera_r/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        )

    robot_config = AgileXConfig(id="agilex_infer_single", mock=mock, cameras=camera_configs)
    return AgileXRobot(robot_config)


@contextlib.contextmanager
def connected_robot(robot: AgileXRobot) -> Iterator[AgileXRobot]:
    robot.connect()
    try:
        yield robot
    finally:
        logger.info("断开机器人连接...")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")


def send_single_arm_action(
    robot: AgileXRobot,
    action: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    """发送单臂动作，另一臂保持当前位置。

    Args:
        robot: 机器人实例
        action: 动作字典（包含7个关节目标位置）
        arm: 'left' 或 'right'

    Returns:
        实际发送的动作字典
    """
    # 获取当前双臂状态
    left_state, right_state = robot.ros_bridge.get_puppet_state()

    if arm == "left":
        # 左臂使用推理动作，右臂保持当前位置
        left_target = np.array(
            [action[f"left_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32
        )
        right_target = np.array(right_state.position, dtype=np.float32)
    else:  # right
        # 右臂使用推理动作，左臂保持当前位置
        left_target = np.array(left_state.position, dtype=np.float32)
        right_target = np.array(
            [action[f"right_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32
        )

    # 应用安全限制
    if robot.config.max_relative_target > 0:
        left_target = _clip_delta(left_state.position, left_target, robot.config.max_relative_target)
        right_target = _clip_delta(right_state.position, right_target, robot.config.max_relative_target)

    # 发送命令
    robot.ros_bridge.send_joint_commands(left_target, right_target)

    # 返回实际发送的动作
    result = {}
    for i, jn in enumerate(JOINT_NAMES):
        result[f"left_{jn}.pos"] = float(left_target[i])
        result[f"right_{jn}.pos"] = float(right_target[i])
    return result


def _clip_delta(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    """限制单步变化以确保安全。"""
    delta = target - current
    delta = np.clip(delta, -max_delta, max_delta)
    return current + delta


def run_inference_loop(
    *,
    robot: AgileXRobot,
    policy: ACTPolicy,
    preprocessor: Any,
    postprocessor: Any,
    device: torch.device,
    fps: int,
    duration_s: float | None,
    stop: StopSignal,
    allow_blank_images: bool,
    arm: str,
    arm_config: dict[str, Any],
    binary_gripper: bool,
    gripper_threshold: float,
) -> tuple[int, float]:
    """运行推理循环。"""
    logger.info(f"开始推理循环 (FPS={fps}, arm={arm})")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # 构建 DS_FEATURES
    ds_features = {"action": {"names": arm_config["state_keys"]}}

    step_count = 0
    start_t = time.perf_counter()
    period_s = 1.0 / fps
    deadline_t = time.monotonic() + duration_s if duration_s is not None else None

    while not stop.stopped:
        loop_start_t = time.perf_counter()

        if deadline_t is not None and time.monotonic() >= deadline_t:
            logger.info(f"达到设定时间 {duration_s}s，停止推理")
            break

        obs = robot.get_observation()
        if not has_valid_images(obs, arm_config["cameras"], allow_blank=allow_blank_images):
            logger.warning("当前帧观测无有效图像，跳过")
        else:
            action = predict_action(
                observation=format_observation(obs, arm_config),
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
            )
            # 转换为机器人动作格式
            robot_action = make_robot_action(action, ds_features)
            # 二值化夹爪
            if binary_gripper:
                robot_action = binarize_gripper(robot_action, arm, gripper_threshold)
            # 发送单臂动作
            send_single_arm_action(robot, robot_action, arm)

            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.perf_counter() - start_t
                fps_actual = step_count / elapsed if elapsed > 0 else 0.0
                logger.info(f"步数: {step_count}, 实际 FPS: {fps_actual:.1f}")

        precise_sleep(period_s - (time.perf_counter() - loop_start_t))

    return step_count, time.perf_counter() - start_t


def main() -> int:
    args = parse_args()

    stop = StopSignal()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        logger.error(f"Checkpoint 路径不存在: {checkpoint_path}")
        return 1

    # 获取臂配置
    arm_config = get_arm_config(args.arm)
    logger.info(f"单臂模式: {args.arm}")
    logger.info(f"使用相机: {arm_config['cameras']}")
    logger.info(f"控制关节: {arm_config['state_keys']}")

    device_str, device = resolve_device(args.device)
    logger.info(f"使用设备: {device}")

    try:
        policy, preprocessor, postprocessor = load_policy_stack(
            checkpoint_path, device=device, device_str=device_str
        )
    except Exception as e:
        logger.error(f"策略/处理器加载失败: {e}")
        return 1

    robot = make_robot(args.mock, args.arm)
    try:
        with connected_robot(robot):
            step_count, total_time = run_inference_loop(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                fps=args.fps,
                duration_s=args.duration,
                stop=stop,
                allow_blank_images=args.mock,
                arm=args.arm,
                arm_config=arm_config,
                binary_gripper=args.binary_gripper,
                gripper_threshold=args.gripper_threshold,
            )
    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        raise

    avg_fps = step_count / total_time if total_time > 0 else 0.0
    logger.info(f"推理结束，共 {step_count} 步，耗时 {total_time:.1f}s，平均 FPS: {avg_fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
