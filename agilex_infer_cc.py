#!/usr/bin/env python3
"""
Agilex 双臂 ACT 策略推理脚本

用于推理双臂训练的 ACT 模型。
支持 3 个相机（camera_left, camera_right, camera_front）和 14 个关节。

用法:
    python agilex_infer_cc.py \
        --checkpoint /home/agilex/cqy/lerobot_dev/lerobot_4_2/lerobot_my_fork/outputs/act_agilex_dual_banana_final/checkpoints/last/pretrained_model \
        --fps 30 --binary-gripper

    # Mock 模式测试
    python agilex_infer_cc.py --checkpoint <path> --mock --duration 5
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


# ============== 双臂配置 ==============

def get_dual_arm_config() -> dict[str, Any]:
    """获取双臂配置信息。

    Returns:
        包含相机名称、状态键等配置的字典
    """
    return {
        "cameras": ("camera_left", "camera_right", "camera_front"),
        "state_keys": [
            f"{side}_{joint}.pos"
            for side in ("left", "right")
            for joint in JOINT_NAMES
        ],  # 14个关节
    }


@dataclass
class StopSignal:
    stopped: bool = False

    def __call__(self, _signum: int, _frame: object | None = None) -> None:
        if not self.stopped:
            logger.info("收到停止信号，正在退出...")
        self.stopped = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agilex 双臂 ACT 策略推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="策略 checkpoint 路径 (包含 config.json 和 model.safetensors)",
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
    """检查观测中是否有有效的图像。

    Args:
        obs: 观测字典
        camera_names: 相机名称元组
        allow_blank: 是否允许空白图像

    Returns:
        是否有有效图像
    """
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


def format_observation(obs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """格式化观测数据为策略输入格式。

    Args:
        obs: 原始观测字典
        config: 配置信息

    Returns:
        格式化后的观测字典
    """
    formatted: dict[str, Any] = {}

    # 处理相机图像（3个相机）
    for cam in config["cameras"]:
        if cam in obs:
            formatted[f"observation.images.{cam}"] = obs[cam]

    # 处理关节状态（14个关节）
    formatted["observation.state"] = np.fromiter(
        (float(obs.get(k, 0.0)) for k in config["state_keys"]),
        dtype=np.float32,
        count=len(config["state_keys"]),
    )
    return formatted


def binarize_gripper(action: dict[str, Any], threshold: float) -> dict[str, Any]:
    """对夹爪值进行二值化处理。

    Args:
        action: 动作字典
        threshold: 阈值 (0-1)

    Returns:
        处理后的动作字典
    """
    GRIPPER_MAX = 0.085
    threshold_value = GRIPPER_MAX * threshold

    for key in ("left_gripper.pos", "right_gripper.pos"):
        if key in action:
            if action[key] < threshold_value:
                action[key] = 0.0
            else:
                action[key] = GRIPPER_MAX
    return action


def resolve_device(device_str: str) -> tuple[str, torch.device]:
    """解析设备字符串。"""
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("请求使用 cuda，但当前环境不可用；已自动回退到 cpu。")
        device_str = "cpu"
    return device_str, torch.device(device_str)


def load_policy_stack(
    checkpoint_path: Path, *, device: torch.device, device_str: str
) -> tuple[ACTPolicy, Any, Any]:
    """加载策略和处理器。

    Args:
        checkpoint_path: checkpoint 路径
        device: PyTorch 设备
        device_str: 设备字符串

    Returns:
        (policy, preprocessor, postprocessor) 元组
    """
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


def make_robot(mock: bool) -> AgileXRobot:
    """创建机器人实例，配置3个相机。

    Args:
        mock: 是否使用 mock 模式

    Returns:
        AgileXRobot 实例
    """
    camera_configs = {
        "camera_left": RosCameraConfig(
            topic_name="/camera_l/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        ),
        "camera_right": RosCameraConfig(
            topic_name="/camera_r/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        ),
        "camera_front": RosCameraConfig(
            topic_name="/camera_f/color/image_raw",
            width=640,
            height=480,
            fps=30,
            mock=mock,
        ),
    }

    robot_config = AgileXConfig(id="agilex_infer_dual", mock=mock, cameras=camera_configs)
    return AgileXRobot(robot_config)


@contextlib.contextmanager
def connected_robot(robot: AgileXRobot) -> Iterator[AgileXRobot]:
    """机器人连接上下文管理器。"""
    robot.connect()
    try:
        yield robot
    finally:
        logger.info("断开机器人连接...")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning(f"断开连接时出错: {e}")


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
    config: dict[str, Any],
    binary_gripper: bool,
    gripper_threshold: float,
) -> tuple[int, float]:
    """运行推理循环。

    Args:
        robot: 机器人实例
        policy: ACT 策略
        preprocessor: 预处理器
        postprocessor: 后处理器
        device: PyTorch 设备
        fps: 推理频率
        duration_s: 持续时间（秒）
        stop: 停止信号
        allow_blank_images: 是否允许空白图像
        config: 配置信息

    Returns:
        (步数, 总时间) 元组
    """
    logger.info(f"开始推理循环 (FPS={fps})")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # 构建 DS_FEATURES
    ds_features = {"action": {"names": config["state_keys"]}}

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
        if not has_valid_images(obs, config["cameras"], allow_blank=allow_blank_images):
            logger.warning("当前帧观测无有效图像，跳过")
        else:
            action = predict_action(
                observation=format_observation(obs, config),
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
            )
            # 转换为机器人动作格式并发送
            robot_action = make_robot_action(action, ds_features)
            if binary_gripper:
                robot_action = binarize_gripper(robot_action, gripper_threshold)
            robot.send_action(robot_action)

            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.perf_counter() - start_t
                fps_actual = step_count / elapsed if elapsed > 0 else 0.0
                logger.info(f"步数: {step_count}, 实际 FPS: {fps_actual:.1f}")

        precise_sleep(period_s - (time.perf_counter() - loop_start_t))

    return step_count, time.perf_counter() - start_t


def main() -> int:
    """主函数。"""
    args = parse_args()

    stop = StopSignal()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_dir():
        logger.error(f"Checkpoint 路径不存在: {checkpoint_path}")
        return 1

    # 获取双臂配置
    config = get_dual_arm_config()
    logger.info(f"双臂模式")
    logger.info(f"使用相机: {config['cameras']}")
    logger.info(f"控制关节: {len(config['state_keys'])} 个")

    device_str, device = resolve_device(args.device)
    logger.info(f"使用设备: {device}")

    try:
        policy, preprocessor, postprocessor = load_policy_stack(
            checkpoint_path, device=device, device_str=device_str
        )
    except Exception as e:
        logger.error(f"策略/处理器加载失败: {e}")
        return 1

    robot = make_robot(args.mock)
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
                config=config,
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
