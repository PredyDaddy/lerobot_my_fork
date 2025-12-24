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

CAMERA_NAMES = ("camera_left", "camera_right", "camera_front")
STATE_KEYS = [f"{side}_{joint}.pos" for side in ("left", "right") for joint in JOINT_NAMES]
DS_FEATURES = {"action": {"names": STATE_KEYS}}


@dataclass
class StopSignal:
    stopped: bool = False

    def __call__(self, _signum: int, _frame: object | None = None) -> None:
        if not self.stopped:
            logger.info("收到停止信号，正在退出...")
        self.stopped = True


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--fps", type=int, default=30, help="推理频率")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--duration", type=float, default=None, help="推理持续时间(秒)，不指定则无限循环")
    parser.add_argument("--mock", action="store_true", help="使用 mock 模式（不连接真实硬件）")

    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps 必须为正整数")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须为正数")
    return args


def has_valid_images(obs: dict[str, Any], *, allow_blank: bool) -> bool:
    image_keys = [k for k in CAMERA_NAMES if k in obs]
    if not image_keys:
        return False
    if allow_blank:
        return True

    for key in image_keys:
        img = obs.get(key)
        if img is not None and np.any(img):
            return True
    return False


def format_observation(obs: dict[str, Any]) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    for cam in CAMERA_NAMES:
        if cam in obs:
            formatted[f"observation.images.{cam}"] = obs[cam]

    formatted["observation.state"] = np.fromiter(
        (float(obs.get(k, 0.0)) for k in STATE_KEYS),
        dtype=np.float32,
        count=len(STATE_KEYS),
    )
    return formatted


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


def make_robot(mock: bool) -> AgileXRobot:
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

    robot_config = AgileXConfig(id="agilex_infer", mock=mock, cameras=camera_configs)
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
) -> tuple[int, float]:
    logger.info(f"开始推理循环 (FPS={fps})")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

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
        if not has_valid_images(obs, allow_blank=allow_blank_images):
            logger.warning("当前帧观测无有效图像，跳过")
        else:
            action = predict_action(
                observation=format_observation(obs),
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
            )
            robot.send_action(make_robot_action(action, DS_FEATURES))

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
            )
    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        raise

    avg_fps = step_count / total_time if total_time > 0 else 0.0
    logger.info(f"推理结束，共 {step_count} 步，耗时 {total_time:.1f}s，平均 FPS: {avg_fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
