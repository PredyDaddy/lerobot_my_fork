#!/usr/bin/env python3
"""
Agilex ACT 单臂策略推理脚本

用于推理通过 split_dual_arm.py 分割出的单臂数据训练得到的 ACT checkpoint（7 维 state/action）。
脚本会根据 checkpoint 的 input_features 自动选择相机输入（left/right + front），并仅控制对应手臂；
另一只手臂保持在当前关节角度（发送当前状态作为目标）。

用法:
    # 直接传训练输出目录（默认取 checkpoints/last/pretrained_model）
    python agilex_infer_single.py --checkpoint outputs/act_agilex_left_yellow_bottle

    # 也支持传具体 checkpoint
    python agilex_infer_single.py --checkpoint outputs/act_agilex_left_yellow_bottle/checkpoints/100000/pretrained_model

    # 指定 FPS / 设备 / mock
    python agilex_infer_single.py --checkpoint <path> --fps 30 --device cuda
    python agilex_infer_single.py --checkpoint <path> --mock
"""

from __future__ import annotations

import argparse
import contextlib
import json
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CAMERA_TOPICS: dict[str, str] = {
    "camera_left": "/camera_l/color/image_raw",
    "camera_right": "/camera_r/color/image_raw",
    "camera_front": "/camera_f/color/image_raw",
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
        description="Agilex ACT 单臂策略推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="策略 checkpoint 路径（可传训练输出目录或 pretrained_model 目录）",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="auto",
        choices=["auto", "left", "right"],
        help="控制的手臂；auto 会根据 checkpoint 的 input_features 自动推断",
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


def resolve_device(device_str: str) -> tuple[str, torch.device]:
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("请求使用 cuda，但当前环境不可用；已自动回退到 cpu。")
        device_str = "cpu"
    return device_str, torch.device(device_str)


def resolve_checkpoint_dir(checkpoint_arg: Path) -> Path:
    candidates = [
        checkpoint_arg,
        checkpoint_arg / "pretrained_model",
        checkpoint_arg / "checkpoints" / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            return candidate

    tried = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        "未找到可用的 pretrained_model 目录（需要包含 config.json 与 model.safetensors）。尝试路径:\n"
        f"{tried}"
    )


def _policy_input_feature_keys(policy: ACTPolicy) -> list[str]:
    input_features = getattr(policy.config, "input_features", None)
    if input_features is None:
        return []
    if isinstance(input_features, dict):
        return list(input_features.keys())
    try:
        return list(input_features.keys())
    except Exception:
        return []


def extract_required_cameras(policy: ACTPolicy) -> tuple[str, ...]:
    cameras: list[str] = []
    for key in _policy_input_feature_keys(policy):
        if key.startswith("observation.images."):
            cameras.append(key.split("observation.images.", 1)[1])
    return tuple(cameras)


def infer_arm_from_policy(policy: ACTPolicy, checkpoint_dir: Path) -> str | None:
    keys = set(_policy_input_feature_keys(policy))
    has_left = "observation.images.camera_left" in keys
    has_right = "observation.images.camera_right" in keys
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"

    train_cfg_path = checkpoint_dir / "train_config.json"
    if train_cfg_path.exists():
        try:
            train_cfg = json.loads(train_cfg_path.read_text())
            repo_id = str(train_cfg.get("dataset", {}).get("repo_id", ""))
            if "left" in repo_id and "right" not in repo_id:
                return "left"
            if "right" in repo_id and "left" not in repo_id:
                return "right"
        except Exception:
            pass

    return None


def has_valid_images(obs: dict[str, Any], *, required_cameras: tuple[str, ...], allow_blank: bool) -> bool:
    if not required_cameras:
        return True

    missing = [cam for cam in required_cameras if cam not in obs]
    if missing:
        return False

    if allow_blank:
        return True

    for cam in required_cameras:
        img = obs.get(cam)
        if img is not None and np.any(img):
            return True
    return False


def format_observation(
    obs: dict[str, Any],
    *,
    required_cameras: tuple[str, ...],
    state_keys: list[str],
) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    for cam in required_cameras:
        formatted[f"observation.images.{cam}"] = obs[cam]

    formatted["observation.state"] = np.fromiter(
        (float(obs.get(k, 0.0)) for k in state_keys),
        dtype=np.float32,
        count=len(state_keys),
    )
    return formatted


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


def make_robot(*, mock: bool, fps: int, required_cameras: tuple[str, ...]) -> AgileXRobot:
    camera_configs: dict[str, RosCameraConfig] = {}
    for cam in required_cameras:
        topic = CAMERA_TOPICS.get(cam)
        if topic is None:
            raise ValueError(f"未知相机名称: {cam}，请在 CAMERA_TOPICS 中配置对应 topic。")
        camera_configs[cam] = RosCameraConfig(
            topic_name=topic,
            width=640,
            height=480,
            fps=fps,
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
    required_cameras: tuple[str, ...],
) -> tuple[int, float]:
    if arm not in {"left", "right"}:
        raise ValueError(f"arm 必须为 left/right，收到: {arm}")

    controlled_keys = [f"{arm}_{joint}.pos" for joint in JOINT_NAMES]
    other_arm = "right" if arm == "left" else "left"
    other_keys = [f"{other_arm}_{joint}.pos" for joint in JOINT_NAMES]

    ds_features = {"action": {"names": controlled_keys}}

    logger.info(
        "开始推理循环 (FPS=%d, arm=%s, cameras=%s)", fps, arm, ",".join(required_cameras) or "(none)"
    )

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
            logger.info("达到设定时间 %.2fs，停止推理", duration_s)
            break

        obs = robot.get_observation()
        if not has_valid_images(obs, required_cameras=required_cameras, allow_blank=allow_blank_images):
            logger.warning("当前帧观测无有效图像/缺少相机输入，跳过")
        else:
            action_tensor = predict_action(
                observation=format_observation(
                    obs, required_cameras=required_cameras, state_keys=controlled_keys
                ),
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=False,
            )

            action = make_robot_action(action_tensor, ds_features)
            for k in other_keys:
                action[k] = float(obs.get(k, 0.0))
            robot.send_action(action)

            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.perf_counter() - start_t
                fps_actual = step_count / elapsed if elapsed > 0 else 0.0
                logger.info("步数: %d, 实际 FPS: %.1f", step_count, fps_actual)

        precise_sleep(period_s - (time.perf_counter() - loop_start_t))

    return step_count, time.perf_counter() - start_t


def main() -> int:
    args = parse_args()

    stop = StopSignal()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        checkpoint_dir = resolve_checkpoint_dir(Path(args.checkpoint))
    except Exception as e:
        logger.error(str(e))
        return 1

    device_str, device = resolve_device(args.device)
    logger.info("使用设备: %s", device)

    try:
        policy, preprocessor, postprocessor = load_policy_stack(
            checkpoint_dir, device=device, device_str=device_str
        )
    except Exception as e:
        logger.error("策略/处理器加载失败: %s", e)
        return 1

    required_cameras = extract_required_cameras(policy)
    for cam in required_cameras:
        if cam not in CAMERA_TOPICS:
            logger.error("Checkpoint 需要相机 %s，但 CAMERA_TOPICS 未配置对应 topic。", cam)
            return 1

    arm = args.arm
    if arm == "auto":
        inferred = infer_arm_from_policy(policy, checkpoint_dir)
        if inferred is None:
            logger.error("无法自动判断控制手臂，请显式指定 --arm left/right。")
            return 1
        arm = inferred

    robot = make_robot(mock=args.mock, fps=args.fps, required_cameras=required_cameras)
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
                arm=arm,
                required_cameras=required_cameras,
            )
    except Exception as e:
        logger.error("推理过程出错: %s", e)
        raise

    avg_fps = step_count / total_time if total_time > 0 else 0.0
    logger.info("推理结束，共 %d 步，耗时 %.1fs，平均 FPS: %.1f", step_count, total_time, avg_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

