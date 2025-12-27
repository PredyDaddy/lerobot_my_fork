#!/usr/bin/env python3
"""
AgileX + GR00T N1.6 推理脚本（直接控机器人，不通过 lerobot-record）。

这个脚本复用 lerobot 的 AgileXRobot 采集观测，并通过 Isaac-GR00T 的 `Gr00tPolicy`
加载/推理 GR00T N1.6 checkpoint，输出动作为关节目标位置。

推荐把 `--checkpoint` 指向 Isaac-GR00T 训练输出的某个 `checkpoint-xxxx/` 目录，
因为该目录通常同时包含：
  - config.json / model*.safetensors
  - processor_config.json / statistics.json / embodiment_id.json

用法示例：
  python agilex_infer_gr00t_n1d6.py \
    --checkpoint /mnt/data2/cqy/workspace/Isaac-GR00T-test/outputs/agilex_proj_diff_multi/checkpoint-2000 \
    --task "Wipe the bottle with a sponge" \
    --device cuda --fps 30

  # Mock 模式（不连硬件）
  python agilex_infer_gr00t_n1d6.py --checkpoint <path> --task "..." --mock
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from lerobot.cameras.ros_camera import RosCameraConfig
from lerobot.policies.utils import make_robot_action
from lerobot.robots.agilex import AgileXConfig, AgileXRobot
from lerobot.robots.agilex.config_agilex import JOINT_NAMES
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CAMERA_NAMES = ("camera_left", "camera_right", "camera_front")
STATE_KEYS = [f"{side}_{joint}.pos" for side in ("left", "right") for joint in JOINT_NAMES]
DS_FEATURES = {"action": {"names": STATE_KEYS}}

# GR00T 侧的 NEW_EMBODIMENT（AgileX 数据集/模态配置通常使用这个 tag）
DEFAULT_EMBODIMENT_TAG = "NEW_EMBODIMENT"

# GR00T 侧的模态 key（例如 "front"）到 AgileXRobot obs key（例如 "camera_front"）的映射
DEFAULT_VIDEO_KEY_MAP = {
    "front": "camera_front",
    "left": "camera_left",
    "right": "camera_right",
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
        description="AgileX + GR00T N1.6 推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Isaac-GR00T checkpoint 目录（包含 config.json / model*.safetensors / processor_config.json）",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="语言指令/任务描述（建议与训练数据 tasks.jsonl 一致）",
    )
    parser.add_argument("--fps", type=int, default=30, help="控制频率（Hz）")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备（cuda / cuda:0 / cpu）")
    parser.add_argument("--duration", type=float, default=None, help="运行时长(秒)，不指定则无限循环")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=1,
        help="一次规划后连续执行的步数（1 表示每步都重规划；>1 表示执行多步再重规划）",
    )
    parser.add_argument(
        "--isaac-groot-dir",
        type=str,
        default=None,
        help="Isaac-GR00T 仓库路径（如未安装 gr00t 包，可通过该路径注入 PYTHONPATH）",
    )
    parser.add_argument("--mock", action="store_true", help="使用 mock 模式（不连接真实硬件）")

    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps 必须为正整数")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须为正数")
    if args.execution_horizon <= 0:
        parser.error("--execution-horizon 必须为正整数")
    return args


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("请求使用 cuda，但当前环境不可用；已自动回退到 cpu。")
        return torch.device("cpu")
    return torch.device(device_str)


def ensure_gr00t_importable(isaac_groot_dir: str | None) -> None:
    try:
        import gr00t  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    candidates: list[Path] = []
    if isaac_groot_dir:
        candidates.append(Path(isaac_groot_dir))
    # 常见：lerobot_my_fork 和 Isaac-GR00T-test 在同级目录
    candidates.append(Path(__file__).resolve().parent.parent / "Isaac-GR00T-test")

    for cand in candidates:
        if (cand / "gr00t").is_dir():
            sys.path.insert(0, str(cand))
            logger.info(f"已将 Isaac-GR00T 路径加入 PYTHONPATH: {cand}")
            return

    raise ModuleNotFoundError(
        "未找到可导入的 gr00t 包。请先安装 Isaac-GR00T，或使用 --isaac-groot-dir 指定其仓库路径。"
    )


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
    robot_config = AgileXConfig(id="agilex_gr00t_infer", mock=mock, cameras=camera_configs)
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


def _state_vec_from_robot_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.fromiter(
        (float(obs.get(k, 0.0)) for k in STATE_KEYS),
        dtype=np.float32,
        count=len(STATE_KEYS),
    )


def _split_state_for_gr00t(state_vec: np.ndarray, state_keys: list[str]) -> dict[str, np.ndarray]:
    if set(state_keys) != {"left_arm", "left_gripper", "right_arm", "right_gripper"}:
        raise ValueError(
            f"当前脚本只支持 AgileX/new_embodiment 的 state keys {['left_arm','left_gripper','right_arm','right_gripper']}，"
            f"但 checkpoint 需要: {state_keys}"
        )

    # 维度与 agilex_dataset1_v21_gr00t_abs/meta/modality.json 一致：6+1+6+1=14
    return {
        "left_arm": state_vec[None, None, 0:6],
        "left_gripper": state_vec[None, None, 6:7],
        "right_arm": state_vec[None, None, 7:13],
        "right_gripper": state_vec[None, None, 13:14],
    }


def _build_gr00t_observation(
    *,
    robot_obs: dict[str, Any],
    task: str,
    video_keys: list[str],
    state_keys: list[str],
    language_key: str,
) -> dict[str, Any]:
    video: dict[str, np.ndarray] = {}
    for key in video_keys:
        robot_key = DEFAULT_VIDEO_KEY_MAP.get(key, key)
        img = robot_obs.get(robot_key)
        if img is None:
            raise KeyError(f"观测中缺少相机图像: {robot_key}（用于 GR00T video key: {key}）")
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"{robot_key} 期望为 HWC/RGB 图像，实际 shape={img.shape}")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        video[key] = img[None, None, ...]  # (B=1, T=1, H, W, C)

    state_vec = _state_vec_from_robot_obs(robot_obs)
    state = _split_state_for_gr00t(state_vec, state_keys)

    # GR00T policy 期望语言为 (B,T) 形状的 list[list[str]]
    language = {language_key: [[task]]}
    return {"video": video, "state": state, "language": language}


def _flatten_action_chunk(action: dict[str, np.ndarray]) -> np.ndarray:
    # action[key]: (B, T, D)
    left_arm = action["left_arm"][0]  # (T, 6)
    left_gripper = action["left_gripper"][0]  # (T, 1)
    right_arm = action["right_arm"][0]  # (T, 6)
    right_gripper = action["right_gripper"][0]  # (T, 1)
    return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)  # (T, 14)


def run_inference_loop(
    *,
    robot: AgileXRobot,
    policy: Any,
    fps: int,
    duration_s: float | None,
    execution_horizon: int,
    task: str,
    stop: StopSignal,
    allow_blank_images: bool,
) -> tuple[int, float]:
    logger.info(f"开始推理循环 (FPS={fps}, execution_horizon={execution_horizon})")

    policy.reset()

    step_count = 0
    start_t = time.perf_counter()
    period_s = 1.0 / fps
    deadline_t = time.monotonic() + duration_s if duration_s is not None else None

    planned_actions: list[np.ndarray] = []
    planned_idx = 0

    video_keys = policy.modality_configs["video"].modality_keys
    state_keys = policy.modality_configs["state"].modality_keys
    language_key = policy.language_key

    while not stop.stopped:
        loop_start_t = time.perf_counter()

        if deadline_t is not None and time.monotonic() >= deadline_t:
            logger.info(f"达到设定时间 {duration_s}s，停止推理")
            break

        obs = robot.get_observation()
        if not has_valid_images(obs, allow_blank=allow_blank_images):
            logger.warning("当前帧观测无有效图像，跳过")
            precise_sleep(period_s - (time.perf_counter() - loop_start_t))
            continue

        # 重规划条件：计划为空 或 已执行到 execution_horizon
        if (
            (not planned_actions)
            or (planned_idx >= len(planned_actions))
            or (planned_idx >= execution_horizon)
        ):
            gr00t_obs = _build_gr00t_observation(
                robot_obs=obs,
                task=task,
                video_keys=video_keys,
                state_keys=state_keys,
                language_key=language_key,
            )
            action, _info = policy.get_action(gr00t_obs)
            action_chunk = _flatten_action_chunk(action)  # (T, 14)
            planned_actions = [action_chunk[i] for i in range(action_chunk.shape[0])]
            planned_idx = 0

        next_action = planned_actions[planned_idx]
        planned_idx += 1

        # 发送给机器人
        action_tensor = torch.from_numpy(next_action).to(dtype=torch.float32).unsqueeze(0)
        robot.send_action(make_robot_action(action_tensor, DS_FEATURES))

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

    ensure_gr00t_importable(args.isaac_groot_dir)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    device = resolve_device(args.device)
    logger.info(f"使用设备: {device}")

    try:
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag[DEFAULT_EMBODIMENT_TAG],
            model_path=str(checkpoint_path),
            device=str(device),
            strict=True,
        )
    except Exception as e:
        logger.error(f"GR00T policy 加载失败: {e}")
        return 1

    robot = make_robot(args.mock)
    try:
        with connected_robot(robot):
            step_count, total_time = run_inference_loop(
                robot=robot,
                policy=policy,
                fps=args.fps,
                duration_s=args.duration,
                execution_horizon=args.execution_horizon,
                task=args.task,
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
