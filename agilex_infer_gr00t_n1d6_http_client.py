#!/usr/bin/env python3
"""
AgileX 侧“接收端”脚本：从 GR00T HTTP 推理服务获取动作，并直接下发到机器人。

配套服务端（在 Isaac-GR00T-test 仓库中）：
  python gr00t/eval/run_gr00t_http_server.py --checkpoint <ckpt_dir> --host 0.0.0.0 --port 8000

该脚本会：
- 通过 lerobot 的 AgileXRobot 从 ROS 相机/关节状态读观测
- 将观测（3 路相机 + 14 维关节位置）发给服务端 /act
- 接收动作 chunk（T x 14），按 `--execution-horizon` 执行若干步后再请求下一次动作
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import logging
import signal
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


AVAILABLE_CAMERA_KEYS = ("camera_left", "camera_right", "camera_front")
STATE_KEYS: list[str] = []
DS_FEATURES: dict[str, Any] = {}


@dataclass
class StopSignal:
    stopped: bool = False

    def __call__(self, _signum: int, _frame: object | None = None) -> None:
        if not self.stopped:
            logger.info("收到停止信号，正在退出...")
        self.stopped = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgileX -> GR00T HTTP 推理客户端",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="GR00T HTTP 服务地址（不含末尾斜杠）",
    )
    parser.add_argument("--task", type=str, required=True, help="语言指令/任务描述")
    parser.add_argument("--fps", type=int, default=10, help="控制频率（Hz）。建议先用 1~10 做联调")
    parser.add_argument("--duration", type=float, default=None, help="运行时长(秒)，不指定则无限循环")
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=1,
        help="一次请求得到 action chunk 后连续执行的步数（1=每步都请求；>1=执行多步再请求）",
    )
    parser.add_argument(
        "--image-downsample",
        type=int,
        default=2,
        help="对输入图像做简单下采样（2 表示 H/W 各减半；1 表示不下采样）",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP 请求超时（秒）")
    parser.add_argument("--reset-on-start", action="store_true", help="启动时请求服务端 /reset")
    parser.add_argument(
        "--lerobot-dir",
        type=str,
        default=None,
        help="lerobot 仓库路径（例如 /mnt/data2/cqy/workspace/lerobot_my_fork）。若未安装 lerobot，可用此参数注入 PYTHONPATH。",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default="camera_left,camera_right,camera_front",
        help="要发送给服务端的相机 key，逗号分隔（例如仅前视：camera_front）",
    )
    parser.add_argument("--mock", action="store_true", help="mock 模式（不连接真实硬件/ROS）")
    args = parser.parse_args()

    camera_keys = [k.strip() for k in args.camera_keys.split(",") if k.strip()]
    if not camera_keys:
        parser.error("--camera-keys 不能为空")
    unknown = [k for k in camera_keys if k not in AVAILABLE_CAMERA_KEYS]
    if unknown:
        parser.error(f"--camera-keys 包含未知相机 key: {unknown}")
    # 去重但保持顺序
    args.camera_keys = tuple(dict.fromkeys(camera_keys))

    if args.fps <= 0:
        parser.error("--fps 必须为正整数")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须为正数")
    if args.execution_horizon <= 0:
        parser.error("--execution-horizon 必须为正整数")
    if args.image_downsample <= 0:
        parser.error("--image-downsample 必须为正整数")
    return args


def ensure_lerobot_importable(lerobot_dir: str | None) -> None:
    try:
        import lerobot  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    candidates: list[Path] = []
    if lerobot_dir:
        candidates.append(Path(lerobot_dir))
    # 常见：Isaac-GR00T-test 与 lerobot_my_fork 在同级目录
    candidates.append(Path(__file__).resolve().parent.parent / "lerobot_my_fork")

    for cand in candidates:
        if (cand / "src" / "lerobot").is_dir():
            sys.path.insert(0, str(cand / "src"))
            logger.info("已将 lerobot 路径加入 PYTHONPATH: %s", cand / "src")
            return
        if (cand / "lerobot").is_dir():
            sys.path.insert(0, str(cand))
            logger.info("已将 lerobot 路径加入 PYTHONPATH: %s", cand)
            return

    raise ModuleNotFoundError(
        "未找到可导入的 lerobot 包。请先安装 lerobot（pip install -e .），或使用 --lerobot-dir 指定其仓库路径。"
    )


def _encode_npy_zlib_b64(arr: np.ndarray) -> dict[str, Any]:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    compressed = zlib.compress(buf.getvalue())
    return {
        "__ndarray__": True,
        "encoding": "npy+zlib+base64",
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _downsample_hwc(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    return image[::factor, ::factor, :]


def _http_post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def has_valid_images(obs: dict[str, Any], *, allow_blank: bool, camera_keys: tuple[str, ...]) -> bool:
    image_keys = [k for k in camera_keys if k in obs]
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


def _build_request_payload(
    *,
    obs: dict[str, Any],
    task: str,
    image_downsample: int,
    camera_keys: tuple[str, ...],
) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for cam_key in camera_keys:
        img = obs.get(cam_key)
        if img is None:
            raise KeyError(f"观测缺少相机: {cam_key}")
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"{cam_key} 期望 HWC/RGB，实际 shape={img.shape}")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        img = _downsample_hwc(img, image_downsample)
        images[cam_key] = _encode_npy_zlib_b64(img)

    joint_positions = _state_vec_from_robot_obs(obs).tolist()
    return {"task": task, "images": images, "joint_positions": joint_positions}


def make_robot(mock: bool, camera_keys: tuple[str, ...]):
    from lerobot.cameras.ros_camera import RosCameraConfig
    from lerobot.robots.agilex import AgileXConfig, AgileXRobot

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
    camera_configs = {k: v for k, v in camera_configs.items() if k in camera_keys}
    robot_config = AgileXConfig(id="agilex_gr00t_http_client", mock=mock, cameras=camera_configs)
    return AgileXRobot(robot_config)


@contextlib.contextmanager
def connected_robot(robot) -> Iterator[Any]:
    robot.connect()
    try:
        yield robot
    finally:
        logger.info("断开机器人连接...")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning("断开连接时出错: %s", e)


def main() -> int:
    args = parse_args()
    ensure_lerobot_importable(args.lerobot_dir)

    from lerobot.policies.utils import make_robot_action
    from lerobot.robots.agilex.config_agilex import JOINT_NAMES
    from lerobot.utils.robot_utils import precise_sleep

    global STATE_KEYS, DS_FEATURES
    STATE_KEYS = [f"{side}_{joint}.pos" for side in ("left", "right") for joint in JOINT_NAMES]
    DS_FEATURES = {"action": {"names": STATE_KEYS}}

    stop = StopSignal()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    server_url = args.server_url.rstrip("/")
    act_url = f"{server_url}/act"
    reset_url = f"{server_url}/reset"

    camera_keys = args.camera_keys
    logger.info("启用相机: %s", ", ".join(camera_keys))
    robot = make_robot(args.mock, camera_keys)
    period_s = 1.0 / args.fps

    planned_actions: list[np.ndarray] = []
    planned_idx = 0

    with connected_robot(robot):
        if args.reset_on_start:
            try:
                _http_post_json(reset_url, {"options": None}, timeout_s=args.timeout)
                logger.info("已请求服务端 reset")
            except Exception as e:
                logger.warning("请求 reset 失败（可忽略，继续执行）: %s", e)

        step_count = 0
        start_t = time.perf_counter()
        deadline_t = time.monotonic() + args.duration if args.duration is not None else None

        while not stop.stopped:
            loop_start_t = time.perf_counter()

            if deadline_t is not None and time.monotonic() >= deadline_t:
                logger.info("达到设定时间 %.1fs，停止推理", args.duration)
                break

            obs = robot.get_observation()
            if not has_valid_images(obs, allow_blank=args.mock, camera_keys=camera_keys):
                logger.warning("当前帧无有效图像，跳过")
                precise_sleep(period_s - (time.perf_counter() - loop_start_t))
                continue

            # 重规划条件：计划为空 或 计划用完 或 已执行到 execution_horizon
            if (
                (not planned_actions)
                or (planned_idx >= len(planned_actions))
                or (planned_idx >= args.execution_horizon)
            ):
                payload = _build_request_payload(
                    obs=obs,
                    task=args.task,
                    image_downsample=args.image_downsample,
                    camera_keys=camera_keys,
                )
                try:
                    resp = _http_post_json(act_url, payload, timeout_s=args.timeout)
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                    logger.error("请求服务端失败: %s", e)
                    precise_sleep(period_s - (time.perf_counter() - loop_start_t))
                    continue

                if resp.get("status") != "ok":
                    logger.error("服务端返回错误: %s", resp)
                    precise_sleep(period_s - (time.perf_counter() - loop_start_t))
                    continue

                action_chunk = np.asarray(resp.get("action_chunk", []), dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] != len(STATE_KEYS):
                    logger.error(
                        "action_chunk 形状不正确：期望 (T,%d)，得到 %s",
                        len(STATE_KEYS),
                        action_chunk.shape,
                    )
                    precise_sleep(period_s - (time.perf_counter() - loop_start_t))
                    continue

                planned_actions = [action_chunk[i] for i in range(action_chunk.shape[0])]
                planned_idx = 0

            next_action = planned_actions[planned_idx]
            planned_idx += 1

            action_tensor = torch.from_numpy(next_action).to(dtype=torch.float32).unsqueeze(0)
            robot.send_action(make_robot_action(action_tensor, DS_FEATURES))

            step_count += 1
            if step_count % 50 == 0:
                elapsed = time.perf_counter() - start_t
                fps_actual = step_count / elapsed if elapsed > 0 else 0.0
                logger.info("步数: %d, 实际 FPS: %.2f", step_count, fps_actual)

            precise_sleep(period_s - (time.perf_counter() - loop_start_t))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
