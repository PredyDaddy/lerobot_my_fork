#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_agilex import AgilexBimanualConfig

logger = logging.getLogger(__name__)

try:
    from piper_sdk import C_PiperInterface_V2
except Exception:  # pragma: no cover - hardware dependency
    C_PiperInterface_V2 = None  # type: ignore


class AgilexBimanual(Robot):
    """Agilex Piper 双臂从臂实现。"""

    config_class = AgilexBimanualConfig
    name = "agilex_bimanual"

    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "wrist_rotate",
    ]

    def __init__(self, config: AgilexBimanualConfig):
        super().__init__(config)
        self.config = config

        self._left_arm: C_PiperInterface_V2 | None = None
        self._right_arm: C_PiperInterface_V2 | None = None

        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        for joint in self.JOINT_NAMES[: self.config.num_joints_per_arm]:
            features[f"left_{joint}.pos"] = float
            features[f"right_{joint}.pos"] = float

        if self.config.use_gripper:
            features["left_gripper.pos"] = float
            features["right_gripper.pos"] = float

        for cam_key, cam_cfg in self.config.cameras.items():
            features[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for joint in self.JOINT_NAMES[: self.config.num_joints_per_arm]:
            features[f"left_{joint}.pos"] = float
            features[f"right_{joint}.pos"] = float

        if self.config.use_gripper:
            features["left_gripper.pos"] = float
            features["right_gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        arms = self._left_arm is not None and self._right_arm is not None
        cams = all(cam.is_connected for cam in self.cameras.values())
        return arms and cams

    @property
    def is_calibrated(self) -> bool:
        # 使用绝对编码器，无需额外校准
        return True

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - calibrate unused by design
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is required to use AgilexBimanual.")

        self._left_arm = self._connect_arm(self.config.left_arm_port)
        self._right_arm = self._connect_arm(self.config.right_arm_port)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info("%s connected.", self)

    def _connect_arm(self, port: str) -> C_PiperInterface_V2:
        arm = C_PiperInterface_V2(port)
        arm.ConnectPort()

        try:
            status = arm.GetArmStatus().arm_status
            if getattr(status, "motion_status", None) not in (None, 0):
                arm.EmergencyStop(0x02)
                time.sleep(0.2)
        except Exception as exc:  # pragma: no cover - hardware dependency
            logger.debug("GetArmStatus failed on %s: %s", port, exc)

        deadline = time.time() + self.config.enable_timeout
        while time.time() < deadline:
            try:
                if arm.EnablePiper():
                    break
            except Exception as exc:  # pragma: no cover - hardware dependency
                logger.debug("EnablePiper failed on %s: %s", port, exc)
            time.sleep(0.1)
        else:
            arm.DisconnectPort()
            raise TimeoutError(f"EnablePiper timed out on {port}")

        speed = max(10, min(100, int(self.config.motion_speed)))
        try:
            arm.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        except Exception as exc:  # pragma: no cover - hardware dependency
            logger.warning("MotionCtrl_2 failed on %s: %s", port, exc)

        return arm

    def calibrate(self) -> None:
        # Piper 使用绝对编码器，无需校准
        return

    def configure(self) -> None:
        # 预留未来参数下发
        return

    def _read_joint_deg(self, arm: C_PiperInterface_V2) -> list[float]:
        joint_state = arm.GetArmJointMsgs().joint_state
        readings = []
        for idx in range(1, self.config.num_joints_per_arm + 1):
            readings.append(getattr(joint_state, f"joint_{idx}", 0.0) / 1000.0)
        return readings

    def _read_gripper_mm(self, arm: C_PiperInterface_V2) -> float:
        gripper_state = arm.GetArmGripperMsgs().gripper_state
        return getattr(gripper_state, "grippers_angle", 0.0) / 10000.0

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}
        left_joints = self._read_joint_deg(self._left_arm)  # type: ignore[arg-type]
        right_joints = self._read_joint_deg(self._right_arm)  # type: ignore[arg-type]

        for idx, joint in enumerate(self.JOINT_NAMES[: self.config.num_joints_per_arm]):
            obs[f"left_{joint}.pos"] = left_joints[idx]
            obs[f"right_{joint}.pos"] = right_joints[idx]

        if self.config.use_gripper:
            obs["left_gripper.pos"] = self._read_gripper_mm(self._left_arm)  # type: ignore[arg-type]
            obs["right_gripper.pos"] = self._read_gripper_mm(self._right_arm)  # type: ignore[arg-type]

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def _send_joint_deg(self, arm: C_PiperInterface_V2, joints: list[float]) -> None:
        ints = [int(round(val * 1000.0)) for val in joints[: self.config.num_joints_per_arm]]
        arm.JointCtrl(*ints)

    def _send_gripper_mm(self, arm: C_PiperInterface_V2, gripper_mm: float) -> None:
        arm.GripperCtrl(int(round(gripper_mm * 10000.0)), 1000, 0x01, 0x00)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left = [float(action[f"left_{j}.pos"]) for j in self.JOINT_NAMES[: self.config.num_joints_per_arm]]
        right = [float(action[f"right_{j}.pos"]) for j in self.JOINT_NAMES[: self.config.num_joints_per_arm]]

        self._send_joint_deg(self._left_arm, left)  # type: ignore[arg-type]
        self._send_joint_deg(self._right_arm, right)  # type: ignore[arg-type]

        if self.config.use_gripper:
            if "left_gripper.pos" in action:
                self._send_gripper_mm(self._left_arm, float(action["left_gripper.pos"]))  # type: ignore[arg-type]
            if "right_gripper.pos" in action:
                self._send_gripper_mm(self._right_arm, float(action["right_gripper.pos"]))  # type: ignore[arg-type]

        return action

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                logger.debug("Camera disconnect failed", exc_info=True)

        for arm in (self._left_arm, self._right_arm):
            if arm is None:
                continue
            try:
                if self.config.disable_torque_on_disconnect:
                    arm.DisablePiper()
            except Exception:
                logger.debug("DisablePiper failed", exc_info=True)
            try:
                arm.DisconnectPort()
            except Exception:
                logger.debug("DisconnectPort failed", exc_info=True)

        self._left_arm = None
        self._right_arm = None
