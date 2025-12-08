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
from functools import cached_property
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_agilex_leader import AgilexBimanualLeaderConfig

logger = logging.getLogger(__name__)

try:
    from piper_sdk import C_PiperInterface_V2
except Exception:  # pragma: no cover - hardware dependency
    C_PiperInterface_V2 = None  # type: ignore


class AgilexBimanualLeader(Teleoperator):
    """Agilex 双臂主臂遥操作器（被动监听 CAN）。"""

    config_class = AgilexBimanualLeaderConfig
    name = "agilex_bimanual_leader"

    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "wrist_rotate",
    ]

    def __init__(self, config: AgilexBimanualLeaderConfig):
        super().__init__(config)
        self.config = config
        self._left_arm: C_PiperInterface_V2 | None = None
        self._right_arm: C_PiperInterface_V2 | None = None

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
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._left_arm is not None and self._right_arm is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - calibrate unused by design
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is required to use AgilexBimanualLeader.")

        self._left_arm = self._connect_passive(self.config.left_arm_port)
        self._right_arm = self._connect_passive(self.config.right_arm_port)
        logger.info("%s connected.", self)

    def _connect_passive(self, port: str) -> C_PiperInterface_V2:
        arm = C_PiperInterface_V2(port)
        arm.ConnectPort(piper_init=False, start_thread=True)
        return arm

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _read_joint_deg(self, arm: C_PiperInterface_V2) -> list[float]:
        joint_state = arm.GetArmJointMsgs().joint_state
        return [
            getattr(joint_state, f"joint_{idx}", 0.0) / 1000.0
            for idx in range(1, self.config.num_joints_per_arm + 1)
        ]

    def _read_gripper_mm(self, arm: C_PiperInterface_V2) -> float:
        gripper_state = arm.GetArmGripperMsgs().gripper_state
        return getattr(gripper_state, "grippers_angle", 0.0) / 10000.0

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action: dict[str, Any] = {}
        left_joints = self._read_joint_deg(self._left_arm)  # type: ignore[arg-type]
        right_joints = self._read_joint_deg(self._right_arm)  # type: ignore[arg-type]
        for idx, joint in enumerate(self.JOINT_NAMES[: self.config.num_joints_per_arm]):
            action[f"left_{joint}.pos"] = left_joints[idx]
            action[f"right_{joint}.pos"] = right_joints[idx]

        if self.config.use_gripper:
            action["left_gripper.pos"] = self._read_gripper_mm(self._left_arm)  # type: ignore[arg-type]
            action["right_gripper.pos"] = self._read_gripper_mm(self._right_arm)  # type: ignore[arg-type]

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: ARG002
        # Piper SDK 主臂暂不支持反馈
        return

    def disconnect(self) -> None:
        for arm in (self._left_arm, self._right_arm):
            if arm is None:
                continue
            try:
                arm.DisconnectPort()
            except Exception:
                logger.debug("DisconnectPort failed", exc_info=True)
        self._left_arm = None
        self._right_arm = None
