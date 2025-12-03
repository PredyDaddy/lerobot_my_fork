#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Agilex Piper teleoperator (master/leader arms) implementation for LeRobot.

This module provides the AgileXTeleoperator class that reads joint positions
from the master/leader arms for teleoperation.
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.robots.agilex.agilex_ros_bridge import (
    AgileXROSBridge,
    MockAgileXROSBridge,
)
from lerobot.robots.agilex.config_agilex import JOINT_NAMES
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_agilex_teleop import AgileXTeleoperatorConfig

logger = logging.getLogger(__name__)


class AgileXTeleoperator(Teleoperator):
    """Agilex master arm teleoperator.

    This teleoperator reads joint positions from the master/leader arms
    and provides them as actions for controlling the follower/puppet arms.

    Example:
        ```python
        from lerobot.teleoperators.agilex import AgileXTeleoperator, AgileXTeleoperatorConfig

        config = AgileXTeleoperatorConfig(mock=True)
        teleop = AgileXTeleoperator(config)
        teleop.connect()

        action = teleop.get_action()
        print(action.keys())

        teleop.disconnect()
        ```
    """

    config_class = AgileXTeleoperatorConfig
    name = "agilex_teleop"

    def __init__(self, config: AgileXTeleoperatorConfig):
        """Initialize the teleoperator.

        Args:
            config: Teleoperator configuration.
        """
        super().__init__(config)
        self.config = config

        # Create bridge for teleoperator:
        # - Subscribe to master arms to get leader input
        # - Don't need puppet topics or command publishing
        if config.mock:
            self._bridge = MockAgileXROSBridge()
        else:
            self._bridge = AgileXROSBridge(
                node_name=config.node_name,
                master_left_topic=config.master_left_topic,
                master_right_topic=config.master_right_topic,
                # Don't need puppet topics for teleoperator
                puppet_left_topic="",
                puppet_right_topic="",
                puppet_left_cmd_topic="",
                puppet_right_cmd_topic="",
                subscribe_to_master=True,  # Teleoperator needs to read master arm state
            )

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """Define the action features (master arm joint positions).

        Returns:
            Dictionary mapping action names to their types.
        """
        features = {}
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                features[f"{side}_{joint_name}.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, Any]:
        """Define feedback features (not used for Agilex).

        Returns:
            Empty dictionary as Agilex doesn't support force feedback.
        """
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if the teleoperator is connected."""
        return self._bridge.is_connected()

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the master arms.

        Args:
            calibrate: Whether to run calibration (not used for Agilex).

        Raises:
            DeviceAlreadyConnectedError: If already connected.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._bridge.connect()
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """Disconnect from the master arms.

        Raises:
            DeviceNotConnectedError: If not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._bridge.disconnect()
        logger.info(f"{self} disconnected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if calibrated (always True for Agilex)."""
        return True

    def calibrate(self) -> None:
        """Run calibration (handled by ROS driver for Agilex)."""
        logger.info("Agilex calibration is handled by ROS driver")

    def get_action(self) -> dict[str, float]:
        """Read master arm joint positions as action.

        Returns:
            Dictionary mapping joint names to their positions.

        Raises:
            DeviceNotConnectedError: If not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        left_state, right_state = self._bridge.get_master_state()

        action = {}
        for i, joint_name in enumerate(JOINT_NAMES):
            action[f"left_{joint_name}.pos"] = float(left_state.position[i])
            action[f"right_{joint_name}.pos"] = float(right_state.position[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send feedback to the master arms (not supported).

        Args:
            feedback: Feedback values (ignored).
        """
        # Agilex doesn't support force feedback
        pass

    def configure(self) -> None:
        """Apply runtime configuration.

        For Agilex, configuration is handled by the ROS driver.
        """
        pass

