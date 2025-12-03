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

"""Agilex Piper dual-arm robot implementation for LeRobot.

This module provides the AgileXRobot class that implements the LeRobot Robot
interface for the Agilex Piper dual-arm teleoperation system.
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .agilex_ros_bridge import make_ros_bridge
from .config_agilex import JOINT_NAMES, AgileXConfig

logger = logging.getLogger(__name__)


class AgileXRobot(Robot):
    """Agilex Piper dual-arm robot.

    This robot supports the Agilex Piper dual-arm teleoperation system with:
    - 7 DOF per arm (6 joints + 1 gripper)
    - Total 14 DOF for dual arms
    - ROS-based communication
    - Optional camera integration

    Example:
        ```python
        from lerobot.robots.agilex import AgileXRobot, AgileXConfig

        config = AgileXConfig(mock=True)  # Use mock mode for testing
        robot = AgileXRobot(config)
        robot.connect()

        obs = robot.get_observation()
        print(obs.keys())

        robot.disconnect()
        ```
    """

    config_class = AgileXConfig
    name = "agilex"

    def __init__(self, config: AgileXConfig):
        """Initialize the Agilex robot.

        Args:
            config: Robot configuration.
        """
        super().__init__(config)
        self.config = config
        # Robot only needs to:
        # - Subscribe to puppet topics (to read current state)
        # - Publish to master topics (to send commands)
        # It does NOT need to subscribe to master topics (that's for teleoperator)
        self.ros_bridge = make_ros_bridge(config, subscribe_to_master=False)
        self.cameras = make_cameras_from_configs(config.cameras)

    # ===== Feature Definitions =====

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """Define the observation space.

        Returns:
            Dictionary mapping feature names to their types/shapes.
            - Joint positions: float values for each of 14 joints
            - Camera images: (height, width, 3) tuples
        """
        features = {}

        # Dual arm joint positions (14 DOF)
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                features[f"{side}_{joint_name}.pos"] = float

        # Camera images
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)

        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """Define the action space (controls puppet/follower arms only).

        Returns:
            Dictionary mapping action names to their types.
        """
        features = {}
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                features[f"{side}_{joint_name}.pos"] = float
        return features

    # ===== Connection Management =====

    @property
    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        return self.ros_bridge.is_connected()

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot.

        Args:
            calibrate: Whether to run calibration (not used for Agilex as it uses
                      built-in calibration from the ROS driver).

        Raises:
            DeviceAlreadyConnectedError: If already connected.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.ros_bridge.connect()

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """Disconnect from the robot.

        Raises:
            DeviceNotConnectedError: If not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        self.ros_bridge.disconnect()
        logger.info(f"{self} disconnected.")

    # ===== Core I/O =====

    def get_observation(self) -> dict[str, Any]:
        """Get the current observation from the robot.

        Returns:
            Dictionary containing:
            - Joint positions for all 14 joints
            - Camera images (if cameras are configured)

        Raises:
            DeviceNotConnectedError: If not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = {}

        # Get puppet (follower) arm joint states
        start = time.perf_counter()
        left_state, right_state = self.ros_bridge.get_puppet_state()

        for i, joint_name in enumerate(JOINT_NAMES):
            obs[f"left_{joint_name}.pos"] = float(left_state.position[i])
            obs[f"right_{joint_name}.pos"] = float(right_state.position[i])

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read joints: {dt_ms:.1f}ms")

        # Get camera images
        for cam_name, cam in self.cameras.items():
            start = time.perf_counter()
            img = cam.async_read()
            if img is not None:
                obs[cam_name] = img
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_name}: {dt_ms:.1f}ms")

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send control commands to the robot.

        Args:
            action: Dictionary mapping joint names to target positions.

        Returns:
            Dictionary of actually sent actions (may be clipped for safety).

        Raises:
            DeviceNotConnectedError: If not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Parse action into left and right arm targets
        left_target = np.array(
            [action[f"left_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32
        )
        right_target = np.array(
            [action[f"right_{jn}.pos"] for jn in JOINT_NAMES], dtype=np.float32
        )

        # Apply safety clipping if configured
        if self.config.max_relative_target > 0:
            left_state, right_state = self.ros_bridge.get_puppet_state()
            left_target = self._clip_delta(left_state.position, left_target)
            right_target = self._clip_delta(right_state.position, right_target)

        # Send commands
        self.ros_bridge.send_joint_commands(left_target, right_target)

        # Return the actually sent actions
        result = {}
        for i, jn in enumerate(JOINT_NAMES):
            result[f"left_{jn}.pos"] = float(left_target[i])
            result[f"right_{jn}.pos"] = float(right_target[i])
        return result

    def _clip_delta(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Clip the target to limit single-step change.

        Args:
            current: Current joint positions.
            target: Target joint positions.

        Returns:
            Clipped target positions.
        """
        delta = target - current
        delta = np.clip(delta, -self.config.max_relative_target, self.config.max_relative_target)
        return current + delta

    # ===== Calibration =====

    @property
    def is_calibrated(self) -> bool:
        """Check if the robot is calibrated.

        Agilex uses built-in calibration from the ROS driver.
        """
        return True

    def calibrate(self) -> None:
        """Run calibration.

        Agilex calibration is handled by the ROS driver.
        """
        logger.info("Agilex calibration is handled by ROS driver")

    def configure(self) -> None:
        """Apply runtime configuration.

        For Agilex, configuration is handled by the ROS driver.
        """
        pass

