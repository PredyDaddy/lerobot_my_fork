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

"""Unit tests for Agilex Piper robot using mock mode."""

import numpy as np
import pytest

from lerobot.robots.agilex.agilex import AgileXRobot
from lerobot.robots.agilex.agilex_ros_bridge import JointState, MockAgileXROSBridge
from lerobot.robots.agilex.config_agilex import JOINT_NAMES, AgileXConfig


@pytest.fixture
def mock_robot():
    """Create a mock mode robot without cameras."""
    config = AgileXConfig(mock=True, cameras={})
    robot = AgileXRobot(config)
    robot.connect()
    yield robot
    robot.disconnect()


class TestAgileXConfig:
    """Tests for AgileXConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgileXConfig(mock=True)
        assert config.mock is True
        assert config.node_name == "lerobot_agilex"
        assert config.max_relative_target == 0.2
        assert len(config.joint_limits) == 7

    def test_joint_names(self):
        """Test joint names constant."""
        assert len(JOINT_NAMES) == 7
        assert "shoulder_pan" in JOINT_NAMES
        assert "gripper" in JOINT_NAMES


class TestMockAgileXROSBridge:
    """Tests for MockAgileXROSBridge."""

    def test_connect_disconnect(self):
        """Test connection lifecycle."""
        bridge = MockAgileXROSBridge()
        assert not bridge.is_connected()

        bridge.connect()
        assert bridge.is_connected()

        bridge.disconnect()
        assert not bridge.is_connected()

    def test_get_puppet_state(self):
        """Test getting puppet state."""
        bridge = MockAgileXROSBridge()
        bridge.connect()

        left, right = bridge.get_puppet_state()
        assert isinstance(left, JointState)
        assert isinstance(right, JointState)
        assert left.position.shape == (7,)
        assert right.position.shape == (7,)

    def test_send_joint_commands(self):
        """Test sending joint commands."""
        bridge = MockAgileXROSBridge()
        bridge.connect()

        left_cmd = np.array([0.1] * 7, dtype=np.float32)
        right_cmd = np.array([0.2] * 7, dtype=np.float32)
        bridge.send_joint_commands(left_cmd, right_cmd)

        # Verify state was updated
        left, right = bridge.get_puppet_state()
        np.testing.assert_array_almost_equal(left.position, left_cmd)
        np.testing.assert_array_almost_equal(right.position, right_cmd)


class TestAgileXRobot:
    """Tests for AgileXRobot."""

    def test_observation_features(self, mock_robot):
        """Test observation feature definitions."""
        features = mock_robot.observation_features

        # Check all joints are present
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                assert f"{side}_{joint_name}.pos" in features

        # Total 14 joint features (no cameras in this test)
        assert len(features) == 14

    def test_action_features(self, mock_robot):
        """Test action feature definitions."""
        features = mock_robot.action_features
        assert len(features) == 14  # 7 joints * 2 arms

    def test_get_observation(self, mock_robot):
        """Test getting observations."""
        obs = mock_robot.get_observation()

        # Check joint data exists
        assert "left_shoulder_pan.pos" in obs
        assert "right_gripper.pos" in obs

        # Check all 14 joints
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                key = f"{side}_{joint_name}.pos"
                assert key in obs
                assert isinstance(obs[key], float)

    def test_send_action(self, mock_robot):
        """Test sending actions."""
        action = {}
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                action[f"{side}_{joint_name}.pos"] = 0.1

        result = mock_robot.send_action(action)

        # Verify returned action
        assert len(result) == 14
        assert result["left_shoulder_pan.pos"] == pytest.approx(0.1, abs=0.01)

    def test_safety_clipping(self, mock_robot):
        """Test safety clipping of large actions."""
        # Send a large action that should be clipped
        action = {}
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                action[f"{side}_{joint_name}.pos"] = 10.0  # Way beyond limit

        result = mock_robot.send_action(action)

        # Verify clipping was applied
        max_delta = mock_robot.config.max_relative_target
        assert result["left_shoulder_pan.pos"] <= max_delta

    def test_is_calibrated(self, mock_robot):
        """Test calibration status."""
        assert mock_robot.is_calibrated is True

