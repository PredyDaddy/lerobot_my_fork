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

"""Unit tests for Agilex Piper teleoperator using mock mode."""

import pytest

from lerobot.robots.agilex.config_agilex import JOINT_NAMES
from lerobot.teleoperators.agilex.agilex_teleoperator import AgileXTeleoperator
from lerobot.teleoperators.agilex.config_agilex_teleop import AgileXTeleoperatorConfig


@pytest.fixture
def mock_teleop():
    """Create a mock mode teleoperator."""
    config = AgileXTeleoperatorConfig(mock=True)
    teleop = AgileXTeleoperator(config)
    teleop.connect()
    yield teleop
    teleop.disconnect()


class TestAgileXTeleoperatorConfig:
    """Tests for AgileXTeleoperatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgileXTeleoperatorConfig(mock=True)
        assert config.mock is True
        assert config.node_name == "lerobot_agilex_teleop"
        assert config.master_left_topic == "/master/joint_left"
        assert config.master_right_topic == "/master/joint_right"


class TestAgileXTeleoperator:
    """Tests for AgileXTeleoperator."""

    def test_connect_disconnect(self):
        """Test connection lifecycle."""
        config = AgileXTeleoperatorConfig(mock=True)
        teleop = AgileXTeleoperator(config)

        assert not teleop.is_connected

        teleop.connect()
        assert teleop.is_connected

        teleop.disconnect()
        assert not teleop.is_connected

    def test_action_features(self, mock_teleop):
        """Test action feature definitions."""
        features = mock_teleop.action_features
        assert len(features) == 14  # 7 joints * 2 arms

        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                assert f"{side}_{joint_name}.pos" in features

    def test_feedback_features(self, mock_teleop):
        """Test feedback features (should be empty for Agilex)."""
        features = mock_teleop.feedback_features
        assert len(features) == 0

    def test_get_action(self, mock_teleop):
        """Test getting action from master arms."""
        action = mock_teleop.get_action()

        # Check all joints are present
        assert len(action) == 14
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                key = f"{side}_{joint_name}.pos"
                assert key in action
                assert isinstance(action[key], float)

    def test_send_feedback(self, mock_teleop):
        """Test sending feedback (should not raise)."""
        # Agilex doesn't support force feedback, but should not raise
        mock_teleop.send_feedback({"test": 1.0})

    def test_is_calibrated(self, mock_teleop):
        """Test calibration status."""
        assert mock_teleop.is_calibrated is True

