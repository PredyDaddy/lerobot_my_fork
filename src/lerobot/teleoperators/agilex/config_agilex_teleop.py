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

"""Configuration for Agilex Piper teleoperator (master/leader arms)."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("agilex_teleop")
@dataclass
class AgileXTeleoperatorConfig(TeleoperatorConfig):
    """Agilex master arm teleoperator configuration.

    This configuration is for the leader/master arms that are used for
    teleoperation input.

    Attributes:
        ros_master_uri: URI of the ROS master node.
        node_name: Name for the ROS node.
        master_left_topic: ROS topic for left master arm state.
        master_right_topic: ROS topic for right master arm state.
        mock: If True, use mock mode without actual ROS connection.
    """

    ros_master_uri: str = "http://localhost:11311"
    node_name: str = "lerobot_agilex_teleop"

    # Master arm topics
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"

    # Mock mode
    mock: bool = False

