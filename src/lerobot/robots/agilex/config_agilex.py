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

"""Configuration for Agilex Piper dual-arm robot."""

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig

# Joint names for each arm (7 DOF: 6 joints + 1 gripper)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "shoulder_roll",
    "elbow",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
]


@RobotConfig.register_subclass("agilex")
@dataclass
class AgileXConfig(RobotConfig):
    """Agilex Piper dual-arm robot configuration.

    This configuration supports the Agilex Piper dual-arm teleoperation system
    with 7 DOF per arm (6 joints + 1 gripper), totaling 14 DOF.

    The AgileX control stack expects joint *commands* to be published as
    ``sensor_msgs/JointState`` messages on the master arm topics
    (``/master/joint_left`` and ``/master/joint_right``), exactly like the
    original ``aiglex_origin_code/replay_data.py`` script.

    Attributes:
        ros_master_uri: URI of the ROS master node.
        node_name: Name for the ROS node.
        puppet_left_topic: ROS topic for left follower arm state.
        puppet_right_topic: ROS topic for right follower arm state.
        master_left_topic: ROS topic for left master arm state / commands.
        master_right_topic: ROS topic for right master arm state / commands.
        puppet_left_cmd_topic: Optional dedicated follower command topic
            (unused for the current AgileX setup, kept for flexibility).
        puppet_right_cmd_topic: Optional dedicated follower command topic
            (unused for the current AgileX setup, kept for flexibility).
        enable_flag_topic: ROS topic for robot enable/disable flag.
        cameras: Dictionary of camera configurations.
        max_relative_target: Maximum single-step joint change (radians) for safety.
        joint_limits: Joint position limits in radians.
        mock: If True, use mock mode without actual ROS connection.
    """

    # === ROS Configuration ===
    ros_master_uri: str = "http://localhost:11311"
    node_name: str = "lerobot_agilex"

    # === Topic Configuration ===
    puppet_left_topic: str = "/puppet/joint_left"
    puppet_right_topic: str = "/puppet/joint_right"

    # Master arm topics (used both for observing teleop and for replay commands)
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"

    # Optional dedicated command topics (not used by default for AgileX)
    puppet_left_cmd_topic: str = ""
    puppet_right_cmd_topic: str = ""

    enable_flag_topic: str = "/enable_flag"

    # === Camera Configuration ===
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # === Safety Parameters ===
    max_relative_target: float = 0.2  # Maximum single-step joint change (radians)

    # === Joint Limits (radians) ===
    joint_limits: dict = field(
        default_factory=lambda: {
            "shoulder_pan": (-2.618, 2.618),
            "shoulder_lift": (-1.571, 1.571),
            "shoulder_roll": (-1.571, 1.571),
            "elbow": (-1.745, 1.745),
            "wrist_pitch": (-1.571, 1.571),
            "wrist_roll": (-2.094, 2.094),
            "gripper": (0.0, 0.085),  # Gripper stroke in meters
        }
    )

    # === Mock Mode (for testing) ===
    mock: bool = False

