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

"""Configuration for ROS Camera adapter."""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("ros_camera")
@dataclass
class RosCameraConfig(CameraConfig):
    """ROS Camera configuration.

    This camera adapter subscribes to a ROS Image topic and converts
    the images to numpy arrays compatible with LeRobot.

    Attributes:
        topic_name: The ROS topic name to subscribe to (e.g., "/camera/color/image_raw").
        mock: If True, use mock mode without actual ROS connection.
    """

    topic_name: str = "/camera/color/image_raw"
    mock: bool = False

    def __post_init__(self):
        if not self.topic_name.startswith("/"):
            raise ValueError(f"topic_name must start with '/', got: {self.topic_name}")

