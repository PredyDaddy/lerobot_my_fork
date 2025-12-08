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

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig

__all__ = ["RosCameraConfig"]


@CameraConfig.register_subclass("ros")
@dataclass
class RosCameraConfig(CameraConfig):
    """Configuration for ROS image topics."""

    topic: str = "/camera/color/image_raw"
    use_depth: bool = False
    depth_topic: str | None = None
    queue_size: int = 1

    def __post_init__(self) -> None:
        # Normalize topics and fail fast on illegal spaces to avoid ROS errors downstream.
        self.topic = self.topic.strip()
        if " " in self.topic:
            raise ValueError(f"Invalid ROS topic (contains spaces): {self.topic}")
        if self.depth_topic:
            self.depth_topic = self.depth_topic.strip()
            if " " in self.depth_topic:
                raise ValueError(f"Invalid ROS depth topic (contains spaces): {self.depth_topic}")
        if self.use_depth and not self.depth_topic:
            raise ValueError("depth_topic is required when use_depth=True")
