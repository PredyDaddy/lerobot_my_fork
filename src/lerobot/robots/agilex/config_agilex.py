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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig

__all__ = ["AgilexBimanualConfig"]


@RobotConfig.register_subclass("agilex_bimanual")
@dataclass(kw_only=True)
class AgilexBimanualConfig(RobotConfig):
    """Agilex Piper 双臂从臂配置。"""

    left_arm_port: str = "can_left"
    right_arm_port: str = "can_right"
    enable_timeout: float = 5.0
    motion_speed: int = 50
    disable_torque_on_disconnect: bool = True
    use_degrees: bool = True
    use_gripper: bool = True
    num_joints_per_arm: int = 6
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
