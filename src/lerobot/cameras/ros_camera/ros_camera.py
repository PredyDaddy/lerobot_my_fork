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

import threading
from typing import Any

import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .config_ros import RosCameraConfig

__all__ = ["RosCamera"]


class RosCamera(Camera):
    """ROS image topic wrapper compatible with LeRobot's Camera interface."""

    config_class = RosCameraConfig
    name = "ros"

    def __init__(self, config: RosCameraConfig):
        super().__init__(config)
        self.config = config

        self._bridge = None
        self._latest_frame: NDArray[Any] | None = None
        self._latest_depth: NDArray[Any] | None = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._is_connected = False
        self._subs: list[Any] = []

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.topic})"

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # ROS 相机通过话题而非设备发现
        return []

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        if not rospy.core.is_initialized():
            rospy.init_node("lerobot_ros_camera", anonymous=True)

        self._bridge = CvBridge()
        self._subs = [
            rospy.Subscriber(
                self.config.topic,
                Image,
                self._image_callback,
                queue_size=self.config.queue_size,
                tcp_nodelay=True,
            )
        ]
        if self.config.use_depth and self.config.depth_topic:
            self._subs.append(
                rospy.Subscriber(
                    self.config.depth_topic,
                    Image,
                    self._depth_callback,
                    queue_size=self.config.queue_size,
                    tcp_nodelay=True,
                )
            )

        self._is_connected = True

        if warmup:
            # 等待首帧到达，避免返回 None
            self._frame_event.wait(timeout=2.0)

    def _image_callback(self, msg) -> None:
        # Lazy import to avoid ROS deps on import time
        frame = self._bridge.imgmsg_to_cv2(msg, "rgb8")
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_event.set()

    def _depth_callback(self, msg) -> None:
        depth = self._bridge.imgmsg_to_cv2(msg, "passthrough")
        with self._frame_lock:
            self._latest_depth = depth

    def async_read(self, timeout_ms: float = 0.0) -> NDArray[Any] | None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if timeout_ms and not self._frame_event.wait(timeout_ms / 1000.0):
            return None

        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return np.copy(self._latest_frame)

    def async_read_depth(self, timeout_ms: float = 0.0) -> NDArray[Any] | None:
        if not self.config.use_depth:
            return None
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if timeout_ms and not self._frame_event.wait(timeout_ms / 1000.0):
            return None
        with self._frame_lock:
            if self._latest_depth is None:
                return None
            return np.copy(self._latest_depth)

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any] | None:
        frame = self.async_read()
        if frame is None:
            return None
        if color_mode == ColorMode.BGR:
            import cv2

            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:
                pass
        self._subs = []
        self._is_connected = False
        self._frame_event.clear()
