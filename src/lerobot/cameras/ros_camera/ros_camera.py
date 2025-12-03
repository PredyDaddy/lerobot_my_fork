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

"""ROS Camera adapter for LeRobot.

This module provides a camera implementation that subscribes to ROS Image topics
and converts them to numpy arrays compatible with LeRobot's camera interface.
"""

import logging
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .configuration_ros_camera import RosCameraConfig

logger = logging.getLogger(__name__)


class RosCamera(Camera):
    """ROS Image Topic to LeRobot Camera adapter.

    This camera subscribes to a ROS sensor_msgs/Image topic and provides
    the images through LeRobot's standard camera interface.

    Example:
        ```python
        from lerobot.cameras.ros_camera import RosCamera, RosCameraConfig

        config = RosCameraConfig(
            topic_name="/camera/color/image_raw",
            fps=30,
            width=640,
            height=480
        )
        camera = RosCamera(config)
        camera.connect()
        image = camera.read()
        camera.disconnect()
        ```
    """

    config_class = RosCameraConfig
    name = "ros_camera"

    def __init__(self, config: RosCameraConfig):
        """Initialize the ROS camera.

        Args:
            config: Camera configuration including topic name and resolution.
        """
        super().__init__(config)
        self.config = config
        self._sub = None
        self._cv_bridge = None
        self._latest_image: NDArray[Any] | None = None
        self._lock = threading.Lock()
        self._connected = False

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.topic_name})"

    @property
    def is_connected(self) -> bool:
        """Check if the camera is connected."""
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        """Connect to the ROS image topic.

        Args:
            warmup: If True, wait for the first image before returning.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            TimeoutError: If no image is received within timeout.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        if self.config.mock:
            self._connect_mock()
        else:
            self._connect_ros()

        if warmup:
            self._wait_for_first_image(timeout=10.0)

        self._connected = True
        logger.info(f"{self} connected.")

    def _connect_mock(self) -> None:
        """Connect in mock mode (no ROS required)."""
        # Create a dummy image
        h = self.height or 480
        w = self.width or 640
        self._latest_image = np.zeros((h, w, 3), dtype=np.uint8)
        logger.info(f"{self} connected in mock mode.")

    def _connect_ros(self) -> None:
        """Connect to actual ROS topic."""
        try:
            import rospy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image
        except ImportError as e:
            raise ImportError(
                "ROS dependencies not found. Please install ROS and cv_bridge: "
                "sudo apt install ros-noetic-cv-bridge"
            ) from e

        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber(
            self.config.topic_name,
            Image,
            self._image_callback,
            queue_size=1,
        )
        logger.info(f"Subscribed to {self.config.topic_name}")

    def _wait_for_first_image(self, timeout: float) -> None:
        """Wait for the first image to arrive."""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._latest_image is not None:
                    return
            time.sleep(0.01)
        raise TimeoutError(f"No image received from {self.config.topic_name} within {timeout}s")

    def _image_callback(self, msg) -> None:
        """ROS callback for incoming images."""
        try:
            # Convert ROS Image to OpenCV format (BGR)
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert BGR to RGB
            rgb_image = cv_image[:, :, ::-1].copy()
            with self._lock:
                self._latest_image = rgb_image
        except Exception as e:
            logger.error(f"Image conversion error: {e}")

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Read the latest image synchronously.

        Args:
            color_mode: Desired color mode (RGB or BGR). Defaults to RGB.

        Returns:
            The latest captured image as a numpy array.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If no image is available.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._lock:
            if self._latest_image is None:
                raise RuntimeError(f"No image available from {self}")
            image = self._latest_image.copy()

        # Handle color mode conversion
        requested_mode = color_mode or ColorMode.RGB
        if requested_mode == ColorMode.BGR:
            image = image[:, :, ::-1].copy()

        return image

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Read the latest image asynchronously (non-blocking).

        This method returns the most recent image without waiting for a new one.

        Args:
            timeout_ms: Maximum time to wait for an image (not used in this implementation
                       as we always return the latest cached image).

        Returns:
            The latest captured image as a numpy array.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
        """
        return self.read()

    def disconnect(self) -> None:
        """Disconnect from the ROS topic and release resources.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._sub is not None:
            self._sub.unregister()
            self._sub = None

        self._latest_image = None
        self._cv_bridge = None
        self._connected = False
        logger.info(f"{self} disconnected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Find available ROS image topics.

        Returns:
            A list of dictionaries containing information about available image topics.
        """
        try:
            import rospy

            topics = rospy.get_published_topics()
            image_topics = []
            for topic_name, topic_type in topics:
                if topic_type == "sensor_msgs/Image":
                    image_topics.append({
                        "name": f"ROS Camera @ {topic_name}",
                        "type": "ros_camera",
                        "id": topic_name,
                    })
            return image_topics
        except Exception as e:
            logger.warning(f"Failed to find ROS cameras: {e}")
            return []

