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

"""ROS Bridge for Agilex Piper robot communication.

This module provides the communication layer between LeRobot and the Agilex Piper
robot through ROS topics. It supports both real hardware and mock mode for testing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ROS joint names used by the original AgileX replay script.
# The controller relies primarily on the ordering, but we keep names
# for better compatibility with the vendor stack.
ROS_JOINT_NAMES = [f"joint{i}" for i in range(7)]


@dataclass
class JointState:
    """Joint state data structure.

    Attributes:
        position: Joint positions array of shape (7,).
        velocity: Joint velocities array of shape (7,).
        effort: Joint efforts/torques array of shape (7,).
        timestamp: Timestamp of the measurement.
    """

    position: np.ndarray  # shape: (7,)
    velocity: np.ndarray  # shape: (7,)
    effort: np.ndarray  # shape: (7,)
    timestamp: float


@runtime_checkable
class AgileXROSBridgeProtocol(Protocol):
    """Protocol defining the ROS Bridge interface."""

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def is_connected(self) -> bool:
        ...

    def get_puppet_state(self) -> tuple[JointState, JointState]:
        ...

    def get_master_state(self) -> tuple[JointState, JointState]:
        ...

    def send_joint_commands(self, left: np.ndarray, right: np.ndarray) -> None:
        ...


class AgileXROSBridge:
    """Real ROS communication implementation for Agilex Piper robot.

    This class handles all ROS communication including:
    - Subscribing to joint state topics for both puppet (follower) and master (leader) arms
    - Publishing joint commands to the puppet arms
    """

    def __init__(
        self,
        node_name: str = "lerobot_agilex",
        puppet_left_topic: str = "/puppet/joint_left",
        puppet_right_topic: str = "/puppet/joint_right",
        master_left_topic: str = "/master/joint_left",
        master_right_topic: str = "/master/joint_right",
        puppet_left_cmd_topic: str = "/puppet/joint_left/command",
        puppet_right_cmd_topic: str = "/puppet/joint_right/command",
        enable_flag_topic: str = "/enable_flag",
        subscribe_to_master: bool = True,
        disable_on_disconnect: bool = True,
    ):
        """Initialize the ROS bridge.

        Args:
            node_name: Name for the ROS node.
            puppet_left_topic: Topic for left follower arm state.
            puppet_right_topic: Topic for right follower arm state.
            master_left_topic: Topic for left leader arm state / command output.
            master_right_topic: Topic for right leader arm state / command output.
            puppet_left_cmd_topic: Topic for left follower arm commands (unused for AgileX).
            puppet_right_cmd_topic: Topic for right follower arm commands (unused for AgileX).
            enable_flag_topic: Topic for robot enable/disable flag.
            subscribe_to_master: If True, subscribe to master topics to read leader
                arm state (for teleoperator). If False, only publish commands to
                master topics (for robot replay).
        """
        self._subscribe_to_master = subscribe_to_master
        self._node_name = node_name
        self._puppet_left_topic = puppet_left_topic
        self._puppet_right_topic = puppet_right_topic
        self._master_left_topic = master_left_topic
        self._master_right_topic = master_right_topic
        self._puppet_left_cmd_topic = puppet_left_cmd_topic
        self._puppet_right_cmd_topic = puppet_right_cmd_topic
        self._enable_flag_topic = enable_flag_topic
        self._disable_on_disconnect = disable_on_disconnect

        # ROS objects (lazy initialization)
        self._ros_initialized = False
        self._sub_puppet_left = None
        self._sub_puppet_right = None
        self._sub_master_left = None
        self._sub_master_right = None
        self._pub_left = None
        self._pub_right = None
        self._pub_enable = None

        # Data cache
        self._puppet_left: JointState | None = None
        self._puppet_right: JointState | None = None
        self._master_left: JointState | None = None
        self._master_right: JointState | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Initialize ROS node and subscribe to topics.

        Raises:
            RuntimeError: If already connected.
            ImportError: If ROS dependencies are not available.
            TimeoutError: If no data is received within timeout.
        """
        if self._ros_initialized:
            raise RuntimeError("Already connected")

        try:
            import rospy
            from sensor_msgs.msg import JointState as RosJointState
            from std_msgs.msg import Bool
        except ImportError as e:
            raise ImportError(
                "ROS dependencies not found. Please install ROS Noetic: "
                "sudo apt install ros-noetic-desktop-full"
            ) from e

        # Initialize ROS node only if not already initialized
        if not rospy.core.is_initialized():
            rospy.init_node(self._node_name, anonymous=True)
        self._ros_initialized = True

        # Subscribe to puppet (follower) arm states
        if self._puppet_left_topic:
            self._sub_puppet_left = rospy.Subscriber(
                self._puppet_left_topic,
                RosJointState,
                lambda msg: self._joint_callback(msg, "puppet_left"),
                queue_size=1,
            )
        if self._puppet_right_topic:
            self._sub_puppet_right = rospy.Subscriber(
                self._puppet_right_topic,
                RosJointState,
                lambda msg: self._joint_callback(msg, "puppet_right"),
                queue_size=1,
            )

        # Subscribe to master (leader) arm states (only if requested)
        # For teleoperator: subscribe to get leader arm input
        # For robot replay: don't subscribe, only publish commands
        if self._subscribe_to_master:
            if self._master_left_topic:
                self._sub_master_left = rospy.Subscriber(
                    self._master_left_topic,
                    RosJointState,
                    lambda msg: self._joint_callback(msg, "master_left"),
                    queue_size=1,
                )
            if self._master_right_topic:
                self._sub_master_right = rospy.Subscriber(
                    self._master_right_topic,
                    RosJointState,
                    lambda msg: self._joint_callback(msg, "master_right"),
                    queue_size=1,
                )

        # Create publishers for master (leader) arm commands.
        #
        # The AgileX control stack moves the puppet arms based on commands
        # sent to the master topics (/master/joint_left and /master/joint_right),
        # exactly like `aiglex_origin_code/replay_data.py`.
        if self._master_left_topic:
            self._pub_left = rospy.Publisher(
                self._master_left_topic, RosJointState, queue_size=1
            )
        if self._master_right_topic:
            self._pub_right = rospy.Publisher(
                self._master_right_topic, RosJointState, queue_size=1
            )

        # Create publisher for enable flag
        if self._enable_flag_topic:
            self._pub_enable = rospy.Publisher(
                self._enable_flag_topic, Bool, queue_size=1, latch=True
            )
            # Enable the robot after a short delay
            rospy.sleep(0.5)
            self._pub_enable.publish(Bool(data=True))
            logger.info(f"[{self._node_name}] Robot enabled")

        # Wait for initial data
        self._wait_for_data(timeout=10.0)
        logger.info(f"[{self._node_name}] ROS Bridge connected")

    def disconnect(self) -> None:
        """Disconnect from ROS and unsubscribe from topics."""
        import rospy
        from std_msgs.msg import Bool

        # Disable robot before disconnecting
        if self._disable_on_disconnect and self._pub_enable is not None:
            try:
                self._pub_enable.publish(Bool(data=False))
                rospy.sleep(0.1)
                logger.info(f"[{self._node_name}] Robot disabled")
            except Exception:
                pass
        elif not self._disable_on_disconnect:
            logger.info(f"[{self._node_name}] Skip disable_on_disconnect; leaving enable_flag as-is")

        if self._sub_puppet_left:
            self._sub_puppet_left.unregister()
        if self._sub_puppet_right:
            self._sub_puppet_right.unregister()
        if self._sub_master_left:
            self._sub_master_left.unregister()
        if self._sub_master_right:
            self._sub_master_right.unregister()
        if self._pub_left:
            self._pub_left.unregister()
        if self._pub_right:
            self._pub_right.unregister()
        if self._pub_enable:
            self._pub_enable.unregister()
        self._ros_initialized = False
        logger.info(f"[{self._node_name}] ROS Bridge disconnected")

    def is_connected(self) -> bool:
        """Check if the bridge is connected."""
        return self._ros_initialized

    def _joint_callback(self, msg, arm_id: str) -> None:
        """ROS callback for joint state messages.

        Args:
            msg: ROS JointState message.
            arm_id: Identifier for the arm (puppet_left, puppet_right, master_left, master_right).
        """
        state = JointState(
            position=np.array(msg.position[:7], dtype=np.float32),
            velocity=np.array(msg.velocity[:7], dtype=np.float32) if msg.velocity else np.zeros(7, dtype=np.float32),
            effort=np.array(msg.effort[:7], dtype=np.float32) if msg.effort else np.zeros(7, dtype=np.float32),
            timestamp=msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
        )
        with self._lock:
            setattr(self, f"_{arm_id}", state)

    def _wait_for_data(self, timeout: float) -> None:
        """Wait for initial joint state data.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            TimeoutError: If no data is received within timeout.
        """
        import rospy

        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                # Check based on which topics we're subscribed to
                puppet_ok = True
                master_ok = True

                # Only check puppet if we subscribed to puppet topics
                if self._sub_puppet_left is not None or self._sub_puppet_right is not None:
                    puppet_ok = (self._puppet_left is not None and self._puppet_right is not None)

                # Only check master if we subscribed to master topics
                if self._sub_master_left is not None or self._sub_master_right is not None:
                    master_ok = (self._master_left is not None and self._master_right is not None)

                if puppet_ok and master_ok:
                    return
            rospy.sleep(0.01)

        # Build error message based on what's missing
        missing = []
        if self._sub_puppet_left is not None and self._puppet_left is None:
            missing.append(f"puppet_left ({self._puppet_left_topic})")
        if self._sub_puppet_right is not None and self._puppet_right is None:
            missing.append(f"puppet_right ({self._puppet_right_topic})")
        if self._sub_master_left is not None and self._master_left is None:
            missing.append(f"master_left ({self._master_left_topic})")
        if self._sub_master_right is not None and self._master_right is None:
            missing.append(f"master_right ({self._master_right_topic})")

        raise TimeoutError(f"Timeout waiting for joint states: {', '.join(missing)}")

    def get_puppet_state(self) -> tuple[JointState, JointState]:
        """Get the current state of the puppet (follower) arms.

        Returns:
            Tuple of (left_state, right_state) JointState objects.

        Raises:
            RuntimeError: If no puppet state is available.
        """
        with self._lock:
            if self._puppet_left is None or self._puppet_right is None:
                raise RuntimeError("No puppet state available")
            return self._puppet_left, self._puppet_right

    def get_master_state(self) -> tuple[JointState, JointState]:
        """Get the current state of the master (leader) arms.

        Returns:
            Tuple of (left_state, right_state) JointState objects.

        Raises:
            RuntimeError: If no master state is available.
        """
        with self._lock:
            if self._master_left is None or self._master_right is None:
                raise RuntimeError("No master state available")
            return self._master_left, self._master_right

    def send_joint_commands(self, left: np.ndarray, right: np.ndarray) -> None:
        """Send joint position commands through master arm topics.

        This mirrors the behavior of the original AgileX ``replay_data.py``
        script: commands are sent as ``sensor_msgs/JointState`` messages on
        ``/master/joint_left`` and ``/master/joint_right``. The vendor
        controller then drives the puppet arms accordingly.

        Args:
            left: Target joint positions for left arm, shape (7,).
            right: Target joint positions for right arm, shape (7,).
        """
        import rospy
        from sensor_msgs.msg import JointState as RosJointState

        if self._pub_left is not None:
            msg_left = RosJointState()
            msg_left.header.stamp = rospy.Time.now()
            msg_left.name = ROS_JOINT_NAMES
            msg_left.position = left.tolist()
            self._pub_left.publish(msg_left)

        if self._pub_right is not None:
            msg_right = RosJointState()
            msg_right.header.stamp = rospy.Time.now()
            msg_right.name = ROS_JOINT_NAMES
            msg_right.position = right.tolist()
            self._pub_right.publish(msg_right)


class MockAgileXROSBridge:
    """Mock implementation for testing without hardware.

    This class simulates the ROS bridge behavior for unit testing
    and development without actual robot hardware.
    """

    def __init__(self, **kwargs):
        """Initialize the mock bridge."""
        self._connected = False
        self._puppet_left = np.zeros(7, dtype=np.float32)
        self._puppet_right = np.zeros(7, dtype=np.float32)
        self._master_left = np.zeros(7, dtype=np.float32)
        self._master_right = np.zeros(7, dtype=np.float32)

    def connect(self) -> None:
        """Connect in mock mode."""
        self._connected = True
        logger.info("[MockBridge] Connected (mock mode)")

    def disconnect(self) -> None:
        """Disconnect in mock mode."""
        self._connected = False
        logger.info("[MockBridge] Disconnected")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def get_puppet_state(self) -> tuple[JointState, JointState]:
        """Get mock puppet state."""
        return (
            JointState(self._puppet_left.copy(), np.zeros(7), np.zeros(7), time.time()),
            JointState(self._puppet_right.copy(), np.zeros(7), np.zeros(7), time.time()),
        )

    def get_master_state(self) -> tuple[JointState, JointState]:
        """Get mock master state with small random noise."""
        noise = np.random.randn(7).astype(np.float32) * 0.01
        return (
            JointState(self._master_left + noise, np.zeros(7), np.zeros(7), time.time()),
            JointState(self._master_right + noise, np.zeros(7), np.zeros(7), time.time()),
        )

    def send_joint_commands(self, left: np.ndarray, right: np.ndarray) -> None:
        """Send commands in mock mode (updates internal state)."""
        self._puppet_left = left.astype(np.float32)
        self._puppet_right = right.astype(np.float32)


def make_ros_bridge(config, subscribe_to_master: bool = False) -> AgileXROSBridgeProtocol:
    """Factory function to create the appropriate ROS bridge.

    Args:
        config: Robot or teleoperator configuration.
        subscribe_to_master: If True, subscribe to master topics (for teleoperator).
            If False, only publish commands to master topics (for robot replay).

    Returns:
        An instance of AgileXROSBridge or MockAgileXROSBridge.
    """
    if config.mock:
        return MockAgileXROSBridge()

    # Determine which topics to use based on config type
    puppet_left_topic = getattr(config, "puppet_left_topic", "")
    puppet_right_topic = getattr(config, "puppet_right_topic", "")
    puppet_left_cmd_topic = getattr(config, "puppet_left_cmd_topic", "")
    puppet_right_cmd_topic = getattr(config, "puppet_right_cmd_topic", "")
    master_left_topic = getattr(config, "master_left_topic", "")
    master_right_topic = getattr(config, "master_right_topic", "")
    enable_flag_topic = getattr(config, "enable_flag_topic", "/enable_flag")
    disable_on_disconnect = getattr(config, "disable_on_disconnect", True)

    return AgileXROSBridge(
        node_name=config.node_name,
        puppet_left_topic=puppet_left_topic,
        puppet_right_topic=puppet_right_topic,
        master_left_topic=master_left_topic,
        master_right_topic=master_right_topic,
        puppet_left_cmd_topic=puppet_left_cmd_topic,
        puppet_right_cmd_topic=puppet_right_cmd_topic,
        enable_flag_topic=enable_flag_topic,
        subscribe_to_master=subscribe_to_master,
        disable_on_disconnect=disable_on_disconnect,
    )
