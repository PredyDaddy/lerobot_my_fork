#!/usr/bin/env python

"""Send a single fixed joint pose to both AgileX follower arms.

Example (keep motors enabled after exit):
 python agilex_scripts/send_pose_to_follower.py --shoulder_pan=0.05 --robot.disable_on_disconnect=false
"""

import logging
import os
import sys
from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import make_robot_from_config
from lerobot.robots.agilex import AgileXConfig


@dataclass
class SendPoseConfig:
    # Robot configuration; defaults to real hardware. You can override
    # --robot.disable_on_disconnect=false to keep motors enabled after exit.
    robot: AgileXConfig = field(default_factory=lambda: AgileXConfig(mock=False, max_relative_target=0.0))
    # Target pose: move the first joint a little; rest stay at zero.
    shoulder_pan: float = 0.05  # radians, applied to left arm
    right_shoulder_pan: float | None = None  # if None, mirror left


@parser.wrap()
def main(cfg: SendPoseConfig) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ros_master = os.environ.get("ROS_MASTER_URI")
    if ros_master:
        logging.info("Using ROS_MASTER_URI=%s", ros_master)

    robot = make_robot_from_config(cfg.robot)
    robot_action_processor = make_default_robot_action_processor()

    # Build a constant action: all zeros except shoulder_pan.
    target_action = {name: 0.0 for name in robot.action_features}
    target_action["left_shoulder_pan.pos"] = cfg.shoulder_pan
    right_pan = cfg.shoulder_pan if cfg.right_shoulder_pan is None else cfg.right_shoulder_pan
    target_action["right_shoulder_pan.pos"] = right_pan

    try:
        robot.connect(calibrate=False)
        logging.info(
            "Connected. Sending pose once (left_pan=%.3f, right_pan=%.3f).",
            cfg.shoulder_pan,
            right_pan,
        )

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((target_action, robot_obs))
        robot.send_action(processed_action)
        logging.info("Pose command sent.")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
