#!/usr/bin/env python

"""Continuously send zero joint targets to both AgileX follower arms.

This mirrors the `lerobot-replay` pipeline (robot connect -> get_observation ->
process action -> send_action) and publishes several frames instead of a single
one, making it more reliable when the Piper driver starts slightly later.


python agilex_scripts/send_zero_to_follower.py --duration_s=3 --fps=30
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import make_robot_from_config
from lerobot.robots.agilex import AgileXConfig
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class SendZeroConfig:
    # Robot configuration; defaults to real hardware.
    robot: AgileXConfig = field(default_factory=lambda: AgileXConfig(mock=False, max_relative_target=0.0))
    # Publish rate and duration (seconds) for the zero command stream.
    fps: int = 30
    duration_s: float = 2.0


@parser.wrap()
def main(cfg: SendZeroConfig) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ros_master = os.environ.get("ROS_MASTER_URI")
    if ros_master:
        logging.info("Using ROS_MASTER_URI=%s", ros_master)

    robot = make_robot_from_config(cfg.robot)
    robot_action_processor = make_default_robot_action_processor()

    # Build a zero action using the robot's declared action features.
    zero_action = {name: 0.0 for name in robot.action_features}

    try:
        robot.connect(calibrate=False)
        logging.info("Connected. Streaming zero commands at %s FPS for %ss.", cfg.fps, cfg.duration_s)

        num_steps = int(cfg.duration_s * cfg.fps)
        for step in range(num_steps):
            start_t = time.perf_counter()

            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((zero_action, robot_obs))
            _ = robot.send_action(processed_action)

            dt = time.perf_counter() - start_t
            precise_sleep(max(0.0, 1 / cfg.fps - dt))

            if step == 0:
                logging.info("First zero frame sent.")

        logging.info("Finished streaming zero commands (%s frames).", num_steps)
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
