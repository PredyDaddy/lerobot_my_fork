#!/usr/bin/env python

"""Send a single zero joint target to both AgileX follower arms.

This mirrors the `lerobot-replay` pipeline (robot connect -> get_observation ->
process action -> send_action) to make sure the ROS bridge is fully initialized
before sending the zero command.

  python agilex_scripts/send_zero_to_follower.py \
    --wait_after_send_s=2.0 \
    --robot.disable_on_disconnect=false
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


@dataclass
class SendZeroConfig:
    # Robot configuration; defaults to real hardware.
    # disable_on_disconnect=False 确保断开连接后保持使能状态，防止机器人掉落
    robot: AgileXConfig = field(default_factory=lambda: AgileXConfig(
        mock=False,
        max_relative_target=0.0,
        disable_on_disconnect=False,  # 保持使能，不掉落
    ))
    # Seconds to keep the process alive after sending the zero command.
    # This helps the follower controller receive/execute the target before we disconnect.
    wait_after_send_s: float = 1.0


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
        logging.info("Connected. Sending a single zero joint target.")

        robot_obs = robot.get_observation()
        processed_action = robot_action_processor((zero_action, robot_obs))
        robot.send_action(processed_action)
        logging.info("Zero joint target sent.")
        if cfg.wait_after_send_s > 0:
            logging.info("Waiting %.2fs before disconnect...", cfg.wait_after_send_s)
            time.sleep(cfg.wait_after_send_s)
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
