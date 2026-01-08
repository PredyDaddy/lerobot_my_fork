#!/usr/bin/env python3
"""
获取 Agilex 机械臂当前关节角度的脚本。

用法:
    python get_state.py
"""

import numpy as np

from lerobot.robots.agilex import AgileXConfig, AgileXRobot
from lerobot.robots.agilex.config_agilex import JOINT_NAMES


def main():
    # 创建机器人配置（不需要相机，断开时不掉使能）
    robot_config = AgileXConfig(
        id="get_state",
        mock=False,
        cameras={},
        disable_on_disconnect=False,  # 断开连接时不发送 disable 信号
    )
    robot = AgileXRobot(robot_config)

    print("正在连接机器人...")
    robot.connect()

    try:
        # 获取左右臂状态
        left_state, right_state = robot.ros_bridge.get_puppet_state()

        print("\n" + "=" * 60)
        print("当前机械臂关节角度")
        print("=" * 60)

        # 打印左臂状态
        print("\n【左臂 (Left Arm)】")
        left_positions = np.array(left_state.position)
        for i, name in enumerate(JOINT_NAMES):
            print(f"  left_{name}.pos: {left_positions[i]:.6f}")

        # 打印右臂状态
        print("\n【右臂 (Right Arm)】")
        right_positions = np.array(right_state.position)
        for i, name in enumerate(JOINT_NAMES):
            print(f"  right_{name}.pos: {right_positions[i]:.6f}")

        # 打印可复制的数组格式
        print("\n" + "=" * 60)
        print("可复制的数组格式:")
        print("=" * 60)
        print(f"\n# 左臂初始位置")
        print(f"LEFT_INIT_POS = {left_positions.tolist()}")
        print(f"\n# 右臂初始位置")
        print(f"RIGHT_INIT_POS = {right_positions.tolist()}")

    finally:
        print("\n断开机器人连接...")
        robot.disconnect()


if __name__ == "__main__":
    main()
