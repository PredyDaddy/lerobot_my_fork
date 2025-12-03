#!/usr/bin/env python
"""AgileX 机器人重置测试脚本。

该脚本连接到 AgileX 机器人，读取当前关节状态，
并将双臂重置到零位（home position）。

需要在 lerobot_v4 环境运行：
    conda activate lerobot_v4
"""

import time
import numpy as np
from lerobot.robots.agilex import AgileXRobot, AgileXConfig
from lerobot.robots.utils import make_robot_from_config

# 创建配置对象（使用 mock=False 连接真实机器人）
agilex_config = AgileXConfig(
    mock=False,  # 设为 True 可在无机器人时测试
    # ROS 配置（根据实际情况修改）
    ros_master_uri="http://localhost:11311",
    node_name="lerobot_agilex_test",
)

# 使用配置对象创建机器人实例
robot = make_robot_from_config(agilex_config)

def reset_to_zero_position(robot, speed=0.5, tolerance=0.05):
    """将机器人双臂重置到零位。

    Args:
        robot: AgileX 机器人实例
        speed: 运动速度系数 (0-1)
        tolerance: 位置误差容忍度（弧度）

    Returns:
        是否成功重置
    """
    from lerobot.robots.agilex.config_agilex import JOINT_NAMES

    print("\n开始重置机器人到零位...")

    # 创建零位动作（所有关节位置为0）
    zero_action = {}
    for side in ["left", "right"]:
        for joint_name in JOINT_NAMES:
            zero_action[f"{side}_{joint_name}.pos"] = 0.0

    # 发送零位命令
    print("发送零位命令...")
    sent_action = robot.send_action(zero_action)

    # 等待机器人到达目标位置
    print("等待机器人运动到零位...")
    max_wait_time = 10.0  # 最长等待10秒
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        # 获取当前状态
        obs = robot.get_observation()

        # 检查是否接近零位
        max_error = 0.0
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                current_pos = obs[f"{side}_{joint_name}.pos"]
                error = abs(current_pos - 0.0)
                max_error = max(max_error, error)

        print(f"  当前最大误差: {max_error:.4f} 弧度", end="\r")

        if max_error < tolerance:
            print(f"\n重置完成！最终误差: {max_error:.4f} 弧度")
            return True

        time.sleep(0.1)

    print(f"\n警告：未在 {max_wait_time} 秒内到达零位，当前最大误差: {max_error:.4f} 弧度")
    return False

def get_joint_positions(obs):
    """从观测中提取关节位置数组。"""
    from lerobot.robots.agilex.config_agilex import JOINT_NAMES

    positions = []
    for side in ["left", "right"]:
        for joint_name in JOINT_NAMES:
            positions.append(obs[f"{side}_{joint_name}.pos"])
    return np.array(positions)

try:
    # 连接机器人
    print("=" * 50)
    print("正在连接 AgileX 机器人...")
    print("=" * 50)
    robot.connect()
    print("✓ 连接成功!\n")

    # 获取初始观测
    print("获取初始关节状态...")
    initial_obs = robot.get_observation()
    initial_joints = get_joint_positions(initial_obs)
    print(f"初始关节角度:")
    for i, pos in enumerate(initial_joints):
        print(f"  关节 {i:2d}: {pos:7.4f} 弧度 ({pos*180/np.pi:7.2f}°)")

    # 调用重置函数
    print("\n" + "=" * 50)
    success = reset_to_zero_position(robot, tolerance=0.05)
    print("=" * 50)

    # 获取重置后的观测
    print("\n获取重置后的关节状态...")
    reset_obs = robot.get_observation()
    reset_joints = get_joint_positions(reset_obs)
    print(f"重置后关节角度:")
    for i, pos in enumerate(reset_joints):
        print(f"  关节 {i:2d}: {pos:7.4f} 弧度 ({pos*180/np.pi:7.2f}°)")

    # 检查重置结果
    max_error = np.max(np.abs(reset_joints))
    print(f"\n统计信息:")
    print(f"  最大误差: {max_error:.4f} 弧度 ({max_error*180/np.pi:.2f}°)")
    print(f"  平均误差: {np.mean(np.abs(reset_joints)):.4f} 弧度")

    if success:
        print("\n✓ 验证成功: 机器人已成功重置到零位!")
    else:
        print("\n⚠ 验证警告: 部分关节可能未正确到达零位")

except Exception as e:
    print(f"\n✗ 发生错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 断开连接
    print("\n" + "=" * 50)
    print("断开机器人连接...")
    print("=" * 50)
    robot.disconnect()
    print("测试完成!")