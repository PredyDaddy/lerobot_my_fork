# Piper 双臂机器人录制与回放完整指南

## 1. 环境准备与SDK安装

### 1.1 系统要求
- **操作系统**: Ubuntu 18.04/20.04/22.04
- **Python版本**: 3.6+
- **硬件**: USB-CAN转换模块（波特率固定1MHz）
- **机械臂**: Agilex Piper 系列（固件版本 V1.5-2 或更高）

### 1.2 安装依赖包

```bash
# 安装系统工具
sudo apt update
sudo apt install can-utils ethtool iproute2 -y

# 安装Python依赖
pip3 install python-can>=3.3.4
```

### 1.3 安装Piper SDK（三选一）

**推荐方式：从PyPI安装**
```bash
pip3 install piper_sdk
```

**开发方式：从源码安装**
```bash
cd /home/agilex/cqy/lerobot_dev/lerobot_4_2_try/lerobot_my_fork/piper_sdk
pip3 install -e .
```

**离线方式：安装whl文件**
```bash
pip3 install piper_sdk-X.X.X-py3-none-any.whl
```

### 1.4 验证安装
```bash
pip3 show piper_sdk
```

---

## 2. CAN总线配置（关键步骤）

### 2.1 查找CAN设备

```bash
bash find_all_can_port.sh
```

**输出示例：**
```
Both ethtool and can-utils are installed.
Interface can0 is connected to USB port 3-1.4:1.0
Interface can1 is connected to USB port 3-1.1:1.0
```

**注意**：记录每个CAN设备对应的USB端口地址（如 `3-1.4:1.0`）。

### 2.2 配置双臂CAN接口

**重要**：两主两从模式需要同时激活两个CAN接口

```bash
# 编辑多设备激活脚本
nano can_muti_activate.sh
```

**修改内容：**
```bash
# 在文件中找到 USB_PORTS 部分，修改为：
USB_PORTS["3-1.4:1.0"]="can_left:1000000"   # 左臂CAN接口
USB_PORTS["3-1.1:1.0"]="can_right:1000000"  # 右臂CAN接口
```

**执行激活：**
```bash
bash can_muti_activate.sh
```

**验证配置：**
```bash
ifconfig | grep can
```

**应看到输出：**
```
can_left: flags=193<UP,RUNNING,NOARP>  mtu 16
can_right: flags=193<UP,RUNNING,NOARP>  mtu 16
```

---

## 3. 两主两从臂配置

### 3.1 硬件连接说明

```
主臂1 (Master1) ── CAN_H/L ──┐
                             │
主臂2 (Master2) ── CAN_H/L ──┼── 控制盒 ── USB ── PC
                             │
从臂1 (Slave1)  ── CAN_H/L ──┤
                             │
从臂2 (Slave2)  ── CAN_H/L ──┘
```

**连接要点：**
- 两个主臂并联在同一CAN总线
- 两个从臂并联在另一CAN总线
- 两条CAN总线通过控制盒汇总到单个USB-CAN适配器

### 3.2 主从模式配置脚本

创建文件：`setup_master_slave.py`
```python
#!/usr/bin/env python3
from piper_sdk import *
import time

def setup_arm(piper, can_name, arm_name, is_master=True):
    """配置单臂为主臂或从臂"""
    print(f"配置 {arm_name} ({can_name})...")

    # 创建接口实例
    piper_arm = C_PiperInterface_V2(
        can_name=can_name,
        judge_flag=False,
        can_auto_init=True,
        dh_is_offset=1,  # V1.6-3后固件版本
        logger_level=LogLevel.WARNING
    )

    # 连接CAN端口
    piper_arm.ConnectPort()
    time.sleep(0.1)

    # 配置主从模式
    mode_code = 0xFA if is_master else 0xFC  # 0xFA=主臂, 0xFC=从臂
    piper_arm.MasterSlaveConfig(mode_code, 0, 0, 0)

    mode_str = "主臂" if is_master else "从臂"
    print(f"✓ {arm_name} 设置为{mode_str}模式")

    return piper_arm

if __name__ == "__main__":
    print("=== 配置两主两从臂模式 ===\n")

    # 配置左主臂
    master1 = setup_arm(None, "can_left", "左主臂", is_master=True)

    # 配置右主臂
    master2 = setup_arm(None, "can_right", "右主臂", is_master=True)

    print("\n=== 主从配置完成 ===")
    print("⚠️  重要：请先开启从臂电源，然后开启主臂电源")
```

**执行配置：**
```bash
python3 setup_master_slave.py
```

---

## 4. 录制双臂关节角度

### 4.1 录制脚本

创建文件：`record_dual_arms.py`
```python
#!/usr/bin/env python3
from piper_sdk import *
import time
import json
import datetime
import os

class DualArmRecorder:
    def __init__(self, left_can="can_left", right_can="can_right"):
        """初始化双臂录制器"""
        self.left_arm = C_PiperInterface_V2(
            can_name=left_can,
            judge_flag=False,
            dh_is_offset=1,
            logger_level=LogLevel.WARNING
        )
        self.right_arm = C_PiperInterface_V2(
            can_name=right_can,
            judge_flag=False,
            dh_is_offset=1,
            logger_level=LogLevel.WARNING
        )

        self.is_recording = False
        self.recorded_data = []

    def connect_arms(self):
        """连接双臂"""
        print("连接左臂CAN接口...")
        self.left_arm.ConnectPort()
        time.sleep(0.1)

        print("连接右臂CAN接口...")
        self.right_arm.ConnectPort()
        time.sleep(0.1)

        # 确认主臂模式
        print("\n确认主臂模式...")
        self.left_arm.MasterSlaveConfig(0xFA, 0, 0, 0)
        self.right_arm.MasterSlaveConfig(0xFA, 0, 0, 0)

        print("✓ 双臂连接成功！")

    def start_recording(self):
        """开始录制数据"""
        print("\n开始录制... 按 Ctrl+C 停止")
        print("请操作主臂进行遥操作演示")

        self.is_recording = True
        self.recorded_data = []
        self.start_time = time.time()

        try:
            while self.is_recording:
                # 读取主臂控制指令（单位：0.001度）
                left_ctrl = self.left_arm.GetArmJointCtrl()
                right_ctrl = self.right_arm.GetArmJointCtrl()

                # 构建数据帧（转换为度）
                frame = {
                    "timestamp": round(time.time() - self.start_time, 3),
                    "left_arm": {
                        "joint_1": round(left_ctrl.joint_ctrl.joint_1 * 0.001, 3),
                        "joint_2": round(left_ctrl.joint_ctrl.joint_2 * 0.001, 3),
                        "joint_3": round(left_ctrl.joint_ctrl.joint_3 * 0.001, 3),
                        "joint_4": round(left_ctrl.joint_ctrl.joint_4 * 0.001, 3),
                        "joint_5": round(left_ctrl.joint_ctrl.joint_5 * 0.001, 3),
                        "joint_6": round(left_ctrl.joint_ctrl.joint_6 * 0.001, 3),
                    },
                    "right_arm": {
                        "joint_1": round(right_ctrl.joint_ctrl.joint_1 * 0.001, 3),
                        "joint_2": round(right_ctrl.joint_ctrl.joint_2 * 0.001, 3),
                        "joint_3": round(right_ctrl.joint_ctrl.joint_3 * 0.001, 3),
                        "joint_4": round(right_ctrl.joint_ctrl.joint_4 * 0.001, 3),
                        "joint_5": round(right_ctrl.joint_ctrl.joint_5 * 0.001, 3),
                        "joint_6": round(right_ctrl.joint_ctrl.joint_6 * 0.001, 3),
                    }
                }

                self.recorded_data.append(frame)

                # 实时状态显示
                print(f"\r录制中... 帧数: {len(self.recorded_data)} | "
                      f"时间: {frame['timestamp']:.1f}秒", end="")

                # 控制采样频率（约50Hz）
                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n\n录制已停止")
            self.stop_recording()

    def stop_recording(self):
        """停止录制"""
        self.is_recording = False

    def save_to_json(self, filename=None):
        """保存录制数据到JSON文件"""
        if not self.recorded_data:
            print("⚠️  没有录制到数据！")
            return None

        # 生成文件名
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dual_arm_record_{timestamp}.json"

        # 创建目录
        os.makedirs("recordings", exist_ok=True)
        filepath = os.path.join("recordings", filename)

        # 构建元数据
        record_info = {
            "metadata": {
                "version": "1.0",
                "date": datetime.datetime.now().isoformat(),
                "duration": self.recorded_data[-1]["timestamp"],
                "total_frames": len(self.recorded_data),
                "frequency": "50Hz",
                "arm_configuration": "dual_master",
                "units": {
                    "joint_angle": "degrees",
                    "timestamp": "seconds"
                }
            },
            "data": self.recorded_data
        }

        # 保存文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record_info, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 录制数据已保存到: {filepath}")
        print(f"  - 总帧数: {len(self.recorded_data)}")
        print(f"  - 时长: {record_info['metadata']['duration']:.2f} 秒")

        return filepath

if __name__ == "__main__":
    print("=== 双臂遥操作录制工具 ===\n")

    # 创建录制器实例
    recorder = DualArmRecorder(left_can="can_left", right_can="can_right")

    # 连接双臂
    print("正在连接双臂...")
    recorder.connect_arms()

    # 等待用户准备
    print("\n准备就绪！")
    input("请操作主臂，准备好后按回车开始录制...")

    # 开始录制
    recorder.start_recording()

    # 保存数据
    filepath = recorder.save_to_json()

    if filepath:
        print(f"\n✅ 录制完成！数据文件: {filepath}")
    else:
        print("\n❌ 录制失败！")
```

### 4.2 录制操作步骤

```bash
# 1. 进入工作目录
cd /your/workspace

# 2. 确保CAN接口已激活
ifconfig | grep can

# 3. 执行录制
python3 record_dual_arms.py

# 4. 按提示操作主臂进行演示

# 5. 按 Ctrl+C 停止录制

# 6. 数据自动保存到 recordings/ 目录
```

**录制输出示例：**
```
=== 双臂遥操作录制工具 ===

连接左臂CAN接口...
连接右臂CAN接口...

确认主臂模式...
✓ 双臂连接成功！

准备就绪！
请操作主臂，准备好后按回车开始录制...
```

---

## 5. 回放录制数据

### 5.1 简易回放（不拔主臂）

创建文件：`playback_simple.py`
```python
#!/usr/bin/env python3
from piper_sdk import *
import time
import json

class DualArmSimplePlayer:
    def __init__(self, playback_file, left_can="can_left", right_can="can_right"):
        """初始化简易播放器"""
        self.playback_file = playback_file

        # 加载录制数据
        with open(playback_file, "r", encoding="utf-8") as f:
            self.record_data = json.load(f)

        self.data_frames = self.record_data["data"]
        self.metadata = self.record_data["metadata"]

        # 初始化从臂（注意：主臂不拔）
        self.left_slave = C_PiperInterface_V2(
            can_name=left_can,
            judge_flag=False,
            dh_is_offset=1,
            logger_level=LogLevel.WARNING
        )

        self.right_slave = C_PiperInterface_V2(
            can_name=right_can,
            judge_flag=False,
            dh_is_offset=1,
            logger_level=LogLevel.WARNING
        )

        print(f"加载录制文件: {playback_file}")
        print(f"总帧数: {self.metadata['total_frames']}")
        print(f"时长: {self.metadata['duration']:.2f} 秒")

    def connect_slave_arms(self):
        """连接从臂"""
        print("\n连接左从臂...")
        self.left_slave.ConnectPort()
        time.sleep(0.1)

        print("连接右从臂...")
        self.right_slave.ConnectPort()
        time.sleep(0.1)

        # 配置为从臂模式
        print("配置从臂模式...")
        self.left_slave.MasterSlaveConfig(0xFC, 0, 0, 0)
        self.right_slave.MasterSlaveConfig(0xFC, 0, 0, 0)

        print("✓ 从臂连接成功！")

    def play(self, speed=1.0):
        """播放录制数据

        Args:
            speed: 播放速度倍数（0.5=半速，2.0=双倍速）
        """
        print(f"\n开始回放... 速度: {speed}x")
        print("⚠️  重要：回放时主臂控制指令会被从臂忽略")
        print("按 Ctrl+C 停止")

        try:
            # 启动从臂
            print("启动从臂电机...")
            self.left_slave.EnablePiper()
            self.right_slave.EnablePiper()
            time.sleep(0.5)

            # 设置为高跟随模式
            print("设置高跟随模式...")
            self.left_slave.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
            self.right_slave.MotionCtrl_2(0x01, 0x01, 100, 0xAD)

            # 开始回放
            print("\n开始执行动作...")
            start_time = time.time()

            for i, frame in enumerate(self.data_frames):
                # 计算期望时间
                expected_time = frame["timestamp"] / speed
                elapsed_time = time.time() - start_time

                # 等待到期望时间点
                while elapsed_time < expected_time:
                    time.sleep(0.001)
                    elapsed_time = time.time() - start_time

                # 发送左从臂控制指令
                left_joints = frame["left_arm"]
                self.left_slave.JointCtrl(
                    int(left_joints["joint_1"] * 1000),
                    int(left_joints["joint_2"] * 1000),
                    int(left_joints["joint_3"] * 1000),
                    int(left_joints["joint_4"] * 1000),
                    int(left_joints["joint_5"] * 1000),
                    int(left_joints["joint_6"] * 1000)
                )

                # 发送右从臂控制指令
                right_joints = frame["right_arm"]
                self.right_slave.JointCtrl(
                    int(right_joints["joint_1"] * 1000),
                    int(right_joints["joint_2"] * 1000),
                    int(right_joints["joint_3"] * 1000),
                    int(right_joints["joint_4"] * 1000),
                    int(right_joints["joint_5"] * 1000),
                    int(right_joints["joint_6"] * 1000)
                )

                # 显示进度
                progress = (i + 1) / len(self.data_frames) * 100
                current_time = elapsed_time * speed
                print(f"\r回放进度: {progress:.1f}% | "
                      f"时间: {current_time:.1f}/{self.metadata['duration']:.1f}秒", end="")

            print("\n\n✅ 回放完成！")

        except KeyboardInterrupt:
            print("\n\n停止回放")
            self.stop()
        except Exception as e:
            print(f"\n回放错误: {e}")
            self.stop()

    def stop(self):
        """停止回放"""
        print("停止从臂...")
        self.left_slave.DisableArm()
        self.right_slave.DisableArm()

if __name__ == "__main__":
    # 使用示例
    playback_file = "recordings/dual_arm_record_20241204_143022.json"

    print("=== 双臂回放工具（简易模式）===\n")

    try:
        # 创建播放器
        player = DualArmSimplePlayer(
            playback_file=playback_file,
            left_can="can_left",
            right_can="can_right"
        )

        # 连接从臂（主臂保持连接）
        player.connect_slave_arms()

        # 等待用户准备
        print("\n准备就绪！")
        input("按回车开始回放（无需拔掉主臂）...")

        # 设置回放速度
        speed = 1.0  # 正常速度

        # 开始回放
        player.play(speed=speed)

    except FileNotFoundError:
        print(f"❌ 错误：找不到录制文件 '{playback_file}'")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
```

### 5.2 使用说明

**前置条件**：
1. 主臂已上电（无需拔线）
2. 从臂已上电并设置为从臂模式
3. 两条CAN线路连接正常

**运行回放：**

```bash
# 回放最近一次录制的数据
python3 playback_simple.py

# 回放指定文件
playback_file="recordings/dual_arm_record_20241204_143022.json"
python3 playback_simple.py
```

**控制回放速度：**
```python
# 在脚本中修改 speed 参数
player.play(speed=1.0)  # 正常速度
player.play(speed=0.5)  # 半速播放
player.play(speed=2.0)  # 双倍速播放
```

---

## 6. 高级回放模式（拔主臂）

### 6.1 完整回放模式

如果需要完全断开主臂，使用标准回放模式：

**操作步骤：**
1. 关闭主臂电源
2. 拔掉主臂CAN连接线
3. 执行回放脚本（同上）

### 6.2 回放安全注意事项

**重要安全提醒：**
1. **碰撞检测**：回放前确保工作空间无障碍物
2. **速度限制**：初次回放建议使用0.5倍速
3. **紧急停止**：随时按Ctrl+C停止回放
4. **从臂状态**：确保从臂已正确配置为从臂模式（0xFC）
5. **主臂影响**：不拔主臂时，主臂指令被从臂忽略，不会冲突

---

## 7. JSON数据格式说明

### 7.1 录制文件结构

```json
{
  "metadata": {
    "version": "1.0",
    "date": "2024-12-04T14:30:22.123456",
    "duration": 45.678,
    "total_frames": 2284,
    "frequency": "50Hz",
    "arm_configuration": "dual_master",
    "units": {
      "joint_angle": "degrees",
      "timestamp": "seconds"
    }
  },
  "data": [
    {
      "timestamp": 0.0,
      "left_arm": {
        "joint_1": 0.0,
        "joint_2": 15.5,
        "joint_3": -30.2,
        "joint_4": 45.1,
        "joint_5": -10.8,
        "joint_6": 0.5
      },
      "right_arm": {
        "joint_1": 2.1,
        "joint_2": 14.8,
        "joint_3": -29.5,
        "joint_4": 44.3,
        "joint_5": -11.2,
        "joint_6": 0.8
      }
    }
  ]
}
```

### 7.2 数据字段说明

| 字段 | 类型 | 单位 | 描述 |
|------|------|------|------|
| timestamp | float | 秒 | 从录制开始的时间戳 |
| joint_1-6 | float | 度 | 各关节角度（J1-J6） |

**关节角度范围：**
- J1: [-150°, 150°]
- J2: [0°, 180°]
- J3: [-170°, 0°]
- J4: [-100°, 100°]
- J5: [-70°, 70°]
- J6: [-120°, 120°]

---

## 8. 故障排查

### 8.1 常见问题

**问题1：找不到CAN设备**
```bash
# 检查USB连接
lsusb

# 重新激活CAN
bash can_muti_activate.sh

# 查看CAN接口
ifconfig -a | grep can
```

**问题2：无法读取关节数据**
- 确认机械臂为主臂模式（0xFA）
- 检查CAN总线连接
- 确认电源已开启

**问题3：回放时从臂不动作**
- 检查从臂是否为从臂模式（0xFC）
- 确认高跟随模式已设置（0xAD）
- 验证从臂电机已使能

**问题4：录制数据为空**
- 确认主臂在运动中
- 检查CAN接口是否正确（can_left/can_right）
- 确认SDK版本支持主从模式

### 8.2 调试技巧

**查看实时CAN数据：**
```bash
candump can_left
candump can_right
```

**检查主从模式配置：**
```python
from piper_sdk import *
piper = C_PiperInterface_V2("can_left")
piper.ConnectPort()

# 读取当前模式（需要固件支持）
status = piper.GetArmStatus()
print(status)
```

---

## 9. 安全警告

⚠️ **重要安全事项：**

1. **MIT协议警告**：单关节MIT控制为高级功能，使用不当可能导致机械臂损坏
2. **碰撞风险**：回放前确保工作空间清空障碍物
3. **速度限制**：初次使用建议使用较低速度（0.5倍速）
4. **紧急停止**：保持急停按钮可触及范围内
5. **人员安全**：回放时保持安全距离，不要站在机械臂运动范围内

### 9.1 使用建议

1. 先在空载状态下测试录制回放功能
2. 逐步增加回放速度和负载
3. 定期检查机械臂状态和关节限位
4. 重要操作前备份录制数据

---

## 10. 扩展功能

### 10.1 添加夹爪控制

修改录制脚本，添加夹爪数据：
```python
# 在录制循环中添加
left_gripper = self.left_arm.GetArmGripperCtrl()
right_gripper = self.right_arm.GetArmGripperCtrl()

# 在数据帧中添加
"left_gripper": {
    "position": left_gripper.gripper_ctrl.position * 1e-6,  # 转换为米
    "force": left_gripper.gripper_ctrl.force
}
```

### 10.2 力矩数据录制

如需录制关节力矩，使用：
```python
joint_torque = piper.GetArmJointMsgs()
torque_1 = joint_torque.joint_state.torque_1  # 单位：mN·m
```

### 10.3 末端位姿录制

录制笛卡尔空间位姿：
```python
end_pose = piper.GetArmEndPoseMsgs()
position = {
    "x": end_pose.end_pose.x * 0.001,  # mm to m
    "y": end_pose.end_pose.y * 0.001,
    "z": end_pose.end_pose.z * 0.001,
    "roll": end_pose.end_pose.roll * 0.001,  # 0.001deg to deg
    "pitch": end_pose.end_pose.pitch * 0.001,
    "yaw": end_pose.end_pose.yaw * 0.001
}
```

---

## 11. 相关资源

- **SDK文档**：`/home/agilex/cqy/lerobot_dev/lerobot_4_2_try/lerobot_my_fork/piper_sdk/README.MD`
- **接口文档**：`/home/agilex/cqy/lerobot_dev/lerobot_4_2_try/lerobot_my_fork/piper_sdk/asserts/V2/INTERFACE_V2.MD`
- **双臂配置**：`/home/agilex/cqy/lerobot_dev/lerobot_4_2_try/lerobot_my_fork/piper_sdk/asserts/double_piper.MD`
- **Demo源码**：`/home/agilex/cqy/lerobot_dev/lerobot_4_2_try/lerobot_my_fork/piper_sdk/piper_sdk/demo/V2/`

---

## 12. 技术支持

- **GitHub Issues**: https://github.com/agilexrobotics/piper_sdk/issues
- **Discord社区**: https://discord.gg/wrKYTxwDBd
- **邮箱**: support@agilex.ai

**版本信息**
- SDK版本: 0.6.1
- 支持固件: V1.5-2 及更高版本
- 最后更新: 2024-12-04
