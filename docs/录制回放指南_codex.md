# Piper 双臂主从模式录制/回放指南

本文档基于仓库内的 `piper_sdk/` 提供的 `C_PiperInterface_V2` 接口（见 `piper_sdk/piper_sdk/interface/piper_interface_v2.py`），讲述如何在两主两从臂同时插在 CAN 总线、处于主从联动模式时录制双臂关节角度到 JSON，并在离线状态下回放。流程拆分为**环境安装 → 硬件准备与模式切换 → 录制脚本编写 → 回放脚本编写 → 操作注意事项**五部分。

---

## 1. 环境准备与 SDK 安装

1. **Python 版本**：建议 Python ≥ 3.8，对应仓库默认开发环境。
2. **依赖安装**  
   ```bash
   # 建议在虚拟环境中安装
   pip install -e .[dev,test]  # 仓库整体依赖（含 numpy/pytest 等）

   # Piper SDK 运行时依赖
   pip install python-can>=4.3.1
   pip install ./piper_sdk  # 或者 pip install piper-sdk==0.6.1
   ```
   - 仓库内 `piper_sdk/piper_sdk/__init__.py` 会把硬件端口、协议解析、正解算法全部导出，安装后可直接 `from piper_sdk import C_PiperInterface_V2`。
3. **CAN 配置**  
   Piper 臂默认使用 `socketcan`，波特率 1 Mbps。可以手动配置，也可以使用 SDK 自带脚本（位于 `piper_sdk/piper_sdk/can_config.sh` 等）。
   ```bash
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set can0 up
   sudo ip link set can1 type can bitrate 1000000
   sudo ip link set can1 up
   ```
   - 录制阶段需要两主两从四条 CAN 线，可以在 `/etc/network/interfaces` 中写入持久化配置，确保开机即为 `UP` 状态。

---

## 2. 硬件连接与主从模式

### 2.1 主从模式回顾

- SDK 的 `C_PiperInterface_V2.MasterSlaveConfig()`（`piper_sdk/piper_sdk/interface/piper_interface_v2.py:2944-2997`）通过 CAN 指令 `0x470` 切换“示教输入臂/运动输出臂”以及反馈、控制报文的 ID 偏移量。
- 官方 demo `piper_sdk/piper_sdk/demo/V2/piper_set_master.py`/`piper_set_slave.py` 展示了两种典型配置：
  - `MasterSlaveConfig(0xFA, 0, 0, 0)`：设置为示教输入臂（主臂）。
  - `MasterSlaveConfig(0xFC, 0, 0, 0)`：设置为运动输出臂（从臂）。
- README (`piper_sdk/piper_sdk/demo/V2/README.MD`) 特别强调：**当机械臂处于主臂模式时，发送“设为从臂”指令后需要重启机械臂才会生效**。因此，同一条 CAN 上同时插两主两从时，必须先按顺序上电：先插主臂（示教）再插从臂（运动），否则 CAN 报文 ID 映射会错乱。

### 2.2 录制/回放时的连接策略

| 阶段 | CAN 连接 | 主从状态 | 说明 |
| ---- | -------- | -------- | ---- |
| 录制 | 主臂 + 从臂都插在各自 CAN 口 | 维持现有联动配置 | 录制就是抓取主臂发送给从臂的目标以及从臂反馈。无需拔线。 |
| 回放（只靠 PC 驱动） | **拔掉主臂**，仅 PC → 从臂 | 将目标从臂通过 `MasterSlaveConfig(0xFC, 0, 0, 0)` 切换为“运动输出臂”，重启硬件 | 如果主臂仍在，`PiperSDKInterface` 无法抢占 CAN 控制权，会一直 Enable 失败。 |
| 回放（仍需示教） | 主臂插着 | 需要对 SDK 进行二次开发，处理偏移后的 CAN ID | 仓库内的 `PiperSDKInterface` 尚未实现此模式，本文档默认回放时断开主臂。 |

---

## 3. 录制脚本设计

### 3.1 思路

1. 每个从臂接一个 `C_PiperInterface_V2` 实例（如 `can_left`、`can_right`）。
2. 在主从联动模式下，主臂负责发布目标，CAN 总线上可以直接读取：
   - `GetArmJointMsgs()` → 从臂反馈（0.001°）。
   - `GetArmJointCtrl()` → 主臂发送给从臂的目标（若已启用联动 ID）。
3. 将上述数据按时间戳写入 JSON 行或数组，结构建议：
   ```json
   {
     "timestamp": 1700000000.123,
     "left_joint_deg": [...],
     "right_joint_deg": [...],
     "left_ctrl_deg": [...],
     "right_ctrl_deg": [...]
   }
   ```
4. 录制停止时保存到 `records/<session>.json`.

### 3.2 示例骨架

```python
#!/usr/bin/env python3
import json, time
from pathlib import Path
from piper_sdk import C_PiperInterface_V2

def create_interface(port: str) -> C_PiperInterface_V2:
    piper = C_PiperInterface_V2(port)
    piper.ConnectPort(start_thread=True, piper_init=False)  # 录制只读状态，可关闭初始化动作
    return piper

def read_deg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    return [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]

def read_ctrl(piper: C_PiperInterface_V2) -> list[int]:
    # 只有主从联动时才有值，否则保持 0
    ctrl = piper.GetArmJointCtrl().joint_ctrl
    return [ctrl.joint_1, ctrl.joint_2, ctrl.joint_3, ctrl.joint_4, ctrl.joint_5, ctrl.joint_6]

def main():
    left = create_interface("can_left")
    right = create_interface("can_right")
    try:
        records = []
        start = time.time()
        while time.time() - start < 30:  # 录 30 秒
            ts = time.time()
            records.append({
                "timestamp": ts,
                "left_joint": read_deg(left),
                "right_joint": read_deg(right),
                "left_ctrl": read_ctrl(left),
                "right_ctrl": read_ctrl(right),
            })
            time.sleep(0.01)  # 100 Hz
    finally:
        left.DisconnectPort()
        right.DisconnectPort()

    out = Path("records")
    out.mkdir(exist_ok=True)
    with (out / f"session_{int(start)}.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

if __name__ == "__main__":
    main()
```

> **注意**：录制阶段只读状态，无需调用 `EnablePiper()`，否则会尝试抢占控制权导致主从联动失效。`ConnectPort(..., piper_init=False)` 即可。

---

## 4. 回放脚本设计

### 4.1 回放前置步骤

1. 拔掉示教主臂，或者让示教主臂彻底断电，确保 CAN 上仅剩被控制的双臂。
2. 将每个从臂切换至“运动输出臂”模式并重启（若之前一直是示教输入臂）：
   ```bash
   python piper_sdk/piper_sdk/demo/V2/piper_set_slave.py  # 对每条 CAN 分别执行
   # 断电重启机械臂
   ```
3. 打开 `send_zero_pose.py` 之类的脚本，确认 `PiperSDKInterface` 可以正常 `EnablePiper()`。

### 4.2 回放骨架

```python
#!/usr/bin/env python3
import json, time
from pathlib import Path
from piper_sdk import C_PiperInterface_V2

def enable(port: str) -> C_PiperInterface_V2:
    robot = C_PiperInterface_V2(port)
    robot.ConnectPort()
    start = time.time()
    while not robot.EnablePiper():
        if time.time() - start > 5:
            raise TimeoutError(f"{port} enable timeout")
        time.sleep(0.01)
    robot.MotionCtrl_2(0x01, 0x01, 50, 0x00)
    return robot

def to_sdk_units(deg_list: list[float]) -> list[int]:
    return [int(round(d * 1000.0)) for d in deg_list]

def playback(json_file: Path):
    data = json.loads(json_file.read_text())
    left = enable("can_left")
    right = enable("can_right")
    try:
        start_ts = data[0]["timestamp"]
        base_time = time.time()
        for frame in data:
            target_time = base_time + (frame["timestamp"] - start_ts)
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
            left.JointCtrl(*to_sdk_units(frame["left_joint"]))
            right.JointCtrl(*to_sdk_units(frame["right_joint"]))
            # 如需夹爪，可附带调用 GripperCtrl
    finally:
        left.DisablePiper()
        left.DisconnectPort()
        right.DisablePiper()
        right.DisconnectPort()

if __name__ == "__main__":
    playback(Path("records/session_1700000000.json"))
```

> **提示**：`C_PiperInterface_V2.JointCtrl()` 需传入 0.001° 单位整数。若录制时保存的是整数，可省略 `to_sdk_units`。回放前建议先通过 `agilex_script/send_zero_pose.py` 把臂回到零位，避免直接跳转导致碰撞。

---

## 5. 操作注意事项

1. **录制端**  
   - CAN 要求：主臂与从臂严格匹配，避免两个主臂接在同一总线上。
   - 脚本不要调用 `EnablePiper()`，保持监听状态即可。
   - 建议在录制数据中同时存储 `GetArmStatus()` 的 `ctrl_mode`、`arm_status` 字段，以便排查急停/奇异点。

2. **回放端**  
   - 必须断开示教主臂，否则 `PiperSDKInterface` 中的 `EnablePiper()`（`agilex_script/agilex_infer.py:34-64`）会因为主臂持续发送控制而失败。
   - 回放时若需要插回主臂，需要修改 SDK 支持 `MasterSlaveConfig` 的偏移 ID。仓库现有实现默认 CAN ID，不适用于“PC 与主臂同时在线”场景。
   - 在回放脚本中加入限位检查（可使用 `PiperSDKInterface.min_pos/max_pos`）以确保目标角度在安全范围内。

3. **数据格式**  
   - 建议保存为 JSON Lines 或者版本化的 JSON 文件，在头部写入元数据，例如采样频率、臂型号、主从配置。
   - 若后续要与 LeRobot 训练脚本打通，可转存为 `npz`/`hdf5` 再喂入数据集。

4. **故障排查**  
   - `EnablePiper()` 超时：检查主臂是否仍在、`MasterSlaveConfig` 是否生效、以及 CAN 是否 `UP`。
   - `GetArmJointCtrl()` 全为零：说明目前处于单臂控制或未启用联动，需要在触摸示教器上切换到主从模式。
   - `JointCtrl()` 报错：确认 `MotionCtrl_2()` 已设置为 `move_mode=0x01`（关节模式）且 `ctrl_mode=0x01`。

---

通过以上步骤即可实现“两主两从联动录制 → 拔掉主臂后回放”的完整流程。录制、回放脚本可以以 `piper_sdk` 为依赖，放在 `agilex_script/` 或自建 `tools/` 目录下，便于与仓库的其它调试脚本（如 `get_state.py`、`send_zero_pose.py`）一起管理。后续若要支持“主臂不断开即可回放”，需要在 `PiperSDKInterface` 内部增加对 `MasterSlaveConfig` 偏移 ID 的解析与发送，这超出了本文档范畴。***
