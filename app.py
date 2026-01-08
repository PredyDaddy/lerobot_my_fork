#!/usr/bin/env python3
"""
Agilex 机械臂 Web 控制界面

提供网页界面控制机械臂执行各种抓取任务。

用法:
    python app.py
    # 然后访问 http://localhost:5000
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)


# ============== 任务配置 ==============

class TaskType(Enum):
    LEFT_BOX = "left_box"
    RIGHT_BOX = "right_box"
    LEFT_BLACK_CUP = "left_black_cup"
    RIGHT_BLACK_CUP = "right_black_cup"
    LEFT_YELLOW_BOTTLE = "left_yellow_bottle"
    RIGHT_YELLOW_BOTTLE = "right_yellow_bottle"


@dataclass
class TaskConfig:
    name: str
    script: str
    checkpoint: str
    arm: str
    binary_gripper: bool
    fps: int = 30


TASKS = {
    TaskType.LEFT_BOX: TaskConfig(
        name="左边抓盒子",
        script="agilex_infer_single_cc_vertical.py",
        checkpoint="outputs/act_agilex_left_box/checkpoints/last/pretrained_model",
        arm="left",
        binary_gripper=True,
    ),
    TaskType.RIGHT_BOX: TaskConfig(
        name="右边抓盒子",
        script="agilex_infer_single_cc_vertical.py",
        checkpoint="outputs/act_agilex_right_box/checkpoints/last/pretrained_model",
        arm="right",
        binary_gripper=True,
    ),
    TaskType.LEFT_BLACK_CUP: TaskConfig(
        name="左边抓黑色杯子",
        script="agilex_infer_single_cc_vertical.py",
        checkpoint="outputs/act_agilex_left_black_cup/checkpoints/last/pretrained_model",
        arm="left",
        binary_gripper=False,
    ),
    TaskType.RIGHT_BLACK_CUP: TaskConfig(
        name="右边抓黑色杯子",
        script="agilex_infer_single_cc_vertical.py",
        checkpoint="outputs/act_agilex_right_black_cup/checkpoints/last/pretrained_model",
        arm="right",
        binary_gripper=False,
    ),
    TaskType.LEFT_YELLOW_BOTTLE: TaskConfig(
        name="左边抓黄色瓶",
        script="agilex_infer_single_cc_horizal.py",
        checkpoint="outputs/act_agilex_left_yellow_bottle/checkpoints/last/pretrained_model",
        arm="left",
        binary_gripper=True,
    ),
    TaskType.RIGHT_YELLOW_BOTTLE: TaskConfig(
        name="右边抓黄色瓶子",
        script="agilex_infer_single_cc_horizal.py",
        checkpoint="outputs/act_agilex_right_yellow_bottle/checkpoints/last/pretrained_model",
        arm="right",
        binary_gripper=True,
    ),
}


# ============== 全局状态 ==============

class RobotController:
    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.current_task: Optional[str] = None
        self.status: str = "idle"  # idle, running, stopping, returning_zero
        self.lock = threading.Lock()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def get_status(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "current_task": self.current_task,
            }

    def start_task(self, task_type: TaskType) -> dict:
        with self.lock:
            if self.status != "idle":
                return {"success": False, "message": f"机器人正忙: {self.status}"}

            self.status = "running"
            self.current_task = TASKS[task_type].name

        # 在后台线程中运行任务
        thread = threading.Thread(target=self._run_task, args=(task_type,))
        thread.daemon = True
        thread.start()

        return {"success": True, "message": f"已启动: {TASKS[task_type].name}"}

    def _run_task(self, task_type: TaskType):
        config = TASKS[task_type]

        # 构建命令
        cmd = [
            "python3",
            os.path.join(self.base_dir, config.script),
            "--checkpoint", os.path.join(self.base_dir, config.checkpoint),
            "--arm", config.arm,
            "--fps", str(config.fps),
        ]
        if config.binary_gripper:
            cmd.append("--binary-gripper")

        try:
            # 启动推理进程
            self.current_process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

            # 等待进程结束
            self.current_process.wait()

        except Exception as e:
            print(f"任务执行出错: {e}")

        finally:
            # 回零位
            self._return_to_zero()

            with self.lock:
                self.current_process = None
                self.current_task = None
                self.status = "idle"

    def _return_to_zero(self):
        with self.lock:
            self.status = "returning_zero"

        try:
            zero_cmd = ["python3", os.path.join(self.base_dir, "agilex_scripts/send_zero_to_follower.py")]
            subprocess.run(zero_cmd, cwd=self.base_dir, timeout=30)
        except Exception as e:
            print(f"回零位出错: {e}")

    def stop_task(self) -> dict:
        with self.lock:
            if self.status == "idle":
                return {"success": False, "message": "没有正在运行的任务"}

            if self.current_process is None:
                return {"success": False, "message": "进程不存在"}

            self.status = "stopping"

        try:
            # 发送 SIGINT 信号给进程组
            os.killpg(os.getpgid(self.current_process.pid), signal.SIGINT)
            return {"success": True, "message": "正在停止任务..."}
        except Exception as e:
            return {"success": False, "message": f"停止失败: {e}"}

    def return_to_zero_only(self) -> dict:
        with self.lock:
            if self.status != "idle":
                return {"success": False, "message": f"机器人正忙: {self.status}"}
            self.status = "returning_zero"
            self.current_task = "回零位"

        thread = threading.Thread(target=self._return_to_zero_thread)
        thread.daemon = True
        thread.start()

        return {"success": True, "message": "正在回零位..."}

    def _return_to_zero_thread(self):
        self._return_to_zero()
        with self.lock:
            self.current_task = None
            self.status = "idle"


controller = RobotController()


# ============== HTML 模板 ==============

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agilex 机械臂控制</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .status-bar {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.idle { background: #4ade80; }
        .status-dot.running { background: #fbbf24; animation: pulse 1s infinite; }
        .status-dot.stopping { background: #f87171; }
        .status-dot.returning_zero { background: #60a5fa; animation: pulse 1s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .task-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .task-btn {
            background: linear-gradient(145deg, #3b82f6, #2563eb);
            border: none;
            border-radius: 12px;
            padding: 20px;
            color: white;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }
        .task-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        .task-btn:active:not(:disabled) {
            transform: translateY(0);
        }
        .task-btn:disabled {
            background: #4b5563;
            cursor: not-allowed;
            box-shadow: none;
        }
        .task-btn.left { background: linear-gradient(145deg, #10b981, #059669); box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3); }
        .task-btn.left:hover:not(:disabled) { box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4); }
        .task-btn.right { background: linear-gradient(145deg, #8b5cf6, #7c3aed); box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3); }
        .task-btn.right:hover:not(:disabled) { box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4); }
        .control-btns {
            display: flex;
            gap: 15px;
        }
        .stop-btn {
            flex: 1;
            background: linear-gradient(145deg, #ef4444, #dc2626);
            border: none;
            border-radius: 12px;
            padding: 15px;
            color: white;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stop-btn:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        .stop-btn:disabled {
            background: #4b5563;
            cursor: not-allowed;
        }
        .zero-btn {
            flex: 1;
            background: linear-gradient(145deg, #6b7280, #4b5563);
            border: none;
            border-radius: 12px;
            padding: 15px;
            color: white;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .zero-btn:hover:not(:disabled) {
            transform: translateY(-2px);
        }
        .zero-btn:disabled {
            background: #374151;
            cursor: not-allowed;
        }
        .section-title {
            font-size: 0.9em;
            color: #9ca3af;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .task-section {
            margin-bottom: 25px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Agilex 机械臂控制</h1>

        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">空闲</span>
            </div>
            <span id="currentTask"></span>
        </div>

        <div class="task-section">
            <div class="section-title">📦 抓盒子</div>
            <div class="task-grid">
                <button class="task-btn left" onclick="startTask('left_box')">左臂抓盒子</button>
                <button class="task-btn right" onclick="startTask('right_box')">右臂抓盒子</button>
            </div>
        </div>

        <div class="task-section">
            <div class="section-title">☕ 抓黑色杯子</div>
            <div class="task-grid">
                <button class="task-btn left" onclick="startTask('left_black_cup')">左臂抓黑杯</button>
                <button class="task-btn right" onclick="startTask('right_black_cup')">右臂抓黑杯</button>
            </div>
        </div>

        <div class="task-section">
            <div class="section-title">🍾 抓黄色瓶子</div>
            <div class="task-grid">
                <button class="task-btn left" onclick="startTask('left_yellow_bottle')">左臂抓黄瓶</button>
                <button class="task-btn right" onclick="startTask('right_yellow_bottle')">右臂抓黄瓶</button>
            </div>
        </div>

        <div class="control-btns">
            <button class="stop-btn" id="stopBtn" onclick="stopTask()" disabled>⏹ 停止任务</button>
            <button class="zero-btn" id="zeroBtn" onclick="returnToZero()">🏠 回零位</button>
        </div>
    </div>

    <script>
        const statusMap = {
            'idle': { text: '空闲', class: 'idle' },
            'running': { text: '运行中', class: 'running' },
            'stopping': { text: '停止中', class: 'stopping' },
            'returning_zero': { text: '回零位中', class: 'returning_zero' }
        };

        function updateUI(data) {
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const currentTask = document.getElementById('currentTask');
            const stopBtn = document.getElementById('stopBtn');
            const zeroBtn = document.getElementById('zeroBtn');
            const taskBtns = document.querySelectorAll('.task-btn');

            const status = statusMap[data.status] || statusMap['idle'];
            statusDot.className = 'status-dot ' + status.class;
            statusText.textContent = status.text;
            currentTask.textContent = data.current_task || '';

            const isIdle = data.status === 'idle';
            stopBtn.disabled = isIdle;
            zeroBtn.disabled = !isIdle;
            taskBtns.forEach(btn => btn.disabled = !isIdle);
        }

        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateUI(data);
            } catch (e) {
                console.error('获取状态失败:', e);
            }
        }

        async function startTask(taskType) {
            try {
                const response = await fetch('/api/start/' + taskType, { method: 'POST' });
                const data = await response.json();
                if (!data.success) {
                    alert(data.message);
                }
                fetchStatus();
            } catch (e) {
                alert('启动失败: ' + e);
            }
        }

        async function stopTask() {
            try {
                const response = await fetch('/api/stop', { method: 'POST' });
                const data = await response.json();
                if (!data.success) {
                    alert(data.message);
                }
                fetchStatus();
            } catch (e) {
                alert('停止失败: ' + e);
            }
        }

        async function returnToZero() {
            try {
                const response = await fetch('/api/zero', { method: 'POST' });
                const data = await response.json();
                if (!data.success) {
                    alert(data.message);
                }
                fetchStatus();
            } catch (e) {
                alert('回零位失败: ' + e);
            }
        }

        // 定时刷新状态
        setInterval(fetchStatus, 1000);
        fetchStatus();
    </script>
</body>
</html>
"""


# ============== 路由 ==============

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/status")
def get_status():
    return jsonify(controller.get_status())


@app.route("/api/start/<task_type>", methods=["POST"])
def start_task(task_type: str):
    try:
        task = TaskType(task_type)
        return jsonify(controller.start_task(task))
    except ValueError:
        return jsonify({"success": False, "message": f"未知任务类型: {task_type}"})


@app.route("/api/stop", methods=["POST"])
def stop_task():
    return jsonify(controller.stop_task())


@app.route("/api/zero", methods=["POST"])
def return_to_zero():
    return jsonify(controller.return_to_zero_only())


if __name__ == "__main__":
    print("=" * 50)
    print("Agilex 机械臂 Web 控制界面")
    print("访问: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
