"""启动脚本：启动 Streamlit 和 Agent 服务"""

import sys
import os
import subprocess
import time

if __name__ == "__main__":
    # 启动 Agent 服务（后台）
    agent_proc = subprocess.Popen(
        [sys.executable, "agent_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"✅ Agent 服务已启动 (PID: {agent_proc.pid})")

    # 等待 Agent 服务就绪
    time.sleep(2)

    # 启动 Streamlit（前台，阻塞）
    print("🚀 启动 Streamlit...")
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "web_app.py", "--server.port=8501"])
