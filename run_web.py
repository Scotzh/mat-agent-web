"""启动脚本：启动 Streamlit 和 Agent 服务"""

import sys
import os
import subprocess
import time
import requests
import signal

def kill_existing_agent():
    """终止已存在的 Agent 服务进程"""
    try:
        # 查找并终止占用 8765 端口的进程
        result = subprocess.run(
            ["fuser", "-k", "8765/tcp"],
            capture_output=True,
            text=True
        )
        time.sleep(1)
    except:
        pass

if __name__ == "__main__":
    # 先清理可能存在的旧进程
    print("🧹 清理旧进程...")
    kill_existing_agent()
    
    # 启动 Agent 服务（后台，输出到当前终端）
    print("🚀 启动 Agent 服务...")
    agent_proc = subprocess.Popen(
        [sys.executable, "agent_server.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )
    print(f"✅ Agent 服务进程已创建 (PID: {agent_proc.pid})")

    # 等待 Agent 服务就绪（轮询健康检查）
    print("⏳ 等待 Agent 服务就绪...")
    max_retries = 60
    agent_ready = False
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:8765/health", timeout=2)
            if response.status_code == 200 and response.json().get("agent_ready"):
                print("✅ Agent 服务已就绪")
                agent_ready = True
                break
        except Exception as e:
            # 检查进程是否还在运行
            if agent_proc.poll() is not None:
                print(f"❌ Agent 服务进程已退出 (返回码: {agent_proc.returncode})")
                sys.exit(1)
        time.sleep(1)
    
    if not agent_ready:
        print("⚠️ Agent 服务启动超时，但继续启动 Streamlit...")

    # 启动 Streamlit（前台，阻塞）
    print("🚀 启动 Streamlit...")
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "web_app.py", "--server.port=8501"])
