"""启动脚本：启动 Streamlit"""

import sys
import os
import time

# 直接启动 Streamlit
if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "web_app.py", "--server.port=8501"])
