"""
MatAgent 服务进程
提供 HTTP API 供多个 Streamlit 客户端共享同一个 Agent 实例
"""

import os
import json
import sys
import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入 Agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.langchain_agent import create_langchain_agent

app = FastAPI(title="MatAgent API", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 Agent 实例
_agent = None
_agent_lock = False

# 会话管理
_sessions: Dict[str, Dict[str, Any]] = {}

# 线程池执行器
_executor = ThreadPoolExecutor(max_workers=4)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: Optional[list] = None


class ChatResponse(BaseModel):
    type: str
    message: str
    tool_results: Optional[list] = None
    duration: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    active_sessions: int


def get_agent():
    """获取或创建全局 Agent 实例"""
    global _agent, _agent_lock
    
    if _agent is not None:
        return _agent
    
    if _agent_lock:
        raise HTTPException(status_code=503, detail="Agent 正在初始化中")
    
    try:
        _agent_lock = True
        _agent = create_langchain_agent()
        _agent_lock = False
        return _agent
    except Exception as e:
        _agent_lock = False
        raise HTTPException(status_code=500, detail=f"Agent 初始化失败: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化 Agent"""
    try:
        get_agent()
        print("✅ Agent 服务已启动")
    except Exception as e:
        print(f"⚠️ Agent 初始化失败: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        agent_ready=_agent is not None,
        active_sessions=len(_sessions)
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    agent = get_agent()
    
    try:
        # 恢复会话历史
        if request.history:
            from langchain_core.messages import HumanMessage, AIMessage
            agent._message_history = []
            for msg in request.history:
                if msg["role"] == "user":
                    agent._message_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    agent._message_history.append(AIMessage(content=msg["content"]))
        
        # 在线程池中执行同步的 agent.chat() 调用，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, agent.chat, request.message)
        
        # 打印工具调用日志
        if result.get("tool_results"):
            print(f"\n🔧 [{datetime.now().strftime('%H:%M:%S')}] 会话 {request.session_id[:8]}... 调用了 {len(result['tool_results'])} 个工具:")
            for i, tr in enumerate(result["tool_results"], 1):
                print(f"  {i}. 工具: {tr.get('tool_name', 'unknown')}")
                print(f"     参数: {json.dumps(tr.get('tool_args', {}), ensure_ascii=False)}")
                result_preview = str(tr.get('result', ''))[:200]
                if len(str(tr.get('result', ''))) > 200:
                    result_preview += "..."
                print(f"     结果: {result_preview}")
            print()
        
        # 保存会话状态
        _sessions[request.session_id] = {
            "last_active": datetime.now().isoformat(),
            "message_count": len(agent._message_history)
        }
        
        return ChatResponse(
            type=result.get("type", "text"),
            message=result.get("message", ""),
            tool_results=result.get("tool_results"),
            duration=result.get("duration")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_bandgap")
async def predict_bandgap(formula: str):
    """预测带隙"""
    agent = get_agent()
    try:
        result = agent.predict_band_gap(formula)
        return {"formula": formula, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """列出活跃会话"""
    return {
        "sessions": [
            {"session_id": sid, **info}
            for sid, info in _sessions.items()
        ]
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """清除会话状态"""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared"}


# ========== VASP 任务 API ==========

@app.get("/vasp/task_directories")
async def list_task_directories():
    """列出任务目录"""
    agent = get_agent()
    try:
        result = agent.list_task_directories()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vasp/squeue")
async def check_squeue():
    """检查 Slurm 队列"""
    agent = get_agent()
    try:
        result = agent.check_squeue()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/create_task")
async def create_task(formula: str, cif_path: str):
    """创建任务目录"""
    agent = get_agent()
    try:
        result = agent.create_task(formula, cif_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/create_mission")
async def create_mission(task_directory: str, mission_type: str):
    """生成任务输入文件"""
    agent = get_agent()
    try:
        result = agent.create_mission(task_directory, mission_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/modify_incar")
async def modify_incar(task_directory: str, mission: str, key: str, value: str):
    """修改 INCAR 参数"""
    agent = get_agent()
    try:
        result = agent.modify_incar(task_directory, mission, key, value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/submit")
async def submit_mission(task_directory: str, mission: str):
    """提交任务"""
    agent = get_agent()
    try:
        result = agent.submit_mission(task_directory, mission)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/extract")
async def extract_result(task_directory: str, mission: str, plot: bool = False):
    """提取结果"""
    agent = get_agent()
    try:
        result = agent.extract_result(task_directory, mission, plot)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
