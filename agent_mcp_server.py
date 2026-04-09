"""
MatAgent MCP 版本服务进程
提供 HTTP API 供多个客户端共享同一个 MCP Agent 实例
使用 langchain_mcp_adapters 连接 MCP Server
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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入 MCP Agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.langchain_mcp_agent import create_agent as create_mcp_agent


# 全局 Agent 实例
_agent = None
_agent_lock = False

# 会话管理
_sessions: Dict[str, Dict[str, Any]] = {}

# 线程池执行器
_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化 Agent
    global _agent
    try:
        _agent = create_mcp_agent()
        # 在线程池中执行同步连接
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _agent.connect)
        print("✅ MCP Agent 服务已启动")
    except Exception as e:
        print(f"⚠️ Agent 初始化失败: {e}")
    
    yield
    
    # 关闭时清理资源
    if _agent:
        print("🔄 正在断开 MCP Server 连接...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _agent.disconnect)
        _agent = None
        print("✅ MCP Server 连接已断开")


app = FastAPI(title="MatAgent MCP API", version="2.0.0", lifespan=lifespan)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    mcp_server_connected: bool


def get_agent():
    """获取全局 MCP Agent 实例"""
    global _agent
    
    if _agent is not None:
        return _agent
    
    raise HTTPException(status_code=503, detail="Agent 尚未初始化完成，请稍后重试")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        agent_ready=_agent is not None,
        active_sessions=len(_sessions),
        mcp_server_connected=_agent is not None and _agent._connected
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    agent = get_agent()
    
    start_time = time.time()
    
    try:
        # 在线程池中执行同步的 agent.chat() 调用，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            _executor, 
            agent.chat, 
            request.message,
            request.session_id
        )
        
        duration = int((time.time() - start_time) * 1000)
        
        # 保存会话状态
        _sessions[request.session_id] = {
            "last_active": datetime.now().isoformat(),
        }
        
        return ChatResponse(
            type="text",
            message=message,
            duration=duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def list_tools():
    """列出所有可用工具"""
    agent = get_agent()
    try:
        tools_info = agent._async_agent.get_tools_info()
        return {"tools": tools_info}
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


# ========== 材料查询 API (通过 MCP) ==========

@app.get("/materials/search")
async def search_materials(
    elements: str = None,
    exclude_elements: str = None,
    formula: str = None,
    max_results: int = 10
):
    """搜索 Materials Project 材料 (通过 MCP)"""
    agent = get_agent()
    try:
        # 构建查询消息
        query_parts = []
        if elements:
            query_parts.append(f"包含 {elements}")
        if exclude_elements:
            query_parts.append(f"排除 {exclude_elements}")
        if formula:
            query_parts.append(f"化学式 {formula}")
        
        query = f"搜索材料：{', '.join(query_parts)}，最多返回 {max_results} 个"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, agent.chat, query)
        
        return {
            "query": query,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/materials/structure/{material_id}")
async def get_material_structure(material_id: str):
    """获取材料结构 (通过 MCP)"""
    agent = get_agent()
    try:
        query = f"获取材料 {material_id} 的结构信息"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, agent.chat, query)
        
        return {
            "material_id": material_id,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/time")
async def get_time():
    """获取当前时间 (测试 MCP 连接)"""
    agent = get_agent()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, 
            agent.chat, 
            "现在几点了？"
        )
        return {"time": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8766)
