"""
MatAgent MCP 版本服务进程
提供 HTTP API 供多个客户端共享同一个 MCP Agent 实例
使用 langchain_mcp_adapters 连接 MCP Server
支持持久化会话记忆
"""

import os
import json
import sys
import time
import asyncio
import sqlite3
from typing import Dict, Any, Optional, List
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

# 数据库配置
DB_PATH = "matagent_server_history.db"

# ============ 数据库操作函数 ============

def _ensure_chat_table():
    """确保聊天记录表存在"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history(
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR,
                role VARCHAR,
                content TEXT,
                tool_results TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id)')
        conn.commit()
    finally:
        conn.close()

def _ensure_session_table():
    """确保会话信息表存在"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions(
                session_id VARCHAR PRIMARY KEY,
                session_name VARCHAR DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    finally:
        conn.close()

def init_database():
    """初始化数据库表"""
    _ensure_chat_table()
    _ensure_session_table()
    print(f"✅ 数据库已初始化: {DB_PATH}")

def add_chat_message(session_id: str, role: str, content: str, tool_results: list[dict] | None = None):
    """添加聊天消息到数据库"""
    _ensure_chat_table()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        tool_results_json = json.dumps(tool_results, ensure_ascii=False) if tool_results else None
        conn.execute('''
            INSERT INTO chat_history (session_id, role, content, tool_results)
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, tool_results_json))
        conn.commit()
    finally:
        conn.close()

def get_chat_history(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """获取会话的聊天历史"""
    _ensure_chat_table()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        results = conn.execute('''
            SELECT role, content, tool_results, timestamp 
            FROM chat_history 
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit)).fetchall()
        return [
            {
                "role": r[0], 
                "content": r[1], 
                "tool_results": json.loads(r[2]) if r[2] else None,
                "timestamp": r[3]
            } 
            for r in results
        ]
    finally:
        conn.close()

def update_session_name(session_id: str, session_name: str):
    """更新会话名称"""
    _ensure_session_table()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('''
            INSERT INTO sessions (session_id, session_name)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                session_name = excluded.session_name,
                updated_at = CURRENT_TIMESTAMP
        ''', (session_id, session_name))
        conn.commit()
    finally:
        conn.close()

def get_session_name(session_id: str) -> str | None:
    """获取会话名称"""
    _ensure_session_table()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        result = conn.execute(
            'SELECT session_name FROM sessions WHERE session_id = ?',
            (session_id,)
        ).fetchone()
        return result[0] if result else None
    finally:
        conn.close()

def delete_session(session_id: str):
    """删除会话及其所有消息"""
    _ensure_chat_table()
    _ensure_session_table()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
        conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
    finally:
        conn.close()

def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """列出所有会话"""
    _ensure_chat_table()
    _ensure_session_table()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        results = conn.execute('''
            SELECT 
                c.session_id,
                MAX(c.timestamp) as last_time,
                s.session_name,
                COUNT(c.ID) as message_count
            FROM chat_history c
            LEFT JOIN sessions s ON c.session_id = s.session_id
            GROUP BY c.session_id
            ORDER BY last_time DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        return [
            {
                "session_id": r[0], 
                "last_time": r[1],
                "session_name": r[2],
                "message_count": r[3]
            } 
            for r in results
        ]
    finally:
        conn.close()

def clear_all_sessions():
    """清除所有会话数据"""
    _ensure_chat_table()
    _ensure_session_table()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute('DELETE FROM chat_history')
        conn.execute('DELETE FROM sessions')
        conn.commit()
    finally:
        conn.close()


# 全局 Agent 实例
_agent = None
_agent_lock = False

# 内存中的会话状态（用于快速访问）
_sessions: Dict[str, Dict[str, Any]] = {}

# 线程池执行器
_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化数据库和 Agent
    global _agent
    try:
        # 初始化数据库
        init_database()
        
        print("🔄 正在初始化 MCP Agent...")
        _agent = create_mcp_agent()
        # 在线程池中执行同步连接
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _agent.connect)
        print("✅ MCP Agent 服务已启动")
    except Exception as e:
        print(f"⚠️ Agent 初始化失败: {e}")
        import traceback
        traceback.print_exc()
    
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

# 线程池执行器
_executor = ThreadPoolExecutor(max_workers=4)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[Any] | None = None


class ChatResponse(BaseModel):
    type: str
    message: str
    tool_results: list[dict[str, Any]] | None = None
    duration: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    active_sessions: int
    mcp_server_connected: bool


class RenameSessionRequest(BaseModel):
    session_name: str


def get_agent():
    """获取或创建全局 MCP Agent 实例"""
    global _agent, _agent_lock
    
    print(f"[DEBUG] get_agent() called, _agent={_agent is not None}, _agent_lock={_agent_lock}")
    
    if _agent is not None:
        print(f"[DEBUG] Returning existing agent: {type(_agent)}")
        return _agent
    
    if _agent_lock:
        raise HTTPException(status_code=503, detail="Agent 正在初始化中")
    
    try:
        _agent_lock = True
        print("🔄 正在初始化 MCP Agent...")
        _agent = create_mcp_agent()
        # 连接 MCP Server
        _agent.connect()
        _agent_lock = False
        print("✅ MCP Agent 初始化完成")
        return _agent
    except Exception as e:
        _agent_lock = False
        raise HTTPException(status_code=500, detail=f"Agent 初始化失败: {str(e)}")





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
    """对话接口 - 支持持久化记忆和工具调用记录"""
    agent = get_agent()
    
    start_time = time.time()
    session_id = request.session_id
    
    try:
        # 1. 加载历史对话上下文
        history = get_chat_history(session_id, limit=20)
        
        # 2. 构建带上下文的查询（如果有历史记录）
        if history and len(history) > 0:
            # 构建上下文提示
            context_parts = []
            for msg in history[-10:]:  # 最近10条消息
                role_prefix = "用户" if msg["role"] == "user" else "助手"
                context_parts.append(f"{role_prefix}: {msg['content']}")
            
            context_str = "\n".join(context_parts)
            enhanced_message = f"以下是我们之前的对话:\n{context_str}\n\n用户当前问题: {request.message}\n\n请根据上下文回答。"
        else:
            enhanced_message = request.message
        
        # 3. 在线程池中执行同步的 agent.chat() 调用
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, 
            agent.chat, 
            enhanced_message,
            session_id
        )
        
        duration = int((time.time() - start_time) * 1000)
        
        # 4. 提取消息和工具调用结果
        message = result.get("message", "无响应")
        tool_results = result.get("tool_results", [])
        
        # 5. 保存用户消息到数据库
        add_chat_message(session_id, "user", request.message)
        
        # 6. 保存助手回复到数据库（包含工具调用）
        add_chat_message(session_id, "assistant", message, tool_results)
        
        # 7. 更新内存中的会话状态
        _sessions[session_id] = {
            "last_active": datetime.now().isoformat(),
            "message_count": len(history) + 2
        }
        
        # 8. 如果是新会话，自动生成会话名称（基于第一条用户消息）
        if len(history) == 0 and request.message:
            # 使用用户第一条消息的前20个字符作为会话名称
            session_name = request.message[:30] + "..." if len(request.message) > 30 else request.message
            update_session_name(session_id, session_name)
        
        return ChatResponse(
            type="text",
            message=message,
            tool_results=tool_results,
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
async def api_list_sessions(limit: int = 20):
    """列出所有会话（从数据库）"""
    sessions = list_sessions(limit=limit)
    return {
        "sessions": sessions,
        "total": len(sessions)
    }


@app.get("/sessions/{session_id}/history")
async def api_get_session_history(session_id: str, limit: int = 50):
    """获取指定会话的聊天历史"""
    history = get_chat_history(session_id, limit=limit)
    session_name = get_session_name(session_id)
    return {
        "session_id": session_id,
        "session_name": session_name,
        "messages": history,
        "message_count": len(history)
    }


@app.post("/sessions/{session_id}/rename")
async def api_rename_session(session_id: str, request: RenameSessionRequest):
    """重命名会话"""
    if request.session_name:
        update_session_name(session_id, request.session_name)
        return {"status": "success", "session_id": session_id, "session_name": request.session_name}
    else:
        raise HTTPException(status_code=400, detail="session_name is required")


@app.delete("/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """删除会话及其所有消息"""
    delete_session(session_id)
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.delete("/sessions")
async def api_clear_all_sessions():
    """清除所有会话数据（危险操作）"""
    clear_all_sessions()
    _sessions.clear()
    return {"status": "all_cleared"}


# ========== 统一工具调用处理函数 ==========

async def invoke_mcp_tool_direct(
    tool_name: str,
    params: dict[str, Any]
) -> dict[str, Any] | list[Any] | str:
    """
    直接调用 MCP 工具的底层函数
    
    Args:
        tool_name: MCP 工具名称
        params: 调用参数字典
    
    Returns:
        解析后的结果 (dict/list/str)
    
    Raises:
        HTTPException: 工具不存在或调用失败
    """
    agent = get_agent()
    print(f"[DEBUG] invoke_mcp_tool_direct: agent type={type(agent)}, has _async_agent={hasattr(agent, '_async_agent')}")
    
    # 获取原始工具
    if not hasattr(agent, '_async_agent'):
        raise HTTPException(status_code=500, detail=f"Agent 类型错误: {type(agent)}")
    
    mcp_client = agent._async_agent.mcp_client
    print(f"[DEBUG] mcp_client={mcp_client is not None}")
    
    if mcp_client is None:
        raise HTTPException(status_code=500, detail="MCP 客户端未初始化")
    
    raw_tools = await mcp_client.get_tools()
    target_tool = None
    for t in raw_tools:
        if t.name == tool_name:
            target_tool = t
            break
    
    if target_tool is None:
        available = [t.name for t in raw_tools]
        raise HTTPException(
            status_code=500, 
            detail=f"工具 '{tool_name}' 不存在。可用工具: {available}"
        )
    
    # 调用工具
    try:
        print(f"[DEBUG] Invoking tool '{tool_name}' with params: {params}")
        result = await target_tool.ainvoke(params)
        print(f"[DEBUG] Tool result type: {type(result)}, result: {result[:200] if isinstance(result, str) else result}")
    except Exception as e:
        print(f"[DEBUG] Tool invocation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"工具调用失败: {str(e)}")
    
    # 解析 MCP 返回的 list[dict] 格式: [{"type": "text", "text": "..."}]
    return _parse_mcp_result(result)


def _parse_mcp_result(result: Any) -> dict[str, Any] | list[Any] | str:
    """
    解析 MCP 工具返回的结果
    支持 list[dict] 格式和直接 dict 格式
    """
    # list[dict] 格式: [{"type": "text", "text": "..."}]
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "text" in result[0]:
            text_content = result[0]["text"]
            try:
                return json.loads(text_content)
            except json.JSONDecodeError:
                return {"text": text_content}
        return result
    
    # 直接是字典格式
    elif isinstance(result, dict):
        return result
    
    # 字符串或其他类型
    return result


async def invoke_tool_with_agent_chat(
    query_template: str,
    **kwargs: Any
) -> dict[str, Any]:
    """
    通过 Agent Chat 方式调用工具的包装函数
    
    Args:
        query_template: 查询模板字符串，可使用 kwargs 格式化
        **kwargs: 模板参数
    
    Returns:
        {"query": 实际查询, "result": 结果}
    """
    agent = get_agent()
    query = query_template.format(**kwargs)
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, agent.chat, query)
    
    return {"query": query, "result": result}


# ========== 材料查询 API (通过 MCP) ==========

@app.get("/materials/search")
async def search_materials(
    elements: str | None = None,
    exclude_elements: str | None = None,
    formula: str | None = None,
    max_results: int = 10
):
    """搜索 Materials Project 材料 (直接调用 MCP 工具)"""
    try:
        # 构建参数
        params: dict[str, Any] = {"chunk_size": max_results}
        if elements:
            params["elements"] = [e.strip() for e in elements.split(",")]
        if exclude_elements:
            params["exclude_elements"] = [e.strip() for e in exclude_elements.split(",")]
        if formula:
            params["formula"] = formula.strip()
        
        # 直接调用 MCP 工具获取结构化数据
        result = await invoke_mcp_tool_direct("search_materials_from_mp", params)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/materials/structure/{material_id}")
async def get_material_structure(material_id: str):
    """获取材料结构 (直接调用 MCP 工具)"""
    try:
        result = await invoke_mcp_tool_direct(
            "get_material_structure_from_mp", 
            {"material_id": material_id, "get_plot": True, "get_sites": True}
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/time")
async def get_time():
    """获取当前时间 (测试 MCP 连接)"""
    try:
        result = await invoke_mcp_tool_direct("get_time", {})
        return {"time": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 带隙预测 API ==========

@app.get("/predict_bandgap")
async def predict_bandgap(formula: str):
    """预测材料带隙"""
    try:
        result = await invoke_mcp_tool_direct("predict_band_gap", {"formula": formula})
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 结构建模 API ==========

@app.post("/structure/build")
async def build_structure_endpoint(
    a: float = 5.0,
    b: float = 5.0,
    c: float = 5.0,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
    elements: str = "",
    frac_coords: str = "",
    scaling_matrix: str | None = None,
    save_to_cif: bool = True
):
    """构建晶体结构"""
    try:
        # 解析参数
        element_list = [e.strip() for e in elements.split(",") if e.strip()]
        coords_list = json.loads(frac_coords) if frac_coords else []
        
        params = {
            "a": a, "b": b, "c": c,
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "elements": element_list,
            "frac_coord": coords_list,  # MCP 工具使用单数形式
            "save_to_cif": save_to_cif
        }
        
        if scaling_matrix:
            try:
                params["scaling_matrix"] = json.loads(scaling_matrix)
            except json.JSONDecodeError:
                # 如果是单个数字，转换为各向同性缩放
                try:
                    scale = int(scaling_matrix)
                    params["scaling_matrix"] = [scale, scale, scale]
                except ValueError:
                    pass
        
        result = await invoke_mcp_tool_direct("build_structure", params)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== VASP 任务管理 API ==========

@app.get("/vasp/task_directories")
async def vasp_list_dirs():
    """列出 VASP 任务目录"""
    try:
        result = await invoke_mcp_tool_direct("list_task_directories", {})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vasp/squeue")
async def vasp_check_queue():
    """检查 SLURM 队列"""
    try:
        result = await invoke_mcp_tool_direct("check_squeue", {})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/create_task")
async def vasp_create_task(formula: str, cif_path: str):
    """创建 VASP 任务"""
    try:
        result = await invoke_mcp_tool_direct("create_task", {
            "formula": formula,
            "cif_path": cif_path
        })
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/create_mission")
async def vasp_create_mission(
    task_directory: str,
    mission_type: str  # relax, scf, band, dos
):
    """生成 VASP 输入文件"""
    try:
        result = await invoke_mcp_tool_direct("create_mission", {
            "task_directory": task_directory,
            "mission": mission_type
        })
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/modify_incar")
async def vasp_modify_incar(
    task_directory: str,
    mission: str,
    key: str,
    value: str
):
    """读取或修改 INCAR 文件
    
    key="__read__" 时读取 INCAR
    key="__write__" 时写入 INCAR (value 为完整内容)
    """
    try:
        if key == "__read__":
            result = await invoke_mcp_tool_direct("modify_incar", {
                "task_directory": task_directory,
                "mission": mission,
                "read": True
            })
            return result
        elif key == "__write__":
            result = await invoke_mcp_tool_direct("modify_incar", {
                "task_directory": task_directory,
                "mission": mission,
                "read": False,
                "write": value
            })
            return {"result": result}
        else:
            # 修改单个参数
            result = await invoke_mcp_tool_direct("modify_incar", {
                "task_directory": task_directory,
                "mission": mission,
                "read": False,
                "write": f"{key} = {value}"
            })
            return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/submit")
async def vasp_submit(
    task_directory: str,
    mission: str
):
    """提交 VASP 任务"""
    try:
        result = await invoke_mcp_tool_direct("submit_mission", {
            "task_directory": task_directory,
            "mission": mission
        })
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vasp/extract")
async def vasp_extract(
    task_directory: str,
    mission: str,
    plot: bool = True
):
    """提取 VASP 计算结果"""
    try:
        result = await invoke_mcp_tool_direct("extract_result", {
            "task_directory": task_directory,
            "mission": mission,
            "plot": plot
        })
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8766)
