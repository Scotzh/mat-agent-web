"""
MatAgent MCP Web 应用
连接 agent_mcp_server.py (端口 8766)
现代化界面设计
"""

import streamlit as st
import os
import sys
import uuid
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import requests

# 页面配置
st.set_page_config(
    page_title="MatAgent MCP - 材料智能设计平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# MCP Agent API 地址
MCP_AGENT_URL = "http://127.0.0.1:8766"


# ============ Logo 配置 ============
# 方式1: 使用本地图片文件（推荐）
LOGO_PATH = "web/assets/logo.png"

# 方式2: 使用网络图片 URL
# LOGO_URL = "https://your-domain.com/logo.png"

# 方式3: 使用 base64 编码的图片（内嵌）
# LOGO_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."

# 初始化数据库
from db import databasemanage

if "db" not in st.session_state:
    st.session_state.db = databasemanage.DatabaseManager("db/matagent.db")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mcp_connected" not in st.session_state:
    st.session_state.mcp_connected = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "deepseek-chat"


# ============ Logo 显示函数 ============
def show_logo():
    """在页面顶部显示 Logo"""
    try:
        # 方式1: 本地图片
        if os.path.exists(LOGO_PATH):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(LOGO_PATH, use_container_width=True)
            return

        # 方式2: 网络图片 URL（取消下面的注释）
        # if 'LOGO_URL' in globals():
        #     col1, col2, col3 = st.columns([1, 2, 1])
        #     with col2:
        #         st.image(LOGO_URL, use_container_width=True)
        #     return

        # 方式3: Base64 编码图片（取消下面的注释）
        # if 'LOGO_BASE64' in globals():
        #     col1, col2, col3 = st.columns([1, 2, 1])
        #     with col2:
        #         st.image(LOGO_BASE64, use_container_width=True)
        #     return

        # 如果没有找到 logo，显示默认标题
        st.markdown(
            '<h1 class="main-title">🔬 MatAgent MCP</h1>', unsafe_allow_html=True
        )

    except Exception as e:
        # 出错时显示默认标题
        st.markdown(
            '<h1 class="main-title">🔬 MatAgent MCP</h1>', unsafe_allow_html=True
        )


# ============ 自定义样式 ============
st.markdown(
    """
<style>
    /* Logo 容器样式 */
    .logo-container {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    /* 副标题 */
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 - 透明背景 */
    .card {
        background: transparent;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: none;
    }
    
    /* 状态指示器 */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .status-online {
        background: #d1fae5;
        color: #065f46;
    }
    .status-offline {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* 工具调用卡片 - 透明背景 */
    .tool-card {
        background: transparent;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 结果展示 - 透明背景 */
    .result-box {
        background: transparent;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 侧边栏样式 */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: inherit;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 1rem;
    }
    
    /* 回到底部浮动按钮 */
    .scroll-to-bottom {
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        cursor: pointer;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        transition: all 0.3s ease;
    }
    .scroll-to-bottom:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    .scroll-to-bottom:active {
        transform: translateY(-1px);
    }
    
    /* 功能按钮 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* 输入框美化 */
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* 为固定输入框腾出空间 */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* 隐藏默认的chat input容器边距 */
    .stChatInputContainer {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 模型选择下拉框与输入框对齐 */
    div[data-testid="stSelectbox"] > label {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stSelectbox"] > div > div[data-baseweb="select"] > div {
        min-height: 44px !important;
        height: 44px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============ API 调用函数 ============


def check_mcp_server() -> bool:
    """检查 MCP Agent 服务状态"""
    try:
        response = requests.get(f"{MCP_AGENT_URL}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get("agent_ready", False) and data.get(
                "mcp_server_connected", False
            )
    except:
        pass
    return False


def call_mcp_api(
    endpoint: str, method: str = "GET", params: dict = None, json_data: dict = None
) -> dict:
    """调用 MCP Agent Server API"""
    url = f"{MCP_AGENT_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        else:
            response = requests.post(url, params=params, json=json_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def extract_returns(result: Any) -> Any:
    """
    从 MCP API 返回结果中提取 returns 字段

    处理流程:
    1. 如果是列表格式 [{"type": "text", "text": "..."}], 先提取 text 中的 JSON
    2. 如果是 {"args": ..., "returns": ...} 格式, 返回 returns
    3. 如果是 {"prediction": ...}, {"result": ...} 或 {"data": ...} 格式, 递归解析内部值

    Args:
        result: API 返回的原始数据

    Returns:
        提取并解析后的 returns 数据
    """
    if result is None:
        return None

    # 处理字符串类型（可能是JSON）
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return result

    # 处理列表格式: [{"type": "text", "text": "..."}]
    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict) and "text" in first_item:
            try:
                inner = json.loads(first_item["text"])
                # 递归处理内部值
                return extract_returns(inner)
            except json.JSONDecodeError:
                return first_item["text"]
        return result

    # 处理字典格式
    if isinstance(result, dict):
        # 如果是 {"args": ..., "returns": ...} 格式, 提取 returns 并继续处理
        if "returns" in result:
            returns_data = result["returns"]
            # 如果 returns 里有 data 键, 直接返回 data(材料列表)
            if isinstance(returns_data, dict) and "data" in returns_data:
                return returns_data["data"]
            return returns_data

        # 如果有 result 或 data 键, 递归解析
        if "result" in result:
            return extract_returns(result["result"])
        if "data" in result:
            return extract_returns(result["data"])

        # 其他字典直接返回
        return result

    return result


def parse_mcp_result(result: dict, key: str = "result") -> Any:
    """
    统一解析 MCP API 返回的结果（使用 extract_returns 简化处理）

    Args:
        result: API 返回的原始字典
        key: 要提取的键名，默认为 "result"

    Returns:
        解析后的数据，如果出错则返回包含 error 键的字典
    """
    if not isinstance(result, dict):
        return {"error": f"无效的返回类型: {type(result)}"}

    if "error" in result:
        return result

    # 使用 extract_returns 提取 returns
    return extract_returns(result)


def safe_json_loads(text: str, default: Any = None) -> Any:
    """安全地解析 JSON 字符串"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else text


def chat_with_mcp(message: str) -> dict:
    """通过 MCP 与 Agent 对话（非流式）"""
    try:
        response = requests.post(
            f"{MCP_AGENT_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": message},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"type": "error", "message": "请求超时，请稍后重试"}
    except Exception as e:
        return {"type": "error", "message": f"请求失败: {str(e)}"}


def chat_with_mcp_stream(message: str, model: str = "deepseek-chat"):
    """
    通过 MCP 与 Agent 进行流式对话
    生成器yield格式: (event_type, data)
    """
    try:
        response = requests.post(
            f"{MCP_AGENT_URL}/chat/stream",
            json={
                "session_id": st.session_state.session_id,
                "message": message,
                "model": model,
            },
            stream=True,
            timeout=300,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                # SSE 格式: data: {...}
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])  # 去掉 "data: " 前缀
                        yield event
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.Timeout:
        yield {"type": "error", "data": "请求超时，请稍后重试"}
    except Exception as e:
        yield {"type": "error", "data": f"请求失败: {str(e)}"}


def update_message_blocks(session_id: str, content_blocks: list[dict]) -> bool:
    """更新消息的内容块信息到服务器"""
    try:
        response = requests.post(
            f"{MCP_AGENT_URL}/sessions/{session_id}/messages/update-blocks",
            json={"content_blocks": content_blocks},
            timeout=10,
        )
        if response.status_code != 200:
            print(f"更新 content_blocks 失败: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"更新 content_blocks 异常: {e}")
        return False


def add_chat_message_via_api(
    session_id: str,
    role: str,
    content: str,
    tool_results: list = None,
    content_blocks: list = None,
    model: str = None,
    duration: int = None,
) -> bool:
    """通过API添加聊天消息到服务器数据库"""
    try:
        payload = {
            "role": role,
            "content": content,
            "tool_results": tool_results,
            "content_blocks": content_blocks,
        }
        if model is not None:
            payload["model"] = model
        if duration is not None:
            payload["duration"] = duration

        response = requests.post(
            f"{MCP_AGENT_URL}/sessions/{session_id}/messages", json=payload, timeout=10
        )
        if response.status_code != 200:
            print(f"添加消息失败: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"添加消息异常: {e}")
        return False


def predict_bandgap(formula: str) -> dict:
    """预测带隙"""
    return call_mcp_api("/predict_bandgap", params={"formula": formula})


def search_materials(
    elements: str = None, exclude: str = None, formula: str = None, chunk_size: int = 10
) -> dict:
    """搜索材料"""
    params = {"chunk_size": chunk_size}
    if elements:
        params["elements"] = elements
    if exclude:
        params["exclude_elements"] = exclude
    if formula:
        params["formula"] = formula
    return call_mcp_api("/materials/search", params=params)


def get_material_structure(material_id: str) -> dict:
    """获取材料结构"""
    return call_mcp_api(f"/materials/structure/{material_id}")


def build_structure(**kwargs) -> dict:
    """构建结构"""
    return call_mcp_api("/structure/build", method="POST", params=kwargs)


# VASP 相关
def vasp_list_dirs() -> dict:
    return call_mcp_api("/vasp/task_directories")


def vasp_check_queue() -> dict:
    return call_mcp_api("/vasp/squeue")


def vasp_create_task(formula: str, cif_path: str) -> dict:
    return call_mcp_api(
        "/vasp/create_task",
        method="POST",
        params={"formula": formula, "cif_path": cif_path},
    )


def vasp_create_mission(task_dir: str, mission_type: str) -> dict:
    return call_mcp_api(
        "/vasp/create_mission",
        method="POST",
        params={"task_directory": task_dir, "mission_type": mission_type},
    )


def vasp_modify_incar(task_dir: str, mission: str, key: str, value: str) -> dict:
    return call_mcp_api(
        "/vasp/modify_incar",
        method="POST",
        params={
            "task_directory": task_dir,
            "mission": mission,
            "key": key,
            "value": value,
        },
    )


def vasp_submit(task_dir: str, mission: str) -> dict:
    return call_mcp_api(
        "/vasp/submit",
        method="POST",
        params={"task_directory": task_dir, "mission": mission},
    )


def vasp_extract(task_dir: str, mission: str, plot: bool = True) -> dict:
    return call_mcp_api(
        "/vasp/extract",
        method="POST",
        params={"task_directory": task_dir, "mission": mission, "plot": plot},
    )


# ============ 侧边栏 ============


def load_session_history(session_id: str):
    """从服务器加载会话历史"""
    try:
        response = requests.get(
            f"{MCP_AGENT_URL}/sessions/{session_id}/history", timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            messages = data.get("messages", [])
            # 调试输出
            for msg in messages:
                if msg.get("content_blocks"):
                    print(
                        f"加载消息包含 content_blocks: {len(msg['content_blocks'])} 个块"
                    )
                else:
                    print(f"加载消息不包含 content_blocks")
            return messages
        else:
            print(f"加载会话历史失败: {response.status_code}")
    except Exception as e:
        print(f"加载会话历史失败: {e}")
    return []


def delete_session(session_id: str):
    """删除会话"""
    try:
        response = requests.delete(f"{MCP_AGENT_URL}/sessions/{session_id}", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"删除会话失败: {e}")
    return False


def rename_session(session_id: str, new_name: str):
    """重命名会话"""
    try:
        response = requests.post(
            f"{MCP_AGENT_URL}/sessions/{session_id}/rename",
            json={"session_name": new_name},
            timeout=5,
        )
        return response.status_code == 200
    except Exception as e:
        print(f"重命名会话失败: {e}")
    return False


def sidebar():
    with st.sidebar:
        # 侧边栏 Logo（小尺寸）
        try:
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, use_container_width=True)
            else:
                st.markdown(
                    '<div class="sidebar-header">🔬 MatAgent MCP</div>',
                    unsafe_allow_html=True,
                )
        except:
            st.markdown(
                '<div class="sidebar-header">🔬 MatAgent MCP</div>',
                unsafe_allow_html=True,
            )

        # 连接状态
        if check_mcp_server():
            st.markdown(
                '<span class="status-badge status-online">🟢 服务已连接</span>',
                unsafe_allow_html=True,
            )
            st.session_state.mcp_connected = True
        else:
            st.markdown(
                '<span class="status-badge status-offline">🔴 服务未连接</span>',
                unsafe_allow_html=True,
            )
            st.session_state.mcp_connected = False
            if st.button("🔄 重新连接", use_container_width=True):
                st.rerun()

        st.divider()

        # 功能导航
        page = st.radio(
            "选择功能",
            [
                "💬 智能体对话",
                "🔍 材料查询",
                "📊 结构建模",
                "🧪 ML 预测",
                "💻 VASP 任务",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # 会话管理
        st.markdown("**📜 会话管理**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 新建会话", use_container_width=True):
                # 清除当前会话的 history_loaded 标记
                old_session_id = st.session_state.get("session_id")
                if old_session_id:
                    old_key = f"history_loaded_{old_session_id}"
                    if old_key in st.session_state:
                        del st.session_state[old_key]
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                # 清除编辑状态
                if "editing_session" in st.session_state:
                    del st.session_state["editing_session"]
                st.rerun()

        with col2:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # 显示当前会话
        st.caption(f"会话: `{st.session_state.session_id[:8]}...`")

        st.divider()

        # 历史会话列表
        st.markdown("**📚 历史会话**")

        # 初始化编辑状态
        if "editing_session" not in st.session_state:
            st.session_state.editing_session = None

        try:
            response = requests.get(f"{MCP_AGENT_URL}/sessions", timeout=5)
            if response.status_code == 200:
                data = response.json()
                sessions = data.get("sessions", [])

                if sessions:
                    for sess in sessions:
                        session_id = sess.get("session_id", "")
                        session_name = sess.get("session_name", "")
                        message_count = sess.get("message_count", 0)
                        last_time = (
                            sess.get("last_time", "")[:10]
                            if sess.get("last_time")
                            else ""
                        )

                        # 显示名称
                        display_name = (
                            session_name
                            if session_name
                            else f"会话 {session_id[:6]}..."
                        )

                        # 判断是否是当前会话
                        is_current = session_id == st.session_state.session_id
                        current_marker = "✓ " if is_current else ""

                        with st.container():
                            # 会话行
                            cols = st.columns([4, 1, 1])

                            with cols[0]:
                                # 点击切换到该会话
                                btn_label = f"{current_marker}{display_name}"
                                if st.button(
                                    btn_label,
                                    key=f"select_{session_id}",
                                    use_container_width=True,
                                ):
                                    st.session_state.session_id = session_id
                                    # 从服务器加载历史消息
                                    history = load_session_history(session_id)
                                    st.session_state.messages = [
                                        {
                                            "role": msg["role"],
                                            "content": msg["content"],
                                            "tool_results": msg.get("tool_results"),
                                            "content_blocks": msg.get("content_blocks"),
                                            "model": msg.get("model"),
                                            "duration": msg.get("duration"),
                                        }
                                        for msg in history
                                    ]
                                    # 标记已加载，避免 chat_page() 重复加载
                                    st.session_state[f"history_loaded_{session_id}"] = (
                                        True
                                    )
                                    st.rerun()

                            with cols[1]:
                                # 重命名按钮
                                if st.button("✏️", key=f"edit_{session_id}"):
                                    st.session_state.editing_session = session_id
                                    st.rerun()

                            with cols[2]:
                                # 删除按钮
                                if st.button("🗑️", key=f"del_{session_id}"):
                                    if delete_session(session_id):
                                        # 如果删除的是当前会话，重置会话
                                        if session_id == st.session_state.session_id:
                                            # 清除 history_loaded 标记
                                            old_key = f"history_loaded_{session_id}"
                                            if old_key in st.session_state:
                                                del st.session_state[old_key]
                                            st.session_state.session_id = str(
                                                uuid.uuid4()
                                            )
                                            st.session_state.messages = []
                                        st.success("已删除")
                                        st.rerun()
                                    else:
                                        st.error("删除失败")

                            # 显示消息数和最后活动时间
                            st.caption(f"💬 {message_count}条消息 | 📅 {last_time}")

                            # 重命名输入框
                            if st.session_state.editing_session == session_id:
                                new_name = st.text_input(
                                    "新名称",
                                    value=session_name if session_name else "",
                                    key=f"rename_input_{session_id}",
                                )
                                col_save, col_cancel = st.columns(2)
                                with col_save:
                                    if st.button("保存", key=f"save_{session_id}"):
                                        if rename_session(session_id, new_name):
                                            st.session_state.editing_session = None
                                            st.rerun()
                                with col_cancel:
                                    if st.button("取消", key=f"cancel_{session_id}"):
                                        st.session_state.editing_session = None
                                        st.rerun()

                            st.divider()
                else:
                    st.info("暂无历史会话")
            else:
                st.info("无法加载历史会话")
        except Exception as e:
            st.info(f"无法加载历史会话: {e}")

        st.divider()

        # 使用提示
        with st.expander("💡 使用提示"):
            st.markdown("""
            - **智能体对话**: 用自然语言描述材料需求
            - **材料查询**: 从 Materials Project 搜索
            - **结构建模**: 自定义晶体结构
            - **ML 预测**: 快速预测材料性质
            - **VASP 任务**: 管理计算任务
            """)

        return page


# ============ 页面组件 ============


def display_tool_result(tr: dict):
    """显示单个工具调用结果"""
    tool_name = tr.get("tool_name", "未知工具")
    raw_result = tr.get("result")

    # 先处理字符串类型（可能是JSON）
    result = raw_result
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass

    # 处理 [[{"type": "text", "text": "..."}], {...}] 格式
    if isinstance(result, list) and len(result) >= 1:
        # 第一个元素是 text 列表
        if isinstance(result[0], list) and len(result[0]) > 0:
            first_item = result[0][0]
            if isinstance(first_item, dict) and first_item.get("type") == "text":
                text_str = first_item.get("text")
                # 尝试解析 text 中的 JSON
                if text_str:
                    try:
                        result = json.loads(text_str)
                    except json.JSONDecodeError:
                        result = text_str
        # 如果上面的解析失败，尝试第二个元素
        if (
            isinstance(result, list)
            and len(result) >= 2
            and isinstance(result[1], dict)
        ):
            structured = result[1].get("structured_content")
            if structured:
                result = structured

    # 统一从 result 中提取 args 和 returns
    tool_args = None
    if isinstance(result, dict):
        tool_args = result.get("args")
        if "returns" in result:
            result = result.get("returns")
    elif isinstance(result, str):
        # 如果结果是字符串，尝试解析为JSON
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                tool_args = parsed.get("args")
                if "returns" in parsed:
                    result = parsed.get("returns")
        except json.JSONDecodeError:
            pass

    with st.expander(f"🔧 {tool_name}", expanded=False):
        # 显示工具参数
        st.markdown("**输入参数:**")
        if tool_args is not None and tool_args != {}:
            st.json(tool_args)
        else:
            st.json({})

        # 显示工具返回结果
        st.markdown("**返回结果:**")

        # 处理结果是字典的情况（提取图片、链接等）
        if isinstance(result, dict):
            # 从 result 中提取图片和链接
            img_url = (
                result.get("2d_image_url")
                or result.get("image_url")
                or result.get("image")
            )
            html_url = result.get("3d_html_url") or result.get("3d_image_url")

            # 显示图片
            if img_url:
                st.markdown("**🖼️ 结构图:**")
                try:
                    st.image(img_url, use_container_width=True)
                except Exception as e:
                    st.error(f"加载图片失败: {e}")

            # 显示3D链接
            if html_url:
                st.markdown(f"**🌐 3D 可视化:** [点击查看]({html_url})")
                if html_url.startswith("/"):
                    html_url = f"http://127.0.0.1:5001{html_url}"
                st.components.v1.iframe(html_url, height=400, scrolling=True)

            # CIF 文件下载
            cif_path = result.get("cif_path")
            if cif_path and os.path.exists(cif_path):
                st.markdown("**📥 CIF 文件:**")
                with open(cif_path, "r") as f:
                    st.download_button(
                        label="下载 CIF",
                        data=f.read(),
                        file_name=os.path.basename(cif_path),
                        mime="chemical/x-cif",
                    )

            # 显示简化后的内容（排除已处理的字段）
            display_data = {
                k: v
                for k, v in result.items()
                if k
                not in [
                    "2d_image_url",
                    "3d_html_url",
                    "3d_image_url",
                    "image_url",
                    "image",
                    "cif_path",
                    "structure_dict",
                ]
            }
            if display_data:
                st.json(display_data)
        elif result is None:
            st.markdown("*等待结果...*")
        else:
            st.code(str(result))


def chat_page():
    # 只显示副标题（Logo 只在侧边栏显示）
    st.markdown(
        '<h1 class="main-title">💬 MatAgent材料智能设计平台</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">基于 MCP 协议的材料科学智能助手</p>',
        unsafe_allow_html=True,
    )

    # 添加回到底部的浮动按钮 - 使用 HTML anchor 链接
    st.markdown(
        """
    <style>
    .scroll-bottom-link {
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        cursor: pointer;
        z-index: 999999;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
    }
    .scroll-bottom-link:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    </style>
    <a href="#chat-input-anchor" class="scroll-bottom-link" title="回到底部">⬇️</a>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.mcp_connected:
        st.warning("⚠️ MCP 服务未连接，请确保 agent_mcp_server.py 已启动")
        return

    # 首次加载时从服务器获取历史消息
    # 使用 session_id 作为 key 的一部分，确保切换会话时重新加载
    history_key = f"history_loaded_{st.session_state.session_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = True
        try:
            print(f"页面加载: 尝试加载会话 {st.session_state.session_id} 的历史消息")
            history = load_session_history(st.session_state.session_id)
            print(f"页面加载: 加载到 {len(history)} 条消息")
            if history:
                st.session_state.messages = [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "tool_results": msg.get("tool_results"),
                        "content_blocks": msg.get("content_blocks"),
                        "model": msg.get("model"),
                        "duration": msg.get("duration"),
                    }
                    for msg in history
                ]
                print(
                    f"页面加载: 已设置 session_state.messages，共 {len(st.session_state.messages)} 条"
                )
        except Exception as e:
            print(f"页面加载: 加载历史消息失败: {e}")

    # 显示历史消息（包含工具调用）
    print(f"渲染消息: 共 {len(st.session_state.messages)} 条消息")
    for i, msg in enumerate(st.session_state.messages):
        print(
            f"  消息 {i}: role={msg['role']}, content_len={len(msg.get('content', ''))}, has_blocks={bool(msg.get('content_blocks'))}"
        )
        with st.chat_message(msg["role"]):
            # 如果有内容块（包含工具调用位置信息），按顺序渲染
            if msg.get("content_blocks"):
                print(f"  消息 {i}: 有 {len(msg['content_blocks'])} 个 content_blocks")
                for j, block in enumerate(msg["content_blocks"]):
                    print(f"    块 {j}: type={block.get('type')}")
                    if block["type"] == "text":
                        st.markdown(block["content"])
                    elif block["type"] == "tool":
                        display_tool_result(block["data"])
            else:
                print(f"  消息 {i}: 没有 content_blocks，使用兼容模式")
                # 兼容旧格式：先显示工具调用，再显示消息内容
                if msg.get("tool_results"):
                    for tr in msg["tool_results"]:
                        display_tool_result(tr)
                st.markdown(msg["content"])

            # 显示模型和响应时间信息（如果有）
            if msg.get("model") and msg.get("duration"):
                st.caption(f"{msg['model']} | 响应时间: {msg['duration']}ms")

    # 模型选择和输入框（放在消息列表之后，显示在底部）
    st.markdown("---")
    input_container = st.container()

    with input_container:
        col_model, col_input = st.columns([1, 4])

        with col_model:
            st.session_state.selected_model = st.selectbox(
                "模型",
                options=["deepseek-chat", "deepseek-reasoner", "glm-5"],
                index=["deepseek-chat", "deepseek-reasoner", "glm-5"].index(
                    st.session_state.selected_model
                ),
                label_visibility="collapsed",
            )

        with col_input:
            # 添加锚点用于回到底部按钮
            st.markdown('<div id="chat-input-anchor"></div>', unsafe_allow_html=True)
            prompt = st.chat_input("描述您的材料科学问题...")

    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 Agent（流式）
        with st.chat_message("assistant"):
            # 创建占位符列表，按顺序存储文本和工具框
            content_placeholders: list[Any] = []  # 存储文本和工具框的占位符信息
            caption_placeholder = st.empty()

            full_message: str = ""
            duration: int = 0

            # 当前文本的占位符
            current_text_placeholder = st.empty()
            current_text = ""

            # 流式接收响应
            last_update_time = time.time()
            is_tool_running = False  # 标记是否正在执行工具

            for event in chat_with_mcp_stream(
                prompt, model=st.session_state.selected_model
            ):
                event_type = event.get("type")
                data = event.get("data")

                if event_type == "token" and isinstance(data, str):
                    # 如果正在执行工具，不累积 token（这些是工具返回的 JSON，不是回复）
                    if is_tool_running:
                        continue
                    # 后端发送的是单个 token，需要累积
                    full_message += data
                    current_text += data
                    # 每50ms更新一次显示，减少闪烁
                    current_time = time.time()
                    if current_time - last_update_time > 0.05:
                        current_text_placeholder.markdown(current_text + "▌")
                        last_update_time = current_time

                elif event_type == "tool_start" and isinstance(data, dict):
                    # 标记工具开始执行
                    is_tool_running = True

                    tool_name = data.get("tool_name", "未知工具")
                    print(f"[FRONTEND DEBUG] tool_start received: {data}")
                    # 使用后端传递的 tool_id 作为唯一标识
                    tool_id = data.get("tool_id")
                    if not tool_id:
                        # 兼容旧数据，使用 tool_name + args 哈希
                        tool_args = data.get("tool_args", {})
                        tool_id = f"{tool_name}_{hash(str(tool_args))}"

                    # 检查是否已经创建过这个工具框（避免重复）
                    already_exists = False
                    for item in content_placeholders:
                        if (
                            item.get("type") == "tool"
                            and item.get("tool_id") == tool_id
                            and item.get("status") == "running"
                        ):
                            already_exists = True
                            break

                    if already_exists:
                        continue

                    # 保存当前文本占位符到列表（去掉光标）
                    if current_text:
                        current_text_placeholder.markdown(current_text)
                        content_placeholders.append(
                            {
                                "type": "text",
                                "placeholder": current_text_placeholder,
                                "content": current_text,
                            }
                        )

                    # 创建新的工具占位符（在当前位置）
                    tool_placeholder = st.empty()
                    with tool_placeholder.container():
                        with st.expander(f"🔧 {tool_name}", expanded=True):
                            st.markdown("*正在执行...*")
                            # 工具开始时先不显示参数，等工具结束后再显示

                    content_placeholders.append(
                        {
                            "type": "tool",
                            "placeholder": tool_placeholder,
                            "data": data,
                            "tool_id": tool_id,
                            "status": "running",
                        }
                    )

                    # 创建新的文本占位符用于后续内容
                    current_text = ""
                    current_text_placeholder = st.empty()

                elif event_type == "tool_end" and isinstance(data, dict):
                    # 工具调用完成，更新对应工具框
                    tool_id = data.get("tool_id")
                    tool_name = data.get("tool_name")
                    result = data.get("result")

                    for item in content_placeholders:
                        if (
                            item.get("type") == "tool"
                            and item.get("status") == "running"
                        ):
                            # 使用 tool_id 匹配
                            item_tool_id = item.get("tool_id")

                            matched = False
                            if tool_id and item_tool_id:
                                matched = tool_id == item_tool_id
                            elif tool_name:
                                # 回退：使用 tool_name 匹配第一个未完成的同名工具
                                item_tool_name = item.get("data", {}).get("tool_name")
                                matched = tool_name == item_tool_name

                            if matched:
                                # 更新工具框显示结果
                                tool_placeholder = item["placeholder"]
                                tool_data = item["data"].copy()
                                tool_data["result"] = result

                                tool_placeholder.empty()
                                with tool_placeholder.container():
                                    display_tool_result(tool_data)

                                # 更新状态
                                item["status"] = "completed"
                                item["data"] = tool_data
                                break

                    # 重置工具运行标记
                    is_tool_running = False

                elif event_type == "complete" and isinstance(data, dict):
                    # 流结束，使用后端返回的完整消息（已清理工具JSON）
                    full_message = data.get("message", full_message)
                    duration = data.get("duration", 0)

                elif event_type == "done" and isinstance(data, dict):
                    # 兼容旧格式的流结束事件
                    duration = data.get("duration", 0)

                elif event_type == "error":
                    # 错误
                    error_msg = str(data) if data else "未知错误"
                    full_message += f"\n\n❌ 错误: {error_msg}"
                    current_text += f"\n\n❌ 错误: {error_msg}"
                    break

            # 保存最后的文本
            if current_text:
                current_text_placeholder.markdown(current_text)
                content_placeholders.append(
                    {
                        "type": "text",
                        "placeholder": current_text_placeholder,
                        "content": current_text,
                    }
                )
            elif not content_placeholders:
                # 如果没有内容，显示空消息
                current_text_placeholder.markdown(full_message)
                content_placeholders.append(
                    {
                        "type": "text",
                        "placeholder": current_text_placeholder,
                        "content": full_message,
                    }
                )

            if duration:
                model_name = st.session_state.selected_model
                caption_placeholder.caption(f" {model_name} | 响应时间: {duration}ms")

            # 构建 content_blocks 用于历史记录（移除 placeholder 对象，只保留可序列化的数据）
            content_blocks_for_history = []

            for block in content_placeholders:
                if block["type"] == "text":
                    content_blocks_for_history.append(
                        {"type": "text", "content": block["content"]}
                    )
                elif block["type"] == "tool":
                    # 只提取可序列化的字段
                    tool_data = block["data"]
                    result = tool_data.get("result")
                    tool_args = tool_data.get("tool_args")

                    # 处理新的格式 {"args": ..., "returns": ...}
                    if isinstance(result, dict) and "returns" in result:
                        # 从结果中提取参数（如果之前没有）
                        if tool_args is None or tool_args == {}:
                            tool_args = result.get("args")
                        # 使用实际的返回内容
                        result = result.get("returns")

                    serializable_data = {
                        "tool_name": tool_data.get("tool_name"),
                        "tool_args": tool_args,
                        "result": result,
                    }
                    content_blocks_for_history.append(
                        {"type": "tool", "data": serializable_data}
                    )

            # 从 content_blocks_for_history 中提取 tool_results（用于兼容旧代码）
            tool_results_for_api = [
                block["data"]
                for block in content_blocks_for_history
                if block["type"] == "tool"
            ]

            # 保存消息到session_state（包含内容块顺序信息）
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_message,
                    "tool_results": tool_results_for_api,
                    "content_blocks": content_blocks_for_history,
                    "model": st.session_state.selected_model,
                    "duration": duration,
                }
            )

            # 通过API保存到服务器数据库（确保前后端数据一致）
            add_chat_message_via_api(st.session_state.session_id, "user", prompt)
            add_chat_message_via_api(
                st.session_state.session_id,
                "assistant",
                full_message,
                tool_results=tool_results_for_api,
                content_blocks=content_blocks_for_history,
                model=st.session_state.selected_model,
                duration=duration,
            )

            # 流式传输结束后刷新页面，让聊天框回到最底部
            st.rerun()


def material_search_page():
    st.markdown('<h1 class="main-title">🔍 材料查询</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">从 Materials Project 数据库搜索材料</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.mcp_connected:
        st.warning("⚠️ MCP 服务未连接")
        return

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**🔬 搜索条件**")

            elements = st.text_input("包含元素 (逗号分隔)", placeholder="Li, Fe, O")
            exclude = st.text_input("排除元素", placeholder="H, He")
            formula = st.text_input("化学式", placeholder="LiFeO2")
            chunk_size = st.slider("返回数量", 1, 50, 10)

            search_btn = st.button(
                "🔍 开始搜索", type="primary", use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📊 搜索结果**")

            # 搜索按钮处理
            if search_btn:
                if not any([elements, exclude, formula]):
                    st.warning("请至少输入一个搜索条件")
                else:
                    with st.spinner("搜索中..."):
                        result = search_materials(
                            elements, exclude, formula, chunk_size
                        )

                        if "error" in result:
                            st.error(f"搜索失败: {result['error']}")
                            st.session_state.search_results = None
                        else:
                            # 使用统一解析函数处理结果
                            materials = parse_mcp_result(result)
                            st.session_state.search_results = materials

            # 显示搜索结果（独立于搜索按钮状态）
            if st.session_state.get("search_results"):
                materials = st.session_state.search_results

                if isinstance(materials, dict) and "error" in materials:
                    st.error(f"解析失败: {materials['error']}")
                elif isinstance(materials, list) and len(materials) > 0:
                    st.success(f"找到 {len(materials)} 个材料")

                    for mat in materials:
                        # 支持新旧两种字段名格式
                        mat_id = mat.get("entry_id") or mat.get("material_id")
                        formula = (
                            mat.get("name")
                            or mat.get("formula_pretty")
                            or mat.get("formula", "Unknown")
                        )
                        with st.expander(f"{formula} ({mat_id})"):
                            col_a, col_b = st.columns(2)
                            col_a.metric("带隙", f"{mat.get('band_gap', 'N/A')} eV")

                            # 处理不同的对称性格式
                            symmetry = mat.get("spacegroup") or mat.get("symmetry", {})
                            if isinstance(symmetry, dict):
                                crystal_system = symmetry.get("crystal_system", "N/A")
                            else:
                                crystal_system = str(symmetry)
                            col_b.metric("晶系", crystal_system)

                            # 使用独立的 key 存储每个材料的结构数据
                            struct_key = f"struct_{mat_id}"

                            if col_b.button("📊 查看结构", key=f"view_{mat_id}"):
                                with st.spinner("获取结构中..."):
                                    struct = get_material_structure(mat_id)
                                    if "error" not in struct:
                                        # 使用统一解析函数处理结果
                                        result_data = parse_mcp_result(struct)
                                        # 存储完整的结果数据（包含 image_url, 3d_image_url, structure_dict）
                                        st.session_state[struct_key] = result_data
                                        # 强制重新运行以显示结构
                                        st.rerun()
                                    else:
                                        st.error(struct.get("error"))

                            # 显示已查看的结构（检查当前材料的结构数据）
                            if struct_key in st.session_state:
                                st.divider()
                                st.markdown("**📐 结构详情**")
                                result_data = st.session_state[struct_key]

                                # 获取 structure_dict（实际的结构数据）
                                struct_data = result_data.get(
                                    "structure_dict", result_data
                                )

                                # 显示晶格参数
                                if "lattice_parameters" in struct_data:
                                    lp = struct_data["lattice_parameters"]
                                    st.markdown(f"""
                                    - a = {lp.get("a", "N/A")} Å
                                    - b = {lp.get("b", "N/A")} Å
                                    - c = {lp.get("c", "N/A")} Å
                                    - α = {lp.get("alpha", "N/A")}°
                                    - β = {lp.get("beta", "N/A")}°
                                    - γ = {lp.get("gamma", "N/A")}°
                                    """)

                                # 显示空间群
                                if "space_group_symbol" in struct_data:
                                    st.markdown(
                                        f"**空间群:** {struct_data.get('space_group_symbol', 'N/A')}"
                                    )

                                # 显示原子位置
                                if "sites" in struct_data:
                                    with st.expander("🔬 原子位置"):
                                        sites = struct_data["sites"]
                                        import pandas as pd

                                        # 处理两种可能的格式
                                        if (
                                            sites
                                            and "fractional_coordinates" in sites[0]
                                        ):
                                            # MCP 格式: [{"element": "Li", "fractional_coordinates": [0,0,0]}]
                                            sites_df = pd.DataFrame(
                                                [
                                                    {
                                                        "元素": s.get("element", "N/A"),
                                                        "x": s.get(
                                                            "fractional_coordinates",
                                                            [0, 0, 0],
                                                        )[0],
                                                        "y": s.get(
                                                            "fractional_coordinates",
                                                            [0, 0, 0],
                                                        )[1],
                                                        "z": s.get(
                                                            "fractional_coordinates",
                                                            [0, 0, 0],
                                                        )[2],
                                                    }
                                                    for s in sites
                                                ]
                                            )
                                        else:
                                            # 旧格式: [{"species": [{"element": "Li"}], "xyz": [0,0,0]}]
                                            sites_df = pd.DataFrame(
                                                [
                                                    {
                                                        "元素": s.get("species", [{}])[
                                                            0
                                                        ].get("element", "N/A"),
                                                        "x": s.get("xyz", [0, 0, 0])[0],
                                                        "y": s.get("xyz", [0, 0, 0])[1],
                                                        "z": s.get("xyz", [0, 0, 0])[2],
                                                    }
                                                    for s in sites
                                                ]
                                            )
                                        st.dataframe(sites_df, use_container_width=True)

                                # 显示结构图片（从 result_data 根级别获取）
                                if (
                                    "image_url" in result_data
                                    and result_data["image_url"]
                                ):
                                    st.divider()
                                    st.markdown("**📊 结构示意图**")
                                    st.image(
                                        result_data["image_url"],
                                        use_container_width=True,
                                    )

                                # 显示3D可视化链接（从 result_data 根级别获取）
                                if (
                                    "3d_image_url" in result_data
                                    and result_data["3d_image_url"]
                                ):
                                    st.divider()
                                    st.markdown("**🔍 3D 结构可视化**")
                                    url_3d = result_data["3d_image_url"]
                                    # 确保URL是完整的
                                    if url_3d.startswith("/"):
                                        url_3d = f"http://127.0.0.1:5001{url_3d}"
                                    st.markdown(f"[在浏览器中打开3D结构]({url_3d})")
                                    # 使用iframe嵌入3D视图
                                    st.components.v1.iframe(
                                        url_3d, height=500, scrolling=True
                                    )

                                # 显示完整数据
                                with st.expander("📋 完整数据"):
                                    st.json(result_data)
                else:
                    st.info("未找到结果或结果格式不正确")

            st.markdown("</div>", unsafe_allow_html=True)


def structure_builder_page():
    st.markdown('<h1 class="main-title">📊 结构建模</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">构建自定义晶体结构</p>', unsafe_allow_html=True)

    if not st.session_state.mcp_connected:
        st.warning("⚠️ MCP 服务未连接")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📐 晶格参数**")

        lattice_col1, lattice_col2, lattice_col3 = st.columns(3)
        with lattice_col1:
            a = st.number_input("a (Å)", value=5.0, step=0.1, format="%.4f")
            alpha = st.number_input("α (°)", value=90.0, step=0.1)
        with lattice_col2:
            b = st.number_input("b (Å)", value=5.0, step=0.1, format="%.4f")
            beta = st.number_input("β (°)", value=90.0, step=0.1)
        with lattice_col3:
            c = st.number_input("c (Å)", value=5.0, step=0.1, format="%.4f")
            gamma = st.number_input("γ (°)", value=90.0, step=0.1)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**⚙️ 高级选项**")

        scaling_type = st.radio("超胞类型", ["各向同性", "各向异性"], horizontal=True)
        if scaling_type == "各向同性":
            scaling = st.number_input(
                "扩展因子 (如 2 表示 2×2×2)", min_value=1, value=1, step=1
            )
            scaling_str = str(scaling) if scaling > 1 else None
        else:
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                sa = st.number_input("a 方向", min_value=1, value=1, step=1)
            with col_s2:
                sb = st.number_input("b 方向", min_value=1, value=1, step=1)
            with col_s3:
                sc = st.number_input("c 方向", min_value=1, value=1, step=1)
            if sa > 1 or sb > 1 or sc > 1:
                scaling_str = json.dumps([sa, sb, sc])
            else:
                scaling_str = None
        save_cif = st.checkbox("保存为 CIF 文件", value=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🧬 原子坐标**")

        import pandas as pd

        if "atom_data" not in st.session_state:
            st.session_state.atom_data = pd.DataFrame(
                [
                    {"元素": "Na", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"元素": "Cl", "x": 0.5, "y": 0.5, "z": 0.5},
                ]
            )

        atom_data = st.data_editor(
            st.session_state.atom_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "元素": st.column_config.TextColumn("元素", width="small"),
                "x": st.column_config.NumberColumn("x", format="%.4f"),
                "y": st.column_config.NumberColumn("y", format="%.4f"),
                "z": st.column_config.NumberColumn("z", format="%.4f"),
            },
            key="atom_editor",
        )

        if st.button("🏗️ 构建结构", type="primary", use_container_width=True):
            valid_data = atom_data.dropna(subset=["元素"])
            if valid_data.empty:
                st.warning("请至少输入一个原子")
            else:
                with st.spinner("构建中..."):
                    elements = valid_data["元素"].tolist()
                    frac_coords = valid_data[["x", "y", "z"]].values.tolist()

                    params = {
                        "a": a,
                        "b": b,
                        "c": c,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "elements": ",".join(elements),
                        "frac_coords": json.dumps(frac_coords),
                        "save_to_cif": save_cif,
                    }
                    if scaling_str:
                        params["scaling_matrix"] = scaling_str

                    result = build_structure(**params)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ 结构构建成功!")
                        # 存储结果到 session_state 用于显示
                        st.session_state.build_result = result

        st.markdown("</div>", unsafe_allow_html=True)

        # 显示构建结果
        if st.session_state.get("build_result"):
            result = st.session_state.build_result
            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            # 使用统一解析函数处理结果
            parsed = parse_mcp_result(result)

            if isinstance(parsed, dict):
                # 显示晶格参数
                if "lattice_parameters" in parsed:
                    st.markdown("**📐 晶格参数**")
                    lp = parsed["lattice_parameters"]
                    st.markdown(f"""
                    - a = {lp.get("a", "N/A")} Å
                    - b = {lp.get("b", "N/A")} Å
                    - c = {lp.get("c", "N/A")} Å
                    - α = {lp.get("alpha", "N/A")}°
                    - β = {lp.get("beta", "N/A")}°
                    - γ = {lp.get("gamma", "N/A")}°
                    """)

                # 显示空间群
                if "space_group_symbol" in parsed:
                    st.markdown(
                        f"**空间群:** {parsed.get('space_group_symbol', 'N/A')}"
                    )

                # 显示原子位置
                if "sites" in parsed:
                    with st.expander("🔬 原子位置"):
                        sites = parsed["sites"]
                        # 处理两种可能的格式
                        if sites and "fractional_coordinates" in sites[0]:
                            sites_df = pd.DataFrame(
                                [
                                    {
                                        "元素": s.get("element", "N/A"),
                                        "x": s.get("fractional_coordinates", [0, 0, 0])[
                                            0
                                        ],
                                        "y": s.get("fractional_coordinates", [0, 0, 0])[
                                            1
                                        ],
                                        "z": s.get("fractional_coordinates", [0, 0, 0])[
                                            2
                                        ],
                                    }
                                    for s in sites
                                ]
                            )
                        else:
                            sites_df = pd.DataFrame(
                                [
                                    {
                                        "元素": s.get("species", [{}])[0].get(
                                            "element", "N/A"
                                        ),
                                        "x": s.get("xyz", [0, 0, 0])[0],
                                        "y": s.get("xyz", [0, 0, 0])[1],
                                        "z": s.get("xyz", [0, 0, 0])[2],
                                    }
                                    for s in sites
                                ]
                            )
                        st.dataframe(sites_df, use_container_width=True)

                # 显示结构图片 (build_structure 返回的是 "image" 而不是 "image_url")
                image_url = parsed.get("image_url") or parsed.get("image")
                if image_url:
                    st.divider()
                    st.markdown("**📊 结构示意图**")
                    st.image(image_url, use_container_width=True)

                # 显示3D可视化链接
                if "3d_image_url" in parsed and parsed["3d_image_url"]:
                    st.divider()
                    st.markdown("**🔍 3D 结构可视化**")
                    url_3d = parsed["3d_image_url"]
                    # 确保URL是完整的
                    if url_3d.startswith("/"):
                        url_3d = f"http://127.0.0.1:5001{url_3d}"
                    st.markdown(f"[在浏览器中打开3D结构]({url_3d})")
                    # 使用iframe嵌入3D视图
                    st.components.v1.iframe(url_3d, height=500, scrolling=True)

                # 显示CIF文件路径
                if "cif_path" in parsed:
                    st.markdown(f"**💾 CIF 文件:** `{parsed['cif_path']}`")

                # 显示完整数据
                with st.expander("📋 完整数据"):
                    st.json(parsed)
            else:
                st.json(parsed)

            st.markdown("</div>", unsafe_allow_html=True)


def ml_prediction_page():
    st.markdown('<h1 class="main-title">🧪 ML 预测</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">基于机器学习的材料性质快速预测</p>', unsafe_allow_html=True
    )

    if not st.session_state.mcp_connected:
        st.warning("⚠️ MCP 服务未连接")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**🔮 带隙预测**")

        formula = st.text_input("化学式", placeholder="例如: SiO2, LiFePO4")

        if st.button("开始预测", type="primary", use_container_width=True):
            if not formula:
                st.warning("请输入化学式")
            else:
                with st.spinner("预测中..."):
                    result = predict_bandgap(formula)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state.prediction_result = result
                        st.success("预测完成!")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📊 预测结果**")

        if "prediction_result" in st.session_state:
            result = st.session_state.prediction_result
            # 解析 ML 预测结果
            try:
                # 使用统一解析函数处理结果（使用默认key="result"）
                parsed = parse_mcp_result(result)

                if isinstance(parsed, dict) and "predicted_band_gap" in parsed:
                    gap = parsed["predicted_band_gap"]
                    if isinstance(gap, list) and len(gap) > 0:
                        st.metric("预测带隙", f"{gap[0]:.4f} eV")
                    else:
                        st.metric("预测带隙", f"{gap} eV")
                else:
                    st.json(parsed)
            except Exception as e:
                st.error(f"解析预测结果失败: {e}")
                st.json(result)
        else:
            st.info("请输入化学式并点击预测")

        st.markdown("</div>", unsafe_allow_html=True)


def vasp_task_page():
    st.markdown('<h1 class="main-title">💻 VASP 任务管理</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">管理高性能计算任务</p>', unsafe_allow_html=True)

    if not st.session_state.mcp_connected:
        st.warning("⚠️ MCP 服务未连接")
        return

    tabs = st.tabs(
        [
            "📁 任务目录",
            "📊 任务队列",
            "➕ 创建任务",
            "📝 生成输入",
            "⚙️ 修改INCAR",
            "🚀 提交/提取",
        ]
    )

    # Tab 1: 任务目录
    with tabs[0]:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button("🔄 刷新目录", use_container_width=True):
                with st.spinner("获取中..."):
                    result = vasp_list_dirs()
                    if "error" not in result:
                        # 使用统一解析函数
                        parsed = parse_mcp_result(result)
                        directories = []
                        if isinstance(parsed, dict):
                            # MCP 工具返回的是 "task_directories" 键
                            directories = parsed.get("task_directories", [])
                        elif isinstance(parsed, list):
                            directories = parsed

                        st.session_state.task_dirs = directories
                        st.success(f"刷新成功，共 {len(directories)} 个目录")
                    else:
                        st.error(result["error"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if "task_dirs" in st.session_state:
                dirs = st.session_state.task_dirs
                if dirs:
                    st.markdown(f"**共 {len(dirs)} 个任务目录**")
                    for d in dirs:
                        # 显示时只显示目录名
                        display_name = d.rstrip("/").split("/")[-1] if "/" in d else d
                        st.markdown(f"- `{display_name}`")
                else:
                    st.info("暂无任务目录")
            else:
                st.info("点击刷新获取任务目录")
            st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: 任务队列
    with tabs[1]:
        if st.button("🔄 刷新队列"):
            with st.spinner("获取中..."):
                result = vasp_check_queue()
                if "error" not in result:
                    # 使用统一解析函数
                    parsed = parse_mcp_result(result)
                    squeue_text = ""
                    if isinstance(parsed, dict):
                        squeue_text = parsed.get("squeue", "")
                    elif isinstance(parsed, str):
                        squeue_text = parsed
                    st.session_state.squeue = squeue_text

        if "squeue" in st.session_state:
            st.code(st.session_state.squeue or "无运行任务")

    # Tab 3: 创建任务
    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            formula = st.text_input("化学式", placeholder="SiO2")
            cif_path = st.text_input("CIF 文件路径", placeholder="./cifs/SiO2.cif")

        with col2:
            st.markdown("**说明**")
            st.info("创建任务目录并上传 CIF 文件到远程服务器")

        if st.button("✅ 创建任务", type="primary"):
            if formula and cif_path:
                with st.spinner("创建中..."):
                    result = vasp_create_task(formula, cif_path)
                    if "error" not in result:
                        st.success("创建成功!")
                        # 使用统一解析函数处理结果
                        parsed = parse_mcp_result(result)
                        st.json(parsed)
                    else:
                        st.error(result["error"])
            else:
                st.warning("请填写所有字段")

    # Tab 4: 生成输入
    with tabs[3]:
        col1, col2 = st.columns(2)

        with col1:
            task_dirs_full = st.session_state.get("task_dirs", [])
            task_dir = st.selectbox(
                "选择任务目录",
                task_dirs_full,
                format_func=lambda x: x.rstrip("/").split("/")[-1] if "/" in x else x,
                key="gen_input_dir",
            )
            mission_type = st.selectbox("计算类型", ["relax", "scf", "band", "dos"])

        with col2:
            st.markdown("**计算类型说明**")
            st.markdown("""
            - **relax**: 结构优化
            - **scf**: 自洽计算
            - **band**: 能带计算
            - **dos**: 态密度计算
            """)

        if st.button("🔨 生成输入文件", type="primary"):
            if task_dir:
                with st.spinner("生成中..."):
                    result = vasp_create_mission(task_dir, mission_type)
                    if "error" not in result:
                        st.success("生成成功!")
                        # 使用统一解析函数处理结果
                        parsed = parse_mcp_result(result)
                        st.json(parsed)
                    else:
                        st.error(result["error"])

    # Tab 5: 修改 INCAR
    with tabs[4]:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**⚙️ 选择任务**")

            task_dirs_full = st.session_state.get("task_dirs", [])
            incar_dir = st.selectbox(
                "任务目录",
                task_dirs_full,
                format_func=lambda x: x.rstrip("/").split("/")[-1] if "/" in x else x,
                key="incar_dir",
            )
            incar_mission = st.selectbox(
                "计算类型", ["relax", "scf", "band", "dos"], key="incar_mission"
            )

            if st.button("📖 读取 INCAR", type="primary", use_container_width=True):
                if incar_dir:
                    with st.spinner("读取中..."):
                        result = vasp_modify_incar(
                            incar_dir, incar_mission, "__read__", ""
                        )
                        if "error" not in result:
                            # 使用统一解析函数处理结果
                            parsed = parse_mcp_result(result)
                            st.session_state.incar_content = parsed
                            st.success("读取成功!")
                        else:
                            st.error(result.get("error", "读取失败"))
                            if result.get("hint"):
                                st.info(result["hint"])

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**✏️ 编辑 INCAR**")

            if "incar_content" in st.session_state:
                incar_data = st.session_state.incar_content

                # 显示当前 INCAR 参数
                if isinstance(incar_data, dict) and "incar_params" in incar_data:
                    incar_params = incar_data["incar_params"]

                    # 过滤掉 pymatgen 的内部字段
                    display_params = {
                        k: v
                        for k, v in incar_params.items()
                        if not k.startswith("@") and not k.startswith("_")
                    }

                    st.markdown(
                        f"**任务:** `{incar_data.get('task_directory', 'N/A')}`"
                    )
                    st.markdown(f"**计算类型:** `{incar_data.get('mission', 'N/A')}`")

                    # 转换为 DataFrame 用于表格编辑
                    import pandas as pd

                    params_df = pd.DataFrame(
                        [{"参数": k, "值": str(v)} for k, v in display_params.items()]
                    )

                    st.markdown("**修改参数:**")
                    edited_df = st.data_editor(
                        params_df,
                        num_rows="dynamic",
                        use_container_width=True,
                        key="incar_editor",
                    )

                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button(
                            "💾 保存修改", type="primary", use_container_width=True
                        ):
                            with st.spinner("保存中..."):
                                # 将表格转换回 INCAR 格式
                                write_lines = []
                                for _, row in edited_df.iterrows():
                                    if row.get("参数") and row.get("值") is not None:
                                        write_lines.append(
                                            f"{row['参数']} = {row['值']}"
                                        )
                                write_text = "\n".join(write_lines)

                                result = vasp_modify_incar(
                                    incar_dir, incar_mission, "__write__", write_text
                                )
                                if "error" not in result:
                                    st.success("INCAR 已保存!")
                                    del st.session_state.incar_content
                                    st.rerun()
                                else:
                                    st.error(result.get("error", "保存失败"))

                    with col_btn2:
                        if st.button("❌ 取消", use_container_width=True):
                            del st.session_state.incar_content
                            st.rerun()
                elif isinstance(incar_data, dict) and "incar" in incar_data:
                    # 兼容旧格式
                    incar_params = incar_data["incar"]
                    display_params = {
                        k: v
                        for k, v in incar_params.items()
                        if not k.startswith("@") and not k.startswith("_")
                    }

                    import pandas as pd

                    params_df = pd.DataFrame(
                        [{"参数": k, "值": str(v)} for k, v in display_params.items()]
                    )

                    st.markdown("**修改参数:**")
                    edited_df = st.data_editor(
                        params_df,
                        num_rows="dynamic",
                        use_container_width=True,
                        key="incar_editor",
                    )

                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button(
                            "💾 保存修改", type="primary", use_container_width=True
                        ):
                            with st.spinner("保存中..."):
                                write_lines = []
                                for _, row in edited_df.iterrows():
                                    if row.get("参数") and row.get("值") is not None:
                                        write_lines.append(
                                            f"{row['参数']} = {row['值']}"
                                        )
                                write_text = "\n".join(write_lines)

                                result = vasp_modify_incar(
                                    incar_dir, incar_mission, "__write__", write_text
                                )
                                if "error" not in result:
                                    st.success("INCAR 已保存!")
                                    del st.session_state.incar_content
                                    st.rerun()
                                else:
                                    st.error(result.get("error", "保存失败"))

                    with col_btn2:
                        if st.button("❌ 取消", use_container_width=True):
                            del st.session_state.incar_content
                            st.rerun()
                else:
                    st.json(incar_data)
            else:
                st.info("请先选择任务并读取 INCAR 文件")

            st.markdown("</div>", unsafe_allow_html=True)

    # Tab 6: 提交/提取
    with tabs[5]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🚀 提交任务**")
            task_dirs_full = st.session_state.get("task_dirs", [])
            submit_dir = st.selectbox(
                "任务目录",
                task_dirs_full,
                format_func=lambda x: x.rstrip("/").split("/")[-1] if "/" in x else x,
                key="submit_dir",
            )
            submit_mission = st.selectbox(
                "计算类型", ["relax", "scf", "band", "dos"], key="submit_mission"
            )

            if st.button("🚀 提交", type="primary", use_container_width=True):
                if submit_dir:
                    with st.spinner("提交中..."):
                        result = vasp_submit(submit_dir, submit_mission)
                        if "error" not in result:
                            st.success("提交成功!")
                            # 使用统一解析函数处理结果
                            parsed = parse_mcp_result(result)
                            st.json(parsed)
                        else:
                            st.error(result["error"])

        with col2:
            st.markdown("**📥 提取结果**")
            task_dirs_full = st.session_state.get("task_dirs", [])
            extract_dir = st.selectbox(
                "任务目录",
                task_dirs_full,
                format_func=lambda x: x.rstrip("/").split("/")[-1] if "/" in x else x,
                key="extract_dir",
            )
            extract_mission = st.selectbox(
                "计算类型", ["relax", "scf", "band", "dos"], key="extract_mission"
            )
            plot_result = st.checkbox("生成图表", value=True)

            if st.button("📥 提取", type="primary", use_container_width=True):
                if extract_dir:
                    with st.spinner("提取中..."):
                        result = vasp_extract(extract_dir, extract_mission, plot_result)
                        if "error" not in result:
                            st.success("提取成功!")
                            # 使用统一解析函数处理结果
                            parsed = parse_mcp_result(result)
                            st.json(parsed)
                        else:
                            st.error(result["error"])


# ============ 主程序 ============


def main():
    page = sidebar()

    if page == "💬 智能体对话":
        chat_page()
    elif page == "🔍 材料查询":
        material_search_page()
    elif page == "📊 结构建模":
        structure_builder_page()
    elif page == "🧪 ML 预测":
        ml_prediction_page()
    elif page == "💻 VASP 任务":
        vasp_task_page()


if __name__ == "__main__":
    main()
