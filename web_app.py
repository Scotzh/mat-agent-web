import streamlit as st
import os
import sys
import uuid
import time
from datetime import datetime

st.set_page_config(
    page_title="MatAgent 材料智能设计平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化数据库（用于材料存储）
import databasemanage
if "db" not in st.session_state:
    st.session_state.db = databasemanage.DatabaseManager("matagent.db")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "langchain_connected" not in st.session_state:
    st.session_state.langchain_connected = False

if "selected_material" not in st.session_state:
    st.session_state.selected_material = None

# 加载最近会话（每次页面加载时检查）
current_session_id = st.session_state.get("session_id")
recent_sessions = databasemanage.list_sessions(limit=1)
recent_session_id = recent_sessions[0]["session_id"] if recent_sessions else None

# 辅助函数：加载历史消息（现在 tool_results 已存储在 content 字段中）
def load_history_with_tools(session_id):
    history = databasemanage.get_chat_history(session_id)
    
    messages = []
    for h in history:
        msg = {
            "role": h["role"],
            "content": h["content"],
            "timestamp": h["timestamp"]
        }
        if h.get("tool_results"):
            msg["tool_results"] = h["tool_results"]
        messages.append(msg)
    
    return messages

# 如果没有session_id，或者消息为空时从数据库加载
if not current_session_id:
    if recent_session_id:
        st.session_state.session_id = recent_session_id
        st.session_state.messages = load_history_with_tools(recent_session_id)
    else:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
elif not st.session_state.get("messages"):
    st.session_state.messages = load_history_with_tools(current_session_id)

# 页面加载时自动同步历史消息到 Agent
if st.session_state.get("agent") and st.session_state.get("messages"):
    from langchain_core.messages import HumanMessage, AIMessage
    # 只在 Agent 消息为空或数量不匹配时同步
    if len(st.session_state.agent._message_history) != len(st.session_state.messages):
        st.session_state.agent._message_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.session_state.agent._message_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                st.session_state.agent._message_history.append(AIMessage(content=msg["content"]))

# 确保每次页面加载时同步历史消息到 Agent
def _sync_agent_history():
    """同步消息历史到 Agent"""
    if st.session_state.get("agent") and st.session_state.get("messages"):
        from langchain_core.messages import HumanMessage, AIMessage
        st.session_state.agent._message_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.session_state.agent._message_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                st.session_state.agent._message_history.append(AIMessage(content=msg["content"]))

st.markdown(
    """
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1E88E5, #00ACC1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        color: #1E88E5;
    }
    .stButton>button {
        width: 100%;
    }
    .function-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        margin: 10px 0;
    }
    .error-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffebee;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


import requests

# Agent 服务配置
AGENT_SERVER_URL = "http://127.0.0.1:8765"

def check_agent_server():
    """检查 Agent 服务是否运行"""
    try:
        response = requests.get(f"{AGENT_SERVER_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get("agent_ready")
    except:
        return False

def init_langchain_agent():
    """连接到 Agent 服务"""
    try:
        # 等待服务就绪
        max_retries = 30
        for i in range(max_retries):
            if check_agent_server():
                st.session_state.langchain_connected = True
                return True
            if i == 0:
                st.info("⏳ 正在连接 Agent 服务...")
            time.sleep(0.5)
        
        st.error("❌ 无法连接到 Agent 服务，请确保已运行: python agent_server.py")
        st.session_state.langchain_connected = False
        return False
    except Exception as e:
        st.error(f"Agent 连接失败: {e}")
        st.session_state.langchain_connected = False
        return False

def chat_with_agent(message: str, history: list = None) -> dict:
    """通过 HTTP API 与 Agent 对话"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": message,
                "history": history
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"type": "error", "message": "Agent 服务未运行，请启动: python agent_server.py"}
    except Exception as e:
        return {"type": "error", "message": f"请求失败: {str(e)}"}

def predict_bandgap_via_agent(formula: str) -> dict:
    """通过 HTTP API 预测带隙"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/predict_bandgap",
            params={"formula": formula},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ========== VASP HTTP 客户端函数 ==========

def vasp_list_task_directories() -> dict:
    """列出任务目录"""
    try:
        response = requests.get(f"{AGENT_SERVER_URL}/vasp/task_directories", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def vasp_check_squeue() -> dict:
    """检查 Slurm 队列"""
    try:
        response = requests.get(f"{AGENT_SERVER_URL}/vasp/squeue", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def vasp_create_task(formula: str, cif_path: str) -> dict:
    """创建任务目录"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/vasp/create_task",
            params={"formula": formula, "cif_path": cif_path},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def vasp_create_mission(task_directory: str, mission_type: str) -> dict:
    """生成任务输入文件"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/vasp/create_mission",
            params={"task_directory": task_directory, "mission_type": mission_type},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def vasp_modify_incar(task_directory: str, mission: str, key: str, value: str) -> dict:
    """修改 INCAR 参数"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/vasp/modify_incar",
            params={"task_directory": task_directory, "mission": mission, "key": key, "value": value},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def vasp_submit_mission(task_directory: str, mission: str) -> dict:
    """提交任务"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/vasp/submit",
            params={"task_directory": task_directory, "mission": mission},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def vasp_extract_result(task_directory: str, mission: str, plot: bool = False) -> dict:
    """提取结果"""
    try:
        response = requests.post(
            f"{AGENT_SERVER_URL}/vasp/extract",
            params={"task_directory": task_directory, "mission": mission, "plot": plot},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}



def sidebar_functions():
    with st.sidebar:
        st.markdown('<p class="sidebar-title">📁 功能面板</p>', unsafe_allow_html=True)

        function_tabs = st.radio(
            "选择功能",
            [
                "💬 AI对话",
                "🔍 材料查询",
                "📊 结构建模",
                "🧪 ML预测",
                "💻 VASP任务",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # 自动初始化 Agent（如果未初始化）
        if not st.session_state.langchain_connected:
            with st.spinner("🚀 正在自动初始化 Agent..."):
                init_langchain_agent()
        
        if st.session_state.langchain_connected:
            st.success("✅ Agent 已就绪")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 重置对话"):
                    # 先保存当前会话消息
                    if st.session_state.get("messages"):
                        for msg in st.session_state.messages:
                            tool_results = msg.get("tool_results")
                            databasemanage.add_chat_message(
                                st.session_state.session_id, msg["role"], msg["content"], tool_results
                            )
                    if st.session_state.agent:
                        st.session_state.agent._message_history = []
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("✨ 新建会话"):
                    # 先保存当前会话消息
                    if st.session_state.get("messages"):
                        for msg in st.session_state.messages:
                            tool_results = msg.get("tool_results")
                            databasemanage.add_chat_message(
                                st.session_state.session_id, msg["role"], msg["content"], tool_results
                            )
                    # 清除当前会话的工具调用记录
                    databasemanage.clear_tool_calls(st.session_state.session_id)
                    # 清除当前会话消息
                    st.session_state.messages = []
                    # 生成新会话ID
                    st.session_state.session_id = str(uuid.uuid4())
                    # 重置 Agent 消息
                    if st.session_state.agent:
                        st.session_state.agent._message_history = []
                    st.rerun()
            
            # 显示当前会话ID
            st.caption(f"会话ID: {st.session_state.session_id[:8]}...")

        st.divider()
        
        # 会话历史列表
        st.markdown("**📜 历史会话**")
        sessions = databasemanage.list_sessions(limit=10)
        if sessions:
            for s in sessions:
                # 显示会话名称或ID
                display_name = s.get("session_name") or f"会话 {s['session_id'][:8]}"
                session_label = f"{display_name} | {s['last_time']}"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(session_label, key=f"session_{s['session_id']}"):
                        # 切换到该会话
                        st.session_state.session_id = s["session_id"]
                        st.session_state.messages = load_history_with_tools(s["session_id"])
                        
                        # 更新 Agent 的消息历史
                        if st.session_state.agent:
                            st.session_state.agent._message_history = []
                            from langchain_core.messages import HumanMessage, AIMessage
                            for msg in st.session_state.messages:
                                if msg["role"] == "user":
                                    st.session_state.agent._message_history.append(HumanMessage(content=msg["content"]))
                                elif msg["role"] == "assistant":
                                    st.session_state.agent._message_history.append(AIMessage(content=msg["content"]))
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{s['session_id']}"):
                        databasemanage.delete_session(s["session_id"])
                        # 如果删除的是当前会话，清空消息
                        if st.session_state.get("session_id") == s["session_id"]:
                            st.session_state.messages = []
                            st.session_state.session_id = str(uuid.uuid4())
                        st.rerun()
            
            # 重命名当前会话
            if st.session_state.get("session_id"):
                current_name = databasemanage.get_session_name(st.session_state.session_id)
                new_name = st.text_input(
                    "重命名当前会话",
                    value=current_name or "",
                    placeholder="输入会话名称",
                    key="session_name_input"
                )
                if new_name and new_name != current_name:
                    if st.button("保存名称"):
                        databasemanage.update_session_name(st.session_state.session_id, new_name)
                        st.rerun()
        else:
            st.info("暂无历史会话")

        st.divider()

        st.markdown("**💡 使用提示**")
        st.info("""
        - 使用 AI 对话：用自然语言描述需求
        - 手动操作：选择对应功能面板
        - 支持材料搜索、结构建模、带隙预测、VASP任务管理
        """)

        return function_tabs


def chat_interface():
    st.markdown('<p class="main-header">💬 AI 对话</p>', unsafe_allow_html=True)

    # 立即显示历史消息（页面加载时）
    if st.session_state.get("messages"):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                # 先显示工具调用（如果有）
                if msg.get("tool_results"):
                    for tr in msg["tool_results"]:
                        tool_name = tr.get("tool_name", "")
                        tool_args = tr.get("tool_args", {})
                        result = tr.get("result")
                        
                        with st.expander(f"🔧 {tool_name}"):
                            st.markdown(f"**参数:** `{tool_args}`")
                            
                            if isinstance(result, dict):
                                img_url = result.get("2d_image_url", "")
                                if img_url:
                                    st.markdown("**2D 结构图:**")
                                    try:
                                        st.image(img_url, width=400)
                                    except Exception as e:
                                        st.error(f"图片路径: {img_url}, 错误: {e}")
                                
                                if result.get("3d_html_url"):
                                    st.markdown(f"**3D 结构图:** [查看3D结构]({result['3d_html_url']})")
                                
                                json_data = {k: v for k, v in result.items() 
                                            if k not in ["2d_image_url", "3d_html_url", "cif_path"]}
                                if json_data:
                                    st.json(json_data)
                            else:
                                st.code(str(result))
                
                # 再显示消息内容
                st.markdown(msg["content"])

    # 处理待处理的消息（用于实时显示工具调用）
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 保存用户消息到数据库
        databasemanage.add_chat_message(
            st.session_state.session_id, "user", prompt, None
        )
        
        # 在 assistant 消息位置显示
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 创建占位符用于实时显示
        tool_placeholder = st.empty()
        
        with st.chat_message("assistant"):
            with st.spinner("AI 正在思考..."):
                if st.session_state.langchain_connected:
                    # 准备历史消息
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[:-1]  # 排除刚添加的用户消息
                    ]
                    
                    result = chat_with_agent(prompt, history)
                    
                    # 立即显示工具调用
                    if result.get("type") == "tool_calls":
                        tool_results = result.get("tool_results", [])
                        for tr in tool_results:
                            tool_name = tr.get("tool_name", "")
                            tool_args = tr.get("tool_args", {})
                            result_val = tr.get("result")
                            
                            with st.expander(f"🔧 {tool_name}", expanded=True):
                                st.markdown(f"**参数:** `{tool_args}`")
                                
                                if isinstance(result_val, dict):
                                    img_url = result_val.get("2d_image_url", "")
                                    if img_url:
                                        st.markdown("**2D 结构图:**")
                                        try:
                                            st.image(img_url, width=400)
                                        except Exception as e:
                                            st.error(f"加载图片失败: {e}")
                                    
                                    if result_val.get("3d_html_url"):
                                        st.markdown(f"**3D 结构图:** [查看3D结构]({result_val['3d_html_url']})")
                                    
                                    json_data = {k: v for k, v in result_val.items() 
                                                if k not in ["2d_image_url", "3d_html_url", "cif_path"]}
                                    if json_data:
                                        st.json(json_data)
                                else:
                                    st.code(str(result_val))
                        
                        final_msg = result.get("message", "")
                        if final_msg:
                            st.markdown(final_msg)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_msg,
                            "tool_results": tool_results
                        })
                        # 保存助手回复到数据库（包含工具调用）
                        databasemanage.add_chat_message(
                            st.session_state.session_id, "assistant", final_msg, tool_results
                        )
                    elif result.get("type") == "error":
                        st.error(result.get("message", "请求失败"))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result.get("message", "请求失败")
                        })
                    else:
                        msg = result.get("message", "")
                        st.markdown(msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": msg
                        })
                        # 保存助手回复到数据库
                        databasemanage.add_chat_message(
                            st.session_state.session_id, "assistant", msg
                        )
                else:
                    st.error("❌ Agent 服务未连接")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "请先在左侧点击「初始化 Agent」",
                    })
                    
                    # 保存到数据库
                    databasemanage.add_chat_message(
                        st.session_state.session_id, "user", prompt, None
                    )
                    databasemanage.add_chat_message(
                        st.session_state.session_id, "assistant", "请先在左侧点击「初始化 Agent」", None
                    )

    if prompt := st.chat_input("描述您的材料科学需求..."):
        st.session_state["pending_prompt"] = prompt
        st.rerun()


def material_search_panel():
    st.markdown("### 🔍 材料查询")

    col1, col2 = st.columns(2)

    with col1:
        elements_input = st.text_input("包含元素 (用逗号分隔)", placeholder="Li, Co, O")
        exclude_elements_input = st.text_input(
            "排除元素 (用逗号分隔)", placeholder="H, He"
        )
        chemsys = st.text_input("化学系统", placeholder="Li-Fe-O")

    with col2:
        band_gap_min = st.number_input("带隙最小值 (eV)", min_value=0.0, value=0.0)
        band_gap_max = st.number_input("带隙最大值 (eV)", min_value=0.0, value=5.0)
        chunk_size = st.slider("返回结果数量", 1, 100, 25)

    if st.button("🔍 搜索材料", type="primary"):
        if not st.session_state.langchain_connected:
            st.error("请先初始化 Agent")
            return

        elements = [e.strip() for e in elements_input.split(",") if e.strip()]
        exclude_elements = [e.strip() for e in exclude_elements_input.split(",") if e.strip()]

        if not elements and not exclude_elements and not chemsys and band_gap_min == 0 and band_gap_max == 5:
            st.warning("请至少输入一个搜索条件（元素、排除元素、化学系统或带隙范围）")
            return

        with st.spinner("搜索中..."):
            try:
                result = st.session_state.agent.search_materials(
                    elements=elements if elements else None,
                    exclude_elements=exclude_elements if exclude_elements else None,
                    chemsys=chemsys if chemsys else None,
                    band_gap=(band_gap_min, band_gap_max)
                    if band_gap_min > 0 or band_gap_max < 5
                    else None,
                    chunk_size=chunk_size,
                )

                st.session_state.search_results = result

            except Exception as e:
                st.error(f"搜索失败: {e}")

    if "search_results" in st.session_state:
        results = st.session_state.search_results
        # 处理字典格式 {"materials": [...]} 或列表格式 [...]
        if isinstance(results, dict):
            materials_list = results.get("materials", [])
            # 检查是否有错误
            if "error" in results:
                st.error(f"搜索出错: {results.get('error')}")
                return
        elif isinstance(results, list):
            materials_list = results
        else:
            materials_list = []
        
        if materials_list:
            st.markdown(f"**找到 {len(materials_list)} 个材料:**")
            for r in materials_list:
                with st.expander(
                    f"{r.get('formula_pretty', 'Unknown')} ({r.get('material_id', 'N/A')})"
                ):
                    col1, col2 = st.columns(2)
                    col1.markdown(f"**带隙:** {r.get('band_gap', 'N/A')} eV")
                    col1.markdown(f"**对称性:** {r.get('symmetry', 'N/A')}")
                    
                    mat_id = r.get("material_id")
                    if col2.button("📊 查看结构", key=f"view_{mat_id}"):
                        if st.session_state.langchain_connected:
                            with st.spinner("获取结构中..."):
                                try:
                                    result = st.session_state.agent.get_material_structure(
                                        material_id=mat_id,
                                        get_sites=True,
                                        get_plot=True,
                                        download=False
                                    )
                                    if "error" in result:
                                        st.error(result.get("error"))
                                    else:
                                        st.session_state.viewed_structure = result
                                except Exception as e:
                                    st.error(f"获取失败: {e}")
                    
                    if "viewed_structure" in st.session_state and st.session_state.get("viewed_structure", {}).get("material_id") == mat_id:
                        struct = st.session_state.viewed_structure
                        st.divider()
                        st.markdown("**📐 晶格参数**")
                        lp = struct.get("lattice_parameters", {})
                        st.markdown(f"""
                        - a = {lp.get('a', 'N/A')} Å
                        - b = {lp.get('b', 'N/A')} Å  
                        - c = {lp.get('c', 'N/A')} Å
                        - α = {lp.get('alpha', 'N/A')}°
                        - β = {lp.get('beta', 'N/A')}°
                        - γ = {lp.get('gamma', 'N/A')}°
                        """)
                        st.markdown(f"**空间群:** {struct.get('space_group_symbol', 'N/A')} ({struct.get('space_group_number', 'N/A')})")
                        st.markdown(f"**原子数:** {struct.get('number_of_sites', 'N/A')}")
                        
                        if struct.get("2d_image_url"):
                            st.markdown("**🖼️ 2D 结构图**")
                            st.image(struct.get("2d_image_url"), width=400)
                        
                        if struct.get("3d_html_url"):
                            st.markdown("**🌐 3D 可视化**")
                            st.markdown(f"[点击查看 3D 结构]({struct.get('3d_html_url')})")
                        elif struct.get("3d_image_url"):
                            st.markdown("**🌐 3D 可视化**")
                            st.markdown(f"[点击查看 3D 结构]({struct.get('3d_image_url')})")
                        
                        if struct.get("cif_path"):
                            st.markdown("**📥 下载 CIF 文件**")
                            with open(struct.get("cif_path"), "r") as f:
                                st.download_button(
                                    label="下载 CIF",
                                    data=f.read(),
                                    file_name=f"{mat_id}.cif",
                                    mime="chemical/x-cif"
                                )

        elif isinstance(results, dict) and "error" in results:
            st.error(results.get("error"))
        
        if not materials_list:
            st.info("未找到符合条件的材料，请尝试调整搜索条件")


def structure_builder_panel():
    st.markdown("### 📊 结构建模")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**晶格参数**")
        a = st.number_input("a (Å)", value=5.0, min_value=0.01, step=0.1, format="%.4f")
        b = st.number_input("b (Å)", value=5.0, min_value=0.01, step=0.1, format="%.4f")
        c = st.number_input("c (Å)", value=5.0, min_value=0.01, step=0.1, format="%.4f")
        alpha = st.number_input("α (°)", value=90.0, min_value=0.0, max_value=180.0, step=0.1, format="%.2f")
        beta = st.number_input("β (°)", value=90.0, min_value=0.0, max_value=180.0, step=0.1, format="%.2f")
        gamma = st.number_input("γ (°)", value=90.0, min_value=0.0, max_value=180.0, step=0.1, format="%.2f")

    with col2:
        st.markdown("**原子信息**")
        elements_input = st.text_input("元素符号 (逗号分隔)", placeholder="Na, Cl")
        coords_input = st.text_area(
            "分数坐标 (每行一个坐标，用逗号分隔)",
            placeholder="0, 0, 0\n0.5, 0.5, 0.5",
            height=150,
        )
        scaling = st.number_input("超胞扩展因子", min_value=1, value=1, step=1)
        save_cif = st.checkbox("保存为 CIF 文件", value=True)

    if st.button("🏗️ 构建结构", type="primary"):
        if not st.session_state.langchain_connected:
            st.error("请先初始化 Agent")
            return

        if not elements_input or not coords_input:
            st.warning("请输入元素符号和分数坐标")
            return

        try:
            elements = [e.strip() for e in elements_input.split(",") if e.strip()]
            frac_coords = []
            for line in coords_input.strip().split("\n"):
                if line.strip():
                    coords = [float(x.strip()) for x in line.split(",")]
                    frac_coords.append(coords)

            if not elements or not frac_coords:
                st.warning("元素符号和分数坐标不能为空")
                return

            result = st.session_state.agent.build_structure(
                a=a, b=b, c=c,
                alpha=alpha, beta=beta, gamma=gamma,
                elements=elements,
                frac_coord=frac_coords,
                scaling_matrix=scaling,
                save_to_cif=save_cif
            )

            if isinstance(result, dict):
                if "error" in result:
                    st.error(result.get("error"))
                else:
                    st.success("结构构建成功!")
                    st.json(result)
            else:
                st.success("结构构建成功!")
                st.write(result)

        except Exception as e:
            st.error(f"构建失败: {e}")


def ml_prediction_panel():
    st.markdown("### 🧪 ML 预测")

    tab1, tab2 = st.tabs(["带隙预测", "更多预测"])

    with tab1:
        formula = st.text_input("化学式", placeholder="LiFePO4")

        if st.button("🔮 预测带隙", type="primary"):
            if not st.session_state.langchain_connected:
                st.error("请先初始化 Agent")
                return

            if not formula:
                st.warning("请输入化学式")
                return

            with st.spinner("预测中..."):
                try:
                    result = predict_bandgap_via_agent(formula)
                    if isinstance(result, dict):
                        if "error" in result:
                            st.error(result.get("error"))
                        else:
                            st.success("预测完成!")
                            st.json(result.get("prediction", result))
                    else:
                        st.write(result)
                except Exception as e:
                    st.error(f"预测失败: {e}")

    with tab2:
        st.info("更多 ML 预测功能开发中...")


def vasp_task_panel():
    st.markdown("### 💻 VASP 任务")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 历史任务", "📊 任务队列", "➕ 创建任务", "📝 生成输入", "⚙️ 修改INCAR", "🚀 提交/提取"
    ])
    
    # ========== Tab 1: 历史任务目录 ==========
    with tab1:
        st.markdown("**📁 历史任务目录**")
        if st.button("🔄 刷新任务列表", key="refresh_tasks"):
            if st.session_state.langchain_connected:
                result = vasp_list_task_directories()
                if "error" in result:
                    st.error(result.get("error"))
                else:
                    st.session_state.task_dirs = result.get("task_directories", [])
        
        if "task_dirs" in st.session_state:
            dirs = st.session_state.task_dirs
            if dirs:
                selected_dir = st.selectbox("选择任务目录", dirs, key="select_task_dir")
                st.text(f"已选择: {selected_dir}")
            else:
                st.info("暂无任务目录")
    
    # ========== Tab 2: 任务队列 ==========
    with tab2:
        st.markdown("**📊 Slurm 任务队列**")
        if st.button("🔄 刷新队列", key="refresh_queue"):
            if st.session_state.langchain_connected:
                result = vasp_check_squeue()
                if "error" in result:
                    st.error(result.get("error"))
                else:
                    st.session_state.squeue = result.get("squeue", "")
        
        if "squeue" in st.session_state:
            st.text(st.session_state.squeue or "无运行任务")
    
    # ========== Tab 3: 创建任务目录 ==========
    with tab3:
        st.markdown("**➕ 创建任务目录并上传 CIF**")
        
        col1, col2 = st.columns(2)
        with col1:
            formula = st.text_input("化学式", placeholder="SiO2")
            cif_local_path = st.text_input("本地 CIF 路径", placeholder="cifs/SiO2.cif")
        
        with col2:
            task_dir_name = st.text_input("任务目录名称", placeholder="SiO2_relax")
        
        if st.button("✅ 创建任务目录", key="create_task", type="primary"):
            if not st.session_state.langchain_connected:
                st.error("请先初始化 Agent")
            elif not formula or not cif_local_path or not task_dir_name:
                st.warning("请填写所有字段")
            else:
                with st.spinner("创建中..."):
                    result = vasp_create_task(formula, cif_local_path)
                    if "error" in result:
                        st.error(result.get("error"))
                    else:
                        st.success("任务创建成功!")
                        st.json(result)
    
    # ========== Tab 4: 生成计算任务输入文件 ==========
    with tab4:
        st.markdown("**📝 生成计算任务输入文件**")
        
        selected_dir = st.selectbox(
            "选择任务目录",
            st.session_state.get("task_dirs", []),
            key="create_mission_dir"
        )
        mission_type = st.selectbox(
            "计算任务类型",
            ["relax", "scf", "band", "dos"],
            key="create_mission_type"
        )
        
        if st.button("🔨 生成输入文件", key="gen_inputs", type="primary"):
            if not st.session_state.langchain_connected:
                st.error("请先初始化 Agent")
            elif not selected_dir:
                st.warning("请选择任务目录")
            else:
                with st.spinner("生成中..."):
                    result = vasp_create_mission(selected_dir, mission_type)
                    if "error" in result:
                        st.error(result.get("error"))
                    else:
                        st.success("输入文件生成成功!")
                        st.json(result)
    
    # ========== Tab 5: 修改 INCAR ==========
    with tab5:
        st.markdown("**⚙️ 修改 INCAR 文件**")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_dir = st.selectbox(
                "任务目录",
                st.session_state.get("task_dirs", []),
                key="incar_dir"
            )
            mission_type = st.selectbox(
                "计算类型",
                ["relax", "scf", "band", "dos"],
                key="incar_type"
            )
        
        if not selected_dir:
            st.warning("请先在「历史任务」标签页中选择任务目录")
        else:
            with col2:
                if st.button("📖 加载 INCAR", key="load_incar"):
                    if st.session_state.langchain_connected:
                        result = vasp_modify_incar(selected_dir, mission_type, "__read__", "")
                        if "error" in result:
                            st.error(result.get("error"))
                            if result.get("hint"):
                                st.info(result.get("hint"))
                        else:
                            incar_dict = result.get("incar", {})
                            rows = []
                            for k, v in incar_dict.items():
                                if not k.startswith("@"):
                                    rows.append({"参数": k, "值": str(v)})
                            st.session_state.incar_rows = rows
            
            if "incar_rows" in st.session_state and st.session_state.incar_rows:
                st.markdown("**📝 直接编辑参数（双击单元格修改）**")
                edited_df = st.data_editor(
                    st.session_state.incar_rows,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    key="incar_editor"
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💾 保存修改", key="save_incar", type="primary"):
                        if st.session_state.langchain_connected:
                            write_str = ""
                            for row in edited_df:
                                if row.get("参数") and row.get("值"):
                                    write_str += f"{row['参数']} = {row['值']}\n"
                            result = vasp_modify_incar(selected_dir, mission_type, "__write__", write_str)
                            if "error" in result:
                                st.error(result.get("error"))
                            else:
                                st.success("INCAR 已保存!")
                with col_btn2:
                    if st.button("🔄 重新加载", key="reload_incar"):
                        st.rerun()
                
                with st.expander("➕ 添加新参数"):
                    new_param = st.text_input("参数名", placeholder="ENCUT")
                    new_value = st.text_input("参数值", placeholder="520")
                    if st.button("➕ 添加参数"):
                        if new_param and new_value:
                            st.session_state.incar_rows.append({"参数": new_param, "值": new_value})
                            st.rerun()
            else:
                st.info("请点击「加载 INCAR」读取现有参数")
    
    # ========== Tab 6: 提交任务 / 提取结果 ==========
    with tab6:
        st.markdown("**🚀 提交任务 & 提取结果**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 提交任务")
            submit_dir = st.selectbox(
                "任务目录",
                st.session_state.get("task_dirs", []),
                key="submit_dir"
            )
            submit_mission = st.selectbox(
                "计算类型",
                ["relax", "scf", "band", "dos"],
                key="submit_mission_type"
            )
            
            if st.button("🚀 提交任务", key="submit_task", type="primary"):
                if st.session_state.langchain_connected and submit_dir:
                    with st.spinner("提交中..."):
                        result = vasp_submit_mission(submit_dir, submit_mission)
                        if "error" in result:
                            st.error(result.get("error"))
                        else:
                            st.success("任务提交成功!")
                            st.json(result)
        
        with col2:
            st.markdown("### 提取结果")
            extract_dir = st.selectbox(
                "任务目录",
                st.session_state.get("task_dirs", []),
                key="extract_dir"
            )
            extract_mission = st.selectbox(
                "计算类型",
                ["relax", "scf", "band", "dos"],
                key="extract_mission_type"
            )
            plot_result = st.checkbox("生成图表", value=True)
            
            if st.button("📥 提取结果", key="extract_result"):
                if st.session_state.langchain_connected and extract_dir:
                    with st.spinner("提取中..."):
                        result = vasp_extract_result(extract_dir, extract_mission, plot_result)
                        if "error" in result:
                            st.error(result.get("error"))
                        else:
                            st.success("结果提取成功!")
                            st.json(result)
                            if result.get("image_url"):
                                st.image(result.get("image_url"))


def main():
    st.title("🔬 MatAgent 材料智能设计平台")

    function_tabs = sidebar_functions()

    if function_tabs == "💬 AI对话":
        chat_interface()
    elif function_tabs == "🔍 材料查询":
        material_search_panel()
    elif function_tabs == "📊 结构建模":
        structure_builder_panel()
    elif function_tabs == "🧪 ML预测":
        ml_prediction_panel()
    elif function_tabs == "💻 VASP任务":
        vasp_task_panel()


if __name__ == "__main__":
    main()
