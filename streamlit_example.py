import streamlit as st
import time
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
import plotly.express as px
import plotly.graph_objects as go

# 页面配置
st.set_page_config(
    page_title="LangChain Agent 助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
def load_css():
    st.markdown("""
    <style>
    /* 主容器 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* 聊天气泡 */
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        max-width: 80%;
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0.5rem;
    }
    
    .assistant-message {
        background: white;
        color: #333;
        margin-right: auto;
        border: 1px solid #e5e7eb;
        border-bottom-left-radius: 0.5rem;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.5rem;
    }
    
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    
    .user-avatar {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .assistant-avatar {
        background: #f0f2f6;
    }
    
    .message-sender {
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .message-time {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-left: auto;
    }
    
    .message-content {
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* 工具调用卡片 */
    .tool-call {
        background: #f8f9fa;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        margin: 0.5rem 0 0.5rem 3rem;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .tool-header {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        cursor: pointer;
        background: #f1f3f5;
        border-bottom: 1px solid transparent;
    }
    
    .tool-header:hover {
        background: #e9ecef;
    }
    
    .tool-icon {
        font-size: 1rem;
        margin-right: 0.75rem;
    }
    
    .tool-info {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .tool-name {
        font-weight: 600;
        font-size: 0.85rem;
        color: #333;
    }
    
    .tool-status {
        padding: 0.1rem 0.5rem;
        border-radius: 0.75rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-running { background: #fff3cd; color: #856404; }
    .status-success { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }
    
    .tool-duration {
        font-size: 0.75rem;
        color: #6c757d;
    }
    
    .tool-details {
        padding: 1rem;
        background: white;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .tool-section {
        margin-bottom: 0.75rem;
    }
    
    .tool-section-title {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
        color: #495057;
    }
    
    .tool-section-content {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* 输入区域 */
    .input-container {
        background: white;
        padding: 1.5rem;
        border-top: 1px solid #e5e7eb;
        position: sticky;
        bottom: 0;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.05);
    }
    
    /* 加载动画 */
    .thinking-dots {
        display: flex;
        gap: 0.25rem;
    }
    
    .thinking-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #6c757d;
        animation: bounce 1.4s infinite;
    }
    
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 100% { opacity: 0.3; transform: translateY(0); }
        50% { opacity: 1; transform: translateY(-4px); }
    }
    
    /* 侧边栏 */
    .sidebar .tool-container {
        background: white;
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .tool-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #333;
    }
    
    /* 响应式 */
    @media (max-width: 768px) {
        .chat-message {
            max-width: 90%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化会话状态
def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = create_agent()
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    if "expanded_tools" not in st.session_state:
        st.session_state.expanded_tools = set()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# 定义工具函数
def get_current_time(query: str) -> str:
    """获取当前时间"""
    time.sleep(0.5)  # 模拟延迟
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

def calculate_expression(expression: str) -> str:
    """计算数学表达式"""
    time.sleep(0.5)
    try:
        # 安全计算
        result = eval(expression, {"__builtins__": {}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

def search_weather(city: str) -> str:
    """查询天气"""
    time.sleep(1)  # 模拟API调用延迟
    
    # 模拟数据
    weather_data = {
        "北京": {"temp": "15°C", "condition": "晴", "humidity": "45%", "wind": "3级"},
        "上海": {"temp": "18°C", "condition": "多云", "humidity": "65%", "wind": "2级"},
        "广州": {"temp": "22°C", "condition": "小雨", "humidity": "85%", "wind": "1级"},
        "深圳": {"temp": "24°C", "condition": "阴", "humidity": "75%", "wind": "2级"},
        "杭州": {"temp": "16°C", "condition": "晴", "humidity": "60%", "wind": "3级"}
    }
    
    if city in weather_data:
        data = weather_data[city]
        return f"{city}天气: {data['condition']}, 温度: {data['temp']}, 湿度: {data['humidity']}, 风力: {data['wind']}"
    else:
        return f"未找到{city}的天气信息"

def get_stock_info(symbol: str) -> str:
    """获取股票信息"""
    time.sleep(1)
    
    stock_data = {
        "AAPL": {"name": "苹果公司", "price": 172.35, "change": "+1.2%", "volume": "85.2M"},
        "GOOGL": {"name": "谷歌", "price": 152.12, "change": "+0.8%", "volume": "45.3M"},
        "TSLA": {"name": "特斯拉", "price": 175.79, "change": "-2.1%", "volume": "120.5M"},
        "MSFT": {"name": "微软", "price": 415.86, "change": "+0.5%", "volume": "65.7M"},
        "BABA": {"name": "阿里巴巴", "price": 78.42, "change": "-1.3%", "volume": "35.9M"}
    }
    
    if symbol.upper() in stock_data:
        data = stock_data[symbol.upper()]
        return f"{data['name']}({symbol.upper()}): ${data['price']} ({data['change']}), 成交量: {data['volume']}"
    else:
        return f"未找到股票代码 {symbol}"

def search_wikipedia(query: str) -> str:
    """搜索维基百科（模拟）"""
    time.sleep(1.5)
    
    mock_data = {
        "人工智能": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习": "机器学习是一门多领域交叉学科，专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能。",
        "深度学习": "深度学习是机器学习的一个分支，它试图使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。",
        "大语言模型": "大语言模型是基于海量文本数据训练的深度学习模型，能够生成、理解和处理自然语言文本。"
    }
    
    for key in mock_data:
        if key in query:
            return mock_data[key]
    
    return f"找到关于'{query}'的信息：这是模拟返回的摘要信息，实际应用中会调用真实API。"

def analyze_sentiment(text: str) -> str:
    """情感分析"""
    time.sleep(0.8)
    
    # 简单情感分析逻辑
    positive_words = ["好", "优秀", "棒", "喜欢", "爱", "高兴", "开心", "满意", "成功"]
    negative_words = ["差", "糟糕", "坏", "讨厌", "恨", "伤心", "难过", "失望", "失败"]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        sentiment = "积极"
        score = positive_count / (positive_count + negative_count + 1)
    elif negative_count > positive_count:
        sentiment = "消极"
        score = negative_count / (positive_count + negative_count + 1)
    else:
        sentiment = "中性"
        score = 0.5
    
    return f"情感分析结果: {sentiment} (置信度: {score:.2f})。\n正面词: {positive_count}个，负面词: {negative_count}个"

def create_plot(data_type: str) -> str:
    """创建图表（返回HTML）"""
    import plotly.io as pio
    
    if data_type == "折线图":
        df = pd.DataFrame({
            '月份': ['1月', '2月', '3月', '4月', '5月', '6月'],
            '销售额': [100, 150, 200, 180, 220, 250],
            '利润': [20, 30, 45, 40, 50, 60]
        })
        fig = px.line(df, x='月份', y=['销售额', '利润'], 
                     title='销售额与利润趋势图',
                     markers=True)
        
    elif data_type == "柱状图":
        df = pd.DataFrame({
            '产品': ['A', 'B', 'C', 'D', 'E'],
            '销量': [120, 150, 80, 200, 90]
        })
        fig = px.bar(df, x='产品', y='销量', 
                    title='产品销量对比',
                    color='销量')
        
    elif data_type == "饼图":
        df = pd.DataFrame({
            '类别': ['电子产品', '服装', '食品', '图书', '其他'],
            '占比': [35, 25, 20, 10, 10]
        })
        fig = px.pie(df, values='占比', names='类别',
                    title='销售品类占比',
                    hole=0.3)
    
    else:
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 3, 5, 7, 11]
        })
        fig = px.scatter(df, x='x', y='y', 
                        title='示例散点图',
                        trendline='ols')
    
    return pio.to_html(fig, full_html=False)

# 创建 Agent (使用 LangGraph)
def create_agent():
    """创建 LangGraph Agent"""
    
    # 定义工具
    tools = [
        Tool(
            name="GetCurrentTime",
            func=get_current_time,
            description="当需要获取当前时间、日期、现在几点时使用此工具。输入应为空字符串或'time'。"
        ),
        Tool(
            name="Calculator",
            func=calculate_expression,
            description="当需要进行数学计算时使用此工具。输入应为数学表达式，如'2 + 3 * 4'或'(15 + 7) * 3'。"
        ),
        Tool(
            name="WeatherSearch",
            func=search_weather,
            description="查询城市天气。输入应为城市名称，如'北京'、'上海'。"
        ),
        Tool(
            name="StockInfo",
            func=get_stock_info,
            description="查询股票信息。输入应为股票代码，如'AAPL'、'GOOGL'。"
        ),
        Tool(
            name="WikipediaSearch",
            func=search_wikipedia,
            description="在维基百科中搜索信息。输入应为要搜索的主题，如'人工智能'、'机器学习'。"
        ),
        Tool(
            name="SentimentAnalysis",
            func=analyze_sentiment,
            description="对文本进行情感分析。输入应为要分析的文本内容。"
        ),
        Tool(
            name="CreatePlot",
            func=create_plot,
            description="创建图表。输入应为图表类型，如'折线图'、'柱状图'、'饼图'。"
        )
    ]
    
    # 创建LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=st.secrets.get("OPENAI_API_KEY", "")
    )
    
    # 使用 LangGraph 创建 Agent
    agent = create_react_agent(llm, tools)
    
    return agent

# 显示消息组件
def display_message(message: Dict[str, Any]):
    """显示单条消息"""
    is_user = message["role"] == "user"
    
    # 头像和发送者
    avatar = "👤" if is_user else "🤖"
    sender = "您" if is_user else "AI助手"
    
    # 时间格式化
    if isinstance(message["timestamp"], datetime):
        time_str = message["timestamp"].strftime("%H:%M")
    else:
        time_str = message["timestamp"]
    
    # 消息容器
    with st.container():
        col1, col2 = st.columns([1, 20])
        
        with col1:
            st.markdown(f'<div class="avatar {"user-avatar" if is_user else "assistant-avatar"}">{avatar}</div>', 
                       unsafe_allow_html=True)
        
        with col2:
            # 消息头
            st.markdown(f'''
            <div class="message-header">
                <span class="message-sender">{sender}</span>
                <span class="message-time">{time_str}</span>
            </div>
            ''', unsafe_allow_html=True)
            
            # 消息气泡
            bubble_class = "user-message" if is_user else "assistant-message"
            st.markdown(f'<div class="chat-message {bubble_class}">', unsafe_allow_html=True)
            st.markdown(f'<div class="message-content">{message["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 显示工具调用
            if not is_user and "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    display_tool_call(tool_call)
            
            # 显示图表
            if not is_user and "plot_html" in message:
                st.components.v1.html(message["plot_html"], height=400)

# 显示工具调用组件
def display_tool_call(tool_call: Dict[str, Any]):
    """显示工具调用信息"""
    tool_id = tool_call.get("id", "")
    is_expanded = tool_id in st.session_state.expanded_tools
    
    with st.container():
        st.markdown(f'''
        <div class="tool-call">
            <div class="tool-header" onclick="toggleToolCall('{tool_id}')">
                <div class="tool-icon">{get_tool_icon(tool_call["status"])}</div>
                <div class="tool-info">
                    <span class="tool-name">{tool_call["tool"]}</span>
                    <span class="tool-status status-{tool_call["status"]}">{tool_call["status"]}</span>
                    {tool_call.get("duration") and f'<span class="tool-duration">{tool_call["duration"]}ms</span>'}
                </div>
                <div class="expand-icon">{"▲" if is_expanded else "▼"}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # 如果展开，显示详情
        if is_expanded:
            with st.expander("", expanded=True):
                if "input" in tool_call:
                    st.markdown('<div class="tool-section"><div class="tool-section-title">输入</div>', unsafe_allow_html=True)
                    st.code(tool_call["input"], language="json")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if "result" in tool_call:
                    st.markdown('<div class="tool-section"><div class="tool-section-title">结果</div>', unsafe_allow_html=True)
                    st.code(tool_call["result"], language="text")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def get_tool_icon(status: str) -> str:
    """根据状态获取工具图标"""
    icons = {
        "running": "⏳",
        "success": "✅",
        "error": "❌",
        "pending": "⚡"
    }
    return icons.get(status, "⚙️")

# 处理用户输入
def process_user_input(user_input: str):
    """处理用户输入"""
    if not user_input.strip():
        return
    
    # 添加用户消息
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M")
    }
    st.session_state.messages.append(user_message)
    
    # 显示思考状态
    with st.spinner("思考中..."):
        # 记录开始时间
        start_time = time.time()
        
        # 工具调用记录
        tool_calls = []
        
        def tool_callback(tool_name, input_data, result=None, error=None):
            """工具调用回调函数"""
            tool_call = {
                "id": f"tool_{len(tool_calls)}_{int(time.time())}",
                "tool": tool_name,
                "input": input_data,
                "status": "running" if error is None else "error",
                "result": result if result else str(error) if error else None,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            tool_calls.append(tool_call)
            
            # 更新最后一条消息的工具调用
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                st.session_state.messages[-1]["tool_calls"] = tool_calls.copy()
        
        try:
            # 执行Agent
            response = st.session_state.agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            
            # 计算耗时
            duration = int((time.time() - start_time) * 1000)
            
            # 检查是否有图表生成
            plot_html = None
            if "CreatePlot" in [t["tool"] for t in tool_calls]:
                for tool in tool_calls:
                    if tool["tool"] == "CreatePlot" and tool["result"] and "html" in tool["result"]:
                        plot_html = tool["result"]
                        break
            
            # 添加助手回复
            assistant_message = {
                "role": "assistant",
                "content": response["output"],
                "timestamp": datetime.now().strftime("%H:%M"),
                "tool_calls": tool_calls,
                "duration": duration
            }
            
            if plot_html:
                assistant_message["plot_html"] = plot_html
            
            st.session_state.messages.append(assistant_message)
            
            # 更新聊天历史
            st.session_state.chat_history.append((user_input, response["output"]))
            
        except Exception as e:
            # 错误处理
            error_message = {
                "role": "assistant",
                "content": f"抱歉，处理您的请求时出现了错误：{str(e)}",
                "timestamp": datetime.now().strftime("%H:%M"),
                "error": True
            }
            st.session_state.messages.append(error_message)
            
            st.error(f"发生错误: {str(e)}")

# 侧边栏组件
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("🤖 LangChain Agent")
        
        # API密钥设置
        st.subheader("🔑 设置")
        api_key = st.text_input("OpenAI API Key", 
                               value=st.secrets.get("OPENAI_API_KEY", ""),
                               type="password")
        
        if api_key and api_key != st.session_state.get("api_key", ""):
            st.session_state.api_key = api_key
            st.rerun()
        
        st.divider()
        
        # 可用工具
        st.subheader("🛠️ 可用工具")
        
        tools_info = [
            {"icon": "⏰", "name": "时间查询", "desc": "获取当前时间日期"},
            {"icon": "🧮", "name": "计算器", "desc": "数学表达式计算"},
            {"icon": "🌤️", "name": "天气查询", "desc": "查询城市天气信息"},
            {"icon": "📈", "name": "股票查询", "desc": "查询股票实时信息"},
            {"icon": "📚", "name": "维基百科", "desc": "搜索百科知识"},
            {"icon": "😊", "name": "情感分析", "desc": "分析文本情感倾向"},
            {"icon": "📊", "name": "图表生成", "desc": "创建数据可视化图表"}
        ]
        
        for tool in tools_info:
            with st.expander(f"{tool['icon']} {tool['name']}", expanded=False):
                st.caption(tool["desc"])
        
        st.divider()
        
        # 对话管理
        st.subheader("💬 对话管理")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("💾 导出对话", use_container_width=True):
                export_conversation()
        
        # 示例问题
        st.divider()
        st.subheader("💡 试试问这些")
        
        example_questions = [
            "现在几点了？",
            "计算(15 + 7) * 3等于多少？",
            "北京和上海的天气怎么样？",
            "苹果公司的股票价格是多少？",
            "什么是人工智能？",
            "分析这句话的情感：'这个产品非常好用，我非常满意！'",
            "帮我创建一个柱状图"
        ]
        
        for question in example_questions:
            if st.button(question, use_container_width=True, key=f"example_{question}"):
                st.session_state.user_input = question
                st.rerun()
        
        st.divider()
        
        # 统计信息
        st.subheader("📊 统计")
        st.metric("消息总数", len(st.session_state.messages))
        
        tool_count = sum(1 for msg in st.session_state.messages 
                        if "tool_calls" in msg and msg["tool_calls"])
        st.metric("工具调用次数", tool_count)

def export_conversation():
    """导出对话记录"""
    if not st.session_state.messages:
        st.warning("没有对话记录可导出")
        return
    
    # 生成导出文本
    export_text = "LangChain Agent 对话记录\n"
    export_text += "=" * 50 + "\n\n"
    
    for msg in st.session_state.messages:
        role = "用户" if msg["role"] == "user" else "助手"
        export_text += f"{role} ({msg['timestamp']}):\n"
        export_text += f"{msg['content']}\n\n"
        
        if msg["role"] == "assistant" and "tool_calls" in msg:
            export_text += "使用的工具：\n"
            for tool in msg["tool_calls"]:
                export_text += f"  - {tool['tool']}: {tool.get('result', '无结果')}\n"
            export_text += "\n"
        
        export_text += "-" * 40 + "\n\n"
    
    # 提供下载
    st.download_button(
        label="📥 下载对话记录",
        data=export_text,
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# 主应用
def main():
    """主应用"""
    # 加载CSS
    load_css()
    
    # 初始化状态
    init_session_state()
    
    # 侧边栏
    render_sidebar()
    
    # 主界面
    st.title("💬 LangChain Agent 聊天助手")
    st.caption("一个支持多种工具调用的智能对话助手")
    
    # 聊天容器
    chat_container = st.container()
    
    with chat_container:
        # 显示消息历史
        for message in st.session_state.messages:
            display_message(message)
        
        # 如果正在处理，显示加载状态
        if st.session_state.processing:
            with st.chat_message("assistant"):
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.markdown('<div class="thinking-dots"><span></span><span></span><span></span></div>', 
                               unsafe_allow_html=True)
                with col2:
                    st.write("正在思考中...")
    
    # 输入区域
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "输入您的消息...",
            value=st.session_state.get("user_input", ""),
            key="user_input_widget",
            label_visibility="collapsed",
            placeholder="输入消息，按回车发送...",
            disabled=st.session_state.processing
        )
    
    with col2:
        send_button = st.button(
            "发送",
            use_container_width=True,
            disabled=st.session_state.processing or not user_input.strip(),
            type="primary"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 处理发送
    if send_button or (user_input and "user_input" in st.session_state and st.session_state.user_input):
        st.session_state.processing = True
        
        # 清除输入框
        if "user_input" in st.session_state:
            st.session_state.user_input = ""
        
        # 处理输入
        process_user_input(user_input)
        
        st.session_state.processing = False
        st.rerun()
    
    # JavaScript交互
    st.markdown("""
    <script>
    function toggleToolCall(toolId) {
        // 发送消息到Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: {action: 'toggle_tool', tool_id: toolId}
        }, '*');
    }
    
    // 监听消息
    window.addEventListener('message', function(event) {
        if (event.data.type === 'streamlit:setComponentValue') {
            // 这里可以处理前端交互
        }
    });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()