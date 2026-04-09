"""
MatAgent MCP 适配器版本 - 使用 langchain-mcp-adapters 官方库
通过 MCP 协议连接 mcp_server.py 提供的工具
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain 相关导入
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_lc_agent

# MCP 官方适配器导入
from langchain_mcp_adapters.client import MultiServerMCPClient

# 加载环境变量
load_dotenv()

# 系统提示词 - 定义Agent的角色和行为
# 优先从环境变量读取，否则使用默认提示词
DEFAULT_SYSTEM_PROMPT = """你是 MatAgent，一位专业的材料科学AI助手。

你的专长包括：
1. 材料数据库查询（Materials Project(稳定)、OQMD(连接不太稳定)等）
2. 晶体结构分析与可视化
3. 材料性质预测（带隙、能带结构等）
4. VASP计算任务管理

回复风格：
- 专业、准确、简洁
- 使用中文回复用户
- 对于技术问题，提供清晰的解释
- 如果不确定，坦诚说明

当使用工具时：
- 理解用户需求后选择合适的工具
- 解释工具返回的结果（用自己的话总结，不要重复原始JSON数据）
- 返回结果里有图片url的话请用markdown格式渲染图片
- 如有必要，建议下一步操作

重要：不要在回复中包含工具返回的原始JSON文本或结构化数据。只提供对结果的简洁解释和解读。
"""

SYSTEM_PROMPT = os.getenv("MATAGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


class MatAgentMCP:
    """MatAgent MCP 版本 - 使用官方 langchain-mcp-adapters 库"""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        mcp_server_url: str = "http://localhost:8000/sse",
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("必须提供 api_key 或设置 DEEPSEEK_API_KEY 环境变量")
        
        self.base_url = base_url
        self.model = str(model)
        self.mcp_server_url = mcp_server_url
        
        self.llm = ChatOpenAI(
            model=str(self.model),
            api_key=str(self.api_key),
            base_url=self.base_url,
            temperature=0.7,
            streaming=True,  # 开启流式模式
        )
        
        self.agent = None
        self.tools: List[BaseTool] = []
        self.mcp_client: Optional[MultiServerMCPClient] = None
        
    def _wrap_tool(self, tool: BaseTool) -> BaseTool:
        """包装工具，将结构化结果转换为字符串"""
        from langchain_core.tools import StructuredTool
        
        # 获取原始工具函数
        original_coroutine = tool.coroutine if hasattr(tool, 'coroutine') else None
        if not original_coroutine:
            # 如果没有 coroutine，使用 _arun
            original_arun = tool._arun if hasattr(tool, '_arun') else None
        
        async def wrapped_coroutine(**kwargs):
            # 调用原始工具
            if original_coroutine:
                result = await original_coroutine(**kwargs)
            else:
                result = await tool.ainvoke(kwargs)
            
            # 将结构化结果转换为字符串
            if isinstance(result, list):
                # langchain-mcp-adapters 返回的是 list[dict] 格式
                texts = []
                for item in result:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    else:
                        texts.append(str(item))
                return "\n".join(texts)
            elif isinstance(result, dict):
                return str(result)
            return result
        
        # 创建新的 StructuredTool 包装原工具
        return StructuredTool.from_function(
            coroutine=wrapped_coroutine,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
        )
    
    async def connect(self):
        """连接到 MCP Server 并加载工具"""
        print(f"🔄 正在连接 MCP Server: {self.mcp_server_url}")
        
        server_config = {
            "matagent": {
                "transport": "sse",
                "url": self.mcp_server_url,
            }
        }
        
        self.mcp_client = MultiServerMCPClient(server_config)
        raw_tools = await self.mcp_client.get_tools()
        
        # 包装工具以转换返回格式
        self.tools = [self._wrap_tool(tool) for tool in raw_tools]
        
        print(f"✅ 成功加载 {len(self.tools)} 个工具")
        for tool in self.tools:
            print(f"   - {tool.name}")
        
        self.agent = create_lc_agent(self.llm, self.tools)
        print("✅ Agent 创建成功")
        
    async def disconnect(self):
        """断开 MCP Server 连接"""
        self.mcp_client = None
        self.agent = None
        self.tools = []
        print("✅ 已断开 MCP Server 连接")
    
    def get_tools_info(self) -> List[Dict[str, Any]]:
        """获取已加载工具的信息"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self.tools
        ]
        
    async def chat(self, message: str, thread_id: Optional[str] = None) -> dict:
        """与 Agent 对话，返回包含工具调用信息的结果"""
        if not self.agent:
            raise RuntimeError("Agent 未初始化，请先调用 connect()")
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        # 构建消息列表：系统提示词 + 用户消息
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message)
        ]
        
        response = await self.agent.ainvoke(
            {"messages": messages},
            config=config
        )
        
        response_messages = response.get("messages", [])
        
        # 提取AI回复内容
        ai_content = ""
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage):
                ai_content = msg.content
                break
        
        # 提取工具调用信息
        tool_results = []
        for msg in response_messages:
            # 检查是否有工具调用
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_results.append({
                        "tool_name": tc.get("name", "unknown"),
                        "tool_args": tc.get("args", {}),
                        "result": None  # 将在后续消息中填充
                    })
            # 检查工具返回结果
            if hasattr(msg, 'name') and msg.name:
                # 找到对应的工具调用并填充结果
                for tr in tool_results:
                    if tr["tool_name"] == msg.name and tr["result"] is None:
                        tr["result"] = msg.content
                        break
        
        return {
            "message": ai_content if ai_content else "无响应",
            "tool_results": tool_results
        }
    
    async def chat_stream(self, message: str, thread_id: Optional[str] = None):
        """
        与 Agent 对话，以流式方式返回结果
        生成器yield格式: {"type": "tool_start", "data": {...}} 或 {"type": "token", "data": "..."} 或 {"type": "tool_end", "data": {...}}
        
        使用 stream_mode="updates" 获取实时更新，包括工具调用和结果
        """
        if not self.agent:
            raise RuntimeError("Agent 未初始化，请先调用 connect()")
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        # 构建消息列表：系统提示词 + 用户消息
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message)
        ]
        
        tool_results: list[dict[str, Any]] = []
        full_message = ""
        pending_tool_calls: dict[str, dict[str, Any]] = {}  # 跟踪待完成的工具调用
        
        # 使用 stream_mode="updates" 获取实时更新
        async for chunk in self.agent.astream(
            {"messages": messages},
            config=config,
            stream_mode="updates"
        ):
            # chunk 是一个 dict，key 是节点名称，value 是节点输出
            for node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue
                
                node_messages = node_output.get("messages", [])
                if not node_messages:
                    continue
                
                for msg in node_messages:
                    # 处理 AI 消息（包含 token 和工具调用请求）
                    if isinstance(msg, AIMessage):
                        # 发送 token
                        if hasattr(msg, 'content') and msg.content:
                            content = msg.content
                            if isinstance(content, str):
                                full_message += content
                                yield {"type": "token", "data": content}
                        
                        # 检测工具调用请求
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_name = tc.get("name", "unknown")
                                tool_args = tc.get("args", {})
                                tool_id = tc.get("id") or tool_name
                                
                                tool_info: dict[str, Any] = {
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "result": None
                                }
                                tool_results.append(tool_info)
                                pending_tool_calls[tool_id] = tool_info
                                yield {"type": "tool_start", "data": tool_info}
                    
                    # 处理工具返回结果
                    elif hasattr(msg, 'name') and msg.name:
                        tool_name = msg.name
                        tool_content = msg.content if hasattr(msg, 'content') else None
                        
                        # 找到对应的工具调用并更新
                        for tr in tool_results:
                            if tr.get("tool_name") == tool_name and tr.get("result") is None:
                                tr["result"] = tool_content
                                yield {"type": "tool_end", "data": tr}
                                break
        
        # 清理消息中的工具JSON内容
        cleaned_message = self._clean_tool_json_from_message(full_message)
        
        # 发送最终完整消息
        yield {"type": "complete", "data": {"message": cleaned_message, "tool_results": tool_results}}
    
    def _clean_tool_json_from_message(self, message: str) -> str:
        """
        清理消息中嵌入的工具JSON内容
        移除类似 [[{"type": "text", "text": "{...}"}]] 这样的工具返回内容
        """
        import re
        
        cleaned = message
        
        # 模式1: 匹配 [[{"type": "text", "text": "...", "id": "..."}]]
        # 使用非贪婪匹配和嵌套括号处理
        pattern1 = r'\[\[\s*\{[^{}]*"type"\s*:\s*"text"[^{}]*\}\s*\]\]'
        matches1 = list(re.finditer(pattern1, cleaned))
        for match in reversed(matches1):  # 从后往前替换，避免位置偏移
            matched_text = match.group(0)
            # 检查是否包含工具返回的特征
            if any(keyword in matched_text for keyword in ['"image_url"', '"3d_image_url"', '"structure_dict"', 
                                                            '"formula"', '"band_gap"', '"material_id"']):
                cleaned = cleaned[:match.start()] + cleaned[match.end():]
        
        # 模式2: 匹配 {"structured_content": {...}}
        # 需要处理嵌套的大括号
        def remove_structured_content(text: str) -> str:
            result = text
            pattern2 = r'\{\s*"structured_content"\s*:\s*\{'
            for match in re.finditer(pattern2, result):
                start = match.start()
                # 找到匹配的结束括号
                brace_count = 0
                end = start
                for i, char in enumerate(result[start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = start + i + 1
                            break
                if end > start:
                    matched_text = result[start:end]
                    if any(keyword in matched_text for keyword in ['"image_url"', '"3d_image_url"', 
                                                                    '"structure_dict"', '"formula"']):
                        result = result[:start] + result[end:]
                        return remove_structured_content(result)  # 递归处理
            return result
        
        cleaned = remove_structured_content(cleaned)
        
        # 清理多余的空行和空格
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)  # 移除行尾空格
        cleaned = cleaned.strip()
        
        return cleaned


class MatAgentMCPSync:
    """MatAgent MCP 版本的同步包装类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        mcp_server_url: str = "http://localhost:8000/sse",
    ):
        self._async_agent = MatAgentMCP(
            api_key=api_key,
            base_url=base_url,
            model=model,
            mcp_server_url=mcp_server_url,
        )
        self._connected = False
        
    def _run_async(self, coro):
        """在已有事件循环中运行协程"""
        try:
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，使用 run_coroutine_threadsafe
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                # 创建一个新的事件循环来运行协程
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()
                
                future = pool.submit(run_in_new_loop)
                return future.result()
        except RuntimeError:
            # 没有运行的事件循环，直接使用 asyncio.run
            return asyncio.run(coro)
        
    def connect(self):
        """同步方式连接 MCP Server"""
        if not self._connected:
            self._run_async(self._async_agent.connect())
            self._connected = True
            
    def disconnect(self):
        """同步方式断开连接"""
        if self._connected:
            self._run_async(self._async_agent.disconnect())
            self._connected = False
            
    def chat(self, message: str, thread_id: Optional[str] = None) -> dict:
        """同步方式对话，返回包含工具调用信息的结果"""
        if not self._connected:
            self.connect()
        return self._run_async(self._async_agent.chat(message, thread_id))
    
    def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """直接调用 MCP 工具（绕过 LLM，更快）"""
        if not self._connected:
            self.connect()
        
        # 查找工具（包装后的工具）
        tool = None
        for t in self._async_agent.tools:
            if t.name == tool_name:
                tool = t
                break
        
        if tool is None:
            available = [t.name for t in self._async_agent.tools]
            raise ValueError(f"工具 '{tool_name}' 不存在。可用工具: {available}")
        
        # 直接调用工具
        return self._run_async(tool.ainvoke(kwargs))
    
    def invoke_tool_raw(self, tool_name: str, **kwargs) -> Any:
        """直接调用原始 MCP 工具，返回原始格式（不转换为字符串）"""
        if not self._connected:
            self.connect()
        
        # 通过 mcp_client 直接调用原始工具
        async def _call_raw():
            # 获取原始工具（未包装的）
            raw_tools = await self._async_agent.mcp_client.get_tools()
            target_tool = None
            for t in raw_tools:
                if t.name == tool_name:
                    target_tool = t
                    break
            
            if target_tool is None:
                available = [t.name for t in raw_tools]
                raise ValueError(f"工具 '{tool_name}' 不存在。可用工具: {available}")
            
            # 调用原始工具，获取原始返回格式
            return await target_tool.ainvoke(kwargs)
        
        return self._run_async(_call_raw())
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        for t in self._async_agent.tools:
            if t.name == tool_name:
                return t
        return None
    
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 注意：使用 langchain-mcp-adapters 时，
        # 每个工具调用会自动创建新会话，不需要手动断开
        pass


def create_agent(
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    mcp_server_url: str = "http://localhost:8000/sse",
) -> MatAgentMCPSync:
    """创建 MatAgent MCP 同步实例"""
    return MatAgentMCPSync(
        api_key=api_key,
        model=model,
        mcp_server_url=mcp_server_url,
    )


async def test_async():
    """测试异步版本"""
    print("=" * 50)
    print("测试 MatAgentMCP 异步版本")
    print("=" * 50)
    
    agent = MatAgentMCP()
    try:
        await agent.connect()
        
        print("\n📝 测试: 获取当前时间")
        response = await agent.chat("现在几点了？")
        print(f"🤖 回复: {response}\n")
        
        print("📝 测试: 搜索锂材料")
        response = await agent.chat("帮我搜索包含 Li 的材料，最多返回 3 个")
        print(f"🤖 回复: {response}\n")
        
    finally:
        await agent.disconnect()


def test_sync():
    """测试同步版本"""
    print("=" * 50)
    print("测试 MatAgentMCPSync 同步版本")
    print("=" * 50)
    
    with create_agent() as agent:
        print("\n📝 测试: 获取当前时间")
        response = agent.chat("现在几点了？")
        print(f"🤖 回复: {response}\n")


if __name__ == "__main__":
    # 运行测试
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(test_async())
    else:
        test_sync()
