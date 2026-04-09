"""
MatAgent MCP 适配器版本 - 使用 langchain-mcp-adapters 官方库
通过 MCP 协议连接 mcp_server.py 提供的工具
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain 相关导入
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_lc_agent

# MCP 官方适配器导入
from langchain_mcp_adapters.client import MultiServerMCPClient

# 加载环境变量
load_dotenv()


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
        
    async def chat(self, message: str, thread_id: Optional[str] = None) -> str:
        """与 Agent 对话"""
        if not self.agent:
            raise RuntimeError("Agent 未初始化，请先调用 connect()")
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        response = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        messages = response.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return "无响应"


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
                future = pool.submit(asyncio.run, coro)
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
            
    def chat(self, message: str, thread_id: Optional[str] = None) -> str:
        """同步方式对话"""
        if not self._connected:
            self.connect()
        return self._run_async(self._async_agent.chat(message, thread_id))
    
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
