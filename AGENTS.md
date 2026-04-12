# MatAgent 开发指南

## 快速启动

```bash
# 1. 启动 Flask 文件服务器（2D/3D 可视化，端口 6750）
python flask_server.py

# 2. 启动 Agent Server（端口 8766）
uv run --env-file config/.env python agent_mcp_server.py

# 3. 启动 Web 界面（端口 8501）
streamlit run web_mcp_app.py
```

## 必需环境变量 (.env)

- `DEEPSEEK_API_KEY` - DeepSeek API (必需)
- `mp_API_KEY` - Materials Project API
- `HOST`, `PORT`, `USERNAME`, `PASSWORD`, `base_dir` - SSH/VASP 配置

## 核心模块

| 文件 | 作用 |
|------|------|
| `web_mcp_app.py` | Streamlit 入口，5个功能面板 |
| `agent/langchain_mcp_agent.py` | LangGraph MCP Agent |
| `agent_mcp_server.py` | FastAPI Agent 服务 (端口 8766) |
| `flask_server.py` | 2D 图片 / 3D HTML 文件服务 (端口 6750) |
| `mcp_server.py` | MCP 工具服务器 (端口 8000) |
| `tryssh.py` | SSH 远程连接，VASP 任务管理 |
| `databasemanage.py` | SQLite 数据库 |

## 常见陷阱

### 1. StructuredTool 不可直接调用
`@tool` 装饰的函数变成 `StructuredTool` 对象，调用会报错。解决方案：使用 `MatAgentMCP` 类的 wrapper 方法：
```python
# 错误
from agent.langchain_mcp_agent import search_materials
result = search_materials(elements=['Li'])

# 正确
from agent.langchain_mcp_agent import MatAgentMCP
agent = MatAgentMCP(api_key='xxx')
result = agent.search_materials(elements=['Li'])
```

### 2. SQLite 多线程问题
在 Streamlit 中使用 SQLite 必须用**模块级函数**，每次创建独立连接：
```python
# 正确 - 模块级函数，每次新建连接
def add_chat_message(session_id, role, content):
    conn = sqlite3.connect("xxx.db")
    try:
        conn.execute(...)
    finally:
        conn.close()

# 错误 - 类方法，Streamlit 多线程报错
class DatabaseManager:
    def add_chat_message(self, ...):  # 共享连接会出错
```

### 3. 3D 结构持久化
`flask_server.py` 使用 `structure_info.json` 保存结构元数据，启动时自动加载。

### 4. 聊天历史
- 数据库: `matagent_history.db` 或 `matagent_server_history.db`
- 每次页面加载同步消息到 Agent: `main()` 中检查并更新 `_message_history`

## 测试

```bash
# 测试模块导入
python -c "from agent.langchain_mcp_agent import MatAgentMCP; print('OK')"

# 测试 MCP Server
python mcp_server.py

# 测试 Agent Server
python agent_mcp_server.py
```

## 依赖管理

```bash
uv sync                    # 同步依赖
uv pip install -r config/requirements.txt  # 备用
```
Python 版本: **3.13.4**

## 系统架构

```
用户层 (Streamlit 8501) 
    ↓ HTTP/REST
服务层 (Agent Server 8766) ← FastAPI，会话管理，SQLite
    ↓ MCP Protocol
工具层 (MCP Server 8000) ← FastMCP，20+ 工具函数
    ↓
外部服务: Materials Project API, OQMD, ML模型, SSH/VASP
```

**核心流程**: Web界面 → Agent Server → MCP Server → 各工具/外部服务

## 目录结构

```
mat-agent-web/
├── web_mcp_app.py              # Streamlit Web 应用 (端口 8501)
├── agent_mcp_server.py         # FastAPI Agent 服务 (端口 8766)
├── mcp_server.py               # MCP 工具服务器 (端口 8000)
├── flask_server.py             # 2D/3D 可视化文件服务 (端口 6750)
├── tryssh.py                   # SSH 远程连接，VASP 任务管理
├── oqmd.py                     # OQMD 数据库查询
├── agent/
│   └── langchain_mcp_agent.py  # LangChain MCP Agent 核心
├── myml/                       # ML预测模型 (XGBoost)
├── config/                     # 配置文件
│   ├── loadenv.py
│   ├── pyproject.toml
│   └── requirements.txt
├── db/                        # 数据库
│   ├── databasemanage.py
│   └── *.db
├── cache/                      # 缓存目录
│   ├── temp_images/           # 2D 结构图
│   ├── temp_3d/               # 3D HTML
│   └── structure_info.json
└── web/
    └── assets/               # Web 静态资源
```

## CHANGELOG 维护

每次完成对项目的重大改动后，更新 `CHANGELOG.md`。以下情况需要记录：
- 新增/删除/重命名核心文件
- 重要功能变更（如新增模块、API 接口变化）
- 配置文件结构变化
- 依赖或构建方式变化

以下情况无需记录：
- 小的 bugfix
- 代码风格调整
- 注释/文档更新
- 临时调试代码

## 参考

- 详细文档: `README.md`
- MCP 工具说明: `README.md` (搜索 "MCP Server")
- 软著说明: `软著说明书生成流程.md`