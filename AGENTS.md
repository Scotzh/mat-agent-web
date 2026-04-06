# MatAgent 开发指南

## 快速启动

```bash
# 1. 启动 Flask 文件服务器（2D/3D 可视化）
python flask_server.py

# 2. 启动 Web 界面
streamlit run web_app.py
```

## 必需环境变量 (.env)

- `DEEPSEEK_API_KEY` - DeepSeek API
- `mp_API_KEY` - Materials Project API
- `HOST`, `PORT`, `USERNAME`, `PASSWORD`, `base_dir` - SSH/VASP 配置

## 核心模块

| 文件 | 作用 |
|------|------|
| `web_app.py` | Streamlit 入口，5个功能面板 |
| `agent/langchain_agent.py` | LangGraph Agent，ReAct 模式 |
| `flask_server.py` | 2D 图片 / 3D HTML 文件服务 |
| `tryssh.py` | SSH 远程连接，VASP 任务管理 |
| `databasemanage.py` | SQLite 数据库 |

## 常见陷阱

### 1. StructuredTool 不可直接调用
`@tool` 装饰的函数变成 `StructuredTool` 对象，调用会报错。解决方案：使用 `MatAgent` 类的 wrapper 方法：
```python
# 错误
from agent.langchain_agent import search_materials
result = search_materials(elements=['Li'])

# 正确
agent = MatAgent(api_key='xxx')
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
- 数据库: `matagent_history.db`
- 每次页面加载同步消息到 Agent: `main()` 中检查并更新 `_message_history`

## 测试

```bash
# 测试单个模块
python -c "from agent.langchain_agent import MatAgent; print('OK')"
```

## 目录结构

```
mat-agent-mcp/
├── web_app.py              # Web 入口
├── agent/langchain_agent.py # LangGraph Agent
├── flask_server.py          # 文件服务
├── databasemanage.py        # 数据库
├── tryssh.py               # SSH/VASP
├── myml/bandgap_predict.py # ML 预测
├── temp_images/            # 2D 图片缓存
├── temp_3d/                # 3D HTML 缓存
├── structure_info.json     # 结构元数据
└── matagent_history.db     # 聊天历史
```

## 参考

- 详细文档: `项目概述.md`
- MCP 工具: `README.md`
- 原有指南: `mcp_skill.instructions.md`