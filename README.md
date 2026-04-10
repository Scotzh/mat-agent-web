# MatAgent - 材料科学智能设计平台

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-Protocol-green)](https://modelcontextprotocol.io/)

MatAgent 是一个基于 **MCP (Model Context Protocol)** 协议的材料科学智能助手，集成了材料数据库查询、晶体结构分析、机器学习预测和 VASP 计算任务管理等功能。

## ✨ 核心特性

- 🤖 **多模型支持** - 支持 DeepSeek (chat/reasoner) 和 智谱 GLM-5 等大语言模型
- 🔍 **材料数据库** - 集成 Materials Project、OQMD 等材料数据库查询
- 🏗️ **结构建模** - 晶体结构构建、可视化 (2D/3D) 和 CIF 文件导出
- 🧠 **ML 预测** - 基于机器学习的材料性质快速预测
- 💻 **VASP 集成** - 远程 SSH 连接，支持计算任务提交和管理
- 💬 **会话管理** - 持久化聊天记录，支持多会话切换
- 🌐 **Web 界面** - 基于 Streamlit 的现代化交互界面

## 🏗️ 系统架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web 前端      │────▶│  Agent MCP Server │────▶│   MCP Server    │
│  (Streamlit)    │     │   (FastAPI)      │     │  (工具服务)      │
│  web_mcp_app.py │◄────│ agent_mcp_server │◄────│  mcp_server.py  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  LangChain Agent │
                        │ langchain_mcp_   │
                        │    agent.py      │
                        └──────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd mat-agent-mcp

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# DeepSeek API (必需)
DEEPSEEK_API_KEY=your_deepseek_api_key

# 智谱 AI (可选，用于 GLM-5)
ZAI_API_KEY=your_zhipu_api_key

# Materials Project API (可选)
mp_API_KEY=your_mp_api_key

# SSH/VASP 配置 (可选)
HOST=your_server_host
PORT=22
USERNAME=your_username
PASSWORD=your_password
base_dir=/path/to/vasp/tasks
```

### 3. 启动服务

**方式一：一键启动（推荐）**

```bash
# 启动 Flask 文件服务器（2D/3D 可视化）
python flask_server.py

# 启动 MCP Agent Server（新终端）
python agent_mcp_server.py

# 启动 Web 界面（新终端）
streamlit run web_mcp_app.py
```

**方式二：使用 MCP 协议启动**

```bash
# 启动 MCP Server
python mcp_server.py

# 启动 Agent Server
python agent_mcp_server.py

# 启动 Web
streamlit run web_mcp_app.py
```

### 4. 访问应用

- Web 界面: http://localhost:8501
- Agent API: http://localhost:8766
- Flask 文件服务: http://localhost:5001

## 📁 项目结构

```
mat-agent-mcp/
├── 📄 web_mcp_app.py              # Streamlit Web 应用主入口
├── 📄 agent_mcp_server.py         # FastAPI Agent 服务 (端口 8766)
├── 📄 agent/langchain_mcp_agent.py # LangChain MCP Agent 实现
├── 📄 mcp_server.py               # MCP 工具服务器
├── 📄 flask_server.py             # 2D/3D 可视化文件服务
├── 📄 databasemanage.py           # SQLite 数据库管理
├── 📄 tryssh.py                   # SSH/VASP 远程操作
├── 📁 myml/                       # 机器学习模型
│   ├── bandgap_predict.py         # 带隙预测模型
│   └── *.csv, *.json              # 训练数据
├── 📁 assets/                     # Logo 等静态资源
├── 📁 temp_images/                # 2D 结构图缓存
├── 📁 temp_3d/                    # 3D 可视化 HTML 缓存
├── 📁 .streamlit/                 # Streamlit 配置
│   └── config.toml                # 主题配置
├── 📄 requirements.txt            # Python 依赖
└── 📄 README.md                   # 本文件
```

## 🔧 核心模块说明

### web_mcp_app.py
Streamlit 构建的 Web 界面，提供：
- 💬 智能体对话（支持流式响应）
- 🔍 材料查询（Materials Project）
- 📊 结构建模（自定义晶体结构）
- 🧪 ML 预测（带隙预测）
- 💻 VASP 任务管理

### agent_mcp_server.py
FastAPI 构建的 Agent 服务：
- 提供 HTTP API 供 Web 前端调用
- 管理 MCP Agent 生命周期
- 会话持久化（SQLite）
- 支持流式聊天响应

### agent/langchain_mcp_agent.py
LangChain MCP Agent 核心：
- 基于 `langchain-mcp-adapters` 官方库
- 支持多模型切换（DeepSeek/GLM-5）
- ReAct 推理模式
- 工具调用链管理

### mcp_server.py
MCP 协议工具服务器，提供：
- `search_materials` - 材料搜索
- `get_material_structure` - 获取晶体结构
- `build_structure` - 构建自定义结构
- `predict_bandgap` - 预测带隙
- `vasp_create_task` - 创建 VASP 任务
- `vasp_submit` - 提交计算任务
- `vasp_extract` - 提取计算结果

## 🎨 自定义配置

### 添加 Logo

将 logo 图片放入 `assets/logo.png`，支持 PNG/JPG/SVG 格式。

### 修改主题颜色

编辑 `.streamlit/config.toml`：

```toml
[theme]
primaryColor = "#3B82F6"      # 主色调（按钮等）
backgroundColor = "#FFFFFF"   # 背景色
secondaryBackgroundColor = "#F8F9FA"
textColor = "#1F2937"
```

### 修改系统提示词

设置环境变量或在代码中修改：

```python
# .env
MATAGENT_SYSTEM_PROMPT="你的自定义提示词"
```

## 🔌 API 接口

### 聊天接口

```bash
# 非流式聊天
POST http://localhost:8766/chat
Content-Type: application/json

{
  "session_id": "uuid",
  "message": "搜索含锂的材料",
  "model": "deepseek-chat"
}

# 流式聊天
POST http://localhost:8766/chat/stream
```

### 工具调用接口

```bash
# 搜索材料
GET http://localhost:8766/materials/search?elements=Li,Fe,O

# 获取结构
GET http://localhost:8766/materials/structure/{material_id}

# 预测带隙
GET http://localhost:8766/predict_bandgap?formula=SiO2
```

## 📝 使用示例

### 材料查询

```python
from agent.langchain_mcp_agent import MatAgentMCP

agent = MatAgentMCP(api_key="your_key")
result = agent.chat("搜索带隙大于2eV的氧化物材料")
```

### VASP 任务管理

```python
# 创建任务
POST /vasp/create_task
{
  "formula": "LiFePO4",
  "cif_path": "./LiFePO4.cif"
}

# 提交计算
POST /vasp/submit
{
  "task_directory": "/path/to/task",
  "mission": "relax"
}
```

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用框架
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters) - MCP 官方适配器
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP Python SDK
- [Streamlit](https://streamlit.io/) - Web 应用框架
- [Pymatgen](https://pymatgen.org/) - 材料分析库
- [Materials Project](https://materialsproject.org/) - 材料数据库

---

<p align="center">
  Made with ❤️ for Materials Science
</p>
