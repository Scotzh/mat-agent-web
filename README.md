# MatAgent - 材料科学智能设计平台

![alt text](web/assets/logo.png)


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
cd mat-agent-web

# 推荐使用 uv 管理依赖
uv sync
# 或使用 pip
uv pip install -r config/requirements.txt
```

### 2. 配置环境变量

编辑 `config/.env` 文件：

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

```bash
# 启动 Flask 文件服务器（2D/3D 可视化）
python -m server.flask_server

# 启动 MCP Server (可选，agent_mcp_server 会自动启动)
python mcp_server.py

# 启动 Agent Server
uv run --env-file config/.env python agent_mcp_server.py

# 启动 Web 界面
streamlit run web_mcp_app.py

```

### 4. 访问应用

- Web 界面: http://localhost:8501
- Agent API: http://localhost:8766
- MCP 服务器: http://localhost:8000
- Flask 文件服务: http://localhost:6750

## 📁 项目结构

```
mat-agent-web/
├── web_mcp_app.py              # Streamlit Web 应用主入口
├── agent_mcp_server.py         # FastAPI Agent 服务 (端口 8766)
├── mcp_server.py               # MCP 工具服务器 (端口 8000)
├── oqmd.py                     # OQMD 数据库查询工具
├── agent/
│   └── langchain_mcp_agent.py  # LangChain MCP Agent 实现
├── config/                     # 配置文件
│   ├── loadenv.py              # 环境变量加载
│   ├── config.toml             # Streamlit 主题配置
│   ├── pyproject.toml          # 项目配置
│   ├── requirements.txt        # Python 依赖
│   ├── uv.lock                 # uv 锁文件
│   ├── .env                    # 环境变量 (包含密钥)
│   └── .env.example            # 环境变量模板
├── db/                         # 数据库
│   ├── databasemanage.py       # SQLite 数据库管理
│   ├── matagent.db             # 材料数据库
│   ├── matagent_history.db     # 聊天历史
│   └── matagent_server_history.db  # 服务器聊天历史
├── cache/                      # 缓存/输出
│   ├── temp_images/            # 2D 结构图缓存
│   ├── temp_3d/                # 3D 可视化 HTML 缓存
│   └── structure_info.json     # 结构元数据
├── server/                     # 后端服务
│   ├── flask_server.py         # 2D/3D 可视化文件服务 (端口 6750)
│   └── tryssh.py               # SSH/VASP 远程操作
├── myml/                       # 机器学习模型
│   ├── bandgap_predict.py     # 带隙预测模型
│   ├── featurizer.py          # 特征提取
│   ├── ion_conductivity.py    # 离子电导率预测
│   ├── atomic_orbital_calc.py # 原子轨道计算
│   ├── *.csv, *.json          # 训练数据与模型文件
└── web/
    └── assets/                 # Web 静态资源
        └── logo.png            # Logo 图片
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
- 及其他工具

### server/flask_server.py
提供 2D 图片和 3D HTML 文件服务：
- 2D 结构图生成与缓存
- 3D 交互式结构可视化
- 结构元数据持久化

### server/tryssh.py
SSH 远程连接工具：
- VASP 任务远程提交
- 计算结果远程提取
- 远程文件管理

## 🎨 自定义配置

### 添加 Logo

将 logo 图片放入 `web/assets/logo.png`，支持 PNG/JPG/SVG 格式。

### 修改主题颜色

编辑 `config/config.toml`：

```toml
[theme]
primaryColor = "#3B82F6"      # 主色调（按钮等）
backgroundColor = "#FFFFFF"   # 背景色
secondaryBackgroundColor = "#F8F9FA"
textColor = "#1F2937"
```

### 修改系统提示词

设置环境变量：

```env
# config/.env
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
from dotenv import load_dotenv
load_dotenv("config/.env")

agent = MatAgentMCP()
result = agent.chat("搜索带隙大于2eV的氧化物材料")
```

### VASP 任务管理

```python
# 通过 Web 界面或 API 创建和提交 VASP 任务
POST /vasp/create_task
{
  "formula": "LiFePO4",
  "cif_path": "./LiFePO4.cif"
}

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

### 🧠 AI 与 LLM 框架
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM 应用开发框架
- **[LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)** - Model Context Protocol 的官方适配器
- **[FastMCP](https://github.com/jlowin/fastmcp)** - Model Context Protocol 的 Python 框架

### 🌐 Web 应用框架
- **[Streamlit](https://streamlit.io/)** - 构建数据驱动 Web 应用的开源 Python 框架
- **[FastAPI](https://fastapi.tiangolo.com/)** - 现代 Python Web 框架

### 🧪 材料科学计算
- **[Pymatgen](https://pymatgen.org/)** - 强大的开源 Python 材料分析库
- **[ASE](https://ase-lib.org/)** - 原子模拟环境
- **[Materials Project](https://materialsproject.org/)** - 开源材料数据库
- **[OQMD](https://oqmd.org/)** - 开放量子材料数据库

### 📊 补充工具与框架
- **NumPy** - Python 科学计算的基础库
- **Pandas** - 数据分析和操作工具
- **Matplotlib** - Python 绘图库
- **Plotly** - 交互式可视化库

---

<p align="center">
  Made with ❤️ for Materials Science
</p>