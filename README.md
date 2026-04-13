# MatAgent — AI 驱动的材料科学智能助手

基于 **LangChain + MCP + LLM** 的材料科学研究平台，集成材料数据库查询、晶体结构建模、ML 性质预测、VASP 远程计算全流程。

---

## ✨ 功能特性

### 核心能力

| 模块 | 功能描述 |
|------|----------|
| 🤖 **AI 对话** | 支持 DeepSeek / GLM-5 大模型，流式响应，智能工具调用 |
| 🔍 **材料搜索** | Materials Project / OQMD 数据库查询，多条件筛选 |
| 🧠 **性质预测** | ALIGNN (16种性质) / XGBoost (带隙) |
| 📊 **可视化** | 2D 结构图 + 3D 交互式结构查看器 |
| 💻 **远程计算** | SSH 连接管理，VASP 任务全生命周期控制 |
| 📈 **结果分析** | 能带/DOS 可视化，预测结果对比展示 |

### 新增功能 (v2.x)

- ✅ **ALIGNN 多性质预测**: 支持预测 16 种材料物理化学性质
- ✅ **多模型支持**: deepseek-chat, deepseek-reasoner, glm-5 三种 LLM
- ✅ **SSE 流式响应**: 实时流式输出 AI 回复
- ✅ **Content Blocks**: 工具调用位置精确记录
- ✅ **INCAR 编辑器**: 可视化 VASP 参数配置
- ✅ **DOS 综合分析图**: 2×3 多子图布局（TDOS/PDOS/元素贡献）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户层 (User Interface)                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           Streamlit Web App (端口: 8501)                     │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
│  │  │ AI Chat  │ │Material  │ │Property  │ │  VASP    │       │    │
│  │  │  对话    │ │ Search   │ │ Predict  │ │ Tasks    │       │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────────────┐
│                        服务层 (Service Layer)                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │         Agent Server - FastAPI (端口: 8766)                  │    │
│  │  • 会话管理 (Session Management)                              │    │
│  │  • 聊天历史持久化 (SQLite)                                    │    │
│  │  • SSE 流式响应 (Server-Sent Events)                         │    │
│  │  • Agent 调度与负载均衡                                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ MCP Protocol
┌─────────────────────────────────────────────────────────────────────┐
│                        工具层 (Tool Layer)                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │          MCP Server - FastMCP (端口: 8000)                    │    │
│  │  ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐   │    │
│  │  │ Materials Project│ │  OQMD DB   │ │ ML Prediction   │   │    │
│  │  │     API          │ │   Query     │ │   Models        │   │    │
│  │  ├─────────────────┤ ├─────────────┤ ├─────────────────┤   │    │
│  │  │  Visualization   │ │  File I/O   │ │  ALIGNN Model   │   │    │
│  │  │  (ASE/Matplotlib)│ │  Utilities  │ │  (JARVIS)       │   │    │
│  │  └─────────────────┘ └─────────────┘ └─────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      外部服务 (External Services)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │Materials     │ │ Remote HPC   │ │ Local ML     │               │
│  │Project API   │ │ Cluster      │ │ Models       │               │
│  │(REST API)    │ │(SSH/VASP)    │ │(XGBoost/ALIGNN)│              │
│  └──────────────┘ └──────────────┘ └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
用户输入 → Streamlit UI → Agent Server (HTTP)
                              ↓
                    LangChain Agent 解析意图
                              ↓
                   ┌──────────┴──────────┐
                   ▼                     ▼
            MCP Tool Call          Direct Response
                   ↓
        ┌─────────┼─────────┐
        ▼         ▼         ▼
    MP API    OQMD DB   ML Model
        │         │         │
        └─────────┴─────────┘
                   ↓
            结果聚合 + 格式化
                   ↓
         SSE 流式返回给前端
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.10 (推荐 3.13.4)
- uv (包管理器) 或 pip
- SSH 访问权限 (用于 VASP 计算)

### 1. 克隆仓库

```bash
git clone https://github.com/matagent/mat-agent-web.git
cd mat-agent-web
git checkout file-update-logic
```

### 2. 安装依赖

```bash
# 使用 uv (推荐)
cd config
uv sync

# 或使用 pip
pip install -r config/requirements.txt
```

### 3. 配置环境变量

```bash
cp config/.env.example config/.env
# 编辑 config/.env 填入必要信息
```

### 4. 启动服务

**方式一：按顺序启动（推荐开发调试）**

```bash
# 激活虚拟环境
source ~/mat-agent-web/config/.venv/bin/activate

# 终端 1: 启动 MCP Server（工具服务）
python mcp_server.py
# → http://localhost:8000

# 终端 2: 启动 Agent Server（API 服务）
uv run --env-file config/.env python agent_mcp_server.py
# → http://localhost:8766

# 终端 3: 启动 Web 界面
streamlit run web_mcp_app.py
# → http://localhost:8501
```

**方式二：使用启动脚本**

```bash
cat > start_all.sh << 'EOF'
#!/bin/bash
# 激活虚拟环境
source ~/mat-agent-web/config/.venv/bin/activate

# 终端 1: 启动 MCP Server（工具服务）
python mcp_server.py &
echo "✅ MCP Server 启动中 → http://localhost:8000"
sleep 2

# 终端 2: 启动 Agent Server（API 服务）
uv run --env-file config/.env python agent_mcp_server.py &
echo "✅ Agent Server 启动中 → http://localhost:8766"
sleep 3

# 终端 3: 启动 Web 界面
streamlit run web_mcp_app.py
echo "✅ Web 界面启动中 → http://localhost:8501"
EOF

# 添加执行权限并运行
chmod +x start_all.sh && ./start_all.sh
```


### 5. 访问应用

打开浏览器访问: **http://localhost:8501**

---

## ⚙️ 环境变量配置

复制 `config/.env.example` 为 `config/.env` 并填写：

| 变量 | 说明 | 必需 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | ✅ |
| `ZAI_API_KEY` | 智谱 AI API 密钥 | ✅ |
| `mp_API_KEY` | Materials Project API Key | ✅ |
| `local_HOST` | 本地 IP 地址（Flask 服务绑定） | ✅ |
| `HOST` | 远程服务器 IP | ✅ |
| `PORT` | 远程 SSH 端口 | ✅ |
| `USERNAME` | SSH 用户名 | ✅ |
| `PASSWORD` | SSH 密码 | ✅ |
| `base_dir` | 远程工作目录 | ✅ |

格式参考：
```ini
DEEPSEEK_API_KEY = 'sk-your-key'
ZAI_API_KEY = 'your-zhipu-key'
mp_API_KEY = 'your-mp-key'
local_HOST = '127.0.0.1'
HOST = '192.168.x.x'
PORT = 22
USERNAME = 'user'
PASSWORD = 'pass'
base_dir = '/home/user/work'
```
---

## MCP 工具列表（21 个）

### 材料数据库

| 函数 | 说明 |
|------|------|
| `search_materials_from_mp` | Materials Project 材料搜索 |
| `get_material_all_infomation_by_id` | MP 材料详情（含能带/DOS） |
| `get_band_gap` | MP 获取带隙信息 |
| `get_material_structure_from_mp` | MP 获取晶体结构 |
| `get_material_project_page` | MP 项目页面信息 |
| `search_materials_from_oqmd` | OQMD 数据库搜索 |
| `get_material_structure_from_oqmd` | OQMD 获取晶体结构 |

### 结构建模

| 函数 | 说明 |
|------|------|
| `build_structure` | 构建晶体结构（立方/四方/六方/正交等） |

### VASP 远程计算

| 函数 | 说明 |
|------|------|
| `create_task` | 创建 VASP 计算任务 |
| `list_task_directories` | 列出远程任务目录 |
| `check_squeue` | 查看 SLURM 作业队列 |
| `create_mission` | 创建子任务（relax/scf/band/dos） |
| `submit_mission` | 提交任务到 SLURM |
| `modify_incar` | 编辑 INCAR 参数 |
| `extract_result` | 提取计算结果（能带图/DOS图） |
| `execute_command` | 执行远程命令 |
| `extract_file` | 提取远程文件内容 |
| `read_file` | 读取本地文件 |

### ML 性质预测

| 函数 | 说明 |
|------|------|
| `predict_band_gap` | XGBoost 本地快速带隙预测 |
| `predict_with_alignn` | ALIGNN 远程多性质预测（16 种） |

### 其他

| 函数 | 说明 |
|------|------|
| `get_time` | 获取服务器时间 |

---

## 项目结构

```
mat-agent-web/
├── web_mcp_app.py              # Streamlit Web 入口（5 个功能面板）
├── agent_mcp_server.py         # FastAPI Agent Server（端口 8766）
├── mcp_server.py               # MCP 工具服务器（端口 8000）
├── flask_server.py             # 2D 图片 / 3D HTML 文件服务（端口 6750）
├── agent/
│   └── langchain_mcp_agent.py  # LangChain + MCP Agent 核心
├── server/
│   └── tryssh.py               # SSH 远程连接模块
├── myml/
│   ├── bandgap_predict.py      # XGBoost 带隙预测
│   ├── featurizer.py           # 特征工程
│   └── xgb_model.json          # XGBoost 训练模型
├── db/
│   └── databasemanage.py       # SQLite 数据库管理
├── oqmd.py                     # OQMD 数据库查询
├── config/
│   ├── .env.example            # 环境变量模板
│   ├── loadenv.py              # 环境变量加载
│   ├── pyproject.toml          # Python 项目配置
│   └── requirements.txt        # 依赖列表
├── cache/
│   ├── temp_images/            # 2D 结构图缓存
│   ├── temp_3d/                # 3D HTML 缓存
│   └── structure_info.json     # 结构元数据
├── calculation_output/         # VASP 计算结果输出
├── custom_structures/          # 用户自定义结构（CIF + images）
├── cifs/                       # CIF 文件存储
├── web/
│   └── assets/                 # Web 静态资源（logo 等）
├── matagent.db                 # 材料数据库
├── matagent_history.db         # 客户端聊天记录
└── matagent_server_history.db  # 服务端全局记录
```

---

## 数据库说明

| 文件 | 使用者 | 内容 |
|------|--------|------|
| `matagent.db` | Streamlit 客户端 | 材料（materials）+ 本地聊天（chat_history） |
| `matagent_history.db` | Streamlit 客户端 | 会话（sessions）+ 聊天记录 + 工具调用（tool_calls） |
| `matagent_server_history.db` | Agent Server | 全局会话 + 聊天历史（含 content_blocks/model/duration） |

> 注意：Server 版本聚合所有客户端请求，体积通常更大。

---

## 常见问题

### 1. Agent 启动报错「必须提供 api_key」

确保使用正确命令启动 Agent Server：

```bash
✅ uv run --env-file config/.env python agent_mcp_server.py
❌ python agent_mcp_server.py           # 未加载环境变量
❌ uv run python agent_mcp_server.py    # 缺少 --env-file
```

### 2. SQLite 多线程错误

在 Streamlit 中使用 SQLite 必须用**模块级函数**，每次创建独立连接。不要使用类方法持有共享连接。

### 3. StructuredTool 不可直接调用

`@tool` 装饰的函数变为 `StructuredTool` 对象，直接调用会报错。请通过 `MatAgentMCP` 类的 wrapper 方法调用。

### 4. ALIGNN 预测慢或失败

`predict_with_alignn` 通过 SSH 上传 CIF 到远端服务器运行 ALIGNN，单次预测耗时较长。建议一次只预测 1~3 个结构，并确认远端服务器网络通畅。

### 5. 3D 结构不显示

`flask_server.py` 需要单独启动（端口 6750），负责提供 3D HTML 文件的 HTTP 访问。
