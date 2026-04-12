# MatAgent V1.0 - 材料科学智能设计平台

![alt text](web/assets/logo.png)

## 📖 项目概述

MatAgent（材料科学智能设计平台）是一个基于 **MCP (Model Context Protocol)** 协议的材料科学智能助手，集成了材料数据库查询、晶体结构分析、机器学习预测和 VASP 第一性原理计算任务管理等功能。该系统于 **2025年4月10日** 开发完成，旨在为材料科学研究人员提供一站式的材料设计解决方案。

随着材料科学研究的深入，传统的材料设计方法面临效率低、成本高、周期长等挑战。MatAgent 通过整合大语言模型（LLM）与材料科学计算工具，实现了智能化的材料研究辅助。用户可以通过自然语言与系统交互，快速完成材料数据查询、结构建模、性能预测和计算任务管理等操作，显著提升研究效率。

本系统采用现代化的微服务架构设计，将前端交互、业务逻辑和底层工具解耦，通过 MCP 协议实现标准化的工具调用。这种架构不仅保证了系统的可扩展性和可维护性，还为后续功能扩展提供了良好的基础。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🤖 **多模型支持** | 支持 DeepSeek (chat/reasoner) 和智谱 GLM-5 等大语言模型，可自由切换 |
| 🔍 **材料数据库** | 集成 Materials Project、OQMD 等主流材料数据库，支持元素筛选和精确查询 |
| 🏗️ **结构建模** | 支持自定义晶体结构构建，2D/3D 可视化预览，CIF 文件导出 |
| 🧠 **ML 预测** | 基于机器学习的材料性质快速预测（带隙、离子电导率、能带边缘） |
| 💻 **VASP 集成** | 远程 SSH 连接，支持计算任务创建、提交、监控和结果提取 |
| 💬 **会话管理** | 持久化聊天历史，支持多会话切换和上下文记忆 |
| 🌐 **Web 界面** | 基于 Streamlit 的现代化交互界面，无需编程即可使用 |

---

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                     用户层                                           │
│                              Web 界面 (Streamlit)                                    │
│                          http://localhost:8501                                     │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐  │
│  │  智能体对话   │   材料查询    │   结构建模    │  ML 预测     │   VASP 任务管理   │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ HTTP/REST API
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    服务层 (端口 8766)                                │
│                             Agent MCP Server (FastAPI)                              │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │  • HTTP API 网关  • 会话管理 (SQLite)  • MCP 工具代理  • 流式响应处理       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              │ MCP Protocol
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    工具层 (端口 8000)                                │
│                              MCP Server (FastMCP)                                   │
│  ┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐  │
│  │ 材料搜索   │ 结构获取   │ 结构建模   │ 可视化     │ ML 预测    │ VASP 管理  │  │
│  │            │            │            │            │            │            │  │
│  └────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │                  │                  │                   │
         ▼                  ▼                  ▼                   ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐ ┌──────────────────────┐
│ Materials       │ │  文件服务     │ │  ML 模型        │ │   SSH 远程           │
│ Project API    │ │ flask_server │ │ myml/           │ │   tryssh.py          │
│ OQMD Web       │ │ (端口 6750)  │ │                 │ │                      │
└─────────────────┘ └──────────────┘ └─────────────────┘ └──────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────────┐
                                    │   SQLite 数据库     │
                                    │  databasemanage.py  │
                                    └─────────────────────┘
```

### 分层说明

| 层级 | 组件 | 职责 |
|------|------|------|
| **用户层** | Streamlit Web 界面 | 提供 5 大功能面板的用户交互 |
| **服务层** | FastAPI Agent Server | HTTP API、Session 管理、工具代理 |
| **工具层** | FastMCP Server | MCP 协议工具集、20+ 工具函数 |
| **数据层** | SQLite + 文件系统 | 数据持久化、模型文件、缓存 |

---

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

# Flask 文件服务器 (可选，默认为本机 IP)
FLASK_HOST=127.0.0.1
FLASK_PORT=6750
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

| 服务 | 地址 |
|------|------|
| Web 界面 | http://localhost:8501 |
| Agent API | http://localhost:8766 |
| MCP 服务器 | http://localhost:8000 |
| Flask 文件服务 | http://localhost:6750 |

---

## 📁 项目结构

```
mat-agent-web/
├── web_mcp_app.py                   # Streamlit Web 应用主入口 (2195行)
├── agent_mcp_server.py              # FastAPI Agent 服务 (端口 8766)
├── mcp_server.py                    # MCP 工具服务器 (端口 8000)
├── oqmd.py                          # OQMD 数据库查询工具
│
├── agent/                           # Agent 核心模块
│   └── langchain_mcp_agent.py      # LangChain MCP Agent 实现
│
├── config/                          # 配置文件
│   ├── loadenv.py                   # 环境变量加载
│   ├── config.toml                  # Streamlit 主题配置
│   ├── pyproject.toml               # 项目配置
│   ├── requirements.txt            # Python 依赖
│   ├── uv.lock                      # uv 锁文件
│   ├── .env                         # 环境变量 (包含密钥)
│   └── .env.example                 # 环境变量模板
│
├── db/                              # 数据库模块
│   ├── __init__.py
│   ├── databasemanage.py            # SQLite 数据库管理
│   ├── matagent.db                  # 材料数据库
│   ├── matagent_history.db          # 聊天历史
│   └── matagent_server_history.db   # 服务器聊天历史
│
├── cache/                           # 缓存/输出目录
│   ├── temp_images/                 # 2D 结构图缓存
│   ├── temp_3d/                     # 3D 可视化 HTML 缓存
│   └── structure_info.json          # 结构元数据
│
├── server/                          # 后端服务
│   ├── flask_server.py              # 2D/3D 可视化文件服务 (端口 6750)
│   └── tryssh.py                    # SSH/VASP 远程操作 (40KB+)
│
├── myml/                            # 机器学习模型
│   ├── bandgap_predict.py          # 带隙预测模型
│   ├── featurizer.py                # 特征提取
│   ├── ion_conductivity.py          # 离子电导率预测
│   ├── atomic_orbital_calc.py       # 原子轨道计算
│   ├── element_features.csv         # 元素特征数据
│   ├── element_features_bandgap.csv # 带隙预测特征
│   ├── nist_atomic_data_lda(eV).csv # 原子轨道能量数据
│   ├── xgb_model.json               # 带隙预测 XGBoost 模型
│   └── halide_xgb.json              # 离子电导率 XGBoost 模型
│
└── web/                             # Web 静态资源
    └── assets/
        └── logo.png                 # Logo 图片
```

---

## 🔧 核心模块说明

### 1. web_mcp_app.py - Streamlit Web 应用主入口

**文件大小**：约 2195 行代码

**主要功能**：
- 构建材料科学智能设计平台的图形用户界面
- 提供 5 大功能面板的 Web 交互入口

**核心页面**：

| 页面 | 函数 | 功能描述 |
|------|------|----------|
| 智能体对话 | `chat_page()` | 支持流式响应的 LLM 对话界面 |
| 材料查询 | `material_search_page()` | Materials Project 数据库查询 |
| 结构建模 | `structure_page()` | 自定义晶体结构构建与可视化 |
| ML 预测 | `ml_prediction_page()` | 材料性质机器学习预测 |
| VASP 任务 | `vasp_page()` | VASP 计算任务管理 |

**关键函数**：

```python
# 对话功能
chat_with_mcp()           # 非流式对话
chat_with_mcp_stream()    # SSE 流式对话

# 材料操作
search_materials()        # 搜索材料
get_material_structure()  # 获取晶体结构
build_structure()         # 构建自定义结构

# ML 预测
predict_bandgap()         # 带隙预测

# VASP 任务
vasp_create_task()        # 创建 VASP 任务
vasp_submit()             # 提交计算任务
vasp_extract()            # 提取计算结果
```

**技术特点**：
- 使用 `session_state` 管理多会话状态
- 支持 `content_blocks` 实现消息中文本与工具结果有序展示
- 集成 SSE 流式输出，处理长对话场景

---

### 2. agent_mcp_server.py - FastAPI Agent 服务

**文件大小**：约 32KB

**主要功能**：
- 提供 HTTP API 供 Web 前端调用
- 管理 MCP Agent 生命周期
- 会话持久化（SQLite）
- 支持流式聊天响应

**核心 API 端点**：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat` | POST | 非流式对话 |
| `/chat/stream` | POST | 流式对话 (SSE) |
| `/sessions/*` | GET/POST/DELETE | 会话管理 |
| `/materials/search` | GET | 材料搜索 |
| `/materials/structure/{id}` | GET | 获取结构 |
| `/predict_bandgap` | GET | 带隙预测 |
| `/structure/build` | POST | 结构建模 |
| `/vasp/*` | POST/GET | VASP 任务管理 |

**数据库操作**：
```python
init_database()           # 初始化数据库表
add_chat_message()        # 添加聊天消息
get_chat_history()        # 获取历史消息
list_sessions()           # 列出所有会话
delete_session()          # 删除会话
update_session_name()    # 更新会话名称
```

**工具调用**：
```python
invoke_mcp_tool_direct()     # 直接调用 MCP 工具
invoke_tool_with_agent_chat() # 通过 Agent 调用工具
```

---

### 3. mcp_server.py - MCP 工具服务器

**文件大小**：约 68KB

**主要功能**：
使用 FastMCP 框架提供材料科学计算工具集，是系统的工具层核心。

**MCP 工具列表**：

| 工具名称 | 功能描述 |
|----------|----------|
| `get_time` | 获取当前时间 |
| `read_file` | 读取本地文件 |
| `get_material_project_page` | 获取 MP 页面信息 |
| `search_materials_from_mp` | MP 材料搜索 |
| `get_material_structure_from_mp` | MP 获取晶体结构 |
| `get_band_gap` | 获取 MP 带隙数据 |
| `get_material_all_infomation_by_id` | 获取 MP 材料完整信息 |
| `search_materials_from_oqmd` | OQMD 材料搜索 |
| `get_material_structure_from_oqmd` | OQMD 获取晶体结构 |
| `build_structure` | 构建自定义晶体结构 |
| `get_plot_url` | 获取结构图片 URL |
| `visualize_structure` | 可视化晶体结构 |
| `get_structure_plot` | 获取结构绘图 |
| `plot_vasp_band` | 绘制 VASP 能带图 |
| `plot_vasp_dos_analysis` | 绘制 VASP 态密度图 |
| `predict_band_gap` | 带隙预测 |
| `create_task` | 创建 VASP 任务 |
| `list_task_directories` | 列出任务目录 |
| `check_squeue` | 检查 SLURM 队列 |
| `create_mission` | 创建计算任务 |
| `submit_mission` | 提交计算任务 |
| `modify_incar` | 修改 INCAR 参数 |
| `extract_result` | 提取计算结果 |
| `execute_command` | 执行远程命令 |
| `extract_file` | 提取远程文件 |

**可视化功能**：
```python
get_plot_url()             # 生成结构图片 URL
visualize_structure()      # 生成 3D HTML 可视化
plot_vasp_band()           # VASP 能带图绘制
plot_vasp_dos_analysis()  # VASP 态密度分析
```

---

### 4. agent/langchain_mcp_agent.py - LangChain MCP Agent 核心

**主要功能**：
基于 LangChain 和 langchain-mcp-adapters 实现，通过 MCP 协议连接工具服务器。

**核心类**：

| 类名 | 功能 |
|------|------|
| `MatAgentMCP` | 异步 Agent 实现 |
| `MatAgentMCPSync` | 同步包装类 |
| `ReasoningCallbackHandler` | 捕获推理过程 |

**MatAgentMCP 核心方法**：

```python
class MatAgentMCP:
    async def connect(self)      # 连接 MCP Server
    async def disconnect(self)   # 断开连接
    async def chat(self)         # 对话（返回完整结果）
    async def chat_stream(self)  # 流式对话
```

**支持的模型**：
| 模型 ID | 描述 |
|---------|------|
| `deepseek-chat` | DeepSeek 对话模型 |
| `deepseek-reasoner` | DeepSeek 推理模型 |
| `glm-5` | 智谱 GLM-5 模型 |

**技术特点**：
- 使用 ReAct 推理模式
- 支持工具调用链管理
- 集成 ReasoningCallbackHandler 捕获 reasoning_content

---

### 5. server/flask_server.py - 2D/3D 可视化文件服务

**文件大小**：约 15KB

**主要功能**：
提供 2D 图片和 3D HTML 结构可视化文件服务。

**核心类**：

```python
class MatFileServer:
    def add_image(self)              # 保存图片并返回 URL
    def add_image_file(self)         # 保存已有图片文件
    def add_html_with_info(self)     # 保存 3D HTML 并存储结构信息
    def add_html_file(self)          # 仅保存 HTML 文件
```

**路由端点**：

| 端点 | 功能 |
|------|------|
| `/image/` | 2D 结构图服务 |
| `/3d/` | 3D HTML 可视化服务 |
| `/view/` | 结构预览 |
| `/cif/` | CIF 文件下载 |
| `/` | 根路径服务 |

**持久化功能**：
- `_load_structure_info()`: 加载结构信息
- `_save_structure_info()`: 保存结构信息
- `cleanup_old_files()`: 清理旧文件

---

### 6. server/tryssh.py - SSH/VASP 远程操作

**文件大小**：约 40KB

**主要功能**：
通过 SSH 远程连接管理 VASP 计算任务。

**核心类**：

```python
class VaspTaskInitializer:
    """SSH 连接管理类（上下文管理器）"""
    
    def link(self)                  # 检查连接状态
    def create_task(self)           # 创建任务目录并上传 CIF
    def get_task_directories(self)  # 列出任务目录
    def check_squeue(self)          # 检查 SLURM 队列
```

**计算任务类型**：

| 方法 | 功能 |
|------|------|
| `relax()` | 结构优化 |
| `scf()` | 自洽计算 |
| `band_calc()` | 能带计算 |
| `dos_calc()` | 态密度计算 |

**结果提取方法**：

| 方法 | 功能 |
|------|------|
| `extract_relax_info()` | 提取结构优化结果 |
| `extract_scf_info()` | 提取自洽计算结果 |
| `extract_band_info()` | 提取能带计算结果 |
| `extract_dos_info()` | 提取态密度结果 |
| `extract_file()` | 提取指定文件 |

**INCAR 操作**：
```python
modify_incar_file()     # 修改 INCAR 参数
execute_command()      # 执行远程命令（含安全检查）
excute_python()         # 执行远程 Python 代码
create_mission()        # 创建分步任务
submit_mission()        # 提交任务到队列
```

---

### 7. db/databasemanage.py - SQLite 数据库管理

**文件大小**：约 15KB

**主要功能**：
SQLite 数据库管理，存储材料结构数据和聊天历史。

**核心类**：

```python
class DatabaseManager:
    def add_material()                      # 添加材料
    def get_material_by_ID()                # 按 ID 查询
    def get_material_by_material_id()       # 按 MP ID 查询
    def get_material_by_elements()          # 按元素筛选
    def list_all_materials_by_pages()       # 分页列出
```

**模块级函数（聊天历史）**：

| 函数 | 功能 |
|------|------|
| `add_chat_message()` | 添加聊天消息 |
| `get_chat_history()` | 获取历史消息 |
| `delete_session()` | 删除会话 |
| `list_sessions()` | 列出所有会话 |
| `add_tool_call()` | 记录工具调用 |
| `get_tool_calls()` | 获取工具调用记录 |

---

### 8. myml/ - 机器学习预测模块

#### 8.1 bandgap_predict.py - 带隙预测模型

**主要功能**：
使用 XGBoost 模型预测材料的带隙值。

**核心函数**：

```python
normalize_formula()      # 解析化学式（支持括号和小数系数）
get_max_feature()        # 获取最大值特征
get_min_feature()        # 获取最小值特征
get_avg_feature()        # 获取平均值特征
get_range_feature()      # 获取极差特征
get_std_feature()        # 获取标准差特征
get_all_features()       # 批量计算特征
predict_bandgap()        # 带隙预测入口
```

**依赖文件**：
- `element_features_bandgap.csv`: 元素特征数据
- `xgb_model.json`: 预训练 XGBoost 模型

---

#### 8.2 featurizer.py - 特征提取

**主要功能**：
从化学式提取元素特征，包括统计特征和轨道特征。

**核心函数**：

```python
normalize_formula()      # 化学式解析
get_max_feature()        # 统计特征：最大值
get_min_feature()        # 统计特征：最小值
get_mean_feature()       # 统计特征：平均值
get_range_feature()      # 统计特征：极差
get_std_feature()        # 统计特征：标准差
get_skew_feature()       # 统计特征：偏度
get_all_features()       # 批量特征提取
calc_block_fractions()   # 计算 s/p/d/ds/f 区块比例
calc_column_fractions_df() # 计算各族元素比例
calc_orbital()           # 计算轨道特征
my_featurizer()         # 综合特征提取（集成 matminer）
```

**依赖文件**：
- `element_features.csv`: 元素特征数据
- `nist_atomic_data_lda(eV).csv`: 原子轨道能量数据

---

#### 8.3 ion_conductivity.py - 离子电导率预测

**主要功能**：
卤化物材料的离子电导率预测，包含全面的特征工程和 XGBoost 预测。

**核心函数**：

```python
ion_data                 # 离子半径、电势、电荷、电负性数据
culculate_polarization_factors()   # 极化因子计算
calculate_migration_ion_specific_features() # 迁移离子特征
culculate_ion_statistics()         # 离子统计特征
ion_coulomb_matrix_from_composition() # 库仑矩阵估算
calculate_all_configurational_entropies() # 7种构型熵
add_features_for_df()              # 综合特征添加
mix_materials()                    # 材料混合计算
predict_ionic_conductivity()      # 单材料预测
predict_mixed_ionic_conductivity() # 混合材料预测
```

**依赖文件**：
- `halide_xgb.json`: 卤化物离子电导率 XGBoost 模型

---

#### 8.4 atomic_orbital_calc.py - 原子轨道计算

**主要功能**：
基于 NIST 原子轨道数据库计算材料的能带边缘（HOMO/LUMO）。

**核心类**：

```python
class ImprovedMolecularOrbitals:
    def prepare_atomic_data()     # 预处理轨道数据
    def calculate_total_electrons() # 计算总电子数
    def build_composite_orbitals() # 构建复合轨道列表
    def calculate_band_edges()     # 电子填充算法，计算 HOMO/LUMO
    def get_data()                # 返回 HOMO/LUMO/gap
```

**依赖文件**：
- `nist_atomic_data_lda(eV).csv`: NIST 原子轨道能量数据

---

### 9. oqmd.py - OQMD 数据库查询

**主要功能**：
通过 Web 爬取方式获取 OQMD 材料数据库的晶体结构数据。

**核心函数**：

```python
search_oqmd()             # OQMD 搜索 API 调用
safe_get()               # 带重试机制的 HTTP GET 请求
get_poscar_content()     # 通过 entry_id 获取 POSCAR
parse_poscar_with_pymatgen() # 用 pymatgen 解析 POSCAR
```

---

## 🔌 API 接口详解

### 1. 聊天接口

#### 非流式聊天

```bash
POST http://localhost:8766/chat
Content-Type: application/json

{
    "session_id": "uuid",
    "message": "搜索带隙大于2eV的氧化物材料",
    "model": "deepseek-chat"
}
```

**响应示例**：
```json
{
    "session_id": "uuid",
    "response": "已为您找到以下材料...",
    "tool_calls": [...],
    "timestamp": "2025-04-10T12:00:00"
}
```

#### 流式聊天 (SSE)

```bash
POST http://localhost:8766/chat/stream
Content-Type: application/json

{
    "session_id": "uuid",
    "message": "搜索含锂的材料",
    "model": "deepseek-chat"
}
```

---

### 2. 工具调用接口

#### 材料搜索

```bash
GET http://localhost:8766/materials/search?elements=Li,Fe,O
```

#### 获取结构

```bash
GET http://localhost:8766/materials/structure/{material_id}
```

#### 预测带隙

```bash
GET http://localhost:8766/predict_bandgap?formula=SiO2
```

#### 结构建模

```bash
POST http://localhost:8766/structure/build
Content-Type: application/json

{
    "formula": "LiFePO4",
    "lattice_type": "orthorhombic",
    "a": 10.0, "b": 5.0, "c": 8.0
}
```

---

### 3. 会话管理接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/sessions` | GET | 列出所有会话 |
| `/sessions` | POST | 创建新会话 |
| `/sessions/{id}` | DELETE | 删除会话 |
| `/sessions/{id}/rename` | POST | 重命名会话 |

---

## 📊 数据库设计

### 材料数据库 (matagent.db)

**表：materials**

| 字段 | 类型 | 描述 |
|------|------|------|
| id | INTEGER | 主键 |
| material_id | TEXT | MP 材料 ID |
| formula | TEXT | 化学式 |
| structure | BLOB | pymatgen Structure (pickle) |
| created_at | DATETIME | 创建时间 |

### 聊天历史数据库 (matagent_server_history.db)

**表：sessions**

| 字段 | 类型 | 描述 |
|------|------|------|
| session_id | TEXT | 会话 UUID |
| name | TEXT | 会话名称 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

**表：messages**

| 字段 | 类型 | 描述 |
|------|------|------|
| id | INTEGER | 主键 |
| session_id | TEXT | 会话 ID |
| role | TEXT | user/assistant |
| content | TEXT | 消息内容 |
| timestamp | DATETIME | 时间戳 |

**表：tool_calls**

| 字段 | 类型 | 描述 |
|------|------|------|
| id | INTEGER | 主键 |
| session_id | TEXT | 会话 ID |
| tool_name | TEXT | 工具名称 |
| arguments | TEXT | 参数 JSON |
| result | TEXT | 结果 JSON |
| timestamp | DATETIME | 时间戳 |

---

## 🧠 ML 模型说明

### 1. 带隙预测模型

- **模型文件**：`xgb_model.json`
- **算法**：XGBoost
- **输入**：化学式
- **输出**：带隙值 (eV)
- **特征**：元素理化性质统计特征

### 2. 离子电导率预测模型

- **模型文件**：`halide_xgb.json`
- **算法**：XGBoost
- **输入**：化学式、浓度
- **输出**：离子电导率 (S/cm)
- **特征**：极化因子、离子统计、构型熵

### 3. 原子轨道计算

- **数据文件**：`nist_atomic_data_lda(eV).csv`
- **方法**：电子填充算法
- **输出**：HOMO、LUMO、gap (eV)

---

## ⚙️ 配置说明

### 环境变量 (.env)

| 变量 | 必需 | 描述 |
|------|------|------|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥 |
| `ZAI_API_KEY` | 否 | 智谱 AI API 密钥 |
| `mp_API_KEY` | 否 | Materials Project API 密钥 |
| `HOST` | 否 | VASP 服务器地址 |
| `PORT` | 否 | SSH 端口 (默认 22) |
| `USERNAME` | 否 | SSH 用户名 |
| `PASSWORD` | 否 | SSH 密码 |
| `base_dir` | 否 | VASP 任务根目录 |
| `FLASK_HOST` | 否 | Flask 服务器地址 |
| `FLASK_PORT` | 否 | Flask 服务器端口 |

### Streamlit 配置 (config.toml)

```toml
[theme]
primaryColor = "#3B82F6"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
textColor = "#1F2937"
```

---

## 💻 使用示例

### 材料查询

```python
from agent.langchain_mcp_agent import MatAgentMCP
from dotenv import load_dotenv

load_dotenv("config/.env")

agent = MatAgentMCP()
result = agent.chat("搜索带隙大于2eV的氧化物材料")
print(result)
```

### 带隙预测

```python
import requests

response = requests.get("http://localhost:8766/predict_bandgap?formula=SiO2")
print(response.json())
```

### VASP 任务管理

```python
import requests

# 创建任务
requests.post("http://localhost:8766/vasp/create_task", json={
    "formula": "LiFePO4",
    "cif_path": "./LiFePO4.cif"
})

# 提交计算
requests.post("http://localhost:8766/vasp/submit", json={
    "task_directory": "/path/to/task",
    "mission": "relax"
})
```

---

## 🛠️ 故障排查

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| MCP 连接失败 | 端口 8766 未启动 | 检查 agent_mcp_server 是否运行 |
| 材料搜索无结果 | API 密钥配置错误 | 检查 mp_API_KEY 是否有效 |
| SSH 连接失败 | 网络或配置错误 | 检查 HOST、USERNAME、PASSWORD |
| 预测结果异常 | 模型文件缺失 | 检查 myml/ 目录下的模型文件 |

### 日志查看

```bash
# 查看 agent 服务日志
tail -f logs/agent.log

# 查看 MCP 服务日志
tail -f logs/mcp.log
```

---

## 📝 版本历史

- **V1.0** (2025-04-10): 初始版本发布
  - 5 大功能面板
  - MCP 协议集成
  - ML 预测模型
  - VASP 任务管理

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

### 🧠 AI 与 LLM 框架

- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM 应用开发框架
- **[LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)** - Model Context Protocol 的官方适配器
- **[FastMCP](https://github.com/jlowin/fastmcp)** - Model Context Protocol 的 Python 框架

### 🌐 Web 应用框架

- **[Streamlit](https://streamlit.io/)** - 构建数据驱动 Web 应用的开源 Python 框架
- **[FastAPI](https://www.tiangolo.com/fastapi/)** - 现代 Python Web 框架
- **[Flask](https://flask.palletsprojects.com/)** - 轻量级 Web 框架

### 🧪 材料科学计算

- **[Pymatgen](https://pymatgen.org/)** - 强大的开源 Python 材料分析库
- **[ASE](https://ase-lib.org/)** - 原子模拟环境
- **[Materials Project](https://materialsproject.org/)** - 开源材料数据库
- **[OQMD](https://oqmd.org/)** - 开放量子材料数据库

### 📊 机器学习

- **[XGBoost](https://xgboost.readthedocs.io/)** - 梯度提升树算法
- **[Matminer](https://github.com/materialsproject/matminer)** - 材料特征工程

### 📊 补充工具与框架

- **NumPy** - Python 科学计算的基础库
- **Pandas** - 数据分析和操作工具
- **Matplotlib** - Python 绘图库
- **Plotly** - 交互式可视化库

---

<p align="center">
  Made with ❤️ for Materials Science
</p>