# MatAgent - AI 驱动的材料科学智能助手

<p align="center">
  <strong>基于 LangChain + MCP + LLM 的材料科学研究平台</strong>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#系统架构">系统架构</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#api-接口">API 接口</a> •
  <a href="#mcp工具">MCP 工具</a> •
  <a href="#vasp工作流">VASP 工作流</a>
</p>

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
git clone <repository-url>
cd mat-agent-web
git checkout file-update-logic
```

### 2. 安装依赖

```bash
# 使用 uv (推荐)
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
# 终端 1: 启动 Flask 文件服务器（2D/3D 可视化）
python flask_server.py
# → http://localhost:6750

# 终端 2: 启动 MCP Server（工具服务）
python mcp_server.py
# → http://localhost:8000

# 终端 3: 启动 Agent Server（API 服务）
uv run --env-file config/.env python agent_mcp_server.py
# → http://localhost:8766

# 终端 4: 启动 Web 界面
streamlit run web_mcp_app.py
# → http://localhost:8501
```

**方式二：使用启动脚本**

```bash
# 创建启动脚本
cat > start_all.sh << 'EOF'
#!/bin/bash
python flask_server.py &
sleep 2
python mcp_server.py &
sleep 2
uv run --env-file config/.env python agent_mcp_server.py &
sleep 3
streamlit run web_mcp_app.py
EOF
chmod +x start_all.sh && ./start_all.sh
```

### 5. 访问应用

打开浏览器访问: **http://localhost:8501**

---

## ⚙️ 环境变量配置

编辑 `config/.env` 文件：

```ini
# ========== 必需配置 ==========

# DeepSeek API 密钥 (必需)
DEEPSEEK_API_KEY=sk-your-deepseek-api-key

# Materials Project API 密钥
mp_API_KEY=your-materials-project-api-key

# SSH 远程连接配置 (用于 VASP 计算)
HOST=your-hpc-cluster.com
PORT=22
USERNAME=your_username
PASSWORD=your_password_or_ssh_key_path

# 远程工作目录
base_dir=/home/username/vasp_work

# ========== 可选配置 ==========

# LLM 模型选择 (deepseek-chat | deepseek-reasoner | glm-5)
LLM_MODEL=deepseek-chat

# API 基础 URL (如使用代理)
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1

# ZhipuAI GLM 配置 (使用 glm-5 时需要)
ZHIPUAI_API_KEY=your-zhipuai-api-key
```

### 环境变量说明

| 变量名 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | - | DeepSeek API 密钥 |
| `mp_API_KEY` | ✅ | - | Materials Project API 密钥 |
| `HOST` | ✅ | - | SSH 主机地址 |
| `PORT` | ❌ | `22` | SSH 端口号 |
| `USERNAME` | ✅ | - | SSH 用户名 |
| `PASSWORD` | ✅ | - | SSH 密码或密钥路径 |
| `base_dir` | ✅ | - | 远程 VASP 工作目录 |
| `LLM_MODEL` | ❌ | `deepseek-chat` | LLM 模型选择 |

---

## 📁 项目结构

```
mat-agent-web/
│
├── web_mcp_app.py                 # Streamlit Web 应用主入口 (~2300行)
├── agent_mcp_server.py            # FastAPI Agent 服务 (~980行)
├── mcp_server.py                  # MCP 工具服务器 (~2045行)
├── flask_server.py                # Flask 文件服务 (~407行)
├── tryssh.py                      # SSH 远程操作模块 (~1473行)
├── oqmd.py                        # OQMD 数据库查询接口
│
├── agent/                         # Agent 核心模块
│   └── langchain_mcp_agent.py     # LangChain MCP Agent (~623行)
│
├── myml/                          # 机器学习预测模型
│   ├── bandgap_predict.py         # XGBoost 带隙预测
│   ├── alignn_predict.py          # ALIGNN 多性质预测
│   ├── models/                    # 已训练模型文件
│   │   └── xgboost_bandgap.json
│   └── bandgap_dataset.csv        # 带隙训练数据集
│
├── server/                        # 服务层模块
│   └── flask_server.py            # Flask 文件服务器 (2D/3D 可视化)
│
├── db/                            # 数据库模块
│   └── databasemanage.py          # SQLite 数据库管理 (~427行)
│
├── config/                        # 配置文件
│   ├── .env                       # 环境变量 (不提交到 Git)
│   ├── .env.example               # 环境变量示例
│   ├── loadenv.py                 # 环境变量加载器
│   ├── pyproject.toml             # Python 项目配置 (uv)
│   └── requirements.txt           # pip 依赖列表
│
├── cache/                         # 缓存目录
│   ├── temp_images/               # 2D 结构图缓存
│   ├── temp_3d/                   # 3D HTML 文件缓存
│   └── structure_info.json        # 结构元数据缓存
│
├── db_file/                       # SQLite 数据库文件
│   ├── matagent_history.db        # Web 应用聊天历史
│   └── matagent_server_history.db # Agent Server 聊天历史
│
├── web/                           # Web 静态资源
│   └── assets/                    # 图片、图标等
│
├── AGENTS.md                      # 开发者快速参考指南
└── README.md                      # 本文件
```

---

## 📚 核心模块详解

### 1. web_mcp_app.py - Streamlit Web 应用

**端口**: `8501` | **技术栈**: Streamlit, Requests, Plotly

主要功能面板:

#### Panel 1: AI 对话 (Chat)

- **智能对话界面**: 支持自然语言输入，AI 自动理解材料科学问题
- **流式输出**: 实时显示 AI 回复过程
- **上下文记忆**: 保持对话连贯性，支持多轮对话
- **工具调用展示**: 显示 AI 调用的工具及中间结果
- **快捷操作按钮**:
  - 📋 复制回复
  - 🔄 重新生成
  - 🗑️ 删除消息
  - 💾 导出对话

#### Panel 2: 材料搜索 (Material Search)

**搜索维度**:
- 化学体系 (Elements System): 如 "Li-Fe-O", "Si-Ge"
- 材料 ID: Materials Project ID (如 "mp-1234")
- 公式 (Formula): 化学式搜索

**筛选条件**:
- 能带带隙范围 (Band Gap Range): 0-5 eV
- 是否稳定 (Stability): e_above_hull < 0.05 eV/atom
- 最大结果数量 (Max Results): 默认 20 条

**结果展示**:
- 表格形式: Material ID, Formula, Band Gap, Stability, Space Group
- 详情卡片: 点击查看结构图、详细属性
- 批量导出: CSV/JSON 格式

#### Panel 3: 性质预测 (Property Prediction)

支持两种预测模式:

| 模型 | 预测目标 | 方法 | 说明 |
|------|----------|------|------|
| **XGBoost - 带隙** | band_gap_eV | XGBoost Regression | 基于 composition 特征 |
| **ALIGNN** | 16 种性质 | 图神经网络 | 需要 CIF 结构文件 |

**ALIGNN 可预测的 16 种性质**:

| 性质名称 | 单位 | 类别 |
|----------|------|------|
| formation_energy_per_atom | eV/atom | 热力学 |
| band_gap | eV | 电子 |
| egslmeb | eV | 电子 |
| bulk_modulus_kv | GPa | 力学 |
| shear_modulus_gv | GPa | 力学 |
| max_frequency | THz | 声子 |
| density | g/cm³ | 基本 |
| volume_atoms | Å³/atom | 基本 |
| nsites | - | 基本 |
| ntypes | - | 基本 |
| is_metal | - | 分类 |
| crystal_system | - | 分类 |
| num_space_group | - | 分类 |
| total_magnetization | μB | 磁学 |
| ref_idx_x | - | 光学 |
| ref_idx_y | - | 光学 |

**输入格式**:
- 化学式: "LiFePO4", "NaCl"
- CIF 文件上传: `.cif` 格式
- Materials Project ID: 自动下载结构

**结果展示**:
- 数值预测: 带置信区间的柱状图
- 分类预测: 概率分布饼图
- ALIGNN 结果: 多性质雷达图/表格

#### Panel 4: VASP 任务管理 (VASP Tasks)

**任务类型**:
- **Geometry Optimization (RELAX)**: 结构优化
- **Self-Consistent Field (SCF)**: 自洽场计算
- **Band Structure (BAND)**: 能带结构计算
- **Density of States (DOS)**: 态密度计算

**任务状态流转**:
```
CREATED → SUBMITTED → RUNNING → COMPLETED
                    ↘ CANCELLED / FAILED
```

**功能列表**:
- ✅ 创建任务 (生成 INCAR/POSCAR/KPOINTS/POTCAR)
- ✅ 提交任务 (qsub/sbatch)
- ✅ 监控状态 (实时更新)
- ✅ 取消任务 (qdel/scancel)
- ✅ 查看结果 (OUTCAR/CONTCAR)
- ✅ 下载文件 (从远程到本地)
- ✅ 可视化分析 (能带/DOS 绘制)

**INCAR 参数编辑器**:
- 可视化表单编辑
- 参数说明提示
- 预设模板选择:
  - 高精度 (High Precision)
  - 标准精度 (Standard)
  - 快速测试 (Quick Test)
  - 磁性计算 (Magnetic)
  - SOC 计算 (Spin-Orbit Coupling)

#### Panel 5: 结果可视化 (Visualization)

**2D 结构图**:
- ASE + Matplotlib 渲染
- 支持旋转/缩放
- 多种样式可选 (ball-stick, space-filling, wireframe)
- 原子标签显示

**3D 交互式结构**:
- Three.js / NGL Viewer
- 鼠标拖拽旋转
- 多面体/球棍模式切换
- 晶胞边界显示
- 测量键长/键角

**计算结果可视化**:
- **能带结构**: 高对称 k 点路径, 色彩映射权重
- **态密度 (DOS)**:
  - TDOS (总态密度)
  - PDOS (分波态密度, 按轨道/orbital)
  - 元素贡献 DOS
  - 费米能级标注
- **综合分析图**: 2×3 子图布局 (TDOS + PDOS + 元素贡献 × 自旋向上/向下)

---

### 2. agent_mcp_server.py - FastAPI Agent Server

**端口**: `8766` | **技术栈**: FastAPI, Uvicorn, LangChain, SSE

#### API 端点列表

##### 聊天相关

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/chat` | 发送消息 (非流式) | `{session_id, message}` | `{reply}` |
| POST | `/chat/stream` | 流式对话 (SSE) | `{session_id, message}` | SSE EventStream |
| GET | `/history/{session_id}` | 获取聊天历史 | - | `[messages]` |
| DELETE | `/history/{session_id}` | 清空历史 | - | `{status}` |
| DELETE | `/sessions/{session_id}/message/{message_id}` | 删除单条消息 | - | `{status}` |

##### 会话管理

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| GET | `/sessions` | 列出所有会话 | `[{id, created_at, ...}]` |
| POST | `/sessions` | 创建新会话 | `{session_id}` |
| DELETE | `/sessions/{session_id}` | 删除会话 | `{status}` |

##### 材料搜索

| Method | Endpoint | Description | Request |
|--------|----------|-------------|---------|
| POST | `/search/materials` | 搜索材料 | `{query, filters}` |
| POST | `/search/mp-by-id` | 按 ID 查询 | `{material_id}` |
| POST | `/search/oqmd` | OQMD 查询 | `{formula}` |

##### 性质预测

| Method | Endpoint | Description | Request |
|--------|----------|-------------|---------|
| POST | `/predict/bandgap` | XGBoost 带隙预测 | `{formula}` |
| POST | `/predict_alignn` | ALIGNN 多性质预测 | `{formula or cif_content}` |

##### VASP 任务

| Method | Endpoint | Description | Request |
|--------|----------|-------------|---------|
| POST | `/vasp/tasks` | 创建 VASP 任务 | `{type, structure, params}` |
| GET | `/vasp/tasks` | 列出任务 | `{status?, limit?}` |
| GET | `/vasp/tasks/{task_id}` | 获取任务详情 | - |
| POST | `/vasp/tasks/{task_id}/submit` | 提交任务 | - |
| POST | `/vasp/tasks/{task_id}/cancel` | 取消任务 | - |
| GET | `/vasp/tasks/{task_id}/results` | 获取计算结果 | - |
| GET | `/vasp/tasks/{task_id}/files/{filename}` | 下载文件 | - |
| POST | `/vasp/visualize/band` | 绘制能带图 | `{task_id, options}` |
| POST | `/vasp/visualize/dos` | 绘制 DOS 图 | `{task_id, options}` |

##### 可视化

| Method | Endpoint | Description | Request |
|--------|----------|-------------|---------|
| POST | `/visualization/structure-2d` | 生成 2D 结构图 | `{format: 'cif', content}` |
| POST | `/visualization/structure-3d` | 生成 3D HTML | `{format: 'cif', content}` |
| GET | `/cache/images/{filename}` | 获取缓存的图片 | - |
| GET | `/cache/3d/{filename}` | 获取缓存的 3D 文件 | - |

##### 系统监控

| Method |Endpoint | Description |
|--------|---------|-------------|
| GET | `/health` | 健康检查 |
| GET | `/stats` | 系统统计 (会话数/请求数等) |
| GET | `/models` | 可用 LLM 模型列表 |

#### SSE 流式响应格式

```javascript
// EventSource 监听
const eventSource = new EventSource('/chat/stream');

eventSource.addEventListener('token', (e) => {
  // 收到文本 token
  const data = JSON.parse(e.data);
  console.log(data.content); // 文本片段
});

eventSource.addEventListener('tool_call', (e) => {
  // 工具调用事件
  const data = JSON.parse(e.data);
  console.log(data.tool_name);  // 工具名称
  console.log(data.tool_input); // 工具输入参数
});

eventSource.addEventListener('tool_result', (e) => {
  // 工具返回结果
  const data = JSON.parse(e.data);
  console.log(data.result); // 工具输出
});

eventSource.addEventListener('done', (e) => {
  // 流结束
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  // 错误处理
});
```

#### Content Blocks 消息格式

用于记录工具调用在回复中的精确位置:

```json
{
  "role": "assistant",
  "content": "根据查询结果，LiFePO4的带隙为...\n\n[图表: bandgap_plot.png]",
  "tool_calls": [
    {
      "id": "call_abc123",
      "tool_name": "search_materials",
      "tool_input": {"elements": ["Li", "Fe", "P", "O"]},
      "content_blocks": [
        {
          "type": "text",
          "start_index": 0,
          "end_index": 38
        },
        {
          "type": "image",
          "start_index": 41,
          "end_index": 63,
          "metadata": {"filename": "bandgap_plot.png"}
        }
      ]
    }
  ],
  "model": "deepseek-chat",
  "duration_ms": 2340
}
```

---

### 3. mcp_server.py - MCP 工具服务器

**端口**: `8000` | **技术栈**: FastMCP, Context, Pymatgen, ASE

MCP (Model Context Protocol) 是一种标准化的工具调用协议。本项目的 MCP Server 提供了 25+ 个工具函数供 Agent 调用。

#### 工具分类

##### 材料数据库查询 (6 个)

```python
@mcp.tool()
async def search_materials(
    elements: Optional[List[str]] = None,
    material_id: Optional[str] = None,
    formula: Optional[str] = None,
    band_gap_range: Tuple[float, float] = (0.0, 5.0),
    max_results: int = 20
) -> List[Dict]:
    """
    在 Materials Project 数据库中搜索材料
    
    Args:
        elements: 元素列表, 如 ['Li', 'Fe', 'O']
        material_id: 材料 ID, 如 'mp-1234'
        formula: 化学式, 如 'LiFePO4'
        band_gap_range: 带隙范围 (min_eV, max_eV), 默认 (0, 5)
        max_results: 最大返回数量, 默认 20
    
    Returns:
        材料信息列表, 每项包含:
        - material_id, formula, band_gap, stability
        - energy_above_hull, space_group, structure (CIF)
    """
```

```python
@mcp.tool()
async def get_material_details(material_id: str) -> Dict:
    """获取材料的详细信息"""

@mcp.tool()
async def get_material_structure(material_id: str, format: str = "cif") -> str:
    """获取晶体结构文件 (CIF/POSCAR)"""

@mcp.tool()
async def search_oqmd(
    formula: str,
    property_filter: Optional[str] = None
) -> List[Dict]:
    """在 OQMD 数据库中搜索材料"""

@mcp.tool()
async def get_stability_info(material_id: str) -> Dict:
    """获取材料稳定性信息 (相图凸包距离)"""

@mcp.tool()
async def find_similar_structures(
    material_id: str,
    threshold: float = 0.3
) -> List[Dict]:
    """查找结构相似的材料"""
```

##### 性质预测 (8 个)

```python
@mcp.tool()
async def predict_bandgap(formula: str) -> Dict:
    """
    使用 XGBoost 模型预测带隙
    
    Returns:
        {
            "formula": "LiFePO4",
            "predicted_bandgap_eV": 3.47,
            "confidence_interval": [3.2, 3.7],
            "model_info": {...}
        }
    """

@mcp.tool()
async def predict_with_alignn(
    formula: Optional[str] = None,
    cif_content: Optional[str] = None,
    properties: Optional[List[str]] = None
) -> Dict:
    """
    使用 ALIGNN 模型预测多种材料性质
    
    Args:
        formula: 化学式 (会自动获取 CIF)
        cif_content: 直接提供 CIF 内容
        properties: 指定要预测的性质, 默认全部 16 种
        
    Returns:
        {
            "formula": "...",
            "predictions": {
                "formation_energy_per_atom": {"value": -2.34, "unit": "eV/atom"},
                "band_gap": {"value": 0.0, "unit": "eV"},
                "bulk_modulus_kv": {"value": 120.5, "unit": "GPa"},
                ...
            }
        }
    """

@mcp.tool()
async def compare_properties(
    formulas: List[str],
    properties: List[str]
) -> Dict:
    """比较多个材料的指定性质"""

@mcp.tool()
async def predict_from_composition_only(
    formula: str
) -> Dict:
    """仅从组成预测基本性质 (无结构)"""

@mcp.tool()
async def batch_predict(
    formulas: List[str],
    model: str = "bandgap"
) -> List[Dict]:
    """批量预测多个材料"""

@mcp.tool()
async def explain_prediction(
    formula: str,
    model: str = "bandgap"
) -> Dict:
    """解释预测结果 (特征重要性)"""
```

##### 可视化生成 (5 个)

```python
@mcp.tool()
async def generate_structure_image(
    cif_content: str,
    image_type: str = "2d",
    style: str = "ball+stick",
    show_labels: bool = True
) -> Dict:
    """
    生成晶体结构图像
    
    Args:
        cif_content: CIF 文件内容
        image_type: '2d' (静态图片) 或 '3d' (交互式 HTML)
        style: 'ball+stick', 'spacefilling', 'wireframe'
        show_labels: 是否显示原子标签
    
    Returns:
        {
            "image_url": "/cache/images/xxx.png",  # 2D
            "html_url": "/cache/3d/xxx.html",      # 3D
            "thumbnail_url": "/cache/images/xxx_thumb.png"
        }
    """

@mcp.tool()
async def generate_band_structure_plot(
    vasprun_xml_path: str,
    options: Optional[Dict] = None
) -> Dict:
    """生成能带结构图"""

@mcp.tool()
async def generate_dos_plot(
    vasprun_xml_path: str,
    dos_options: Optional[Dict] = None
) -> Dict:
    """
    生成 DOS 图 (支持多种模式)
    
    dos_options:
        mode: 'tdos' | 'pdos' | 'element' | 'combined'
        spin: 'up' | 'down' | 'both'
        orbital: list of orbitals for PDOS
        energy_range: [emin, emax] in eV
        fermi_level: float (auto-detected if None)
    """

@mcp.tool()
async def generate_combined_analysis(
    task_id: str,
    include_band: bool = True,
    include_dos: bool = True
) -> Dict:
    """生成综合分析图 (2×3 布局)"""

@mcp.tool()
async def create_comparison_figure(
    data_list: List[Dict],
    figure_type: str = "bar"
) -> Dict:
    """创建对比图"""
```

##### VASP 任务管理 (6 个)

```python
@mcp.tool()
async def create_vasp_task(
    task_type: str,  # 'relax' | 'scf' | 'band' | 'dos'
    structure_cif: str,
    incar_params: Optional[Dict] = None,
    kpoints_density: float = 100,
    potcar_choice: str = "PBE"
) -> Dict:
    """创建 VASP 计算任务"""

@mcp.tool()
async def submit_vasp_task(task_id: str) -> Dict:
    """提交任务到集群队列"""

@mcp.tool()
async def check_task_status(task_id: str) -> Dict:
    """检查任务执行状态"""

@mcp.tool()
async def cancel_vasp_task(task_id: str) -> Dict:
    """取消正在运行的任务"""

@mcp.tool()
async def get_calculation_results(task_id: str) -> Dict:
    """获取计算结果和输出文件"""

@mcp.tool()
async def download_remote_file(
    remote_path: str,
    local_path: Optional[str] = None
) -> Dict:
    """从 HPC 集群下载文件"""
```

##### 文件与系统工具 (4 个)

```python
@mcp.tool()
async def read_local_file(file_path: str) -> str:
    """读取本地文件内容"""

@mcp.tool()
async def write_local_file(file_path: str, content: str) -> bool:
    """写入本地文件"""

@mcp.tool()
async def execute_shell_command(
    command: str,
    timeout: int = 60
) -> Dict:
    """安全执行 Shell 命令"""

@mcp.tool()
async def list_directory(path: str) -> List[Dict]:
    """列出目录内容"""
```

---

### 4. agent/langchain_mcp_agent.py - LangChain Agent 核心

**技术栈**: LangChain, LangGraph, Pydantic

#### Agent 架构

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Agent                       │
│                                                         │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Prompt   │ →  │   LLM Call   │ →  │ Output Parse │  │
│  │ Template  │    │  (DeepSeek)  │    │              │  │
│  └───────────┘    └──────────────┘    └──────────────┘  │
│         ↑                                      │         │
│         │                                      ↓         │
│  ┌──────┴──────┐    ┌───────────────────────────────┐    │
│  │  System     │    │      Tool Execution           │    │
│  │  Message    │ ←  │   (MCP Protocol)              │    │
│  │             │    │  • search_materials           │    │
│  │ Role: Expert│    │  • predict_with_alignn        │    │
│  │ in Materials│    │  • generate_structure_image   │    │
│  │ Science &   │    │  • create_vasp_task           │    │
│  │ Computing   │    │  • ... (25+ tools)            │    │
│  └─────────────┘    └───────────────────────────────┘    │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                Memory (Conversation History)       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 类定义: MatAgentMCP

```python
class MatAgentMCP:
    """
    材料科学智能体 - 基于 LangChain 和 MCP 协议
    
    Attributes:
        api_key: DeepSeek API 密钥
        model_name: 模型名称 ('deepseek-chat' | 'deepseek-reasoner' | 'glm-5')
        base_url: API 基础 URL
        temperature: 生成温度 (0-1)
        max_tokens: 最大 token 数
        mcp_client: MCP 客户端连接
        agent: LangChain Agent 实例
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """初始化 Agent"""
    
    async def chat(self, message: str, session_id: str) -> AsyncIterator[Dict]:
        """
        异步流式对话
        
        Yields:
            {
                'type': 'token' | 'tool_call' | 'tool_result' | 'done',
                'content': str,
                'tool_name': Optional[str],
                'tool_input': Optional[Dict],
                'result': Optional[Any]
            }
        """
    
    def search_materials(self, **kwargs) -> List[Dict]:
        """搜索材料 (包装 MCP 工具)"""
    
    def predict_bandgap(self, formula: str) -> Dict:
        """预测带隙 (包装 MCP 工具)"""
    
    def predict_with_alignn(self, formula: str, **kwargs) -> Dict:
        """ALIGNN 多性质预测 (包装 MCP 工具)"""
    
    # ... 其他工具包装方法
```

#### System Prompt 模板

```markdown
你是一个专业的材料科学和研究计算专家助手。你的专长领域包括：

## 核心能力

1. **材料数据库查询**
   - Materials Project 数据库搜索与分析
   - OQMD 数据库查询
   - 材料稳定性评估 (能量凸包分析)
   - 相图与竞争相识别

2. **材料性质预测**
   - 机器学习模型预测 (XGBoost):
     * 带隙 (Band Gap)
   - 深度学习模型预测 (ALIGNN):
     * 16 种物理化学性质
     * 基于晶体结构的图神经网络

3. **第一性原理计算**
   - VASP 输入文件准备 (INCAR/KPOINTS/POSCAR/POTCAR)
   - 任务提交与管理
   - 结果分析与可视化

4. **可视化与报告**
   - 晶体结构 (2D/3D)
   - 能带结构
   - 态密度 (TDOS/PDOS)

## 回答规范

- 使用专业术语，但适当解释
- 提供数值结果时注明单位和不确定度
- 引用数据来源 (MP, OQMD, 实验)
- 建议下一步实验或计算验证
- 对于不确定性，给出概率估计
```

---

### 5. flask_server.py - Flask 文件服务器

**端口**: `6750` | **技术栈**: Flask, CORS

#### 功能

提供静态文件服务和缓存管理:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | 服务首页 (列出可用资源) |
| `/images/<filename>` | GET | 获取 2D 图片 |
| `/3d/<filename>` | GET | 获取 3D HTML 文件 |
| `/structure-info` | GET | 获取所有缓存的结构元数据 |
| `/structure-info/<file_id>` | GET | 获取单个结构信息 |
| `/clear-cache` | DELETE | 清除过期缓存 |

#### 缓存机制

```python
# 缓存配置
CACHE_CONFIG = {
    'temp_images': {
        'dir': 'cache/temp_images',
        'max_size_mb': 500,
        'max_age_hours': 24,
        'formats': ['png', 'jpg', 'svg']
    },
    'temp_3d': {
        'dir': 'cache/temp_3d',
        'max_size_mb': 200,
        'max_age_hours': 48,
        'formats': ['html', 'json']
    }
}

# structure_info.json 格式
{
    "structures": [
        {
            "file_id": "uuid-string",
            "filename": "LiFePO4.png",
            "type": "image",  # or "html"
            "formula": "LiFePO4",
            "material_id": "mp-12345",
            "created_at": "2026-01-01T12:00:00",
            "size_bytes": 152400,
            "url": "/images/LiFePO4.png"
        }
    ]
}
```

---

### 6. tryssh.py - SSH 远程操作模块

**技术栈**: Paramiko, Concurrency

#### 功能概述

封装了通过 SSH 协议远程操作 HPC 集群的功能，主要用于 VASP 计算任务的管理。

#### 核心类: SSHConnector

```python
class SSHConnector:
    """
    SSH 连接管理器
    
    Features:
    - 连接池管理
    - 命令超时控制
    - 安全命令过滤
    - 断线自动重连
    """
    
    def __init__(self, host, username, password, port=22):
        """建立 SSH 连接"""
    
    def execute_command(
        self,
        command: str,
        timeout: int = 300,
        work_dir: Optional[str] = None
    ) -> Dict:
        """
        执行远程命令
        
        Returns:
            {
                'returncode': 0,
                'stdout': '...',
                'stderr': '',
                'execution_time_s': 12.3
            }
        """
    
    def upload_file(self, local_path, remote_path) -> bool:
        """上传本地文件到远程"""
    
    def download_file(self, remote_path, local_path) -> bool:
        """下载远程文件到本地"""
    
    def close(self):
        """关闭连接"""
```

#### 安全机制: 命令过滤

```python
# 危险命令模式黑名单 (正则表达式)
DANGEROUS_PATTERNS = [
    r'\brm\s+-rf\s+/',          # 强制递归删除根目录
    r'\bdd\s+if=/dev/zero',     # 磁盘覆盖
    r'\bchmod\s+777',           # 权限过于宽松
    r'\bfork\s+bomb',           # Fork 炸弹
    r'>\s*/etc/',               # 写入系统目录
    r'\bcurl.*\|\s*bash',       # 远程脚本注入
    r'\bwget.*\|\s*(sh|bash)',  # 同上
    r'__import__',              # Python 导入
    r'eval\(',                  # 代码执行
    r'exec\(',                  # 同上
    # ... 更多模式
]

def sanitize_command(command: str) -> Tuple[bool, str]:
    """
    检查命令安全性
    
    Returns:
        (is_safe, error_message)
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"检测到危险命令模式: {pattern}"
    return True, ""
```

#### VASP 任务管理方法

```python
class VASPTaskManager(SSHConnector):
    """VASP 任务管理器 (继承 SSHConnector)"""
    
    def create_relaxation_task(
        self,
        poscar_content: str,
        incar_params: Dict,
        kpoints: Dict,
        potcar: str
    ) -> Dict:
        """创建结构优化任务"""
    
    def create_scf_task(self, contcar_content, ...) -> Dict:
        """创建 SCF 任务 (使用优化后的结构)"""
    
    def create_band_task(self, ..., kpath: Dict) -> Dict:
        """
        创建能带计算任务
        
        kpath 示例:
        {
            'high_symmetry_points': {
                'Γ': [0, 0, 0],
                'X': [0.5, 0, 0],
                ...
            },
            'path': [['Γ', 'X'], ['X', 'M'], ...]
        }
        """
    
    def create_dos_task(self, ...) -> Dict:
        """创建 DOS 计算任务"""
    
    def submit_task(self, task_id: str, scheduler: str = 'pbs') -> Dict:
        """
        提交任务到调度系统
        
        scheduler: 'pbs' (PBS Pro) | 'slurm' (SLURM)
        """
    
    def get_task_status(self, task_id: str) -> Dict:
        """
        查询任务状态
        
        Returns:
            {
                'status': 'RUNNING',  # CREATED/SUBMITTED/RUNNING/COMPLETED/CANCELLED/FAILED
                'queue_position': 15,
                'walltime_used': '02:30:00',
                'progress_percent': 65
            }
        """
    
    def cancel_task(self, task_id: str) -> Dict:
        """取消任务"""
    
    def fetch_results(self, task_id: str, files: List[str]) -> Dict:
        """
        获取计算结果文件
        
        files: ['OUTCAR', 'CONTCAR', 'vasprun.xml', 'EIGENVAL', 'DOSCAR']
        """
    
    def parse_outcar(self, outcar_content: str) -> Dict:
        """
        解析 OUTCAR 文件
        
        Extracts:
        - Total energy
        - Forces (max/avg)
        - Stress tensor
        - Convergence info
        - Magnetic moments
        """
    
    def parse_prediction_output(self, output_text: str) -> Dict:
        """
        解析 ALIGNN 预测输出
        
        将原始文本解析为结构化的预测字典
        """
    
    def predict_from_local_cif(
        self,
        cif_path: str,
        properties: Optional[List[str]] = None
    ) -> Dict:
        """
        在本地/远程运行 ALIGNN 预测
        
        1. 上传 CIF 到远程
        2. 运行 JARVIS-Tools predict CLI
        3. 下载并解析结果
        """
```

---

### 7. db/databasemanage.py - 数据库管理

**技术栈**: SQLite3, threading

#### 设计原则 (重要!)

由于 Streamlit 的多线程特性，SQLite 操作必须遵循以下规则:

```python
# ✅ 正确做法: 模块级函数，每次新建独立连接
def add_chat_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        """, (session_id, role, content))
        conn.commit()
    finally:
        conn.close()

# ❌ 错误做法: 类实例共享连接
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_PATH)  # 共享连接!
    
    def add_message(self, ...):
        self.conn.execute(...)  # 多线程时会出错!
```

#### 数据库 Schema

##### messages 表 (聊天消息)

```sql
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    model TEXT DEFAULT 'deepseek-chat',
    tool_calls TEXT,  -- JSON array of tool call records
    content_blocks TEXT,  -- JSON array for position tracking
    duration_ms INTEGER,  -- Response time in ms
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_session_timestamp (session_id, timestamp),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

##### sessions 表 (会话)

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,  -- UUID
    title TEXT,  -- Auto-generated from first message
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    metadata TEXT  -- JSON, extensible
);
```

##### vasp_tasks 表 (VASP 任务)

```sql
CREATE TABLE IF NOT EXISTS vasp_tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    task_type TEXT NOT NULL CHECK(task_type IN (
        'relax', 'scf', 'band', 'dos'
    )),
    status TEXT DEFAULT 'CREATED' CHECK(status IN (
        'CREATED', 'SUBMITTED', 'RUNNING',
        'COMPLETED', 'CANCELLED', 'FAILED'
    )),
    input_files TEXT,  -- JSON {poscar, incar, kpoints, potcar}
    remote_work_dir TEXT,
    job_id TEXT,  # PBS/SLURM job ID
    results TEXT,  -- JSON {outcar, contcar, ...}
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    submitted_at DATETIME,
    started_at DATETIME,
    completed_at DATETIME,
    
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

##### predictions_cache 表 (预测缓存)

```sql
CREATE TABLE IF NOT EXISTS predictions_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    result_json TEXT NOT NULL,  -- Full prediction result
    cif_hash TEXT,  # For structure-based predictions
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,  -- TTL
    
    UNIQUE(formula, model_name, prediction_type, cif_hash)
);
```

#### 主要函数列表

```python
# === 会话管理 ===
def create_session() -> str:  # 返回 session_id (UUID)
def get_session(session_id: str) -> Optional[Dict]
def update_session(session_id: str, **kwargs) -> bool
def delete_session(session_id: str) -> bool
def list_sessions(limit: int = 50, offset: int = 0) -> List[Dict]

# === 消息管理 ===
def add_chat_message(session_id, role, content, **kwargs) -> int
def get_chat_history(session_id, limit: int = 100) -> List[Dict]
def delete_message(message_id: int) -> bool
def clear_session_history(session_id: str) -> bool

# === VASP 任务 ===
def create_vasp_task(**task_data) -> str
def update_vasp_task_status(task_id, status, **kwargs) -> bool
def get_vasp_task(task_id: str) -> Optional[Dict]
def list_vasp_tasks(session_id=None, status=None) -> List[Dict]

# === 预测缓存 ===
def cache_prediction(formula, model, ptype, result, **kwargs) -> bool
def get_cached_prediction(formula, model, ptype) -> Optional[Dict]
def clear_expired_cache() -> int  # 返回清理条数

# === 工具调用记录 ===
def log_tool_call(session_id, message_id, tool_data) -> int
def get_tool_calls(session_id=None, tool_name=None) -> List[Dict]

# === 统计 ===
def get_session_stats() -> Dict
def get_model_usage_stats() -> List[Dict]
def cleanup_old_data(days: int = 30) -> Dict
```

---

### 8. myml/ - 机器学习预测模型

#### 模型概述

| 模型 | 文件 | 输入特征 | 输出 | 算法 | 精度 (MAE) |
|------|------|----------|------|------|-----------|
| 带隙预测 | `bandgap_predict.py` | Composition (157维) | band_gap_eV | XGBoost Regressor | ~0.15 eV |
| 多性质预测 | `alignn_predict.py` | Crystal Structure (CIF) | 16 properties | ALIGNN GNN | 视性质而定 |

#### 特征工程: Composition-based

```python
from matminer.featurizers.composition import ElementFraction, \
    Stoichiometry, ElementProperty, ValenceOrbital, \
    IonProperty, ElectronAffinity, TMetalProperty

def generate_features(formula: str) -> np.ndarray:
    """
    从化学式生成 157 维特征向量
    
    Feature Categories:
    - Element fractions (atomic %)
    - Stoichiometric descriptors
    - Atomic properties (mean, std, min, max, range):
        * Atomic number, atomic mass
        * Electronegativity (Pauling)
        * Covalent radius, Van der Waals radius
        * Ionization energies (1st, 2nd, 3rd)
        * Electron affinity
        * Valence electrons
        * s/p/d/f electron counts
        * Melting point, boiling point
        * Density
        * Molar volume
        * Specific heat
        * Thermal conductivity
        * Electrical resistivity
        * Magnetic moment
    - Valence orbital properties
    - Transition metal specific features
    """
    comp = Composition.from_formula(formula)
    
    featurizers = [
        ElementFraction(),
        Stoichiometry(),  # n_elements, atom fraction stats
        ElementProperty(features=[
            'Number', 'AtomicMass', 'Electronegativity',
            'CovalentRadius', 'NsValence', 'NpValence',
            'NdValence', 'NfValence', 'NValence',
            # ... more properties (total 157 features)
        ]),
        ValenceOrbital(props=['avg', 'sum']),
        IonProperty(ion_props=['IonizationEnergy', 'ElectronAffinity']),
        TMetalProperty()
    ]
    
    features = []
    for featurizer in featurizers:
        try:
            feats = featurizer.featurize(comp)
            features.extend(feats if isinstance(feats, list) else [feats])
        except:
            features.extend([0] * len(featurizer.feature_labels()))
    
    return np.array(features).reshape(1, -1)
```

#### 使用示例

```python
from myml.bandgap_predict import BandgapPredictor
from myml.alignn_predict import ALIGNNPredictor

# XGBoost 预测
bg_predictor = BandgapPredictor(model_path='myml/models/xgboost_bandgap.json')
result = bg_predictor.predict("LiFePO4")
print(f"Predicted band gap: {result['band_gap_eV']:.2f} eV")
print(f"Confidence interval: {result['ci_low']:.2f} - {result['ci_high']:.2f} eV")

# ALIGNN 预测
alignn = ALIGNNPredictor()
results = alignn.predict(cif_content=open('LiFePO4.cif').read())
for prop, data in results['predictions'].items():
    print(f"{prop}: {data['value']} {data['unit']}")
```

---

### 9. oqmd.py - OQMD 数据库查询

**技术栈**: Requests, Pandas

#### 功能

```python
class OQMDClient:
    """
    Open Quantum Materials Database 客户端
    
    API Base URL: https://oqmd.org/oqmd/api
    """
    
    def search(
        self,
        formula: str,
        element_set: Optional[str] = None,
        property_filter: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        搜索 OQMD 数据库
        
        Args:
            formula: 化学式 (如 'NaCl')
            element_set: 元素集合 (如 'Li-Fe-O')
            property_filter: 属性过滤 (如 'band_gap>1.0')
        
        Returns:
            DataFrame with columns:
            - oqmid, compound, composition
            - lattice_type, spacegroup
            - stability (delta_e per atom)
            - band_gap, formation_enthalpy
            - volume, natoms
            - calculation_method (DFT functional used)
        """
    
    def get_by_oqmid(self, oqmid: int) -> Dict:
        """根据 OQMD ID 获取详情"""
    
    def get_convex_hull_distance(self, oqmid: int) -> float:
        """获取距凸包的能量距离 (稳定性指标)"""
```

---

## 🔄 VASP 工作流详解

### 典型计算流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VASP 计算完整工作流                               │
└─────────────────────────────────────────────────────────────────────┘

Step 1: 准备初始结构
┌──────────────┐
│ POSCAR       │ ← 来自 MP/OQMD/实验/手动构建
│ (初始结构)    │
└──────┬───────┘
       ▼
Step 2: 几何优化 (Relaxation)
┌────────────────────────────────────────┐
│ Task Type: relax                       │
│ Input: POSCAR + INCAR + KPOINTS + POTCAR│
│                                        │
│ INCAR 关键参数:                         │
│   ENCUT = 520 ( cutoff energy)          │
│   ISMEAR = 0; SIGMA = 0.05             │
│   EDIFF = 1E-5 (电子收敛)              │
│   EDIFFG = -0.02 (力收敛 eV/Å)        │
│   IBRION = 2 (CG 算法)                 │
│   NSW = 100 (最大离子步)               │
│   ISIF = 3 (全优化)                    │
│                                        │
│ Output: CONTCAR (优化后结构)            │
│         OUTCAR (详细信息)               │
└──────────────┬─────────────────────────┘
               ▼
Step 3: 自洽场计算 (SCF)
┌────────────────────────────────────────┐
│ Task Type: scf                         │
│ Input: CONTCAR (来自 Step 2)           │
│                                        │
│ INCAR 关键参数:                         │
│   ICHARG = 2 (从电荷密度读取)           │
│   LCHARG = .TRUE. (保存电荷密度)        │
│   NSW = 0 (无离子步)                   │
│   ISMEAR = -5; SIGMA = ...             │
│                                        │
│ Output: CHGCAR (电荷密度)              │
│         EIGENVAL (本征值)              │
│         DOSCAR (态密度)                │
│         vasprun.xml (完整数据)          │
└──────────────┬─────────────────────────┘
               ▼
Step 4: (可选) 非自洽能带计算
┌────────────────────────────────────────┐
│ Task Type: band                        │
│ Input: CHGCAR + 高对称 k 点路径         │
│                                        │
│ INCAR 关键参数:                         │
│   ICHARG = 11 (读取 CHGCAR)            │
│   LORBIT = 11 (投影到原子)             │
│   NSIMPLE k-points (沿高对称路径)       │
│                                        │
│ Output: EIGENVAL (能带数据)            │
│         PROCAR (投影字符)               │
└──────────────┬─────────────────────────┘
               ▼
Step 5: (可选) 精细 DOS 计算
┌────────────────────────────────────────┐
│ Task Type: dos                         │
│ Input: 与 SCF 相同但更密的 k mesh       │
│                                        │
│ Output: DOSCAR (精细 TDOS/PDOS)        │
└────────────────────────────────────────┘
               ▼
Step 6: 结果分析与可视化
┌──────────┬──────────┬──────────┐
│ Band     │ DOS      │ Combined │
│ Structure│ Plot     │ Analysis │
│ Plot     │ (2x3)    │ Figure   │
└──────────┴──────────┴──────────┘
```

### INCAR 模板预设

#### 高精度 (High Precision)

```ini
SYSTEM = High Precision Calculation
PREC = Accurate
ENCUT = 600
EDIFF = 1E-7
EDIFFG = -0.01
ISMEAR = 0
SIGMA = 0.02
LREAL = .FALSE.
ALGO = Normal
NCORE = 4
KPAR = 2
```

#### 标准精度 (Standard)

```ini
SYSTEM = Standard Calculation
PREC = Medium
ENCUT = 520
EDIFF = 1E-5
EDIFFG = -0.02
ISMEAR = 0
SIGMA = 0.05
IBRION = 2
ISIF = 3
NSW = 80
```

#### 快速测试 (Quick Test)

```ini
SYSTEM = Quick Test
ENCUT = 400
EDIFF = 1E-4
EDIFFG = -0.05
ISMEAR = 0
SIGMA = 0.2
IBRION = 2
ISIF = 3
NSW = 30
```

#### 磁性计算 (Magnetic)

```ini
SYSTEM = Magnetic Calculation
ISPIN = 2
MAGMOM = 10*5.0  # 根据体系调整
SYMPREC = 1E-8
LNONCOLLINEAR = .FALSE.
LSORBIT = .FALSE.
```

#### SOC 计算 (Spin-Orbit Coupling)

```ini
SYSTEM = SOC Calculation
ISPIN = 1
LSORBIT = .TRUE.
MAGMOM = 10*0.0
SYMPREC = 1E-8
```

### 结果可视化示例

#### 能带结构图

```python
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp import Vasprun

# 解析 vasprun.xml
vasprun = Vasprun("vasprun.xml")
bandstructure = vasprun.get_band_structure(kpoints_filename="KPOINTS")

# 绘制
plotter = BSPlotter(bandstructure)
plotter.get_plot(vbm_cbm_marker=True, ylim=(-5, 5))
plt.savefig("band_structure.png", dpi=300)
```

#### DOS 综合分析图 (2×3)

```python
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import DOSPlotter

dos_plotter = DOSPlotter()

# 读取 TDOS/PDOS
tdos = vasprun.complete_dos
pdos = vasprun.pdos

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Spin Up
dos_plotter.dos = tdos
dos_plotter.plot(ax=axes[0][0])  # TDOS up
axes[0][0].set_title("Total DOS (Spin Up)")
dos_plotter.plot_pdos(pdos, ax=axes[0][1])  # PDOS up
axes[0][1].set_title("Projected DOS (Spin Up)")
dos_plotter.plot_element_dos(pdos, ax=axes[0][2])  # Element contribution
axes[0][2].set_title("Element Contribution (Spin Up)")

# Row 2: Spin Down
# ... similar for spin down

plt.tight_layout()
plt.savefig("dos_combined_analysis.png", dpi=300)
```

---

## 🧪 测试

### 单元测试

```bash
# 测试模块导入
python -c "from agent.langchain_mcp_agent import MatAgentMCP; print('OK')"
python -c "from myml.bandgap_predict import BandgapPredictor; print('OK')"

# 测试数据库连接
python -c "from db.databasemanage import create_session; print(create_session())"

# 测试 SSH 连接 (需要配置)
python -c "from tryssh import SSHConnector; c = SSHConnector(...); c.test_connection()"
```

### 集成测试

```bash
# 测试 MCP Server
python mcp_server.py
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "search_materials", "arguments": {"elements": ["Li"]}}'

# 测试 Agent Server
python agent_mcp_server.py
curl -X POST http://localhost:8766/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "Hello"}'

# 测试流式响应
curl -N -X POST http://localhost:8766/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "Find LiFePO4"}'
```

### 测试预测模型

```bash
# XGBoost 带隙预测
python -c "
from myml.bandgap_predict import BandgapPredictor
p = BandgapPredictor()
print(p.predict('LiFePO4'))
print(p.predict('NaCl'))
print(p.predict('Si'))
"

# ALIGNN 预测 (需要安装 jarvis-tools)
python -c "
from myml.alignn_predict import ALIGNNPredictor
p = ALIGNNPredictor()
result = p.predict(formula='LiFePO4')
import json
print(json.dumps(result, indent=2))
"
```

---

## 🐛 故障排查

### 常见问题与解决方案

#### 1. ModuleNotFoundError

```bash
# 问题: ImportError: No module named 'pymatgen'
# 解决:
uv sync  # 或 pip install pymatgen
```

#### 2. SQLite 数据库锁定

```bash
# 问题: sqlite3.OperationalError: database is locked
# 原因: Streamlit 多线程共享连接
# 解决: 确保 databasemanage.py 中每次操作都新建连接
# 参考 AGENTS.md 中的 SQLite 多线程注意事项
```

#### 3. SSH 连接失败

```bash
# 问题: paramiko.ssh_exception.AuthenticationException
# 解决步骤:
# 1. 检查 config/.env 中的 HOST, USERNAME, PASSWORD
# 2. 确认密码正确 (特殊字符需要转义)
# 3. 尝试手动 ssh 测试:
   ssh username@host
# 4. 如果使用密钥:
   # PASSWORD 应指向私钥路径, 如 /home/user/.ssh/id_rsa
```

#### 4. MCP Server 启动失败

```bash
# 问题: FastMCP import error
# 解决:
pip install fastmcp  # 需要特定版本
# 或
uv pip install "fastmcp>=1.0.0"
```

#### 5. Streamlit 页面空白/报错

```bash
# 检查 Agent Server 是否运行
curl http://localhost:8766/health

# 检查 Flask 文件服务是否运行
curl http://localhost:6750/

# 查看 Streamlit 日志
streamlit run web_mcp_app.py --logger.level=debug
```

#### 6. VASP 任务提交失败

```bash
# 问题: qsub: command not found
# 原因: 远程集群环境未加载
# 解决: 在 tryssh.py 的 submit_task 中添加 module load
# 例如:
self.execute_command("module load intel impi vasp/6.3.0")

# 问题: Permission denied writing to work_dir
# 解决: 检查 base_dir 权限
self.execute_command(f"ls -la {self.base_dir}")
self.execute_command(f"mkdir -p {self.base_dir}")
```

#### 7. DeepSeek API 调用失败

```bash
# 问题: Invalid API key / Rate limit exceeded
# 解决:
# 1. 检查 DEEPSEEK_API_KEY 是否有效
# 2. 查看 API 余额: https://platform.deepseek.com/
# 3. 如果使用代理, 检查 DEEPSEEK_API_BASE_URL
```

#### 8. ALIGNN 预测超时/内存不足

```bash
# 问题: CUDA out of memory / Timeout
# 解决:
# 1. 确保远程机器有 GPU (推荐 > 8GB 显存)
# 2. 增加 timeout 参数
# 3. 使用 CPU 版本 (较慢但更稳定):
   export CUDA_VISIBLE_DEVICES=""
```

#### 9. 3D 可视化无法加载

```bash
# 问题: 3D structure viewer shows blank page
# 解决:
# 1. 检查 flask_server.py 是否运行 (端口 6750)
# 2. 检查 CORS 设置允许跨域请求
# 3. 清除浏览器缓存
# 4. 查看浏览器 Console 是否有 JS 错误
```

### 日志调试

```bash
# 启用详细日志

# Agent Server
export LOG_LEVEL=DEBUG
uv run --env-file config/.env python agent_mcp_server.py

# MCP Server
python mcp_server.py --log-level DEBUG

# Streamlit
streamlit run web_mcp_app.py --logger.level=debug

# 查看日志文件
tail -f ~/.streamlit/logs/streamlit.log
```

---

## 📊 性能与扩展性

### 当前性能指标 (参考)

| 指标 | 典型值 | 备注 |
|------|--------|------|
| AI 对话响应时间 | 2-10 秒 | 取决于问题复杂度和工具调用次数 |
| 材料搜索 (MP API) | 1-3 秒 | 取决于结果数量 |
| XGBoost 预测 | < 100ms | 本地推理 |
| ALIGNN 预测 | 30-120 秒 | GPU 加速 |
| VASP SCF 计算 | 小时~天级别 | 取决于体系大小和 k 点 |
| 并发会话支持 | 50+ | 受限于 SQLite 锁竞争 |

### 优化建议

1. **数据库升级**: 高并发场景考虑 PostgreSQL 替代 SQLite
2. **缓存策略**: Redis 缓存热门查询和预测结果
3. **异步队列**: Celery + Redis 处理长时间任务 (ALIGNN/VASP)
4. **负载均衡**: 多个 Agent Server 实例 + Nginx 反向代理
5. **CDN 加速**: 静态资源和图片走 CDN

---

## 🤝 贡献指南

我们欢迎各种形式的贡献!

### 开发流程

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 提交 Pull Request

### 代码风格

- Python: 遵循 PEP 8
- 类型注解: 所有公共函数必须有类型签名
- Docstring: Google Style
- 测试: 重要功能需要单元测试

### Commit Message 规范

<type>(<scope>): <subject>

<body>

<footer>

**Type**: feat, fix, docs, style, refactor, test, chore

**Scope**: agent, mcp, web, ml, vasp, docs, core

**Example**:
```
feat(mcp): add phonon prediction tool

Add new tool to predict phonon spectra using phonopy
integration. Includes frequency and eigenvector extraction.
```

---

## 📄 许可证

本项目采用 MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- **Materials Project** - 开放材料数据库和 API
- **OQDM** - 开放量子材料数据库
- **Pymatgen** - 材料分析 Python 库
- **LangChain** - LLM 应用开发框架
- **DeepSeek** - 大语言模型 API
- **JARVIS / ALIGNN** - 图神经网络材料性质预测
- **ASE** - 原子模拟环境
- **XGBoost** - 梯度提升决策树框架

---

## 📮 联系方式

- **Issues**: [GitHub Issues](https://github.com/your-repo/mat-agent-web/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/mat-agent-web/discussions)
- **Email**: your-email@example.com

---

<p align="center">
  <strong>MatAgent © 2024-2026 | Built with ❤️ for Materials Science Community</strong>
</p>
