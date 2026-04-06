"""LangChain Function Calling Agent for MatAgent

基于 LangGraph 显式图结构实现 Function Calling Agent，
使用 ToolNode 和 tools_condition 处理工具调用。
"""

import os
import json
import io
import time
from typing import Optional, List, Dict, Any, Tuple, TypedDict, Annotated
from datetime import datetime
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
import matplotlib.pyplot as plt

import loadenv
import databasemanage
import tryssh
import myml.bandgap_predict as bandgap_predict
from myml.bandgap_predict import predict_bandgap
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
import flask_server
from flask_server import MatFileServer
from ase.visualize.plot import plot_atoms
import tempfile
from ase.io import write

config = loadenv.Config()
if not config.validate_config():
    raise EnvironmentError("请设置必要的环境变量")


# ============================================================
# LangGraph 状态定义 (使用 MessagesState)
# ============================================================

class AgentState(MessagesState):
    """Agent 状态，包含消息历史"""
    pass


# ============================================================
# 工具函数定义 (与 mpmcp.py 对应)
# ============================================================

_file_server = None
_file_server_started = False

def _get_file_server():
    """获取统一文件服务器"""
    global _file_server, _file_server_started
    if _file_server is None and not _file_server_started:
        try:
            _file_server = MatFileServer(port=6750)
            _file_server_started = True
        except Exception as e:
            print(f"启动文件服务器失败: {e}")
    return _file_server

# 预启动服务
try:
    _get_file_server()
except:
    pass
    pass


_server_port = 6750

def _get_server_url():
    """获取文件服务器地址"""
    return f"http://localhost:{_server_port}"

def get_structure_plot(structure: Structure, rotation: str = '10x,10y,0z') -> dict:
    """生成 2D 结构图，返回图片 URL"""
    try:
        atoms = structure.to_ase_atoms()
        atoms.wrap()
        
        def _enhance_for_plot(atoms):
            cell = atoms.get_cell()
            new_symbols = []
            new_scaled = atoms.get_scaled_positions()
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for idx, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), new_scaled)):
                            new_symbols.append(symbol)
            
            from ase import Atoms
            return Atoms(symbols=new_symbols[:len(atoms)], scaled_positions=new_scaled, cell=cell, pbc=True)
        
        enhanced_atoms = _enhance_for_plot(atoms)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_atoms(enhanced_atoms, ax, rotation=rotation, show_unit_cell=2)
        
        ax.set_axis_off()
        ax.set_title(f"Crystal Structure of {structure.composition.reduced_formula}", fontsize=14, fontweight='bold')
        
        # 保存到文件服务器的图片目录
        server = _get_file_server()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        img_buffer.seek(0)
        
        image_url = server.add_image(img_buffer)
        
        return {"Image": image_url, "error": None}
    except Exception as e:
        return {"Image": None, "error": str(e)}
    except Exception as e:
        return {"Image": None, "error": str(e)}


@tool
def get_time() -> str:
    """获取当前系统时间"""
    import pandas as pd
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_material_project_page(material_id: str) -> dict:
    """获取指定材料的Material Project页面链接"""
    if not material_id:
        return {"error": "材料ID不能为空"}
    url = f"https://next-gen.materialsproject.org/materials/{material_id}/"
    return {"material_id": material_id, "url": url, "message": "获取成功"}


@tool
def search_materials_from_oqmd(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    stability_max: float = 0.1,
    limit: int = 20
) -> List[dict]:
    """在OQMD数据库搜索材料
    
    Args:
        elements: 元素列表，如 ["Fe", "O"]
        band_gap_min: 最小带隙(eV)
        band_gap_max: 最大带隙(eV)
        stability_max: 最大凸包距离，默认0.1
        limit: 返回数量，默认20
    """
    import oqmd
    
    filter_parts = []
    if elements:
        if len(elements) == 1:
            filter_parts.append(f"element={elements[0]}")
        else:
            filter_parts.append(f"element_set={','.join(elements)}")
    if band_gap_min is not None:
        filter_parts.append(f"band_gap>={band_gap_min}")
    if band_gap_max is not None:
        filter_parts.append(f"band_gap<={band_gap_max}")
    if stability_max is not None:
        filter_parts.append(f"stability<={stability_max}")
    
    filter_expr = " AND ".join(filter_parts) if filter_parts else None
    
    return oqmd.search_oqmd(
        fields=["name", "entry_id", "band_gap", "delta_e", "stability", "spacegroup"],
        filter_expr=filter_expr,
        limit=limit,
        sort_by="stability"
    )


@tool
def get_material_structure_from_oqmd(
    entry_id: int,
    mode: str = "conventional",
    get_sites: bool = False,
    get_plot: bool = False,
    download: bool = False
) -> dict:
    """在OQMD数据库获取指定材料的结构
    
    Args:
        entry_id: OQMD材料条目ID
        mode: 下载模式，"conventional"或"primitive"
        get_sites: 是否获取原子位点信息
        get_plot: 是否生成结构图
        download: 是否下载CIF文件
    """
    import oqmd
    
    res = oqmd.parse_poscar_with_pymatgen(entry_id, mode)
    if not res["success"]:
        return {"error": res["error"]}
    
    structure = res["structure"]
    lattice = structure.lattice
    space_group_info = structure.get_space_group_info()
    
    result = {
        "formula": structure.formula,
        "space_group_symbol": space_group_info[0],
        "space_group_number": space_group_info[1],
        "lattice_parameters": {
            "a": round(lattice.a, 4),
            "b": round(lattice.b, 4),
            "c": round(lattice.c, 4),
            "alpha": round(lattice.alpha, 2),
            "beta": round(lattice.beta, 2),
            "gamma": round(lattice.gamma, 2)
        },
        "number_of_sites": len(structure)
    }
    
    if get_sites:
        result["sites"] = [{
            "element": site.species_string,
            "fractional_coordinates": [round(c, 4) for c in site.frac_coords]
        } for site in structure.sites]
    
    if download:
        os.makedirs("cifs", exist_ok=True)
        cif_path = f"cifs/{structure.formula.replace(' ', '_')}-oqmd-{entry_id}.cif"
        CifWriter(structure).write_file(cif_path)
        result["cif_path"] = cif_path
    
    if get_plot:
        server = _get_file_server()
        url = server.show_structure(structure, f"oqmd-{entry_id}.html")
        result["3d_image_url"] = url
    
    return result


@tool
def search_materials_from_mp(
    elements: Optional[List[str]] = None,
    exclude_elements: Optional[List[str]] = None,
    chemsys: Optional[str] = None,
    band_gap: Optional[Tuple[float, float]] = None,
    num_elements: Optional[Tuple[int, int]] = None,
    formula: Optional[str] = None,
    chunk_size: int = 25
) -> List[dict]:
    """从Materials Project数据库搜索材料
    
    Args:
        elements: 元素符号列表
        exclude_elements: 排除的元素
        chemsys: 化学系统
        band_gap: 带隙范围
        num_elements: 元素个数范围
        formula: 化学式
        chunk_size: 返回数量，默认25
    """
    from mp_api.client import MPRester
    
    API_KEY = config.get_api_key()
    if not API_KEY:
        return {"error": "API密钥未设置"}
    
    try:
        with MPRester(API_KEY) as mpr:
            criteria = {}
            if elements:
                criteria["elements"] = elements
            if exclude_elements:
                criteria["exclude_elements"] = exclude_elements
            if chemsys:
                criteria["chemsys"] = chemsys
            if band_gap:
                criteria["band_gap"] = band_gap
            if num_elements:
                criteria["num_elements"] = num_elements
            if formula:
                criteria["formula"] = formula
            
            results = mpr.summary.search(
                **criteria,
                fields=["material_id", "formula_pretty", "band_gap", "symmetry"],
                chunk_size=chunk_size
            )
            
            return [{
                "material_id": r.material_id,
                "formula_pretty": r.formula_pretty,
                "band_gap": r.band_gap,
                "symmetry": r.symmetry
            } for r in results]
    except Exception as e:
        return {"error": str(e)}


@tool
def get_band_gap(material_id: str) -> dict:
    """获取指定材料的带隙值(Material Project)"""
    from mp_api.client import MPRester
    
    API_KEY = config.get_api_key()
    if not API_KEY:
        return {"error": "API密钥未设置"}
    
    try:
        with MPRester(API_KEY) as mpr:
            results = mpr.summary.search(
                material_ids=material_id,
                fields=["band_gap", "formula_pretty"]
            )
            if not results:
                return {"error": f"未找到材料 {material_id}"}
            return {
                "material_id": material_id,
                "band_gap": results[0].band_gap,
                "formula": results[0].formula_pretty
            }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_material_structure_from_mp(
    material_id: str,
    get_sites: bool = False,
    get_plot: bool = True,
    download: bool = False
) -> dict:
    """获取指定材料的晶体结构(Material Project)"""
    from mp_api.client import MPRester
    
    API_KEY = config.get_api_key()
    if not API_KEY:
        return {"error": "API密钥未设置"}
    
    try:
        with MPRester(API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(material_id)
            lattice = structure.lattice
            space_group_info = structure.get_space_group_info()
            
            result = {
                "formula": structure.formula,
                "space_group_symbol": space_group_info[0],
                "space_group_number": space_group_info[1],
                "lattice_parameters": {
                    "a": round(lattice.a, 4),
                    "b": round(lattice.b, 4),
                    "c": round(lattice.c, 4),
                    "alpha": round(lattice.alpha, 2),
                    "beta": round(lattice.beta, 2),
                    "gamma": round(lattice.gamma, 2)
                },
                "number_of_sites": len(structure)
            }
            
            if get_sites:
                result["sites"] = [{
                    "element": site.species_string,
                    "fractional_coordinates": [round(c, 4) for c in site.frac_coords]
                } for site in structure.sites]
            
            if download:
                os.makedirs("cifs", exist_ok=True)
                cif_path = f"cifs/{material_id}.cif"
                CifWriter(structure).write_file(cif_path)
                result["cif_path"] = cif_path
            
            if get_plot:
                import tempfile
                from ase.io import write
                from pymatgen.io.ase import AseAtomsAdaptor
                
                formula = structure.composition.reduced_formula
                atoms = AseAtomsAdaptor.get_atoms(structure)
                
                # 创建临时 HTML 文件
                with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as tmp:
                    write(tmp.name, atoms, format='html')
                    tmp_path = tmp.name
                
                # 通过文件服务器保存并返回 URL（带结构信息）
                server = _get_file_server()
                html_url = server.add_html_with_info(structure, tmp_path)
                
                result["3d_html_url"] = html_url
                result["material_id"] = material_id
                
                plot_result = get_structure_plot(structure)
                if plot_result.get("Image"):
                    result["2d_image_url"] = plot_result["Image"]
            
            return result
    except Exception as e:
        return {"error": str(e)}


@tool
def get_material_all_infomation_by_id(material_id: str) -> dict:
    """获取Material Project指定材料的所有信息"""
    from mp_api.client import MPRester
    
    API_KEY = config.get_api_key()
    if not API_KEY:
        return {"error": "API密钥未设置"}
    
    try:
        with MPRester(API_KEY) as mpr:
            summary = mpr.summary.search(
                material_ids=material_id,
                fields=["material_id", "formula_pretty", "band_gap", "energy", "symmetry"]
            )
            if not summary:
                return {"error": f"未找到材料 {material_id}"}
            
            s = summary[0]
            return {
                "material_id": s.material_id,
                "formula_pretty": s.formula_pretty,
                "band_gap": s.band_gap,
                "energy": s.energy,
                "symmetry": s.symmetry
            }
    except Exception as e:
        return {"error": str(e)}


@tool
def build_structure(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
    elements: List[str],
    frac_coord: List[List[float]],
    scaling_matrix: Optional[int] = None,
    save_to_cif: bool = False
) -> dict:
    """根据晶格参数和坐标构建晶体结构
    
    Args:
        a, b, c: 晶格参数(埃)
        alpha, beta, gamma: 晶格角(度)
        elements: 元素符号列表
        frac_coord: 分数坐标
        scaling_matrix: 超胞扩展因子
        save_to_cif: 是否保存CIF文件
    """
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    structure = Structure(lattice, elements, frac_coord)
    
    if scaling_matrix:
        structure = structure * scaling_matrix
    
    result = {
        "formula": structure.formula,
        "lattice_parameters": {
            "a": structure.lattice.a,
            "b": structure.lattice.b,
            "c": structure.lattice.c,
            "alpha": structure.lattice.alpha,
            "beta": structure.lattice.beta,
            "gamma": structure.lattice.gamma
        },
        "number_of_sites": len(structure)
    }
    
    if save_to_cif:
        os.makedirs("custom_structures", exist_ok=True)
        cif_path = f"custom_structures/{structure.formula.replace(' ', '_')}.cif"
        CifWriter(structure).write_file(cif_path)
        result["cif_path"] = cif_path
    
    return result


@tool
def predict_band_gap(formula: str) -> dict:
    """使用XGBoost模型预测材料的带隙"""
    try:
        result = bandgap_predict.predict_bandgap(formula)
        return {"formula": formula, "predicted_band_gap": result[0] if result else None}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# VASP 远程计算工具
# ============================================================

def _get_vasp_connection():
    host = config.get_host()
    port = config.get_port()
    username = config.get_username()
    password = config.get_password()
    return tryssh.VaspTaskInitializer(host, username, password, port)


@tool
def create_task(formula: str, cif_path: str) -> dict:
    """在远程服务器创建任务目录并上传CIF"""
    try:
        with _get_vasp_connection() as vasp_task:
            base_dir = config.get_base_dir()
            result = vasp_task.create_task(formula, cif_path, base_dir)
            if result:
                return {"message": "任务创建成功", "task_directory": result}
            return {"error": "任务创建失败"}
    except Exception as e:
        return {"error": str(e)}


@tool
def list_task_directories() -> dict:
    """列出远程服务器上的所有任务目录"""
    try:
        with _get_vasp_connection() as vasp_task:
            base_dir = config.get_base_dir()
            result = vasp_task.get_task_directories(base_dir)
            return {"task_directories": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def check_squeue() -> str:
    """检查Slurm任务队列状态"""
    try:
        with _get_vasp_connection() as vasp_task:
            result = vasp_task.check_squeue()
            if result is None:
                return "无法获取任务队列状态"
            return result
    except Exception as e:
        return f"查询失败: {str(e)}"


@tool
def execute_command(command: str) -> dict:
    """在计算服务器上执行linux命令
    
    注意：这是计算服务器，不是MCP服务器
    """
    try:
        with _get_vasp_connection() as vasp_task:
            result = vasp_task.execute_command(command)
            return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def extract_file(file_path: str) -> dict:
    """从计算服务器提取文件，返回下载URL"""
    try:
        with _get_vasp_connection() as vasp_task:
            result = vasp_task.extract_file(file_path)
            return result
    except Exception as e:
        return {"error": str(e)}


@tool
def create_mission(task_directory: str, mission: str) -> dict:
    """创建计算任务的输入文件，但不提交计算
    
    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: 'relax', 'scf', 'band', 'dos'
    """
    try:
        with _get_vasp_connection() as vasp_task:
            vasp_task.create_mission(task_directory, mission)
            return {"message": f"已创建 {mission} 任务输入文件", "task_directory": task_directory}
    except Exception as e:
        return {"error": str(e)}


@tool
def submit_mission(task_directory: str, mission: str) -> dict:
    """提交计算任务
    
    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: 'relax', 'scf', 'band', 'dos'
    """
    try:
        with _get_vasp_connection() as vasp_task:
            vasp_task.submit_mission(task_directory, mission)
            return {"message": f"已提交 {mission} 任务", "task_directory": task_directory}
    except Exception as e:
        return {"error": str(e)}


@tool
def modify_incar(
    task_directory: str,
    mission: str,
    read: bool,
    write: Optional[str] = None
) -> dict:
    """读写修改计算任务的INCAR文件
    
    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: 'relax', 'scf', 'band', 'dos'
        read: True读取，False写入
        write: 写入的INCAR内容（字符串格式或换行分隔的 key=value 格式）
    """
    subdir_map = {
        "relax": "结构优化",
        "scf": "自洽计算",
        "band": "能带计算",
        "dos": "态密度计算"
    }
    
    if mission not in subdir_map:
        return {"error": f"未知的计算类型: {mission}，可选: {list(subdir_map.keys())}"}
    
    remote_path = f"{task_directory}/{subdir_map[mission]}/INCAR"
    
    try:
        with _get_vasp_connection() as vasp_task:
            if read:
                result = vasp_task.modify_incar_file(task_directory, mission, read_mode=True)
                if result.get("status") == "ok":
                    return {"incar": result.get("incar_params", {}), "file_path": remote_path}
                else:
                    return {"error": result.get("message", "读取失败"), "hint": "请先创建任务并生成输入文件"}
            else:
                params = {}
                if write:
                    for line in write.strip().split("\n"):
                        line = line.strip()
                        if "=" in line and line:
                            key, val = line.split("=", 1)
                            params[key.strip()] = val.strip()
                result = vasp_task.modify_incar_file(task_directory, mission, read_mode=False, new_params=params)
                if result.get("status") == "ok":
                    return {"message": "INCAR已更新", "file_path": remote_path}
                else:
                    return {"error": result.get("message", "写入失败")}
    except Exception as e:
        return {"error": str(e), "hint": "请确认任务目录存在且已生成输入文件"}


@tool
def extract_result(task_directory: str, mission: str, plot: bool = True) -> dict:
    """提取计算任务的结果
    
    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: 'relax', 'scf', 'band', 'dos'
        plot: 是否绘图
    """
    mission = mission.lower().strip()
    
    method_map = {
        "relax": lambda vt: vt.extract_relax_info(task_directory),
        "scf": lambda vt: vt.extract_scf_info(task_directory),
        "band": lambda vt: vt.extract_band_info(task_directory),
        "dos": lambda vt: vt.extract_dos_info(task_directory),
    }
    
    if mission not in method_map:
        return {
            "success": False,
            "error": f"未知的计算类型: {mission}，可选: ['relax', 'scf', 'band', 'dos']"
        }
    
    try:
        with _get_vasp_connection() as vasp_task:
            result = method_map[mission](vasp_task)
            
            if isinstance(result, dict) and result.get("error"):
                return {"success": False, "error": result.get("error")}
            
            return {"success": True, "mission": mission, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def read_file(file_path: str) -> dict:
    """读取MCP服务器上的文件"""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return {"content": content, "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 工具列表 - 与 mpmcp.py 一致
# ============================================================

TOOLS = [
    get_time,
    get_material_project_page,
    search_materials_from_oqmd,
    get_material_structure_from_oqmd,
    search_materials_from_mp,
    get_band_gap,
    get_material_structure_from_mp,
    build_structure,
    get_material_all_infomation_by_id,
    create_task,
    list_task_directories,
    check_squeue,
    execute_command,
    extract_file,
    predict_band_gap,
    read_file,
    create_mission,
    submit_mission,
    modify_incar,
    extract_result,
]

# 项目管理工具（mpmcp.py 中被注释，本地实现）
PROJECT_FILE = "material_workflow.json"


@tool
def list_all_projects() -> dict:
    """列出所有材料研发项目"""
    if not os.path.exists(PROJECT_FILE):
        return {"projects": []}
    with open(PROJECT_FILE, "r") as f:
        projects = json.load(f)
    return {"projects": list(projects.keys())}


@tool
def set_task_progress(
    project_name: str,
    description: Optional[str] = None,
    step_name: Optional[str] = None,
    status: Optional[str] = None
) -> dict:
    """记录项目进度"""
    if not os.path.exists(PROJECT_FILE):
        projects = {}
    else:
        with open(PROJECT_FILE, "r") as f:
            projects = json.load(f)
    
    if project_name not in projects:
        projects[project_name] = {}
        if description:
            projects[project_name]["description"] = description
    
    if step_name and status:
        from datetime import datetime
        projects[project_name][step_name] = {
            "status": status,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    with open(PROJECT_FILE, "w") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)
    
    return {"message": f"项目 {project_name} 已更新", "project": projects[project_name]}


@tool
def get_project_workflow(project_name: str) -> dict:
    """查看项目详情"""
    if not os.path.exists(PROJECT_FILE):
        return {"error": "项目文件不存在"}
    
    with open(PROJECT_FILE, "r") as f:
        projects = json.load(f)
    
    if project_name not in projects:
        return {"error": f"项目 {project_name} 不存在"}
    
    return {"project": project_name, "workflow": projects[project_name]}


TOOLS.extend([list_all_projects, set_task_progress, get_project_workflow])


# ============================================================
# LangGraph Agent 实现
# ============================================================

class MatAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=self.api_key,
            temperature=0.7
        ).bind_tools(TOOLS)
        
        self.tools = TOOLS
        self._tool_map = {t.name: t for t in self.tools}
        
        self._graph = self._build_graph()
        self._message_history = []
        
        # 直接导入 tryssh 供方法调用
        import tryssh
        self._tryssh = tryssh
        self._connection = None
    
    def _get_connection(self):
        """获取远程连接"""
        if self._connection is None:
            self._connection = self._tryssh.VaspTaskInitializer(
                config.get_host(),
                config.get_username(),
                config.get_password(),
                config.get_port()
            )
        return self._connection
    
    def _build_graph(self):
        """构建 LangGraph 状态图 (使用标准 ReAct 模式)"""
        
        def assistant_node(state: AgentState) -> dict:
            """Assistant 节点：调用 LLM"""
            system_msg = SystemMessage(
                content="""你是一个专业的材料科学智能助手，帮助用户进行材料设计与计算。

重要规则：
1. 获取材料结构后，必须在回复中用 Markdown 格式嵌入图片链接
2. 2D结构图用: ![2D结构图](http://localhost:6750/image/图片文件名)
3. 3D结构图用: [查看3D交互结构](http://localhost:6750/view/结构ID)
4. 告诉用户点击链接查看

直接使用上述 HTTP URL 格式，不要使用本地文件路径。"""
            )
            response = self.llm.invoke([system_msg] + state["messages"])
            return {"messages": [response]}
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("assistant", assistant_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.add_edge(START, "assistant")
        workflow.add_conditional_edges(
            "assistant",
            tools_condition,
            {"tools": "tools", "__end__": END}
        )
        workflow.add_edge("tools", "assistant")
        
        return workflow.compile()
    
    def chat(self, user_message: str) -> dict:
        """使用 LangGraph 处理对话，支持上下文记忆"""
        start_time = time.time()
        
        try:
            # 添加用户消息到历史
            self._message_history.append(HumanMessage(content=user_message))
            
            # 保留最近 20 条消息
            recent_messages = self._message_history[-20:]
            
            initial_state = {"messages": recent_messages}
            
            final_state = self._graph.invoke(initial_state)
            
            duration = int((time.time() - start_time) * 1000)
            
            messages = final_state.get("messages", [])
            
            # 获取助手回复
            assistant_messages = [m for m in messages if isinstance(m, AIMessage) and not hasattr(m, 'tool_call_id')]
            final_message = assistant_messages[-1].content if assistant_messages else "完成"
            
            # 添加助手回复到历史
            if assistant_messages:
                self._message_history.append(AIMessage(content=final_message))
            
            tool_results = []
            
            # 遍历所有消息，收集 tool_calls 和 tool responses
            for msg in messages:
                # AIMessage: 有 tool_calls，包含参数信息
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_results.append({
                            "tool_name": tc.get("name", ""),
                            "tool_args": tc.get("args", {}),
                            "result": ""
                        })
                
                # ToolMessage: 有 tool_call_id，包含执行结果
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    tool_name = getattr(msg, "name", None) or "unknown"
                    result = msg.content if hasattr(msg, "content") else str(msg)
                    
                    # 匹配已有的 tool_results
                    for tr in tool_results:
                        if tr["tool_name"] == tool_name and tr["result"] == "":
                            tr["result"] = result
                            break
                    else:
                        # 如果没有匹配的，创建新的
                        tool_results.append({
                            "tool_name": tool_name,
                            "tool_args": {},
                            "result": result
                        })
            
            if tool_results:
                return {
                    "type": "tool_calls",
                    "tool_results": tool_results,
                    "message": final_message,
                    "duration": duration
                }
            
            return {
                "type": "text",
                "message": final_message,
                "duration": duration
            }
            
        except Exception as e:
            return {
                "type": "error",
                "message": f"处理请求时出错: {str(e)}",
                "error": str(e)
            }
    
    def stream_chat(self, user_message: str):
        """流式处理对话，用于 Streamlit 实时显示"""
        initial_state = {"messages": [HumanMessage(content=user_message)]}
        
        for step in self._graph.stream(initial_state, stream_mode="values"):
            messages = step.get("messages", [])
            
            if messages:
                last_msg = messages[-1]
                tool_calls = []
                if hasattr(last_msg, "tool_calls"):
                    tool_calls = last_msg.tool_calls
                
                # 返回可迭代的生成器，供 st.write_stream 使用
                if last_msg.content:
                    yield last_msg.content
    
    def get_state(self) -> AgentState:
        """获取当前状态（用于调试）"""
        return self._graph.get_state({})
    
    def list_task_directories(self) -> dict:
        """列出远程任务目录"""
        try:
            with self._get_connection() as vasp_task:
                base_dir = config.get_base_dir()
                result = vasp_task.get_task_directories(base_dir)
                return {"task_directories": result}
        except Exception as e:
            return {"error": str(e)}
    
    def create_task(self, formula: str, cif_path: str) -> dict:
        """创建任务目录"""
        try:
            with self._get_connection() as vasp_task:
                base_dir = config.get_base_dir()
                task_dir = vasp_task.create_task(formula, cif_path, base_dir)
                if task_dir:
                    return {"task_directory": task_dir, "status": "success"}
                return {"error": "创建失败"}
        except Exception as e:
            return {"error": str(e)}
    
    def check_squeue(self) -> dict:
        """检查 Slurm 任务队列"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.check_squeue()
                return {"squeue": result or "无运行任务"}
        except Exception as e:
            return {"error": str(e)}
    
    def submit_opt_mission(self, task_directory: str) -> dict:
        """提交结构优化任务"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.relax(task_directory)
                return {"status": result}
        except Exception as e:
            return {"error": str(e)}
    
    def submit_scf_mission(self, task_directory: str) -> dict:
        """提交自洽计算任务"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.scf(task_directory)
                return {"status": result}
        except Exception as e:
            return {"error": str(e)}
    
    def submit_band_mission(self, task_directory: str) -> dict:
        """提交能带计算任务"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.band_calc(task_directory)
                return {"status": result}
        except Exception as e:
            return {"error": str(e)}
    
    def extract_opt_info(self, task_directory: str) -> dict:
        """提取结构优化结果"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.extract_opt_info(task_directory)
                return result
        except Exception as e:
            return {"error": str(e)}
    
    def extract_scf_info(self, task_directory: str) -> dict:
        """提取自洽计算结果"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.extract_scf_info(task_directory)
                return result
        except Exception as e:
            return {"error": str(e)}
    
    def extract_band_info(self, task_directory: str) -> dict:
        """提取能带计算结果"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.extract_band_info(task_directory)
                return result
        except Exception as e:
            return {"error": str(e)}
    
    def build_structure(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float,
                       elements: List[str], frac_coord: List[List[float]], scaling_matrix: int = None, 
                       save_to_cif: bool = False) -> dict:
        """构建晶体结构"""
        try:
            from pymatgen.core import Structure, Lattice
            from pymatgen.io.cif import CifWriter
            
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            structure = Structure(lattice, elements, frac_coord)
            
            if scaling_matrix:
                structure = structure * scaling_matrix
            
            result = {
                "formula": structure.formula,
                "lattice_parameters": {
                    "a": structure.lattice.a,
                    "b": structure.lattice.b,
                    "c": structure.lattice.c,
                    "alpha": structure.lattice.alpha,
                    "beta": structure.lattice.beta,
                    "gamma": structure.lattice.gamma
                },
                "number_of_sites": len(structure)
            }
            
            if save_to_cif:
                os.makedirs("custom_structures", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                formula_str = structure.formula.replace(' ', '_')
                cif_path = f"custom_structures/{formula_str}_{timestamp}.cif"
                CifWriter(structure).write_file(cif_path)
                result["cif_path"] = cif_path
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def predict_band_gap(self, formula: str) -> dict:
        """预测带隙"""
        try:
            import myml.bandgap_predict as bandgap_predict
            result = bandgap_predict.predict_bandgap(formula)
            return {"formula": formula, "predicted_band_gap": result[0] if result else None}
        except Exception as e:
            return {"error": str(e)}
    
    def search_materials(self, elements=None, exclude_elements=None, chemsys=None, 
                         band_gap=None, num_elements=None, formula=None, chunk_size=25) -> dict:
        """搜索Materials Project材料"""
        try:
            from mp_api.client import MPRester
            API_KEY = config.get_api_key()
            if not API_KEY:
                return {"error": "API密钥未设置"}
            
            with MPRester(API_KEY) as mpr:
                criteria = {}
                if elements:
                    criteria["elements"] = elements
                if exclude_elements:
                    criteria["exclude_elements"] = exclude_elements
                if chemsys:
                    criteria["chemsys"] = chemsys
                if band_gap:
                    criteria["band_gap"] = band_gap
                if num_elements:
                    criteria["num_elements"] = num_elements
                if formula:
                    criteria["formula"] = formula
                
                results = mpr.summary.search(
                    **criteria,
                    fields=["material_id", "formula_pretty", "band_gap", "symmetry"],
                    chunk_size=chunk_size
                )
                
                return {"materials": [{
                    "material_id": r.material_id,
                    "formula_pretty": r.formula_pretty,
                    "band_gap": r.band_gap,
                    "symmetry": r.symmetry
                } for r in results]}
        except Exception as e:
            return {"error": str(e)}
    
    def get_material_structure(self, material_id, get_sites=False, get_plot=True, download=False) -> dict:
        """获取材料结构"""
        try:
            from mp_api.client import MPRester
            from pymatgen.io.cif import CifWriter
            
            API_KEY = config.get_api_key()
            if not API_KEY:
                return {"error": "API密钥未设置"}
            
            with MPRester(API_KEY) as mpr:
                structure = mpr.get_structure_by_material_id(material_id)
                lattice = structure.lattice
                space_group_info = structure.get_space_group_info()
                
                result = {
                    "formula": structure.formula,
                    "space_group_symbol": space_group_info[0],
                    "space_group_number": space_group_info[1],
                    "lattice_parameters": {
                        "a": round(lattice.a, 4),
                        "b": round(lattice.b, 4),
                        "c": round(lattice.c, 4),
                        "alpha": round(lattice.alpha, 2),
                        "beta": round(lattice.beta, 2),
                        "gamma": round(lattice.gamma, 2)
                    },
                    "number_of_sites": len(structure),
                    "material_id": material_id
                }
                
                if get_sites:
                    result["sites"] = [{
                        "element": site.species_string,
                        "fractional_coordinates": [round(c, 4) for c in site.frac_coords]
                    } for site in structure.sites]
                
                if download:
                    os.makedirs("cifs", exist_ok=True)
                    cif_path = f"cifs/{material_id}.cif"
                    CifWriter(structure).write_file(cif_path)
                    result["cif_path"] = cif_path
                
                if get_plot:
                    # 生成2D结构图
                    plot_result = get_structure_plot(structure)
                    if plot_result.get("Image"):
                        result["2d_image_url"] = plot_result["Image"]
                    
                    # 生成3D结构
                    try:
                        import tempfile
                        from ase.io import write
                        from pymatgen.io.ase import AseAtomsAdaptor
                        
                        atoms = AseAtomsAdaptor.get_atoms(structure)
                        with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as tmp:
                            write(tmp.name, atoms, format='html')
                            tmp_path = tmp.name
                        
                        server = _get_file_server()
                        html_url = server.add_html_with_info(structure, tmp_path)
                        result["3d_html_url"] = html_url
                    except Exception as e:
                        result["3d_error"] = str(e)
                
                return result
        except Exception as e:
            return {"error": str(e)}
    
    def create_mission(self, task_directory: str, mission: str) -> dict:
        """创建计算任务输入文件"""
        try:
            with self._get_connection() as vasp_task:
                vasp_task.create_mission(task_directory, mission)
                return {"message": f"已创建 {mission} 任务输入文件", "task_directory": task_directory}
        except Exception as e:
            return {"error": str(e)}
    
    def modify_incar(self, task_directory: str, mission: str, read: bool, write: str = None) -> dict:
        """修改INCAR文件"""
        subdir_map = {"relax": "结构优化", "scf": "自洽计算", "band": "能带计算", "dos": "态密度计算"}
        
        if mission not in subdir_map:
            return {"error": f"未知的计算类型: {mission}"}
        
        try:
            with self._get_connection() as vasp_task:
                if read:
                    result = vasp_task.modify_incar_file(task_directory, mission, read_mode=True)
                    if result.get("status") == "ok":
                        return {"incar": result.get("incar_params", {}), "file_path": f"{task_directory}/{subdir_map[mission]}/INCAR"}
                    else:
                        return {"error": result.get("message", "读取失败")}
                else:
                    params = {}
                    if write:
                        for line in write.strip().split("\n"):
                            line = line.strip()
                            if "=" in line and line:
                                key, val = line.split("=", 1)
                                params[key.strip()] = val.strip()
                    result = vasp_task.modify_incar_file(task_directory, mission, read_mode=False, new_params=params)
                    if result.get("status") == "ok":
                        return {"message": "INCAR已更新", "file_path": f"{task_directory}/{subdir_map[mission]}/INCAR"}
                    else:
                        return {"error": result.get("message", "写入失败")}
        except Exception as e:
            return {"error": str(e)}
    
    def submit_mission(self, task_directory: str, mission: str) -> dict:
        """提交计算任务"""
        try:
            with self._get_connection() as vasp_task:
                vasp_task.submit_mission(task_directory, mission)
                return {"message": f"已提交 {mission} 任务", "task_directory": task_directory}
        except Exception as e:
            return {"error": str(e)}
    
    def extract_result(self, task_directory: str, mission: str, plot: bool = True) -> dict:
        """提取计算结果"""
        mission = mission.lower().strip()
        method_map = {
            "relax": lambda vt: vt.extract_relax_info(task_directory),
            "scf": lambda vt: vt.extract_scf_info(task_directory),
            "band": lambda vt: vt.extract_band_info(task_directory),
            "dos": lambda vt: vt.extract_dos_info(task_directory),
        }
        
        if mission not in method_map:
            return {"success": False, "error": f"未知的计算类型: {mission}"}
        
        try:
            with self._get_connection() as vasp_task:
                result = method_map[mission](vasp_task)
                if isinstance(result, dict) and result.get("error"):
                    return {"success": False, "error": result.get("error")}
                return {"success": True, "mission": mission, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_langchain_agent(api_key: str = None) -> MatAgent:
    """创建 LangChain Agent 实例"""
    return MatAgent(api_key)


def init_services():
    """初始化远程连接和可视化服务"""
    import tryssh
    
    print("正在连接远程服务器...")
    connection = tryssh.VaspTaskInitializer(
        config.get_host(),
        config.get_username(),
        config.get_password(),
        config.get_port()
    )
    
    for i in range(5):
        try:
            with connection as vasp_task:
                if vasp_task.link():
                    print("✅ 已成功连接到远程服务器")
                    break
        except Exception as e:
            print(f"⚠️ 连接远程服务器失败，正在重试... ({i+1}/5), 错误: {e}")
            if i == 4:
                raise e
    
    print("✅ Agent 初始化完成")
    return connection


if __name__ == "__main__":
    try:
        connection = init_services()
    except Exception as e:
        print(f"⚠️ 连接远程服务器失败: {e}")
    
    agent = create_langchain_agent()
    
    print("\n🤖 MatAgent 已就绪，输入消息开始对话 (输入 'exit' 退出)")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.chat(user_input)
        print(f"Assistant: {response.get('message', response)}")
