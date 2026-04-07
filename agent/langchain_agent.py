"""LangChain Function Calling Agent for MatAgent

基于 LangGraph 显式图结构实现 Function Calling Agent，
使用 ToolNode 和 tools_condition 处理工具调用。
"""

# ============================================================
# 标准库导入
# ============================================================
import os
import json
import io
import time
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from operator import add

# ============================================================
# 第三方库导入
# ============================================================
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor

from ase.visualize.plot import plot_atoms
from ase.io import write

# ============================================================
# 项目模块导入
# ============================================================
import loadenv
import databasemanage
import tryssh
import myml.bandgap_predict as bandgap_predict
from myml.bandgap_predict import predict_bandgap
import flask_server
from flask_server import MatFileServer

# ============================================================
# 配置初始化
# ============================================================
config = loadenv.Config()
if not config.validate_config():
    raise EnvironmentError("请设置必要的环境变量")

# ============================================================
# 常量定义
# ============================================================
SERVER_PORT = 6750
PROJECT_FILE = "material_workflow.json"

# ============================================================
# 文件服务器管理
# ============================================================
_file_server = None
_file_server_started = False


def _get_file_server():
    """获取统一文件服务器"""
    global _file_server, _file_server_started
    if _file_server is None and not _file_server_started:
        try:
            _file_server = MatFileServer(port=SERVER_PORT)
            _file_server_started = True
        except Exception as e:
            print(f"启动文件服务器失败: {e}")
    return _file_server


def _get_server_url():
    """获取文件服务器地址"""
    return f"http://localhost:{SERVER_PORT}"


# 预启动服务
try:
    _get_file_server()
except:
    pass


# ============================================================
# LangGraph 状态定义
# ============================================================
class AgentState(MessagesState):
    """Agent 状态，包含消息历史"""
    pass


# ============================================================
# 可视化辅助函数
# ============================================================
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
        if server is None:
            return {"Image": None, "error": "文件服务器未启动"}
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        img_buffer.seek(0)
        
        image_url = server.add_image(img_buffer)
        
        return {"Image": image_url, "error": None}
    except Exception as e:
        return {"Image": None, "error": str(e)}


def _plot_vasp_band(xml_path: str, kpoints_path: str) -> dict:
    """使用 Pymatgen 绘制能带图，返回图片URL"""
    try:
        from pymatgen.io.vasp import Vasprun
        from pymatgen.electronic_structure.plotter import BSPlotter
        
        # 加载数据
        run = Vasprun(xml_path, parse_projected_eigen=False)
        bs = run.get_band_structure(kpoints_filename=kpoints_path, line_mode=True)
        
        # 提取物理量
        is_metal = bs.is_metal()
        gap_info = bs.get_band_gap()
        results = {
            "is_metal": is_metal,
            "gap": gap_info['energy'],
            "fermi_energy": bs.efermi,
        }
        
        # 绘制能带图
        plotter = BSPlotter(bs)
        plt_module = plotter.get_plot()
        fig = plt.gcf()
        ax = plt.gca()
        
        # 美化标签
        xticks = ax.get_xticks()
        labels = [label.get_text() for label in ax.get_xticklabels()]
        fixed_labels = [l.replace('GAMMA', r'$\Gamma$') for l in labels]
        ax.set_xticks(xticks)
        ax.set_xticklabels(fixed_labels, fontsize=20)
        ax.set_ylabel(r'$E - E_f$ (eV)', fontsize=20)
        ax.set_title('Band Structure', fontsize=22, pad=20)
        ax.axhline(y=0, color='#d62728', linestyle='--', linewidth=2, zorder=1)
        
        # 保存到内存
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 上传到文件服务器
        server = _get_file_server()
        if server:
            image_url = server.add_image(buf)
            return {"Image": image_url, "data": results, "error": None}
        return {"Image": None, "data": results, "error": "文件服务器未启动"}
        
    except Exception as e:
        return {"Image": None, "data": None, "error": str(e)}


def _plot_vasp_dos(vasprun_path: str) -> dict:
    """绘制DOS图，返回图片URL"""
    try:
        from pymatgen.io.vasp import Vasprun
        
        # 解析数据
        vr = Vasprun(vasprun_path, parse_dos=True)
        complete_dos = vr.complete_dos
        
        if complete_dos is None:
            return {"Image": None, "error": "无法从vasprun提取CompleteDos"}
        
        energies = complete_dos.energies - complete_dos.efermi
        tdos_array = list(complete_dos.densities.values())[0]
        element_dos = complete_dos.get_element_dos()
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Density of States Analysis', fontweight='bold')
        
        # (A) Total DOS
        ax = axes[0, 0]
        ax.plot(energies, tdos_array, color='black', lw=1.5, label='Total DOS')
        ax.fill_between(energies, 0, tdos_array, where=(energies < 0), color='gray', alpha=0.2)
        ax.axvline(x=0, color='#D55E00', linestyle='--', lw=1, label='$E_F$')
        ax.set_title('Total DOS')
        ax.set_ylabel('DOS (states/eV)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        
        # (B) Element Projected DOS
        ax = axes[0, 1]
        if element_dos:
            for el, dos_obj in element_dos.items():
                dens = list(dos_obj.densities.values())[0]
                ax.plot(energies, dens, label=str(el), lw=1.3)
            ax.axvline(x=0, color='#D55E00', linestyle='--', lw=1)
            ax.set_title('Element Projected DOS')
            ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # (C) Near-Fermi Region
        ax = axes[1, 0]
        fermi_mask = (energies > -5) & (energies < 5)
        ax.plot(energies[fermi_mask], tdos_array[fermi_mask], color='black', lw=1.5)
        ax.axvline(x=0, color='#D55E00', linestyle='--', lw=1.5, label='$E_F$')
        ax.set_title('Near-Fermi Region (-5 to 5 eV)')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # (D) DOS Analysis Info
        ax = axes[1, 1]
        ax.axis('off')
        info_text = f"""
        DOS Analysis Summary:
        
        Fermi Energy: {complete_dos.efermi:.3f} eV
        Energy Range: [{energies.min():.2f}, {energies.max():.2f}] eV
        Elements: {', '.join([str(el) for el in element_dos.keys()]) if element_dos else 'N/A'}
        """
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存到内存
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # 上传到文件服务器
        server = _get_file_server()
        if server:
            image_url = server.add_image(buf)
            return {"Image": image_url, "error": None}
        return {"Image": None, "error": "文件服务器未启动"}
        
    except Exception as e:
        return {"Image": None, "error": str(e)}


# ============================================================
# VASP 连接辅助函数
# ============================================================
def _get_vasp_connection():
    """获取VASP远程连接"""
    host = config.get_host()
    port = config.get_port()
    username = config.get_username()
    password = config.get_password()
    return tryssh.VaspTaskInitializer(host, username, password, port)


# ============================================================
# 基础工具函数
# ============================================================
@tool
def get_time() -> str:
    """获取当前系统时间"""
    import pandas as pd
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def read_file(file_path: str) -> dict:
    """读取本地文件"""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return {"content": content, "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Materials Project 工具
# ============================================================
@tool
def get_material_project_page(material_id: str) -> dict:
    """获取指定材料的Material Project页面链接"""
    if not material_id:
        return {"error": "材料ID不能为空"}
    url = f"https://next-gen.materialsproject.org/materials/{material_id}/"
    return {"material_id": material_id, "url": url, "message": "获取成功"}


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
    """从Materials Project数据库搜索材料"""
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
                formula = structure.composition.reduced_formula
                atoms = AseAtomsAdaptor.get_atoms(structure)
                
                # 创建临时 HTML 文件
                with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as tmp:
                    write(tmp.name, atoms, format='html')
                    tmp_path = tmp.name
                
                # 通过文件服务器保存并返回 URL
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
            )
            if not summary:
                return {"error": f"未找到材料 {material_id}"}
            
            s = summary[0]
            return s
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# OQMD 工具
# ============================================================
@tool
def search_materials_from_oqmd(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    stability_max: float = 0.1,
    limit: int = 20
) -> List[dict]:
    """在OQMD数据库搜索材料"""
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
    """在OQMD数据库获取指定材料的结构"""
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
        try:
            plot_result = get_structure_plot(structure)
            if plot_result.get("Image"):
                result["2d_image_url"] = plot_result["Image"]
            
            # 3D 结构
            atoms = AseAtomsAdaptor.get_atoms(structure)
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as tmp:
                write(tmp.name, atoms, format='html')
                tmp_path = tmp.name
            
            server = _get_file_server()
            if server:
                html_url = server.add_html_with_info(structure, tmp_path)
                result["3d_html_url"] = html_url
        except Exception as e:
            result["plot_error"] = str(e)
    
    return result


# ============================================================
# 结构构建工具
# ============================================================
@tool
def build_structure(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
    elements: List[str],
    frac_coord: List[List[float]],
    scaling_matrix: Optional[int] = None,
    save_to_cif: bool = False
) -> dict:
    """根据晶格参数和坐标构建晶体结构"""
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


# ============================================================
# ML 预测工具
# ============================================================
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
    """在计算服务器上执行linux命令"""
    try:
        with _get_vasp_connection() as vasp_task:
            result = vasp_task.execute_command(command)
            return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def extract_file(file_path: str) -> dict:
    """从计算服务器提取文件，返回下载URL"""
    print(f"\n🔧 [{datetime.now().strftime('%H:%M:%S')}] 调用工具: extract_file")
    print(f"   参数: file_path='{file_path}'")
    try:
        with _get_vasp_connection() as vasp_task:
            result = vasp_task.extract_file(file_path)
            print(f"   结果: {result.get('status', 'unknown')}")
            return result
    except Exception as e:
        print(f"   错误: {e}")
        return {"error": str(e)}


@tool
def create_mission(task_directory: str, mission: str) -> dict:
    """创建计算任务的输入文件，但不提交计算"""
    print(f"\n🔧 [{datetime.now().strftime('%H:%M:%S')}] 调用工具: create_mission")
    print(f"   参数: task_directory='{task_directory}', mission='{mission}'")
    try:
        with _get_vasp_connection() as vasp_task:
            result = vasp_task.create_mission(task_directory, mission)
            if isinstance(result, dict) and result.get("status") == "error":
                print(f"   错误: {result.get('message', '创建任务失败')}")
                return {"error": result.get("message", "创建任务失败")}
            print(f"   结果: 成功创建 {mission} 任务输入文件")
            return {"message": f"已创建 {mission} 任务输入文件", "task_directory": task_directory, "result": result}
    except Exception as e:
        print(f"   错误: {e}")
        return {"error": str(e)}


@tool
def submit_mission(task_directory: str, mission: str) -> dict:
    """提交计算任务"""
    print(f"\n🔧 [{datetime.now().strftime('%H:%M:%S')}] 调用工具: submit_mission")
    print(f"   参数: task_directory='{task_directory}', mission='{mission}'")
    try:
        with _get_vasp_connection() as vasp_task:
            result = vasp_task.submit_mission(task_directory, mission)
            if isinstance(result, dict) and result.get("status") == "error":
                print(f"   错误: {result.get('message', '提交任务失败')}")
                return {"error": result.get("message", "提交任务失败")}
            print(f"   结果: 成功提交 {mission} 任务")
            return {"message": f"已提交 {mission} 任务", "task_directory": task_directory, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def modify_incar(
    task_directory: str,
    mission: str,
    read: bool,
    write: Optional[str] = None
) -> dict:
    """读写修改计算任务的INCAR文件"""
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
    """提取计算任务的结果"""
    print(f"\n🔧 [{datetime.now().strftime('%H:%M:%S')}] 调用工具: extract_result")
    print(f"   参数: task_directory='{task_directory}', mission='{mission}', plot={plot}")
    
    mission = mission.lower().strip()
    
    if mission not in ["relax", "scf", "band", "dos"]:
        return {
            "success": False,
            "error": f"未知的计算类型: {mission}，可选: ['relax', 'scf', 'band', 'dos']"
        }
    
    try:
        with _get_vasp_connection() as vasp_task:
            if mission == "relax":
                result = vasp_task.extract_relax_info(task_directory)
                if plot and result and not result.get("error"):
                    try:
                        # 从 local_files 获取 CONTCAR 路径
                        local_files = result.get('local_files', {})
                        contcar_path = local_files.get('contcar')
                        if contcar_path and os.path.exists(contcar_path):
                            structure = Structure.from_file(contcar_path)
                            plot_result = get_structure_plot(structure)
                            if plot_result.get("Image"):
                                result["image_url"] = plot_result["Image"]
                            if plot_result.get("error"):
                                result["plot_error"] = plot_result["error"]
                    except Exception as e:
                        print(f"   生成结构图失败: {e}")
                        result["plot_error"] = str(e)
                        
            elif mission == "scf":
                result = vasp_task.extract_scf_info(task_directory)
                
            elif mission == "band":
                result = vasp_task.extract_band_info(task_directory)
                if plot and result and not result.get("error"):
                    try:
                        local_files = result.get('local_files', {})
                        vasprun_path = local_files.get('vasprun.xml')
                        kpoints_path = local_files.get('KPOINTS')
                        if vasprun_path and os.path.exists(vasprun_path) and kpoints_path and os.path.exists(kpoints_path):
                            plot_res = _plot_vasp_band(vasprun_path, kpoints_path)
                            if plot_res.get("Image"):
                                result["image_url"] = plot_res["Image"]
                                result["band_data"] = plot_res.get("data", {})
                            if plot_res.get("error"):
                                result["plot_error"] = plot_res["error"]
                    except Exception as e:
                        print(f"   生成能带图失败: {e}")
                        result["plot_error"] = str(e)
                
            elif mission == "dos":
                result = vasp_task.extract_dos_info(task_directory)
                if plot and result and not result.get("error"):
                    try:
                        local_files = result.get('local_files', {})
                        vasprun_path = local_files.get('vasprun.xml') or local_files.get('vasprun')
                        if vasprun_path and os.path.exists(vasprun_path):
                            plot_res = _plot_vasp_dos(vasprun_path)
                            if plot_res.get("Image"):
                                result["image_url"] = plot_res["Image"]
                            if plot_res.get("error"):
                                result["plot_error"] = plot_res["error"]
                    except Exception as e:
                        print(f"   生成DOS图失败: {e}")
                        result["plot_error"] = str(e)
            
            if isinstance(result, dict) and result.get("error"):
                print(f"   错误: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
            
            print(f"   结果: 成功提取 {mission} 任务结果")
            return {"success": True, "mission": mission, "task_directory": task_directory, "result": result}
    except Exception as e:
        print(f"   错误: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# 项目管理工具
# ============================================================
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


# ============================================================
# 工具列表
# ============================================================
TOOLS = [
    # 基础工具
    get_time,
    read_file,
    # Materials Project
    get_material_project_page,
    search_materials_from_mp,
    get_band_gap,
    get_material_structure_from_mp,
    get_material_all_infomation_by_id,
    # OQMD
    search_materials_from_oqmd,
    get_material_structure_from_oqmd,
    # 结构构建
    build_structure,
    # ML预测
    predict_band_gap,
    # VASP计算
    create_task,
    list_task_directories,
    check_squeue,
    execute_command,
    extract_file,
    create_mission,
    submit_mission,
    modify_incar,
    extract_result,
    # 项目管理
    list_all_projects,
    set_task_progress,
    get_project_workflow,
]


# ============================================================
# MatAgent 类 - LangGraph Agent 实现
# ============================================================
class MatAgent:
    """基于 LangGraph 的材料科学智能助手"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        
        self.llm = ChatOpenAI(
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com/v1",
            api_key=self.api_key,
            temperature=0.7
        ).bind_tools(TOOLS)
        
        self.tools = TOOLS
        self._tool_map = {t.name: t for t in self.tools}
        
        self._graph = self._build_graph()
        self._message_history = []
        
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
        """构建 LangGraph 状态图"""
        
        def assistant_node(state: AgentState) -> dict:
            """Assistant 节点：调用 LLM"""
            system_msg = SystemMessage(
                content="""你是一个专业的材料科学智能助手，帮助用户进行材料设计与计算。

重要规则：
0. 务必真实使用工具，不许捏造结果
1. 调用工具后，如果返回结果里有url，必须在回复中用 Markdown 格式嵌入图片链接
2. 2D结构图用: ![2D结构图](http://localhost:6750/image/图片文件名)
3. 3D结构图用: [查看3D交互结构](http://localhost:6750/view/结构ID)
4. 告诉用户点击链接查看
5. submib_mission和extract_result的路径直接用父目录如/data/user/mission/InGaP2_20260407/

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
        """处理对话，支持上下文记忆"""
        start_time = time.time()
        
        try:
            self._message_history.append(HumanMessage(content=user_message))
            recent_messages = self._message_history[-20:]
            
            initial_state = {"messages": recent_messages}
            final_state = self._graph.invoke(initial_state)
            
            duration = int((time.time() - start_time) * 1000)
            messages = final_state.get("messages", [])
            
            assistant_messages = [m for m in messages if isinstance(m, AIMessage) and not hasattr(m, 'tool_call_id')]
            final_message = assistant_messages[-1].content if assistant_messages else "完成"
            
            if assistant_messages:
                self._message_history.append(AIMessage(content=final_message))
            
            tool_results = []
            
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_results.append({
                            "tool_name": tc.get("name", ""),
                            "tool_args": tc.get("args", {}),
                            "result": ""
                        })
                
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    tool_name = getattr(msg, "name", None) or "unknown"
                    result = msg.content if hasattr(msg, "content") else str(msg)
                    
                    for tr in tool_results:
                        if tr["tool_name"] == tool_name and tr["result"] == "":
                            tr["result"] = result
                            break
                    else:
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
        """流式处理对话"""
        initial_state = {"messages": [HumanMessage(content=user_message)]}
        
        for step in self._graph.stream(initial_state, stream_mode="values"):
            messages = step.get("messages", [])
            
            if messages:
                last_msg = messages[-1]
                if last_msg.content:
                    yield last_msg.content
    
    def get_state(self) -> AgentState:
        """获取当前状态（用于调试）"""
        return self._graph.get_state({})
    
    # ============================================================
    # 便捷方法封装（供直接调用）
    # ============================================================
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
    
    def extract_relax_info(self, task_directory: str) -> dict:
        """提取结构优化结果"""
        try:
            with self._get_connection() as vasp_task:
                result = vasp_task.extract_relax_info(task_directory)
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
                    plot_result = get_structure_plot(structure)
                    if plot_result.get("Image"):
                        result["2d_image_url"] = plot_result["Image"]
                    
                    try:
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
        mission = mission.lower().strip()
        
        submit_method_map = {
            "relax": lambda vt, td: vt.submit_relax_calculation(td),
            "scf": lambda vt, td: vt.submit_scf_calculation(td),
            "band": lambda vt, td: vt.submit_band_calculation(td),
            "dos": lambda vt, td: vt.submit_dos_calculation(td),
        }
        
        if mission not in submit_method_map:
            return {"error": f"未知的计算类型: {mission}，可选: relax, scf, band, dos"}
        
        try:
            with self._get_connection() as vasp_task:
                result = submit_method_map[mission](vasp_task, task_directory)
                if isinstance(result, dict) and result.get("status") != "ok":
                    return {"error": result.get("message", "提交失败")}
                return {"message": f"已提交 {mission} 任务", "task_directory": task_directory, "result": result}
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


# ============================================================
# 工厂函数和初始化
# ============================================================
def create_langchain_agent(api_key: str = None) -> MatAgent:
    """创建 LangChain Agent 实例"""
    return MatAgent(api_key)


def init_services():
    """初始化远程连接和可视化服务"""
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


# ============================================================
# 主程序入口
# ============================================================
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
