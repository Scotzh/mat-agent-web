"""
MatAgent MCP Server - 材料科学计算工具集
提供材料查询、结构建模、VASP任务管理等功能
"""

# ============ 标准库导入 ============
import os
import io
import json
import atexit
import signal
import shutil
import tempfile
import multiprocessing
from datetime import datetime
from warnings import simplefilter
from itertools import product
from typing import Optional, Dict, Any, List

from pydantic_core import Url

# 忽略 FutureWarning
simplefilter(action="ignore", category=FutureWarning)

# ============ 第三方库导入 ============
import numpy as np
import pandas as pd
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage
from pydantic import BaseModel

from mp_api.client import MPRester
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.outputs import Vasprun as VasprunOutput
from pymatgen.electronic_structure.plotter import BSPlotter

from ase.io import write
from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.build import bulk

# ============ 本地模块导入 ============
from config import loadenv
from db import databasemanage
from server import tryssh, flask_server
import oqmd

# ============ 全局配置 ============
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = ["serif"]
plt.rcParams["axes.unicode_minus"] = False

# MCP 实例
mcp = FastMCP(name="MatAgent")

# 全局子进程列表
child_processes: list[tuple[multiprocessing.Process, str]] = []

# ============ 环境配置加载 ============
config = loadenv.Config()
if not config.validate_config():
    raise EnvironmentError("请设置必要的环境变量")

MY_API_KEY = config.get_api_key()
IP = config.get_ip()
HOST = config.get_host()
PORT = config.get_port()
USERNAME = config.get_username()
PASSWORD = config.get_password()


# ============ 进程清理函数 ============
def cleanup_child_processes():
    """在主进程退出时尝试优雅终止所有子进程并删除临时文件目录"""
    for p, temp_dir in list(child_processes):
        try:
            if p.is_alive():
                p.terminate()
                p.join(3)
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
                    p.join(1)
        except Exception:
            pass
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            child_processes.remove((p, temp_dir))
        except ValueError:
            pass


def _handle_exit(signum, frame):
    cleanup_child_processes()
    os._exit(0)


atexit.register(cleanup_child_processes)
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


# ============ 工具函数 ============
def get_plot_url(img_buffer: io.BytesIO) -> str:
    """获取图片的 URL"""
    return matfileserver.add_image(img_buffer)


def _create_error_image(error_message: str) -> io.BytesIO:
    """创建错误信息图片"""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(
        0.5,
        0.5,
        f"❌ {error_message}",
        ha="center",
        va="center",
        fontsize=12,
        color="red",
    )
    ax.set_axis_off()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer


def apply_scientific_style():
    """优化后的出版级绘图风格"""
    okabe_ito = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=okabe_ito),
            "figure.dpi": 150,
        }
    )
    return okabe_ito


def _get_density_array(dos_obj):
    """从 Dos 对象中提取密度数组"""
    assert hasattr(dos_obj, "densities") and dos_obj.densities, "Dos 对象不包含密度数据"
    return list(dos_obj.densities.values())[0]


def _enhance_for_plot(atoms: Atoms, tolerance: float = 0.05) -> Atoms:
    """专门为可视化增强 Atoms：将边界原子复制到相对的边界、棱和顶点"""
    cell = atoms.get_cell()
    scaled_positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

    new_scaled = []
    new_symbols = []

    offsets = list(product([0, 1], repeat=3))

    for pos, symbol in zip(scaled_positions, symbols):
        near_zero = np.isclose(pos, 0, atol=tolerance)

        for off in offsets:
            if any(o == 1 and not nz for o, nz in zip(off, near_zero)):
                continue
            new_scaled.append(pos + off)
            new_symbols.append(symbol)

    enhanced = Atoms(
        symbols=new_symbols, scaled_positions=new_scaled, cell=cell, pbc=True
    )
    return enhanced


# ============ 结构可视化 ============
def visualize_structure(structure: Structure) -> str:
    """可视化晶体结构的3D交互式网页"""
    formula = structure.composition.reduced_formula
    atoms = AseAtomsAdaptor.get_atoms(structure)

    temp_dir = tempfile.mkdtemp(prefix=f"{formula}_custom_")
    html_path = os.path.join(temp_dir, f"{formula}_custom_3d.html")
    write(html_path, atoms, format="html")

    url = matfileserver.add_html_with_info(structure, html_path)
    return url


def get_structure_plot(
    structure: Structure, repeat: bool = True, rotation: str = "10x,10y,0z"
) -> dict:
    """输入指定的晶体结构并返回预览图"""
    try:
        atoms = structure.to_ase_atoms()
        atoms.wrap()

        enhanced_atoms = _enhance_for_plot(atoms=atoms)

        fig, ax = plt.subplots(figsize=(16, 16))

        plot_atoms(
            enhanced_atoms,
            ax,
            rotation=rotation,
            show_unit_cell=2,
        )

        analyzer = SpacegroupAnalyzer(structure)
        spacegroup = analyzer.get_space_group_symbol()

        a, b, c = structure.lattice.a, structure.lattice.b, structure.lattice.c
        alpha, beta, gamma = (
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
        )
        formula = structure.composition.formula

        info_text = (
            f"Formula: {formula}\n"
            f"Space group: {spacegroup}\n"
            f"Lattice parameters: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å\n"
            f"Angles: α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°\n"
            f"Atoms in unit cell: {len(structure)}\n"
            f"Total atoms shown: {len(atoms)}"
        )

        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_axis_off()
        ax.set_title(
            f"Crystal Structure Visualization of {structure.composition.reduced_formula}",
            fontsize=14,
            fontweight="bold",
        )

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        return {"Image": get_plot_url(img_buffer), "error": None}

    except Exception as e:
        return {
            "Image": get_plot_url(_create_error_image(f"构建失败: {str(e)}")),
            "error": e,
        }


# ============ VASP 绘图函数 ============
def plot_vasp_band(xml_path, kpoints_path):
    """使用 Pymatgen 绘制高质量能带图"""
    try:
        run = Vasprun(xml_path, parse_projected_eigen=False)
        bs = run.get_band_structure(kpoints_filename=kpoints_path, line_mode=True)

        is_metal = bs.is_metal()
        gap_info = bs.get_band_gap()

        results = {
            "is_metal": is_metal,
            "gap": gap_info["energy"],
            "fermi_energy": bs.efermi,
        }

        plotter = BSPlotter(bs)
        plt_module = plotter.get_plot()
        fig = plt.gcf()
        ax = plt.gca()

        xticks = ax.get_xticks()
        labels = [label.get_text() for label in ax.get_xticklabels()]
        fixed_labels = [l.replace("GAMMA", r"$\Gamma$") for l in labels]

        ax.set_xticks(xticks)
        ax.set_xticklabels(fixed_labels, fontsize=20)
        ax.set_ylabel(r"$E - E_f$ (eV)", fontsize=20)
        ax.set_title("Band Structure", fontsize=22, pad=20)
        ax.axhline(y=0, color="#d62728", linestyle="--", linewidth=2, zorder=1)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return {"Image": get_plot_url(buf), "data": results, "error": None}

    except Exception as e:
        return {"Image": None, "error": str(e)}


def plot_vasp_dos_analysis(vasprun_path="vasprun.xml", material_name="Material"):
    """主接口：解析 VASP 数据并生成 2x3 综合分析图"""
    try:
        print(f"正在解析 {vasprun_path}...")
        vr = VasprunOutput(vasprun_path, parse_dos=True)
        complete_dos = vr.complete_dos

        assert complete_dos is not None, "无法从 vasprun 提取 CompleteDos"
        assert hasattr(complete_dos, "energies"), "CompleteDos 对象缺失能量数据"

        energies = complete_dos.energies - complete_dos.efermi
        tdos_array = _get_density_array(complete_dos)
        element_dos = complete_dos.get_element_dos()

        # DOS数据分析
        dos_analysis = _analyze_dos_data(energies, tdos_array, element_dos)

        # 绘图逻辑 - 2行3列布局
        colors = apply_scientific_style()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Electronic Structure Analysis: {material_name}", fontweight="bold"
        )

        # (A) Total DOS
        ax = axes[0, 0]
        ax.plot(energies, tdos_array, color="black", lw=1.5, label="Total DOS")
        ax.fill_between(
            energies, 0, tdos_array, where=(energies < 0), color="gray", alpha=0.2
        )
        ax.axvline(x=0, color="#D55E00", linestyle="--", lw=1, label="$E_F$")

        if "band_gap" in dos_analysis:
            gap_text = f"Band gap: {dos_analysis['band_gap']:.3f} eV"
            ax.text(
                0.05,
                0.95,
                gap_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        ax.set_title("(A) Total Density of States")
        ax.set_ylabel("DOS (states/eV)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

        # (B) Element Projected DOS
        ax = axes[0, 1]
        if element_dos:
            for i, (el, dos_obj) in enumerate(element_dos.items()):
                dens = _get_density_array(dos_obj)
                ax.plot(energies, dens, label=str(el), lw=1.3)
            ax.axvline(x=0, color="#D55E00", linestyle="--", lw=1)
            ax.set_title("(B) Element Projected DOS")
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(
                0.5, 0.5, "No Element PDOS found", ha="center", transform=ax.transAxes
            )
        ax.grid(True, alpha=0.3)

        # (C) Near-Fermi Region (Zoomed)
        ax = axes[0, 2]
        mask = (energies > -4) & (energies < 4)
        ax.plot(energies[mask], tdos_array[mask], color="black", lw=1.2)
        ax.fill_between(
            energies[mask],
            0,
            tdos_array[mask],
            where=(energies[mask] < 0),
            color="#56B4E9",
            alpha=0.3,
        )
        ax.axvline(x=0, color="#D55E00", linestyle="--", lw=1)

        if "dos_at_ef_exact" in dos_analysis:
            fermi_dos_text = f"DOS(E$_F$) = {dos_analysis['dos_at_ef_exact']:.3f}"
            ax.text(
                0.05,
                0.95,
                fermi_dos_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

        ax.set_title("(C) Near-Fermi Region (±4 eV)")
        ax.set_xlabel("Energy - $E_F$ (eV)")
        ax.set_ylabel("DOS (states/eV)")
        ax.grid(True, alpha=0.3)

        # (D) Integrated DOS
        ax = axes[1, 0]
        if len(energies) > 1:
            de = energies[1] - energies[0]
            integrated = np.cumsum(tdos_array) * de
            ax.plot(energies, integrated, color="#009E73", lw=1.5)

            if "total_integrated_dos" in dos_analysis:
                total_electrons = dos_analysis["total_integrated_dos"]
                ax.text(
                    0.05,
                    0.95,
                    f"Total e$^-$: {total_electrons:.1f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data\nfor integration",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title("(D) Integrated DOS")
        ax.set_ylabel("Cumulative Electrons")
        ax.set_xlabel("Energy - $E_F$ (eV)")
        ax.grid(True, alpha=0.3)

        # (E) 元素贡献饼图
        ax = axes[1, 1]
        if element_dos and "element_contributions" in dos_analysis:
            element_contributions = dos_analysis["element_contributions"]

            elements = []
            fermi_contributions = []

            for el, contrib in element_contributions.items():
                elements.append(el)
                fermi_contributions.append(contrib["fermi_contribution"])

            valid_indices = [i for i, val in enumerate(fermi_contributions) if val > 0]
            if valid_indices and len(valid_indices) > 1:
                elements = [elements[i] for i in valid_indices]
                fermi_contributions = [fermi_contributions[i] for i in valid_indices]
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(elements)))

                wedges, texts, autotexts = ax.pie(
                    fermi_contributions,
                    labels=elements,
                    colors=colors_pie,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"fontsize": 9},
                )

                for autotext in autotexts:
                    autotext.set_color("black")
                    autotext.set_fontsize(8)
                    autotext.set_fontweight("bold")

                ax.set_title("(E) Element Contribution at Fermi Level")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient element\ncontributions data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No element contribution data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )

        # (F) DOS峰位分析图
        ax = axes[1, 2]
        ax.plot(
            energies, tdos_array, color="black", lw=1.2, alpha=0.7, label="Total DOS"
        )

        if "major_peaks" in dos_analysis and dos_analysis["major_peaks"]:
            peaks = dos_analysis["major_peaks"]
            peak_energies = [p["energy"] for p in peaks]
            peak_heights = [p["dos_height"] for p in peaks]

            peak_colors = plt.cm.viridis(np.linspace(0, 1, len(peaks)))
            for i, (energy, height, color) in enumerate(
                zip(peak_energies, peak_heights, peak_colors)
            ):
                ax.scatter(
                    energy,
                    height,
                    color=color,
                    s=80,
                    zorder=5,
                    edgecolors="black",
                    linewidth=1,
                )
                label_text = f"P{i + 1}: {energy:.2f} eV"
                ax.annotate(
                    label_text,
                    xy=(energy, height),
                    xytext=(energy, height * 1.1),
                    ha="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                )

            peak_table_data = []
            for i, peak in enumerate(peaks[:3]):
                peak_table_data.append(
                    [
                        f"P{i + 1}",
                        f"{peak['energy']:.2f} eV",
                        f"{peak['dos_height']:.2f}",
                    ]
                )

            if peak_table_data:
                table = ax.table(
                    cellText=peak_table_data,
                    colLabels=["Peak", "Energy", "DOS"],
                    cellLoc="center",
                    loc="upper right",
                    bbox=[0.65, 0.6, 0.3, 0.3],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
        else:
            ax.plot(energies, tdos_array, color="black", lw=1.5)
            ax.text(
                0.5,
                0.5,
                "No peak analysis available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )

        ax.axvline(x=0, color="#D55E00", linestyle="--", lw=1, label="$E_F$")
        ax.set_title("(F) DOS Peak Analysis")
        ax.set_xlabel("Energy - $E_F$ (eV)")
        ax.set_ylabel("DOS (states/eV)")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        info_dict = {
            "material": material_name,
            "efermi": float(vr.complete_dos.efermi),
            "formula": vr.final_structure.composition.reduced_formula,
            "dos_analysis": dos_analysis,
        }

        if "band_gap" in dos_analysis:
            info_dict["band_gap_summary"] = {
                "value": dos_analysis["band_gap"],
                "type": dos_analysis.get("gap_type", "unknown"),
                "vbm": dos_analysis.get("valence_band_max", None),
                "cbm": dos_analysis.get("conduction_band_min", None),
            }

        if "major_peaks" in dos_analysis:
            info_dict["peak_summary"] = {
                "num_peaks": len(dos_analysis["major_peaks"]),
                "main_peaks": dos_analysis["major_peaks"][:3]
                if len(dos_analysis["major_peaks"]) >= 3
                else dos_analysis["major_peaks"],
            }

        if "element_contributions" in dos_analysis:
            info_dict["element_contribution_summary"] = dos_analysis[
                "element_contributions"
            ]

        return {"info": info_dict, "Image": get_plot_url(buf)}

    except AssertionError as ae:
        print(f"数据检查未通过: {ae}")
        return {"error": str(ae)}
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def _analyze_dos_data(energies, tdos_array, element_dos):
    """分析DOS数据，返回带隙、费米能级处DOS等关键信息"""
    analysis_results = {}

    if len(energies) > 1:
        de = energies[1] - energies[0]
        analysis_results["energy_step"] = de

    valence_mask = energies < 0
    conduction_mask = energies > 0

    if np.any(valence_mask) and np.any(conduction_mask):
        valence_energies = energies[valence_mask]
        valence_dos = tdos_array[valence_mask]
        valence_nonzero = valence_dos > 1e-6
        if np.any(valence_nonzero):
            vbm_index = np.argmax(valence_energies[valence_nonzero])
            vbm_energy = valence_energies[valence_nonzero][vbm_index]
            vbm_dos = valence_dos[valence_nonzero][vbm_index]
            analysis_results["valence_band_max"] = float(vbm_energy)
            analysis_results["vbm_dos"] = float(vbm_dos)

        conduction_energies = energies[conduction_mask]
        conduction_dos = tdos_array[conduction_mask]
        conduction_nonzero = conduction_dos > 1e-6
        if np.any(conduction_nonzero):
            cbm_index = np.argmin(conduction_energies[conduction_nonzero])
            cbm_energy = conduction_energies[conduction_nonzero][cbm_index]
            cbm_dos = conduction_dos[conduction_nonzero][cbm_index]
            analysis_results["conduction_band_min"] = float(cbm_energy)
            analysis_results["cbm_dos"] = float(cbm_dos)

            if "valence_band_max" in analysis_results:
                band_gap = float(cbm_energy - vbm_energy)
                analysis_results["band_gap"] = band_gap
                analysis_results["gap_type"] = (
                    "direct"
                    if abs(band_gap - (cbm_energy - vbm_energy)) < 0.01
                    else "indirect"
                )

    fermi_window = 0.05
    fermi_mask = (energies > -fermi_window) & (energies < fermi_window)
    if np.any(fermi_mask):
        fermi_dos_values = tdos_array[fermi_mask]
        analysis_results["dos_at_fermi"] = float(np.mean(fermi_dos_values))
        analysis_results["fermi_window_avg"] = float(np.mean(fermi_dos_values))
        if len(energies) > 1:
            dos_at_ef = float(np.interp(0, energies, tdos_array))
            analysis_results["dos_at_ef_exact"] = dos_at_ef

    if len(energies) > 1 and "energy_step" in analysis_results:
        de = analysis_results["energy_step"]
        total_electrons = float(np.sum(tdos_array) * de)
        analysis_results["total_integrated_dos"] = total_electrons

    if np.any(valence_mask) and "energy_step" in analysis_results:
        de = analysis_results["energy_step"]
        valence_integral = float(np.sum(tdos_array[valence_mask]) * de)
        analysis_results["valence_integrated_dos"] = valence_integral

    if np.any(conduction_mask) and "energy_step" in analysis_results:
        de = analysis_results["energy_step"]
        conduction_integral = float(np.sum(tdos_array[conduction_mask]) * de)
        analysis_results["conduction_integrated_dos"] = conduction_integral

    element_contributions = {}
    if element_dos:
        for el, dos_obj in element_dos.items():
            el_dens = _get_density_array(dos_obj)
            if np.any(fermi_mask):
                el_fermi_contrib = float(np.mean(el_dens[fermi_mask]))
                if "energy_step" in analysis_results:
                    de = analysis_results["energy_step"]
                    element_contributions[str(el)] = {
                        "fermi_contribution": el_fermi_contrib,
                        "total_contribution": float(np.sum(el_dens) * de),
                    }
                else:
                    element_contributions[str(el)] = {
                        "fermi_contribution": el_fermi_contrib,
                        "total_contribution": float(np.sum(el_dens)),
                    }
        analysis_results["element_contributions"] = element_contributions

    try:
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(tdos_array, height=0.1, distance=10)
        if len(peaks) > 0:
            peak_info = []
            for i, peak_idx in enumerate(peaks[:5]):
                peak_info.append(
                    {
                        "energy": float(energies[peak_idx]),
                        "dos_height": float(tdos_array[peak_idx]),
                        "relative_to_fermi": float(energies[peak_idx]),
                    }
                )
            analysis_results["major_peaks"] = peak_info
    except ImportError:
        print("scipy未安装，跳过峰位分析")
    except Exception as e:
        print(f"峰位分析失败: {e}")

    return analysis_results


# ============ VASP 结果提取函数 ============
def extract_relax_info(
    task_directory: str, get_plot: bool = True, visualize: bool = False
) -> dict:
    """提取结构优化任务的结果信息"""
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.extract_relax_info(task_directory)
                if result:
                    break

            # 从 CONTCAR 文件重新读取 Structure 对象用于可视化
            structure = None
            if (
                result
                and "local_files" in result
                and "contcar" in result["local_files"]
            ):
                try:
                    contcar_path = result["local_files"]["contcar"]
                    structure = Structure.from_file(contcar_path)
                except Exception as e:
                    print(f"读取 CONTCAR 失败: {e}")

            if visualize and structure is not None:
                structure_url = visualize_structure(structure)
                result["3d_image_url"] = structure_url
            if get_plot and structure is not None:
                res = get_structure_plot(structure)
                image = res["Image"]
                result["error"] = res["error"]
                result["image_url"] = image
            result.pop("structure", None)
            return result
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e), "message": "提取任务结果失败"}


def extract_scf_info(task_directory: str) -> dict:
    """提取自洽计算任务的结果信息"""
    try:
        with connection as vasp_task:
            result = vasp_task.extract_scf_info(task_directory)
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}


def extract_band_info(task_directory: str, plot_band: bool = True) -> dict:
    """提取能带计算任务的结果信息"""
    try:
        with connection as vasp_task:
            result = vasp_task.extract_band_info(task_directory)
            if plot_band:
                res = plot_vasp_band(
                    xml_path=result["local_files"]["vasprun.xml"],
                    kpoints_path=result["local_files"]["KPOINTS"],
                )
                if not res["error"]:
                    image = res["Image"]
                    res.pop("Image")
                    result.update(
                        {"image_url": image, "plot_info": res, "message": "绘图成功"}
                    )
                else:
                    res.pop("Image")
                    result.update({"plot_info": res, "message": "绘图失败"})
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}


def extract_dos_info(task_directory: str, plot_dos: bool = True) -> dict:
    """提取态密度计算任务的结果信息"""
    try:
        with connection as vasp_task:
            result = vasp_task.extract_dos_info(task_directory)
            if plot_dos and result and isinstance(result, dict):
                local_files = result.get("local_files", {}) or {}
                vasprun_path = local_files.get("vasprun.xml") or local_files.get(
                    "vasprun"
                )

                if vasprun_path and os.path.exists(vasprun_path):
                    res = plot_vasp_dos_analysis(vasprun_path)

                    image = res.get("Image") if isinstance(res, dict) else None
                    payload = (
                        {k: v for k, v in res.items() if k != "Image"}
                        if isinstance(res, dict)
                        else {}
                    )
                    if not res.get("error"):
                        result.update(
                            {
                                "image_url": image,
                                "plot_info": payload,
                                "message": "绘图成功",
                            }
                        )
                    else:
                        result.update(
                            {
                                "image_url": image,
                                "plot_info": payload,
                                "message": "绘图失败",
                            }
                        )
                else:
                    result.setdefault("warnings", []).append(
                        "vasprun.xml文件缺失，无法绘图。"
                    )
            return result
    except Exception as e:
        return {"error": str(e), "message": "提取任务结果失败"}


# ============ MCP 工具 ============


"""
!!!
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
工具的返回值必须是{"args":dict, "returns":dict}
!!!
"""


# ----- 基础工具 -----
@mcp.tool()
async def get_time() -> dict:
    """获取当前时间，返回格式：YYYY-MM-DD HH:MM:SS"""
    args = {}
    result = {"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
    return {"args": args, "returns": result}


@mcp.tool()
async def get_material_project_page(material_id: str) -> dict:
    """
    获取指定材料的Material Project页面链接

    Args:
        material_id: 材料ID，如 "mp-1234"

    Returns:
        dict: 包含 material_id, url, message, error(可选)
    """
    args = {"material_id": material_id}
    if not material_id:
        return {
            "args": args,
            "returns": {"error": "材料ID不能为空", "message": "请提供有效的材料ID"},
        }

    url = f"https://next-gen.materialsproject.org/materials/{material_id}/"
    return {"args": args, "returns": {"url": url}}


@mcp.tool()
async def read_file(file_path: str) -> dict:
    """
    读取mcp服务器的文件

    Args:
        file_path: 文件的绝对路径或相对路径

    Returns:
        dict: 包含 success, content, file_path, error(可选)
    """
    args = {"file_path": file_path}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        return {
            "args": args,
            "returns": {
                "success": True,
                "content": content,
            },
        }
    except FileNotFoundError:
        return {
            "args": args,
            "returns": {"success": False, "error": f"文件未找到: {file_path}"},
        }
    except PermissionError:
        return {
            "args": args,
            "returns": {"success": False, "error": f"权限不足: {file_path}"},
        }
    except Exception as e:
        return {
            "args": args,
            "returns": {"success": False, "error": f"读取文件时出错: {str(e)}"},
        }


# ----- OQMD 数据库工具 -----
@mcp.tool()
async def search_materials_from_oqmd(
    elements: Optional[List[str]] = None,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    stability_max: float = 0.1,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    在OQMD数据库搜索材料

    Args:
        elements: 元素列表，如 ["Li", "Fe", "O"] 或 ["Si"]
        band_gap_min: 最小带隙值（eV），如 0.0
        band_gap_max: 最大带隙值（eV），如 3.0
        stability_max: 最大稳定性值（eV/atom），默认 0.1
        limit: 返回结果数量限制，默认 20

    Returns:
        Dict: OQMD搜索结果
    """
    args = {}
    if elements is not None:
        args["elements"] = elements
    if band_gap_min is not None:
        args["band_gap_min"] = band_gap_min
    if band_gap_max is not None:
        args["band_gap_max"] = band_gap_max
    if stability_max != 0.1:
        args["stability_max"] = stability_max
    if limit != 20:
        args["limit"] = limit

    filter_parts = []

    if elements:
        if len(elements) == 1:
            filter_parts.append(f"element={elements[0]}")
        else:
            element_set = ",".join(elements)
            filter_parts.append(f"element_set={element_set}")

    if band_gap_min is not None:
        filter_parts.append(f"band_gap>={band_gap_min}")

    if band_gap_max is not None:
        filter_parts.append(f"band_gap<={band_gap_max}")

    if stability_max is not None:
        filter_parts.append(f"stability<={stability_max}")

    filter_expr = " AND ".join(filter_parts) if filter_parts else None

    fields = [
        "name",
        "entry_id",
        "band_gap",
        "delta_e",
        "stability",
        "spacegroup",
        "ntypes",
    ]

    result = oqmd.search_oqmd(
        fields=fields,
        filter_expr=filter_expr,
        limit=limit,
        offset=0,
        sort_by="stability",
        desc=False,
    )
    return {"args": args, "returns": result}


@mcp.tool()
async def get_material_structure_from_oqmd(
    entry_id: int,
    mode: str = "conventional",
    get_sites: bool = False,
    get_plot: bool = False,
    download: bool = False,
) -> dict | list:
    """
    在OQMD数据库获取指定材料的结构

    Args:
        entry_id: OQMD材料条目ID（整数）
        mode: 晶胞类型，"conventional"（常规晶胞，默认）或 "primitive"（原胞）
        get_sites: 是否包含原子位点详细信息，默认 False
        get_plot: 是否生成结构可视化图，默认 False
        download: 是否下载CIF文件，默认 False

    Returns:
        dict | list: 包含 structure_dict, image_url(可选), message, error(可选)
    """
    args = {
        "entry_id": entry_id,
        "mode": mode,
        "get_sites": get_sites,
        "get_plot": get_plot,
        "download": download,
    }
    res = oqmd.parse_poscar_with_pymatgen(entry_id, mode)
    message = []
    if res["success"]:
        structure = res["structure"]
        lattice = structure.lattice
        space_group_info = structure.get_space_group_info()
        formula = structure.formula
        reduced_formula = structure.composition.reduced_formula
        structure_info = {
            "formula": formula,
            "reduced_formula": reduced_formula,
            "space_group_symbol": space_group_info[0] if space_group_info else "未知",
            "space_group_number": space_group_info[1] if space_group_info else "未知",
            "lattice_parameters": {
                "a": round(lattice.a, 4),
                "b": round(lattice.b, 4),
                "c": round(lattice.c, 4),
                "alpha": round(lattice.alpha, 2),
                "beta": round(lattice.beta, 2),
                "gamma": round(lattice.gamma, 2),
                "volume": round(lattice.volume, 4),
            },
            "number_of_sites": len(structure),
            "density": round(structure.density, 4),
            "is_ordered": structure.is_ordered,
        }
        if get_sites:
            structure_info["sites"] = [
                {
                    "element": site.species_string,
                    "fractional_coordinates": [
                        round(coord, 4) for coord in site.frac_coords
                    ],
                }
                for site in structure.sites
            ]
            message.append(f"材料 {entry_id} 的原子位点信息已包含在返回结果中")
        if download:
            CifWriter(structure).write_file(
                f"cifs/{reduced_formula}-oqmd-{entry_id}.cif"
            )
            print(f"获取材料 {entry_id} 的晶体结构成功，已保存为cif文件")
            message.append(
                f"材料 {entry_id} 的晶体结构已保存为cif文件，路径为'cifs/{reduced_formula}-oqmd-{entry_id}.cif'"
            )
        if get_plot:
            structure_url = visualize_structure(structure)
            message.append(
                "生成了2d结构预览图和3d可视化交互式网页，请点击查看晶体结构图"
            )
            message.append(f"3d_image_url: {structure_url}")
            res = get_structure_plot(structure)
            if not res["error"]:
                image = res["Image"]
                return {
                    "args": args,
                    "returns": {
                        "image_url": image,
                        "structure_dict": structure_info,
                        "message": message,
                    },
                }
            else:
                message.append(res["error"])
                return {
                    "args": args,
                    "returns": {"structure_dict": structure_info, "message": message},
                }

        return {
            "args": args,
            "returns": {"structure_dict": structure_info, "message": message},
        }
    else:
        return {
            "args": args,
            "returns": {"error": res["error"], "message": "构建晶体结构失败"},
        }


# ----- Material Project 工具 -----
@mcp.tool()
async def search_materials_from_mp(
    elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    chemsys: str | list[str] | None = None,
    band_gap: tuple[float, float] | None = None,
    num_elements: tuple[int, int] | None = None,
    formula: str | list[str] | None = None,
    chunk_size: int | None = 25,
) -> dict:
    """
    Material Project数据查询工具

    Args:
        elements: 包含的元素列表，如 ["Li", "Fe", "O"]
        exclude_elements: 排除的元素列表，如 ["C", "N"]
        chemsys: 化学系统，如 "Li-Fe-O" 或 ["Li-Fe-O", "Na-Cl"]
        band_gap: 带隙范围元组，如 (0.0, 3.0)
        num_elements: 元素数量范围，如 (2, 4)
        formula: 化学式，如 "LiFeO2" 或 ["LiFeO2", "NaCl"]
        chunk_size: 每块返回数量，默认25

    Returns:
        list[dict]: 材料列表或错误信息
    """
    args = {}
    if elements is not None:
        args["elements"] = elements
    if exclude_elements is not None:
        args["exclude_elements"] = exclude_elements
    if chemsys is not None:
        args["chemsys"] = chemsys
    if band_gap is not None:
        args["band_gap"] = band_gap
    if num_elements is not None:
        args["num_elements"] = num_elements
    if formula is not None:
        args["formula"] = formula
    if chunk_size is not None and chunk_size != 25:
        args["chunk_size"] = chunk_size

    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("API密钥未设置")
    try:
        with MPRester(API_KEY) as mpr:
            search_kwargs = {}

            if elements:
                search_kwargs["elements"] = elements
            if exclude_elements:
                search_kwargs["exclude_elements"] = exclude_elements
            if chemsys:
                search_kwargs["chemsys"] = chemsys
            if band_gap:
                search_kwargs["band_gap"] = band_gap
            if num_elements:
                search_kwargs["num_elements"] = num_elements
            if formula:
                search_kwargs["formula"] = formula

            search_kwargs["fields"] = [
                "material_id",
                "formula_pretty",
                "band_gap",
                "symmetry",
            ]
            chunk_sz = chunk_size if chunk_size else 25

            results = mpr.materials.summary.search(
                **search_kwargs, chunk_size=chunk_sz, num_chunks=1
            )
            print(f"查询到 {len(results)} 个材料")
            data = {
                "data": [
                    {
                        "formula": r.formula_pretty,
                        "material_id": r.material_id,
                        "symmetry": r.symmetry,
                        "band_gap": r.band_gap,
                    }
                    for r in results
                ]
            }
        return {"args": args, "returns": data}
    except Exception as e:
        import traceback

        print(f"[ERROR] search_materials_from_mp: {e}")
        print(traceback.format_exc())
        return {
            "args": args,
            "returns": {"error": str(e), "message": "查询材料数据失败"},
        }


@mcp.tool()
async def get_band_gap(material_id: str) -> dict:
    """
    获取指定材料的带隙值(Material Project)

    Args:
        material_id: 材料ID，如 "mp-1234"

    Returns:
        dict: 包含 material_id, band_gap, formula, error(可选)
    """
    args = {"material_id": material_id}
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")
    try:
        with MPRester(API_KEY) as mpr:
            results = mpr.summary.search(
                material_ids=material_id, fields=["band_gap", "formula_pretty"]
            )
            if not results:
                raise ValueError(f"未找到材料ID为 {material_id} 的材料")
            else:
                print(f"获取材料 {material_id} 的带隙值成功")
            band_gap = results[0].band_gap
            formula = results[0].formula_pretty
        return {"args": args, "returns": {"band_gap": band_gap}}
    except Exception as e:
        return {
            "args": args,
            "returns": {
                "error": str(e),
                "message": f"获取材料 {material_id} 的带隙值失败",
            },
        }


@mcp.tool()
async def get_material_structure_from_mp(
    material_id: str,
    get_sites: bool = False,
    get_plot: bool = False,
    download: bool = False,
) -> dict | list:
    """
    在Material Project上获取指定材料的晶体结构数据

    Args:
        material_id: 材料ID，如 "mp-1234"
        get_sites: 是否包含原子位点详细信息，默认 False
        get_plot: 是否生成结构可视化图，默认 False
        download: 是否下载CIF文件，默认 False

    Returns:
        dict | list: 包含 structure_dict, image_url(可选), message, error(可选)
    """
    args = {
        "material_id": material_id,
        "get_sites": get_sites,
        "get_plot": get_plot,
        "download": download,
    }
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")
    os.makedirs("cifs", exist_ok=True)
    os.makedirs("cifs/images", exist_ok=True)
    message = []
    try:
        with MPRester(API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(
                material_id, conventional_unit_cell=True
            )
            lattice = structure.lattice
            space_group_info = structure.get_space_group_info()
            formula = structure.formula
            reduced_formula = structure.composition.reduced_formula
            structure_info = {
                "formula": formula,
                "reduced_formula": reduced_formula,
                "space_group_symbol": space_group_info[0]
                if space_group_info
                else "未知",
                "space_group_number": space_group_info[1]
                if space_group_info
                else "未知",
                "lattice_parameters": {
                    "a": round(lattice.a, 4),
                    "b": round(lattice.b, 4),
                    "c": round(lattice.c, 4),
                    "alpha": round(lattice.alpha, 2),
                    "beta": round(lattice.beta, 2),
                    "gamma": round(lattice.gamma, 2),
                    "volume": round(lattice.volume, 4),
                },
                "number_of_sites": len(structure),
                "density": round(structure.density, 4),
                "is_ordered": structure.is_ordered,
            }
            message.append(
                f"材料 {material_id} 的晶体结构信息: formula={formula}, space_group={space_group_info[0] if space_group_info else '未知'}"
            )
            if get_sites:
                sites_data = [
                    {
                        "element": site.species_string,
                        "fractional_coordinates": [
                            round(coord, 4) for coord in site.frac_coords
                        ],
                    }
                    for site in structure.sites
                ]
                structure_info["sites"] = sites_data
                structure_info["sites_count"] = len(structure.sites)
                message.append(
                    f"材料 {material_id} 的原子位点信息已包含在返回结果中，共{len(structure.sites)}个)"
                )
            if download:
                CifWriter(structure).write_file(
                    f"cifs/{reduced_formula}-{material_id}.cif"
                )
                print(f"获取材料 {material_id} 的晶体结构成功，已保存为cif文件")
                message.append(
                    f"材料 {material_id} 的晶体结构已保存为cif文件，路径为'cifs/{reduced_formula}-{material_id}.cif'"
                )
            if get_plot:
                structure_url = visualize_structure(structure)
                message.append(
                    "生成了2d结构预览图和3d可视化交互式网页，请点击查看晶体结构图"
                )
                message.append(f"3d_image_url: {structure_url}")
                res = get_structure_plot(structure)
                if not res["error"]:
                    image = res["Image"]
                    return {
                        "args": args,
                        "returns": {
                            "image_url": image,
                            "3d_image_url": structure_url,
                            "structure_dict": structure_info,
                            "message": message,
                        },
                    }
                else:
                    message.append(res["error"])
                    return {
                        "args": args,
                        "returns": {
                            "3d_image_url": structure_url,
                            "structure_dict": structure_info,
                            "message": message,
                        },
                    }

        return {
            "args": args,
            "returns": {"structure_dict": structure_info, "message": message},
        }
    except Exception as e:
        return {
            "args": args,
            "returns": {
                "error": str(e),
                "message": f"获取材料 {material_id} 的晶体结构失败",
            },
        }


@mcp.tool()
async def get_material_all_infomation_by_id(material_id: str) -> dict:
    """获取Material Project指定材料的所有信息"""
    args = {"material_id": material_id}
    API_KEY = MY_API_KEY
    if not API_KEY:
        raise ValueError("MP_API_KEY环境变量未设置")

    try:
        with MPRester(API_KEY) as mpr:
            with mpr.materials as materials:
                material = materials.search(material_ids=material_id)
                if not material:
                    raise ValueError(f"未找到材料ID为 {material_id} 的材料")
                else:
                    print(f"获取材料 {material_id} 的所有信息成功")
            material_dict = material[0]
        return {"args": args, "returns": material_dict}
    except Exception as e:
        return {
            "args": args,
            "returns": {
                "error": str(e),
                "message": f"获取材料 {material_id} 的所有信息失败",
            },
        }


# ----- 结构建模工具 -----
@mcp.tool()
async def build_structure(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    elements: list[str],
    frac_coord: list[list[float]],
    scaling_matrix: int | list = 1,
    save_to_cif: bool = False,
    add_to_database: str = None,
) -> dict | list:
    """
    构建晶体结构并保存为CIF文件，生成晶体结构图

    Args:
        a: 晶格参数a（Å）
        b: 晶格参数b（Å）
        c: 晶格参数c（Å）
        alpha: 晶格角alpha（度）
        beta: 晶格角beta（度）
        gamma: 晶格角gamma（度）
        elements: 元素符号列表，如 ["Na", "Cl"]
        frac_coord: 分数坐标列表，如 [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        scaling_matrix: 超胞，默认整数（int）：表示在 a, b, c 三个方向进行相同的扩胞。例如 scaling_matrix=2表示构建 2×2×2 的超胞。
                    列表（list）：长度为 3 的列表，分别表示 a, b, c 方向的扩胞倍数。例如 scaling_matrix=[2, 1, 1]表示构建 2×1×1 的超胞。
        save_to_cif: 是否保存为CIF文件，默认False
        add_to_database: 数据库文件名，如添加则保存到该数据库

    Returns:
        dict | list: 包含 image, 3d_image_url, message, error(可选)
    """
    args = {
        "a": a,
        "b": b,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "elements": elements,
        "frac_coord": frac_coord,
        "scaling_matrix": scaling_matrix,
        "save_to_cif": save_to_cif,
    }
    if add_to_database is not None:
        args["add_to_database"] = add_to_database

    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        structure = Structure(lattice, elements, frac_coord)
        structure = structure.make_supercell(scaling_matrix=scaling_matrix)
        formula = structure.composition.reduced_formula
        os.makedirs("custom_structures", exist_ok=True)
        os.makedirs("custom_structures/images", exist_ok=True)
        message = []
        current_date = datetime.now().strftime("%Y%m%d%H%M")
        if save_to_cif:
            CifWriter(structure).write_file(
                f"custom_structures/{formula}_custom_{current_date}.cif"
            )
            message.append(
                f"自定义晶体结构已保存为 ./custom_structures/{formula}_custom_{current_date}.cif"
            )
        structure_url = visualize_structure(structure)
        message.append("3d晶体结构可视化交互式网页，请点击查看晶体结构图")

        if add_to_database:
            db = databasemanage.DatabaseManager("db/" + add_to_database)
            db.add_material(
                formula=formula, structure=structure, band_gap=None, material_id=None
            )
            db.close()
            message.append(f"自定义晶体结构已添加到数据库 {add_to_database}")

        res = get_structure_plot(structure)
        if not res["error"]:
            image = res["Image"]
            return {
                "args": args,
                "returns": {
                    "image": image,
                    "3d_image_url": structure_url,
                    "message": message,
                },
            }
        else:
            message.append(res["error"])
            return {
                "args": args,
                "returns": {"3d_image_url": structure_url, "message": message},
            }

    except Exception as e:
        return {
            "args": args,
            "returns": {"error": str(e), "message": "构建晶体结构失败"},
        }


# ----- VASP 任务管理工具 -----
@mcp.tool()
async def create_task(formula: str, cif_path: str) -> dict:
    """
    在远程服务器上创建任务文件夹并上传CIF文件

    Args:
        formula: 材料化学式，如 "LiFeO2"
        cif_path: 本地CIF文件路径

    Returns:
        dict: 包含 message, task_directory, error(可选)
    """
    args = {"formula": formula, "cif_path": cif_path}
    try:
        with connection as vasp_task:
            base_dir = config.get_base_dir()
            if not base_dir:
                raise ValueError("base_dir环境变量未设置")
            result = None
            for _ in range(3):
                result = vasp_task.create_task(formula, cif_path, base_dir)
                if result:
                    break
            if result:
                return {
                    "args": args,
                    "returns": {
                        "message": f"任务目录已创建并上传CIF文件",
                        "task_directory": result,
                    },
                }
            else:
                return {
                    "args": args,
                    "returns": {"error": "任务创建失败", "message": "请再试一次"},
                }
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "任务创建失败"}}


@mcp.tool()
async def list_task_directories() -> dict:
    """
    列出远程服务器上的所有任务目录
    """
    args = {}
    try:
        with connection as vasp_task:
            base_dir = config.get_base_dir()
            if not base_dir:
                raise ValueError("base_dir环境变量未设置")
            result = None
            for _ in range(3):
                result = vasp_task.get_task_directories(base_dir)
                if result:
                    break
            if result:
                return {"args": args, "returns": {"task_directories": result}}
            else:
                return {
                    "args": args,
                    "returns": {
                        "error": "获取任务目录失败",
                        "message": "请检查服务器连接是否正常",
                    },
                }
    except Exception as e:
        return {
            "args": args,
            "returns": {"error": str(e), "message": "获取任务目录失败"},
        }


@mcp.tool()
async def check_squeue() -> dict:
    """
    检查远程服务器上的任务队列
    """
    args = {}
    try:
        with connection as vasp_task:
            result = None
            for _ in range(3):
                result = vasp_task.check_squeue()
                if result:
                    break
            if result:
                return {"args": args, "returns": {"squeue": result}}
            else:
                return {
                    "args": args,
                    "returns": {
                        "error": "检查任务队列失败",
                        "message": "请检查服务器连接是否正常",
                    },
                }
    except Exception as e:
        return {
            "args": args,
            "returns": {"error": str(e), "message": "检查任务队列失败"},
        }


@mcp.tool()
async def execute_command(command: str) -> dict:
    """
    在计算服务器上执行linux命令

    Args:
        command: 要执行的Linux命令，如 "ls -la"

    Returns:
        dict: 命令执行结果或错误信息
    """
    args = {"command": command}
    try:
        with connection as vasp_task:
            result = vasp_task.execute_command(command)
            return {"args": args, "returns": result}
    except Exception as e:
        return {
            "args": args,
            "returns": {"error": str(e), "message": "命令提交或执行失败"},
        }


@mcp.tool()
async def extract_file(file_path: str) -> dict:
    """
    从计算服务器上提取一个文件，并提供下载的URL

    Args:
        file_path: 远程服务器上的文件路径

    Returns:
        dict: 包含以下字段:
            - local_file: 本地保存的文件路径
            - download_url: 文件下载URL
            - error: 错误信息（如果有）
            - message: 操作结果消息
    """
    args = {"file_path": file_path}
    try:
        with connection as vasp_task:
            result = vasp_task.extract_file(file_path=file_path)
            download_url = matfileserver.add_image_file(result["local_file"])
            result["download_url"] = download_url
            return {"args": args, "returns": result}
    except Exception as e:
        return {"args": args, "returns": {"error": str(e), "message": "提取文件失败"}}


@mcp.tool()
async def create_mission(task_directory: str, mission: str) -> dict:
    """
    创建计算任务的输入文件（POSCAR、INCAR、POTCAR、KPOINTS），但不提交计算

    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"（结构优化）、"scf"（自洽计算）、"band"（能带计算）、"dos"（态密度计算）

    Returns:
        dict: 包含 success, mission, task_directory, raw_result, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission}
    mission = mission.lower().strip()
    method_map = {
        "relax": "create_relax_mission",
        "scf": "create_scf_mission",
        "band": "create_band_mission",
        "dos": "create_dos_mission",
    }

    if mission not in method_map:
        return {
            "args": args,
            "returns": {
                "success": False,
                "error": f"未知的计算类型: {mission}，可选: {list(method_map.keys())}",
            },
        }

    try:
        with connection as vasp_task:
            method_name = method_map[mission]
            method = getattr(vasp_task, method_name)
            result = method(task_directory)

            success = result.get("status") == "ok" or "error" not in result
            response = {
                "success": success,
                "mission": mission,
                "task_directory": task_directory,
                "raw_result": result,
            }

            if not success:
                response["error"] = (
                    result.get("error") or result.get("message") or "创建任务失败"
                )

            return {"args": args, "returns": response}

    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


@mcp.tool()
async def submit_mission(task_directory: str, mission: str) -> dict:
    """
    提交已准备好的计算任务

    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"（结构优化）、"scf"（自洽计算）、"band"（能带计算）、"dos"（态密度计算）

    Returns:
        dict: 包含 success, mission, task_directory, job_id(可选), message, raw_result, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission}
    mission = mission.lower().strip()
    method_map = {
        "relax": "submit_relax_calculation",
        "scf": "submit_scf_calculation",
        "band": "submit_band_calculation",
        "dos": "submit_dos_calculation",
    }

    if mission not in method_map:
        return {
            "args": args,
            "returns": {
                "success": False,
                "error": f"未知的计算类型: {mission}，可选: {list(method_map.keys())}",
            },
        }

    try:
        with connection as vasp_task:
            method_name = method_map[mission]
            method = getattr(vasp_task, method_name)
            result = method(task_directory)

            success = result.get("status") == "ok" or "error" not in result
            response = {
                "success": success,
                "mission": mission,
                "task_directory": task_directory,
                "job_id": result.get("job_id"),
                "raw_result": result,
            }

            if success:
                response["message"] = f"{mission}计算任务提交成功"
                if result.get("job_id"):
                    response["message"] += f"，作业ID: {result['job_id']}"
                    response["message"] += (
                        f"使用工具 extract_result {task_directory} {mission} 来提取计算结果"
                    )
            else:
                response["error"] = (
                    result.get("error") or result.get("message") or "提交任务失败"
                )

            return {"args": args, "returns": response}

    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


@mcp.tool()
async def modify_incar(
    task_directory: str, mission: str, read: bool, write: str = None
) -> dict:
    """
    读写修改计算任务的INCAR文件

    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"、"scf"、"band"、"dos"
        read: 是否为读取模式，True为读取，False为写入
        write: 要写入的INCAR参数（JSON格式字符串），仅在read=False时使用，如: '{"ENCUT": 520}'

    Returns:
        dict: 包含 success, mission, task_directory, read_mode, incar_params(可选), updated_params(可选), message, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission, "read": read}
    if write is not None:
        args["write"] = write

    mission = mission.lower().strip()

    new_params = None
    if not read and write:
        try:
            new_params = json.loads(write)
            if not isinstance(new_params, dict):
                return {
                    "args": args,
                    "returns": {
                        "success": False,
                        "error": "write参数必须是JSON对象（字典）",
                    },
                }
        except Exception as e:
            return {
                "args": args,
                "returns": {"success": False, "error": f"解析write参数失败: {str(e)}"},
            }

    try:
        with connection as vasp_task:
            result = vasp_task.modify_incar_file(
                task_directory=task_directory,
                mission=mission,
                read_mode=read,
                new_params=new_params,
            )

            success = result.get("status") == "ok"
            response = {
                "success": success,
                "mission": mission,
                "task_directory": task_directory,
                "read_mode": read,
                "raw_result": result,
            }

            if success:
                if read:
                    response["incar_params"] = result.get("incar_params", {})
                    response["message"] = f"成功读取{mission}任务的INCAR参数"
                else:
                    response["message"] = result.get("message", "INCAR文件更新成功")
                    response["updated_params"] = result.get("updated_params", [])
            else:
                response["error"] = (
                    result.get("error") or result.get("message") or "操作失败"
                )

            return {"args": args, "returns": response}

    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


@mcp.tool()
def extract_result(task_directory: str, mission: str, plot: bool = True) -> dict:
    """
    提取计算任务的结果

    Args:
        task_directory: 任务目录路径
        mission: 计算类型，可选: "relax"（结构优化）、"scf"（自洽计算）、"band"（能带计算）、"dos"（态密度计算）
        plot: 是否生成图表，默认True

    Returns:
        dict: 包含 success, mission, task_directory, result, error(可选)
    """
    args = {"task_directory": task_directory, "mission": mission, "plot": plot}
    mission = mission.lower().strip()
    method_map = {
        "relax": lambda: extract_relax_info(
            task_directory, get_plot=plot, visualize=plot
        ),
        "scf": lambda: extract_scf_info(task_directory),
        "band": lambda: extract_band_info(task_directory, plot_band=plot),
        "dos": lambda: extract_dos_info(task_directory, plot_dos=plot),
    }

    if mission not in method_map:
        return {
            "args": args,
            "returns": {
                "success": False,
                "error": f"未知的计算类型: {mission}，可选: ['relax', 'scf', 'band', 'dos']",
                "task_directory": task_directory,
                "mission": mission,
            },
        }

    try:
        result = method_map[mission]()
        if isinstance(result, dict) and result.get("error"):
            return {"args": args, "returns": {"success": False, "result": result}}
        return {"args": args, "returns": {"success": True, "result": result}}
    except Exception as e:
        return {"args": args, "returns": {"success": False, "error": str(e)}}


# ----- 机器学习工具 -----
@mcp.tool()
async def predict_band_gap(formula: str | list[str]) -> dict:
    """
    使用预训练模型预测指定材料的带隙值

    Args:
        formula: 化学式，可以是单个字符串如 "LiFeO2" 或列表如 ["LiFeO2", "NaCl"]

    Returns:
        dict: 包含 formula, predicted_band_gap, error(可选)
    """
    args = {"formula": formula}
    from myml import bandgap_predict as mm

    try:
        result = mm.predict_bandgap(formula)
        return {"args": args, "returns": {"predicted_band_gap": result}}
    except Exception as e:
        return {
            "args": args,
            "returns": {"error": str(e), "message": f"预测材料 {formula} 的带隙值失败"},
        }


# ============ 主程序入口 ============
if __name__ == "__main__":
    try:
        # 启动文件服务器
        matfileserver = flask_server.MatFileServer()
        # 连接远程服务器
        connection = tryssh.VaspTaskInitializer(HOST, USERNAME, PASSWORD, PORT)
        for i in range(5):
            try:
                with connection as vasp_task:
                    if vasp_task.link():
                        print("已成功连接到远程服务器")
                        break
            except Exception as e:
                print(f"连接远程服务器失败，正在重试... ({i + 1}/5), 错误: {e}")
                if i == 4:
                    raise e

        # 启动 MCP 服务器
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"服务器运行出错: {e}")
        exit()
