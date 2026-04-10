"""统一文件服务器 - 提供图片和3D结构可视化服务"""

import os
import uuid
import time
import threading
from collections import deque
from flask import Flask, send_from_directory, render_template_string
from pymatgen.io.cif import CifWriter
import loadenv

config = loadenv.Config()
local_host = config.get_ip()

app = Flask(__name__)

# 项目根目录
# 用os.path.abspath和os.path.dirname确保无论从哪里运行，路径都正确
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "temp_images")
HTML_DIR = os.path.join(PROJECT_ROOT, "temp_3d")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# ============ 配置 ============
MAX_IMAGE_FILES = 50
MAX_HTML_FILES = 30
MAX_STRUCTURES = 30

# ============ 结构信息存储 ============
STRUCTURE_INFO = {}
STRUCTURE_QUEUE = deque(maxlen=MAX_STRUCTURES)
# 改为json文件存储结构信息，定期清理过期数据
import json
STRUCTURE_INFO_FILE = os.path.join(PROJECT_ROOT, "structure_info.json")

def _load_structure_info():
    """从JSON文件加载结构信息"""
    global STRUCTURE_INFO, STRUCTURE_QUEUE
    if os.path.exists(STRUCTURE_INFO_FILE):
        try:
            with open(STRUCTURE_INFO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 恢复结构信息（pymatgen对象需要重新加载）
                STRUCTURE_INFO = data.get("info", {})
                # 恢复队列顺序
                queue_ids = data.get("queue", [])
                STRUCTURE_QUEUE = deque(queue_ids[-MAX_STRUCTURES:] if len(queue_ids) > MAX_STRUCTURES else queue_ids)
                print(f"✅ 已加载 {len(STRUCTURE_INFO)} 条结构信息")
        except Exception as e:
            print(f"加载结构信息失败: {e}")

def _save_structure_info():
    """保存结构信息到JSON文件"""
    try:
        # 清理不可序列化的对象
        serializable_info = {}
        for sid, info in STRUCTURE_INFO.items():
            serializable_info[sid] = {
                k: v for k, v in info.items()
                if k != 'structure'  # 不保存pymatgen对象
            }
        data = {
            "info": serializable_info,
            "queue": list(STRUCTURE_QUEUE)
        }
        with open(STRUCTURE_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存结构信息失败: {e}")

# 启动时加载
_load_structure_info()


def cleanup_old_files():
    """清理最过时的文件"""
    # 清理图片
    image_files = sorted(
        [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')],
        key=lambda x: os.path.getctime(os.path.join(IMAGE_DIR, x))
    )
    while len(image_files) > MAX_IMAGE_FILES:
        oldest = image_files.pop(0)
        try:
            os.remove(os.path.join(IMAGE_DIR, oldest))
            print(f"🗑️ 清理旧图片: {oldest}")
        except Exception as e:
            print(f"清理图片失败: {e}")
    
    # 清理 HTML
    html_files = sorted(
        [f for f in os.listdir(HTML_DIR) if f.endswith('.html')],
        key=lambda x: os.path.getctime(os.path.join(HTML_DIR, x))
    )
    while len(html_files) > MAX_HTML_FILES:
        oldest = html_files.pop(0)
        try:
            os.remove(os.path.join(HTML_DIR, oldest))
            print(f"🗑️ 清理旧HTML: {oldest}")
        except Exception as e:
            print(f"清理HTML失败: {e}")
    
    # 同步清理 STRUCTURE_INFO 和 STRUCTURE_QUEUE
    current_html_files = set(os.listdir(HTML_DIR))
    removed = False
    for sid in list(STRUCTURE_QUEUE):
        info = STRUCTURE_INFO.get(sid, {})
        html_file = info.get("html_file", "")
        if html_file and html_file not in current_html_files:
            STRUCTURE_QUEUE.remove(sid)
            STRUCTURE_INFO.pop(sid, None)
            removed = True
    if removed:
        _save_structure_info()

# ============ HTML 模板 ============
COMPLETE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{ info.reduced_formula }} - 可视化</title>
    <style>
        body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f0f2f5; color: #333; overflow: hidden; }
        .container { max-width: 1600px; margin: 15px auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; height: calc(100vh - 30px); display: flex; flex-direction: column; }
        .header { background: #2c3e50; color: white; padding: 12px 25px; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
        .content { display: grid; grid-template-columns: 1fr 420px; gap: 0; flex-grow: 1; overflow: hidden; }
        .vis-container { background: #fff; border-right: 1px solid #eee; height: 100%; }
        .info-section { padding: 20px; overflow-y: auto; background: #fafafa; display: flex; flex-direction: column; gap: 20px; }
        .card { background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
        .card h3 { margin: 0 0 15px 0; color: #3498db; font-size: 1.1em; border-left: 4px solid #3498db; padding-left: 12px; }
        .data-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .data-item { background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #f0f0f0; }
        .label { font-size: 0.75em; color: #7f8c8d; display: block; margin-bottom: 4px; }
        .value { font-weight: 600; font-family: 'Consolas', monospace; font-size: 0.95em; color: #2c3e50; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
        th { text-align: left; color: #7f8c8d; padding: 10px 8px; border-bottom: 2px solid #eee; }
        td { padding: 10px 8px; border-bottom: 1px solid #f5f5f5; color: #444; }
        .btn-cif { background: #3498db; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 600; }
        .btn-cif:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h2 style="margin:0; font-size: 1.4em;">{{ info.reduced_formula }}</h2>
                <span style="font-size: 0.85em; opacity: 0.8;">空间群: {{ info.space_group_symbol }} (No. {{ info.space_group_number }})</span>
            </div>
            <a href="/cif/{{ info.id }}" class="btn-cif">📄 下载 CIF 文件</a>
        </div>
        <div class="content">
            <div class="vis-container">
                <iframe src="/3d/{{ info.html_file }}" style="width:100%; height:100%; border:none;"></iframe>
            </div>
            <div class="info-section">
                <div class="card">
                    <h3>晶格参数</h3>
                    <div class="data-grid">
                        <div class="data-item"><span class="label">a (Å)</span><span class="value">{{ info.lattice.a }}</span></div>
                        <div class="data-item"><span class="label">b (Å)</span><span class="value">{{ info.lattice.b }}</span></div>
                        <div class="data-item"><span class="label">c (Å)</span><span class="value">{{ info.lattice.c }}</span></div>
                        <div class="data-item"><span class="label">α (°)</span><span class="value">{{ info.lattice.alpha }}</span></div>
                        <div class="data-item"><span class="label">β (°)</span><span class="value">{{ info.lattice.beta }}</span></div>
                        <div class="data-item"><span class="label">γ (°)</span><span class="value">{{ info.lattice.gamma }}</span></div>
                        <div class="data-item"><span class="label">体积 (Å³)</span><span class="value">{{ info.lattice.volume }}</span></div>
                        <div class="data-item"><span class="label">密度 (g/cm³)</span><span class="value">{{ info.density }}</span></div>
                    </div>
                </div>
                <div class="card">
                    <h3>基本信息</h3>
                    <div class="data-grid">
                        <div class="data-item"><span class="label">化学式</span><span class="value">{{ info.reduced_formula }}</span></div>
                        <div class="data-item"><span class="label">空间群</span><span class="value">{{ info.space_group_symbol }}</span></div>
                        <div class="data-item"><span class="label">原子总数</span><span class="value">{{ info.num_sites }}</span></div>
                        <div class="data-item"><span class="label">是否有序</span><span class="value">{{ '是' if info.is_ordered else '否' }}</span></div>
                    </div>
                </div>
                <div class="card">
                    <h3>原子位点</h3>
                    <table>
                        <thead><tr><th>元素</th><th>分数坐标</th></tr></thead>
                        <tbody>
                            {% for site in info.sites %}
                            <tr><td><b>{{ site.element }}</b></td><td>{{ site.coords|join(', ') }}</td></tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ============ 路由 ============

@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/view/<id>')
def view_structure(id):
    """显示带信息面板的3D结构页面"""
    if id not in STRUCTURE_INFO:
        return f"<h3>结构已失效 (ID: {id})</h3><p>可能是加载了太多新结构，旧结构已被清理。</p>", 404
    
    info = STRUCTURE_INFO[id]
    return render_template_string(COMPLETE_TEMPLATE, info=info)

@app.route('/3d/<path:filename>')
def serve_3d_html(filename):
    """直接显示3D HTML文件"""
    return send_from_directory(HTML_DIR, filename)

@app.route('/cif/<id>')
def download_cif(id):
    if id not in STRUCTURE_INFO:
        return "CIF文件不存在", 404
    info = STRUCTURE_INFO[id]
    struct = info['structure']
    import io
    sio = io.BytesIO(str(CifWriter(struct)).encode())
    from flask import send_file
    return send_file(sio, as_attachment=True, download_name=f"{info['reduced_formula']}.cif")

@app.route('/')
def index():
    return f"""<h1>MatAgent 文件服务</h1>
    <ul>
        <li>图片目录: {IMAGE_DIR}</li>
        <li>3D目录: {HTML_DIR}</li>
        <li>结构存储: {len(STRUCTURE_INFO)}</li>
    </ul>"""


# ============ 管理器类 ============

class MatFileServer:
    _server_started = False
    _port = 6750
    
    def __init__(self, port=None):
        import os
        if port:
            MatFileServer._port = port
            
        if os.environ.get("MAT_FILESERVER_STARTED") == "1":
            MatFileServer._server_started = True
            return
            
        if not MatFileServer._server_started:
            thread = threading.Thread(
                target=lambda: app.run(host="0.0.0.0", port=MatFileServer._port, debug=False, use_reloader=False),
                daemon=True
            )
            thread.start()
            MatFileServer._server_started = True
            os.environ["MAT_FILESERVER_STARTED"] = "1"
            time.sleep(0.5)
            print(f"🚀 MatAgent 文件服务已在端口 {MatFileServer._port} 开启")
    
    def add_image(self, img_buffer) -> str:
        cleanup_old_files()
        
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(IMAGE_DIR, filename)
        img_buffer.seek(0)
        with open(filepath, "wb") as f:
            f.write(img_buffer.getvalue())
        return f"http://{local_host}:{MatFileServer._port}/image/{filename}"
    
    def add_image_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"错误：文件 {filepath} 不存在"
        filename = f"{uuid.uuid4()}_{os.path.basename(filepath)}"
        dest_path = os.path.join(IMAGE_DIR, filename)
        import shutil
        shutil.copy(filepath, dest_path)
        return f"http://{local_host}:{MatFileServer._port}/image/{filename}"
    
    def add_html_with_info(self, structure, html_file_path) -> str:
        """保存3D HTML并存储结构信息，返回带面板的页面URL"""
        cleanup_old_files()
        
        # 复制 HTML 文件
        filename = f"{uuid.uuid4().hex}.html"
        dest_path = os.path.join(HTML_DIR, filename)
        import shutil
        shutil.copy(html_file_path, dest_path)
        
        # 删除临时文件
        try: os.remove(html_file_path)
        except: pass
        
        # 生成唯一 ID
        structure_id = str(uuid.uuid4())
        
        # 清理旧结构
        if len(STRUCTURE_QUEUE) >= 10:
            old_id = STRUCTURE_QUEUE.popleft()
            STRUCTURE_INFO.pop(old_id, None)
        
        # 存储结构信息
        lat = structure.lattice
        sg = structure.get_space_group_info()
        
        STRUCTURE_INFO[structure_id] = {
            'id': structure_id,
            'structure': structure,
            'reduced_formula': structure.reduced_formula,
            'full_formula': structure.formula,
            'space_group_symbol': sg[0],
            'space_group_number': sg[1],
            'num_sites': len(structure),
            'is_ordered': structure.is_ordered,
            'density': round(structure.density, 4),
            'html_file': filename,
            'lattice': {
                'a': round(lat.a, 4), 'b': round(lat.b, 4), 'c': round(lat.c, 4),
                'alpha': round(lat.alpha, 2), 'beta': round(lat.beta, 2), 'gamma': round(lat.gamma, 2),
                'volume': round(lat.volume, 2)
            },
            'sites': [{'element': str(site.specie), 'coords': [round(c, 4) for c in site.frac_coords]} for site in structure.sites]
        }
        STRUCTURE_QUEUE.append(structure_id)
        
        # 保存到文件
        _save_structure_info()
        
        url = f"http://{local_host}:{MatFileServer._port}/view/{structure_id}"
        print(f"✅ 结构已就绪 [{structure.reduced_formula}]: {url}")
        
        return url
    
    def add_html_file(self, filepath: str) -> str:
        """仅保存HTML文件（不存储结构信息）"""
        if not os.path.exists(filepath):
            return f"错误：文件 {filepath} 不存在"
        
        filename = f"{uuid.uuid4().hex}.html"
        dest_path = os.path.join(HTML_DIR, filename)
        
        import shutil
        shutil.copy(filepath, dest_path)
        
        try: os.remove(filepath)
        except: pass
        
        return f"http://{local_host}:{MatFileServer._port}/view/{filename}"


# ============ 兼容接口 ============
MemoryFileServer = MatFileServer
CrystalManager = MatFileServer


if __name__ == "__main__":
    server = MatFileServer(port=6750)
    print("服务已启动，访问 http://localhost:6750 查看状态")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("停止服务")