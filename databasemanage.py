import sqlite3
import pickle
import pandas as pd
from pymatgen.core import Structure
import re

class DatabaseManager:
    def __init__(self, db_path='materials.db', timeout=30):
        self.conn = sqlite3.connect(db_path, timeout=timeout)
        # 启用外键约束
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_tables()

    def create_tables(self):
        # SQLite 使用 AUTOINCREMENT 而不是 SEQUENCE
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS materials(
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                add_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                material_id VARCHAR,
                formula VARCHAR,
                structure BLOB,
                band_gap FLOAT
            )
        ''')
        # 为 material_id 创建唯一索引
        self.conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_material_id ON materials(material_id)')
        
        # 聊天记录表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history(
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR,
                role VARCHAR,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # 为 session_id 创建索引
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id)')
        
        self.conn.commit()

    def add_material(self, formula, structure, band_gap, material_id="COSTOMED"):
        structure_blob = pickle.dumps(structure)
        try:
            self.conn.execute('''
                INSERT INTO materials (material_id, formula, structure, band_gap)
                VALUES (?, ?, ?, ?)
            ''', (material_id, formula, structure_blob, band_gap))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # 如果 material_id 已存在，更新记录
            self.conn.execute('''
                UPDATE materials 
                SET formula = ?, structure = ?, band_gap = ?
                WHERE material_id = ?
            ''', (formula, structure_blob, band_gap, material_id))
            self.conn.commit()
            return False

    def get_material_by_ID(self, ID):
        result = self.conn.execute('''
            SELECT ID, add_time, material_id, formula, structure, band_gap
            FROM materials
            WHERE ID = ?
        ''', (ID,)).fetchone()
        
        if result:
            ID, add_time, material_id, formula, structure_blob, band_gap = result
            structure = pickle.loads(structure_blob)
            return {
                'ID': ID,
                'add_time': add_time,
                'material_id': material_id,
                'formula': formula,
                'structure': structure,
                'band_gap': band_gap
            }
        else:
            return None

    def get_material_by_material_id(self, material_id):
        result = self.conn.execute('''
            SELECT ID, add_time, material_id, formula, structure, band_gap
            FROM materials
            WHERE material_id = ?
        ''', (material_id,)).fetchone()
        
        if result:
            ID, add_time, material_id, formula, structure_blob, band_gap = result
            structure = pickle.loads(structure_blob)
            return {
                'ID': ID,
                'add_time': add_time,
                'material_id': material_id,
                'formula': formula,
                'structure': structure,
                'band_gap': band_gap
            }
        else:
            return None

    def get_material_by_elements(self, chemical_formula, page=1):
        try:
            results = self.conn.execute('''
                SELECT ID, add_time, material_id, formula, structure, band_gap FROM materials
            ''').fetchall()
            
            input_elements = set(re.findall(r'[A-Z][a-z]?', chemical_formula))
            input_element_count = len(input_elements)
            filtered_results = []
            
            for required_count in range(input_element_count, 0, -1):
                for row in results:
                    formula_db = row[3]
                    elements_in_formula_db = set(re.findall(r'[A-Z][a-z]?', formula_db))
                    match_count = len(elements_in_formula_db & input_elements)
                    if len(elements_in_formula_db) == input_element_count and match_count == required_count:
                        filtered_results.append({
                            'ID': row[0],
                            'material_id': row[2],
                            'formula': row[3],
                            'band_gap': row[5]
                        })
                if filtered_results:
                    break
            
            return self.list_results_by_pages(filtered_results, page)
        except Exception as e:
            return {'error': str(e)}

    def remove_material(self, ID):
        self.conn.execute('DELETE FROM materials WHERE ID = ?', (ID,))
        self.conn.commit()

    def update_material(self, ID, **kwargs):
        fields = []
        values = []
        for key, value in kwargs.items():
            if key == 'structure':
                value = pickle.dumps(value)
            fields.append(f"{key} = ?")
            values.append(value)
        values.append(ID)
        set_clause = ', '.join(fields)
        self.conn.execute(f'UPDATE materials SET {set_clause} WHERE ID = ?', values)
        self.conn.commit()

    def list_results_by_pages(self, results, page=1, page_size=10):
        num_page = (len(results) + page_size - 1) // page_size
        if page < 1 or page > num_page:
            return {'error': 'Invalid page number'}
        start = (page - 1) * page_size
        end = start + page_size
        return {
            'page': page,
            'page_size': page_size,
            'num_page': num_page,
            'materials': results[start:end] if isinstance(results[0], dict) else [{'ID': r[0], 'material_id': r[1], 'formula': r[2], 'band_gap': r[3]} for r in results[start:end]]
        }

    def list_all_materials_by_pages(self, page=1, page_size=10):
        results = self.conn.execute('SELECT ID, material_id, formula, band_gap FROM materials').fetchall()
        return self.list_results_by_pages(results, page, page_size)
    
# ============== 聊天记录操作 ==============
def _ensure_chat_table():
    """确保聊天记录表存在"""
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history(
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR,
                role VARCHAR,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id)')
        conn.commit()
    finally:
        conn.close()

def add_chat_message(session_id: str, role: str, content: str):
    """添加聊天消息（独立连接）"""
    _ensure_chat_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        conn.execute('''
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
        ''', (session_id, role, content))
        conn.commit()
    finally:
        conn.close()

def get_chat_history(session_id: str, limit: int = 50) -> list:
    """获取会话的聊天历史（独立连接）"""
    _ensure_chat_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        results = conn.execute('''
            SELECT role, content, timestamp 
            FROM chat_history 
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit)).fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in results]
    finally:
        conn.close()

def list_sessions(limit: int = 20) -> list:
    """列出所有会话ID（独立连接）"""
    _ensure_chat_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        results = conn.execute('''
            SELECT DISTINCT session_id, MAX(timestamp) as last_time
            FROM chat_history
            GROUP BY session_id
            ORDER BY last_time DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        return [{"session_id": r[0], "last_time": r[1]} for r in results]
    finally:
        conn.close()

def clear_chat_history(session_id: str = None):
    """清除聊天历史（独立连接）"""
    _ensure_chat_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        if session_id:
            conn.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
        else:
            conn.execute('DELETE FROM chat_history')
        conn.commit()
    finally:
        conn.close()
    
    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db_manager = DatabaseManager("material_database.db")

    # Example usage
    struct = Structure.from_file("cifs/La3S4-mp-567.cif")
    db_manager.add_material("La3S4", struct, 8.5, "mp-567")

    material = db_manager.get_material_by_material_id("mp-567")
    print(material)

    db_manager.update_material(material['ID'], band_gap=9.0)
    material = db_manager.get_material_by_material_id("mp-567")
    print(material)

    print(db_manager.list_all_materials_by_pages(page=1, page_size=5))

    db_manager.close()