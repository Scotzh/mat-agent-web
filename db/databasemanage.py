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
    """确保聊天记录表存在，并自动迁移旧表结构"""
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        # 检查表是否存在
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # 创建新表
            conn.execute('''
                CREATE TABLE chat_history(
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id VARCHAR,
                    role VARCHAR,
                    content TEXT,
                    tool_results TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id)')
        else:
            # 检查是否需要添加 tool_results 列
            cursor = conn.execute("PRAGMA table_info(chat_history)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'tool_results' not in columns:
                # 添加新列
                conn.execute('ALTER TABLE chat_history ADD COLUMN tool_results TEXT')
        
        conn.commit()
    finally:
        conn.close()

def _ensure_session_table():
    """确保会话信息表存在"""
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions(
                session_id VARCHAR PRIMARY KEY,
                session_name VARCHAR DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    finally:
        conn.close()

def add_chat_message(session_id: str, role: str, content: str, tool_results: list = None):
    """添加聊天消息（独立连接）
    
    Args:
        session_id: 会话ID
        role: 角色 (user/assistant)
        content: 消息内容
        tool_results: 工具调用结果列表，格式为 [{"tool_name": str, "tool_args": dict, "result": str}]
    """
    _ensure_chat_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        import json
        tool_results_json = json.dumps(tool_results, ensure_ascii=False) if tool_results else None
        conn.execute('''
            INSERT INTO chat_history (session_id, role, content, tool_results)
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, tool_results_json))
        conn.commit()
    finally:
        conn.close()

def update_session_name(session_id: str, session_name: str):
    """更新会话名称"""
    _ensure_session_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        conn.execute('''
            INSERT INTO sessions (session_id, session_name)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                session_name = excluded.session_name,
                updated_at = CURRENT_TIMESTAMP
        ''', (session_id, session_name))
        conn.commit()
    finally:
        conn.close()

def get_session_name(session_id: str) -> str:
    """获取会话名称"""
    _ensure_session_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        result = conn.execute(
            'SELECT session_name FROM sessions WHERE session_id = ?',
            (session_id,)
        ).fetchone()
        return result[0] if result else None
    finally:
        conn.close()

def delete_session(session_id: str):
    """删除会话及其所有消息和工具调用记录"""
    _ensure_chat_table()
    _ensure_tool_call_table()
    _ensure_session_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        # 删除聊天记录
        conn.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
        # 删除工具调用记录
        conn.execute('DELETE FROM tool_calls WHERE session_id = ?', (session_id,))
        # 删除会话信息
        conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
    finally:
        conn.close()

def get_chat_history(session_id: str, limit: int = 50) -> list:
    """获取会话的聊天历史（独立连接）"""
    _ensure_chat_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        import json
        results = conn.execute('''
            SELECT role, content, tool_results, timestamp 
            FROM chat_history 
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit)).fetchall()
        return [
            {
                "role": r[0], 
                "content": r[1], 
                "tool_results": json.loads(r[2]) if r[2] else None,
                "timestamp": r[3]
            } 
            for r in results
        ]
    finally:
        conn.close()

def list_sessions(limit: int = 20) -> list:
    """列出所有会话（独立连接）"""
    _ensure_chat_table()
    _ensure_session_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        # 获取所有会话及其最后活动时间
        results = conn.execute('''
            SELECT 
                c.session_id,
                MAX(c.timestamp) as last_time,
                s.session_name
            FROM chat_history c
            LEFT JOIN sessions s ON c.session_id = s.session_id
            GROUP BY c.session_id
            ORDER BY last_time DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        return [
            {
                "session_id": r[0], 
                "last_time": r[1],
                "session_name": r[2]
            } 
            for r in results
        ]
    finally:
        conn.close()

# ============== 工具调用记录操作 ==============
def _ensure_tool_call_table():
    """确保工具调用记录表存在"""
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tool_calls(
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR,
                tool_name VARCHAR,
                tool_args TEXT,
                result TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_tool_session ON tool_calls(session_id)')
        conn.commit()
    finally:
        conn.close()

def add_tool_call(session_id: str, tool_name: str, tool_args: dict, result: str):
    """添加工具调用记录（独立连接）"""
    _ensure_tool_call_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        import json
        tool_args_json = json.dumps(tool_args, ensure_ascii=False)
        conn.execute('''
            INSERT INTO tool_calls (session_id, tool_name, tool_args, result)
            VALUES (?, ?, ?, ?)
        ''', (session_id, tool_name, tool_args_json, result))
        conn.commit()
    finally:
        conn.close()

def get_tool_calls(session_id: str, limit: int = 50) -> list:
    """获取会话的工具调用记录（独立连接）"""
    _ensure_tool_call_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        import json
        results = conn.execute('''
            SELECT tool_name, tool_args, result, timestamp 
            FROM tool_calls 
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit)).fetchall()
        return [{
            "tool_name": r[0],
            "tool_args": json.loads(r[1]) if r[1] else {},
            "result": r[2],
            "timestamp": r[3]
        } for r in results]
    finally:
        conn.close()

def clear_tool_calls(session_id: str = None):
    """清除工具调用记录（独立连接）"""
    _ensure_tool_call_table()
    conn = sqlite3.connect("matagent_history.db", timeout=30)
    try:
        if session_id:
            conn.execute('DELETE FROM tool_calls WHERE session_id = ?', (session_id,))
        else:
            conn.execute('DELETE FROM tool_calls')
        conn.commit()
    finally:
        conn.close()


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