# CHANGELOG

## 2025-04-10
- 初始版本
- 更新 AGENTS.md：修正启动命令（web_app.py → web_mcp_app.py），更新核心模块表文件路径，添加端口号
- AGENTS.md 添加 CHANGELOG 维护说明
- 代码重构：重新组织目录结构
  - 新建 config/：配置、依赖文件、环境变量
  - 新建 db/：数据库及管理模块
  - 新建 cache/：缓存和输出文件
  - 新建 server/：后端服务（flask_server, tryssh）
  - 新建 web/assets/：Web 静态资源
  - 移动 assets/ → web/assets/
  - 移动 loadenv.py, .env, .env.example, config.toml → config/
  - 移动 pyproject.toml, requirements.txt, uv.lock, uv.toml → config/
  - 移动 *.db → db/
  - 移动 temp_images/, temp_3d/, structure_info.json → cache/
  - 移动 flask_server.py, tryssh.py → server/
  - 移动 databasemanage.py → db/
  - 删除 main.py
  - 更新 .gitignore
  - 更新 mcp_server.py, web_mcp_app.py, server/flask_server.py, server/tryssh.py 中的导入路径
- 更新 README.md：反映新的目录结构