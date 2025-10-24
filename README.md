# fastapi-langgraph-minimal

最小可运行的 FastAPI × LangGraph Agent，内置三个演示工具（时间/计算器/FAQ）。

## 1. 环境准备
- Python >= 3.10（推荐 3.11）
- Windows / macOS / Linux 均可
- 一个可用的 OpenAI 或 **OpenAI 兼容** API Key

## 2. 安装与启动
```bash
python -m venv .venv && source .venv/bin/activate     # macOS/Linux
# Windows PowerShell:
# python -m venv .venv; .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 复制 env
cp .env.example .env    # Windows: copy .env.example .env
# 编辑 .env 填写 OPENAI_API_KEY / OPENAI_BASE_URL(可选) / LLM_MODEL

uvicorn app.main:app --reload --port 8000
```

## 3. 测试
```bash
# 询问当前时间（触发 now 工具）
curl -X POST "http://127.0.0.1:8000/chat"   -H "Content-Type: application/json"   -d '{"session_id":"demo-1","message":"现在几点？请用工具获取"}'

# 计算表达式（触发 calc 工具）
curl -X POST "http://127.0.0.1:8000/chat"   -H "Content-Type: application/json"   -d '{"session_id":"demo-1","message":"帮我算(1+2*3)/2"}'

# 本地 FAQ（触发 faq 工具）
curl -X POST "http://127.0.0.1:8000/chat"   -H "Content-Type: application/json"   -d '{"session_id":"demo-2","message":"LangGraph 是什么？"}'
```

## 4. 目录结构
```
fastapi-langgraph-minimal/
├─ app/
│  ├─ main.py
│  ├─ graph.py
│  ├─ tools.py
│  └─ __init__.py
├─ .env.example
├─ requirements.txt
└─ README.md
```

## 5. 常见问题
- 循环过深：复杂任务可能触发递归限制（GRAPH_RECURSION_LIMIT）。优化条件边或增加退出判定。
- 工具异常：ToolNode 会将异常包装为 ToolMessage 返回，便于在日志中定位。
- 记忆持久化：示例使用内存 Saver；生产替换为 SQLite/Redis 的 Checkpointer 更稳。

## 6. 前端（React + AntD + SSE）
在 `frontend/` 目录提供了一个超轻量示例：
```bash
cd frontend
npm install
# 可选：创建 .env.development 设置后端地址（默认 http://127.0.0.1:8000）
# echo "VITE_API_BASE=http://127.0.0.1:8000" > .env.development
npm run dev  # 打开 http://localhost:5173
```
该前端使用浏览器 `EventSource` 连接后端 `/chat/stream`，实时接收 AI 增量与工具节点事件。
