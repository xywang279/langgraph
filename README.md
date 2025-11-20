# fastapi-langgraph-minimal

最小可运行的 FastAPI × LangGraph Agent，现已接入 HTTP 搜索、内部 API、文档洞察与工作流触发工具，并为每个工具定义鉴权/速率/可观测指标。

### 功能亮点
- **HTTP Search**：通过可配置的 API Gateway 获取实时网页/新闻结果，携带 Bearer Token 鉴权与速率保护。
- **Internal API**：统一封装企业 CRM/ITSM 等内部接口，自动注入 `X-User-Id`，输出 JSON 摘要。
- **Doc Insights**：复用上传文档与记忆向量库，实现上下文 RAG 检索。
- **Workflow Trigger**：下发自动化任务/工单，产出 run_id 并记录审计。
- 内置 Prometheus 指标：`tool_invocations_total`、`tool_invocation_duration_seconds`、`tool_rate_limited_total` 等，可在 `/metrics` 拉取。

## 1. 环境准备
- Python >= 3.10（推荐 3.11）
- Windows / macOS / Linux 均可
- 一个可用的 OpenAI 或 **OpenAI 兼容** API Key

## 2. 安装与启动
> HTTP 搜索接入示例  
> - `HTTP_SEARCH_BASE_URL=https://serpapi.com`  
> - `HTTP_SEARCH_AUTH_MODE=query`（将 `HTTP_SEARCH_API_KEY` 注入 query string）  
> - `HTTP_SEARCH_DEFAULT_PARAMS={"engine":"google","google_domain":"google.com"}` 补充分页/引擎等默认参数。

> 如果暂时无法加载 HuggingFace 模型，可在 `.env` 中设置 `ENABLE_DOCUMENT_RETRIEVAL=false`，后端会跳过文档检索，仅依赖实时搜索与其他工具。

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

# HTTP 搜索（需要配置 HTTP_SEARCH_BASE_URL/KEY）
curl -X POST "http://127.0.0.1:8000/chat"   -H "Content-Type: application/json"   -d '{"session_id":"demo-1","message":"帮我检索 LangGraph 最新的 3 条社区动态"}'

# 内网 API & 工作流（需要配置对应 Base URL + Token）
curl -X POST "http://127.0.0.1:8000/chat"   -H "Content-Type: application/json"   -d '{"session_id":"demo-2","user_id":"ops-01","message":"查询一下 ticket INC-1024 是否关闭，若未关闭则触发 follow_up workflow"}'
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
