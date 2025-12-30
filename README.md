# LangGraph Demo (FastAPI + LangGraph + React)

一个最小但功能完整的 Agent Demo：后端使用 FastAPI + LangGraph，前端使用 React + Ant Design。项目集成了工具调用编排、文档 RAG、工作流触发、以及 SSE 流式输出（前端通过 `fetch` 读取 `text/event-stream`）。

## 功能概览
- **Chat + Workflow**：生成执行计划（plan），分步执行，并支持 **interrupt / continue / cancel**。
- **Tool Orchestration**：支持 `now` / `calc` / `http_search` / `doc_insights` / `internal_api` / `workflow_trigger` 等工具，并具备并发与预算控制。
- **RAG（文档检索）**：支持上传文档，向量化后在对话中检索（可配置开关）。
- **Observability**：Prometheus 指标可在 `/metrics` 拉取。
- **安全加固**：thread_id 由服务端生成，线程归属被锁定（避免 thread 被抢占）。

## 目录结构
```
.
├─ app/                 # FastAPI + LangGraph
├─ data/                # SQLite 数据与 checkpoint（默认）
├─ frontend/            # React + Vite + AntD
├─ scripts/             # 辅助脚本
├─ .env.example
├─ requirements.txt
└─ README.md
```

## 环境要求
- Python >= 3.10（推荐 3.11）
- Node.js >= 18（前端开发/构建）
- OpenAI 或 OpenAI-compatible 的 API Key（或关闭文档检索）

## 快速开始（后端）
```bash
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env   # Windows: copy .env.example .env
# 编辑 .env：至少填 OPENAI_API_KEY；如用兼容服务可填 OPENAI_BASE_URL

uvicorn app.main:app --reload --port 8000
```

## 快速开始（前端）
```bash
cd frontend
npm install

# 可选：配置后端地址
# echo "VITE_API_BASE=http://127.0.0.1:8000" > .env.development

npm run dev
```
打开 `http://localhost:5173`。

## 认证模式（后端）
后端支持 3 种模式（见 `app/main.py`）：
- **Dev passthrough（默认）**：不配置 `API_AUTH_TOKEN` 和 `AUTH_SECRET_KEY` 时，允许无 token 访问（仍会限流）。
- **Static token**：配置 `API_AUTH_TOKEN` 后，需要携带 token（Header Bearer / x-api-key / query api_key）。
- **Session token（推荐）**：配置 `AUTH_SECRET_KEY` 后必须登录，`/auth/login` 会返回 session token（HMAC 签名），并强制 token 与 user_id 匹配。

前端默认使用 `Authorization: Bearer <token>`（`VITE_API_KEY` 或登录 token）。

## Chat API（重要：线程必须先创建）
为了防止 thread 被抢占：
- `thread_id` **仅由服务端生成**（`POST /chat/threads` 会拒绝客户端传 `thread_id`）。
- `/chat`、`/chat/stream` 不会隐式创建线程：thread 不存在会返回 404。

### 1) 创建线程
`POST /chat/threads`
```bash
curl -sS -X POST "http://127.0.0.1:8000/chat/threads" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"user_id":"u1","title":"hello"}'
```
返回：`{"thread_id":"...","title":"..."}`

### 2) 流式聊天（SSE，POST + body 传参）
`POST /chat/stream`，返回 `text/event-stream`。
```bash
curl -N -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"session_id":"<thread_id>","user_id":"u1","message":"现在几点？请用 now 工具","remember":false}'
```

SSE 事件常见类型：
- `plan`：计划生成（包含 `plan/current_step/active_step_id`）
- `status`：状态（planning/executing/waiting/completed/cancelled 等）
- `step_update`：步骤完成/等待等变更
- `interrupt`：需要人工确认
- `cancelled`：取消执行
- `error` / `end`：错误/结束

### 3) interrupt 后继续/取消（SSE）
`POST /chat/continue`
```bash
curl -N -X POST "http://127.0.0.1:8000/chat/continue" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"thread_id":"<thread_id>","user_id":"u1","action":"continue"}'
```
`action` 可为 `continue` 或 `cancel`。

### 4) 会话列表与消息
- `GET /chat/threads?user_id=...`
- `PATCH /chat/threads/{thread_id}?user_id=...`（重命名）
- `DELETE /chat/threads/{thread_id}?user_id=...`
- `GET /chat/threads/{thread_id}/messages?user_id=...&limit=200`

说明：消息接口默认返回 **最新 N 条**（N=limit），但返回顺序为时间正序（从旧到新）。

### 5) 消息长度限制
后端默认限制 `CHAT_MAX_MESSAGE_CHARS=8000`（可在 `.env` 修改）。超限会返回 413。

## 文档与 RAG
上传文档后，Agent 可通过 `doc_insights` 工具进行检索并注入上下文。
- 上传：`POST /documents`（multipart）
- 文档列表/详情/下载：`GET /documents`、`GET /documents/{id}`、`GET /documents/{id}/download`
- 版本发布/回滚：`POST /documents/{document_id}/versions/{version_id}/publish`、`POST /documents/{document_id}/versions/{version_id}/rollback`

如暂不需要文档检索：在 `.env` 设置 `ENABLE_DOCUMENT_RETRIEVAL=false`。

## 异步处理（可选：Celery）
文档处理与 ingestion 支持 Celery（未配置 broker 时会 inline 执行）：
- `CELERY_BROKER_URL=redis://localhost:6379/0`
- `CELERY_RESULT_BACKEND=redis://localhost:6379/0`

## 运营与指标
- Tool budget：`GET /config/tool-budget`、`PATCH /config/tool-budget`
- 向量健康检查：`GET /ops/vector/health`
- Prometheus 指标：`GET /metrics`

