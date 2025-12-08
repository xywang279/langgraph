from __future__ import annotations
import os

import io
import json
import logging
import time
import uuid
import base64
import secrets
import hashlib
import hmac
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
import threading

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    Request,
    Response,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Interrupt
from pydantic import BaseModel
from starlette import status

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import storage
from .settings import load_settings, AppSettings
from .graph import (
    build_step_instruction,
    compiled_graph,
    get_tool_budget_settings,
    update_tool_budget_settings,
)
from .rag import retrieve, user_index_size, _rebuild_vector_index, index_ids_for_user
from .task_queue import enqueue_process_document, enqueue_ingestion_job

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

logger = logging.getLogger(__name__)
if not any(
    isinstance(handler, RotatingFileHandler)
    and getattr(handler, "baseFilename", None) == str(_LOG_FILE)
    for handler in logger.handlers
):
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="LangGraph Agent · Tool Orchestration Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
DOCS_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "40"))
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
try:
    API_RATE_LIMIT_PER_MINUTE = max(
        1, int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "120") or 1)
    )
except ValueError:
    API_RATE_LIMIT_PER_MINUTE = 120
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "").strip()
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin").strip()
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin123").strip()
try:
    AUTH_TOKEN_TTL_SECONDS = max(
        300, int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "43200") or 43200)
    )
except ValueError:
    AUTH_TOKEN_TTL_SECONDS = 43200
ALLOW_USER_REGISTRATION = str(os.getenv("AUTH_ALLOW_REGISTRATION", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
}
try:
    settings = load_settings()
except Exception as exc:  # pragma: no cover - startup guard
    logger.warning("config validation failed: %s (continuing with environment defaults for dev)", exc)
    settings = None


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


MAX_UPLOAD_BYTES = max(0, _parse_int_env("UPLOAD_MAX_BYTES", 20_000_000))  # 20 MB default
ALLOWED_UPLOAD_EXTS = {
    ext.strip().lower()
    for ext in (os.getenv("UPLOAD_ALLOWED_EXTS") or ".pdf,.txt,.md,.doc,.docx,.png,.jpg,.jpeg,.bmp,.tiff,.csv")
    .split(",")
    if ext.strip()
}
ALLOWED_MIME_PREFIXES = {
    prefix.strip().lower()
    for prefix in (os.getenv("UPLOAD_ALLOWED_MIME_PREFIXES") or "application/pdf,text/,image/")
    .split(",")
    if prefix.strip()
}
ENABLE_VIRUS_SCAN = _parse_bool_env("UPLOAD_ENABLE_VIRUS_SCAN", False)
BLOCKED_KEYWORDS = [
    keyword.strip().lower()
    for keyword in (os.getenv("UPLOAD_BLOCKED_KEYWORDS") or "").split(",")
    if keyword.strip()
]


class ChatReq(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = None
    remember: bool = False


class ChatResp(BaseModel):
    reply: str


class LoginReq(BaseModel):
    username: str
    password: str


class LoginResp(BaseModel):
    token: str
    expires_at: int
    user: str


class ThreadCreateReq(BaseModel):
    user_id: str
    title: Optional[str] = None
    thread_id: Optional[str] = None


class ThreadUpdateReq(BaseModel):
    title: Optional[str] = None


class IngestionJobCreateReq(BaseModel):
    user_id: str
    source: str
    items: List[str] = []
    tenant_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class ReindexReq(BaseModel):
    user_id: str
    document_ids: List[str] = []
    embedding_model: Optional[str] = None
    tenant_id: Optional[str] = None


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window_seconds: int = 60) -> None:
        self.max_calls = max_calls
        self.window = max(1, window_seconds)
        self._events: Dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = self._events.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.max_calls:
                return False
            bucket.append(now)
            return True


rate_limiter = SlidingWindowRateLimiter(API_RATE_LIMIT_PER_MINUTE)


def _extract_api_token(request: Request, authorization: Optional[str], x_api_key: Optional[str]) -> str:
    header_token = ""
    if authorization:
        prefix = "bearer "
        if authorization.lower().startswith(prefix):
            header_token = authorization[len(prefix):].strip()
    fallback = (x_api_key or "").strip()
    query_token = request.query_params.get("api_key", "").strip()
    return header_token or fallback or query_token


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _issue_session_token(username: str) -> Dict[str, Any]:
    if not AUTH_SECRET_KEY:
        raise RuntimeError("AUTH_SECRET_KEY is not configured.")
    exp = int(time.time()) + AUTH_TOKEN_TTL_SECONDS
    payload = f"{username}:{exp}"
    sig = hmac.new(AUTH_SECRET_KEY.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    token = f"{_b64url_encode(payload.encode('utf-8'))}.{_b64url_encode(sig)}"
    return {"token": token, "expires_at": exp, "user": username}


def _issue_dev_token(username: str) -> Dict[str, Any]:
    now = int(time.time())
    fallback_token = API_AUTH_TOKEN or "dev-mode-token"
    return {"token": fallback_token, "expires_at": now + 365 * 24 * 3600, "user": username}


def _verify_session_token(token: str) -> Optional[str]:
    if not token or not AUTH_SECRET_KEY:
        return None
    if "." not in token:
        return None
    payload_b64, sig_b64 = token.split(".", 1)
    try:
        payload_bytes = _b64url_decode(payload_b64)
        provided_sig = _b64url_decode(sig_b64)
    except Exception:
        return None
    payload = payload_bytes.decode("utf-8", errors="ignore")
    if ":" not in payload:
        return None
    username, exp_raw = payload.rsplit(":", 1)
    try:
        exp = int(exp_raw)
    except ValueError:
        return None
    expected_sig = hmac.new(AUTH_SECRET_KEY.encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        return None
    if exp < int(time.time()):
        return None
    return username.strip() or None


def require_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> str:
    provided = _extract_api_token(request, authorization, x_api_key)

    # Session token path (preferred if configured)
    if AUTH_SECRET_KEY:
        username = _verify_session_token(provided)
        if username:
            storage.upsert_user(username)
            return username

    # Legacy static token path
    if API_AUTH_TOKEN:
        if provided == API_AUTH_TOKEN:
            return "static-token"
        if AUTH_SECRET_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token.")

    # No auth configured — allow passthrough for dev
    if AUTH_SECRET_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required.")
    return provided


def enforce_rate_limit(
    request: Request,
    provided_token: str = Depends(require_api_key),
) -> str:
    subject = (
        provided_token
        or request.query_params.get("user_id")
        or (request.client.host if request.client else "anonymous")
    )
    if not rate_limiter.allow(subject):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    return provided_token


@app.post("/auth/login", response_model=LoginResp)
def login(payload: LoginReq):
    username = payload.username.strip()
    password = payload.password
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required.")

    if not AUTH_SECRET_KEY:
        # Dev mode: auto-create user if missing, otherwise verify
        if not storage.user_exists(username):
            storage.set_user_password(username, password)
        elif not storage.verify_user_password(username, password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")
        storage.record_login(username)
        token_info = _issue_dev_token(username)
        return LoginResp(**token_info)

    if username == AUTH_USERNAME and not storage.user_exists(username):
        if password == AUTH_PASSWORD:
            storage.set_user_password(username, password)

    if not storage.user_exists(username) or not storage.verify_user_password(username, password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    storage.record_login(username)
    token_info = _issue_session_token(username)
    return LoginResp(**token_info)


@app.post("/auth/register", response_model=LoginResp)
def register(payload: LoginReq):
    username = payload.username.strip()
    password = payload.password
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required.")
    if AUTH_SECRET_KEY and not ALLOW_USER_REGISTRATION:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Registration is disabled.")
    if storage.user_exists(username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists.")

    storage.set_user_password(username, password)
    storage.record_login(username)
    token_info = _issue_session_token(username) if AUTH_SECRET_KEY else _issue_dev_token(username)
    return LoginResp(**token_info)


@app.get("/auth/me")
def auth_me(user: str = Depends(require_api_key)):
    return {"status": "ok", "user": user}


def _build_graph_config(thread_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    configurable: Dict[str, Any] = {"thread_id": thread_id}
    if user_id:
        configurable["user_id"] = user_id
    return {"configurable": configurable, "recursion_limit": DEFAULT_RECURSION_LIMIT}


class ToolBudgetConfig(BaseModel):
    max_tasks: int
    max_parallel: int
    total_latency: float


class ToolBudgetUpdate(BaseModel):
    max_tasks: Optional[int] = None
    max_parallel: Optional[int] = None
    total_latency: Optional[float] = None


pending_interrupts: Dict[str, Dict[str, Any]] = {}


def _register_interrupt(session_id: str, payload: Dict[str, Any]) -> None:
    pending_interrupts[session_id] = payload


def _clear_interrupt(session_id: str) -> None:
    pending_interrupts.pop(session_id, None)


def _resolve_user_id(session_id: str, user_id: Optional[str], token_subject: Optional[str] = None) -> str:
    candidate = (user_id or "").strip()
    if not candidate and token_subject and token_subject != "static-token":
        candidate = token_subject.strip()
    resolved = candidate or session_id
    storage.upsert_user(resolved)
    return resolved


def _assert_user_scope(user_id: str, token_subject: Optional[str]) -> None:
    if AUTH_SECRET_KEY and token_subject and token_subject != "static-token" and token_subject != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token does not match user_id")

def _resolve_tenant_id(request: Request, tenant_id: Optional[str] = None) -> str:
    candidate = (tenant_id or "").strip()
    if not candidate:
        candidate = (request.headers.get("x-tenant-id") or "").strip()
    if not candidate:
        candidate = request.query_params.get("tenant_id", "").strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    return candidate


def _parse_bool_flag(value: Optional[Any]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _sanitize_user_folder(user_id: str) -> Path:
    safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in user_id) or '_default'
    folder = DOCS_ROOT / safe
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _is_remote_source(item: str) -> bool:
    lowered = item.lower()
    return lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("s3://")

def _safe_filename(name: Optional[str]) -> str:
    if not name:
        return 'upload.txt'
    clean = Path(name).name.strip()
    return clean or 'upload.txt'


def _scan_upload(content: bytes, filename: str) -> None:
    """Best-effort upload guard: keyword blocklist + optional antivirus hook."""
    if BLOCKED_KEYWORDS:
        try:
            text = content.decode("utf-8", errors="ignore").lower()
            for keyword in BLOCKED_KEYWORDS:
                if keyword and keyword in text:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Upload blocked by content policy: '{keyword}' detected.",
                    )
        except HTTPException:
            raise
        except Exception:  # pragma: no cover
            logger.warning("keyword scan failed filename=%s", filename, exc_info=True)

    if not ENABLE_VIRUS_SCAN:
        return
    try:
        import clamd  # type: ignore

        client = clamd.ClamdNetworkSocket()
        result = client.instream(io.BytesIO(content))
        verdict = result.get("stream") if isinstance(result, dict) else None
        if isinstance(verdict, tuple) and len(verdict) >= 2 and verdict[0] == "FOUND":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Virus detected: {verdict[1]}",
            )
    except HTTPException:
        raise
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("UPLOAD_ENABLE_VIRUS_SCAN=1 but clamd is not installed; skipping AV scan.")
    except Exception:  # pragma: no cover - best effort
        logger.warning("virus scan failed filename=%s", filename, exc_info=True)


def _validate_upload(file: UploadFile, content: bytes) -> Dict[str, Any]:
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    size_bytes = len(content)
    if MAX_UPLOAD_BYTES and size_bytes > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large: {size_bytes} bytes (limit {MAX_UPLOAD_BYTES}).",
        )

    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    mime_type = (file.content_type or "").split(";")[0].strip().lower() or None

    if ALLOWED_UPLOAD_EXTS and ext not in ALLOWED_UPLOAD_EXTS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension '{ext or 'unknown'}' is not allowed. Allowed: {allowed}",
        )
    if ALLOWED_MIME_PREFIXES and mime_type:
        if all(not mime_type.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
            allowed = ", ".join(sorted(ALLOWED_MIME_PREFIXES))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"MIME type '{mime_type}' is not allowed. Allowed prefixes: {allowed}",
            )

    _scan_upload(content, filename)
    return {"size_bytes": float(size_bytes), "mime_type": mime_type}

def _normalize_tool_payload(content: Any, tool_name: str) -> Dict[str, Any]:
    if isinstance(content, dict):
        payload = content
    elif isinstance(content, str):
        try:
            decoded = json.loads(content)
            payload = decoded if isinstance(decoded, dict) else {"observation": decoded}
        except json.JSONDecodeError:
            payload = {"observation": content}
    else:
        payload = {"observation": str(content)}

    payload.setdefault("tool", tool_name)
    payload.setdefault("status", "unknown")
    observation = payload.get("observation")
    if isinstance(observation, str):
        payload.setdefault("content", observation)
    elif observation is not None:
        payload.setdefault("content", json.dumps(observation, ensure_ascii=False))
    else:
        payload.setdefault("content", "")
    return payload


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Interrupt):
        return {
            "value": _to_serializable(value.value),
            "when": getattr(value, "when", "during"),
        }
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def _format_stream_payloads(
    node: str,
    data: Any,
    session_id: str,
    ai_state: Dict[str, str],
) -> Iterable[Dict[str, Any]]:
    if not isinstance(data, dict):
        yield {"event": node, "data": data}
        return

    if "__interrupt__" in data:
        for interrupt in data["__interrupt__"]:
            payload = getattr(interrupt, "value", interrupt)
            if isinstance(payload, dict):
                _register_interrupt(session_id, payload)
            yield {"event": "interrupt", "node": node, "payload": payload}
        return

    plan = data.get("plan")
    if plan is not None:
        event: Dict[str, Any] = {"event": "plan", "plan": plan, "node": node}
        if "current_step" in data:
            event["current_step"] = data["current_step"]
        if "active_step_id" in data:
            event["active_step_id"] = data["active_step_id"]
        yield event

    if data.get("pending_interrupt"):
        payload = data["pending_interrupt"]
        if isinstance(payload, dict):
            _register_interrupt(session_id, payload)
        yield {"event": "interrupt", "node": node, "payload": payload}
    elif "pending_interrupt" in data and session_id in pending_interrupts:
        _clear_interrupt(session_id)

    if data.get("last_step_update"):
        yield {
            "event": "step_update",
            "node": node,
            "payload": data["last_step_update"],
        }

    if "status" in data:
        yield {"event": "status", "node": node, "status": data["status"]}

    if "messages" in data:
        for msg_obj in data["messages"]:
            metadata = getattr(msg_obj, "additional_kwargs", {}) or {}
            if metadata.get("internal_only"):
                continue
            role = getattr(msg_obj, "type", None) or "unknown"
            content = getattr(msg_obj, "content", "")
            payload: Dict[str, Any] = {"event": node}

            if role == "ai" and isinstance(content, str):
                previous = ai_state.get("full", "")
                delta = content[len(previous) :] if content.startswith(previous) else content
                ai_state["full"] = content
                payload.update({"role": "ai", "delta": delta})
            elif role == "tool":
                tool_name = getattr(msg_obj, "name", "") or "tool"
                tool_payload = _normalize_tool_payload(content, tool_name)
                payload.update({"role": "tool", **tool_payload})
            else:
                payload.update({"role": role, "content": content})

            yield payload

    handled: Set[str] = {
        "plan",
        "pending_interrupt",
        "last_step_update",
        "status",
        "messages",
    }
    extra = {
        key: value
        for key, value in data.items()
        if key not in handled and not key.startswith("_")
    }
    if extra:
        yield {"event": node, "payload": extra}


@app.post("/chat", response_model=ChatResp)
def chat(body: ChatReq, provided_token: str = Depends(enforce_rate_limit)) -> ChatResp:
    logger.info("chat request session=%s message=%s", body.session_id, body.message)

    user_id = _resolve_user_id(body.session_id, body.user_id, provided_token)
    _assert_user_scope(user_id, provided_token)
    try:
        storage.ensure_chat_thread(user_id, thread_id=body.session_id, title=body.message[:80], last_message=body.message)
        storage.append_chat_message(body.session_id, user_id, "user", body.message)
    except Exception:
        logger.warning("chat history write failed session=%s user=%s", body.session_id, user_id, exc_info=True)

    initial_state: Dict[str, Any] = {
        "messages": [HumanMessage(content=body.message)],
        "user_id": user_id,
    }
    if body.remember:
        initial_state["remember_current"] = True

    result = compiled_graph.invoke(
        initial_state,
        config=_build_graph_config(body.session_id, user_id),
    )
    logger.info(
        "chat response session=%s messages=%s",
        body.session_id,
        len(result.get("messages", [])),
    )
    reply = "（暂未生成回复）"
    for message in reversed(result["messages"]):
        if getattr(message, "type", None) == "ai":
            reply = message.content
            break
    try:
        storage.append_chat_message(body.session_id, user_id, "ai", reply)
    except Exception:
        logger.warning("chat history write (ai) failed session=%s user=%s", body.session_id, user_id, exc_info=True)
    return ChatResp(reply=reply)


def _process_document_background(document_id: str, user_id: str, dest_path: Path, version_id: Optional[str] = None) -> None:
    # Deprecated placeholder retained for compatibility; now routed through Celery task queue.
    enqueue_process_document(
        document_id=document_id,
        user_id=user_id,
        path=str(dest_path),
        version_id=version_id,
        publish=True,
    )


@app.get("/chat/stream")
def chat_stream(
    session_id: str,
    message: str,
    user_id: Optional[str] = None,
    remember: Optional[str] = None,
    provided_token: str = Depends(enforce_rate_limit),
) -> StreamingResponse:
    resolved_user = _resolve_user_id(session_id, user_id, provided_token)
    _assert_user_scope(resolved_user, provided_token)
    remember_flag = _parse_bool_flag(remember)
    logger.info(
        "chat stream session=%s user=%s message=%s",
        session_id,
        resolved_user,
        message,
    )
    title_hint = message[:80] if isinstance(message, str) else ""
    try:
        storage.ensure_chat_thread(resolved_user, thread_id=session_id, title=title_hint, last_message=message)
        storage.append_chat_message(session_id, resolved_user, "user", message)
    except Exception:
        logger.warning("chat history write failed session=%s user=%s", session_id, resolved_user, exc_info=True)

    def event_gen():
        cfg = _build_graph_config(session_id, resolved_user)
        initial_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=message)],
            "user_id": resolved_user,
        }
        if remember_flag:
            initial_state["remember_current"] = True
        ai_state = {"full": ""}
        ended = False
        try:
            for update in compiled_graph.stream(
                initial_state,
                config=cfg,
                stream_mode="updates",
            ):
                for node, data in update.items():
                    for chunk in _format_stream_payloads(node, data, session_id, ai_state):
                        yield "data: " + json.dumps(_to_serializable(chunk), ensure_ascii=False) + "\n\n"
        except Exception as exc:  # pragma: no cover - SSE 仅做演示
            logger.exception(
                "chat stream error session=%s user=%s message=%s exc=%s",
                session_id,
                resolved_user,
                message,
                exc,
            )
            if isinstance(exc, KeyError) and exc.args and exc.args[0] == "__end__":
                yield 'data: {"event": "end"}\n\n'
                ended = True
            elif isinstance(exc, RuntimeError) and str(exc) == "Connection error.":
                yield 'data: {"event": "end"}\n\n'
                ended = True
            else:
                yield "data: " + json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False) + "\n\n"
        finally:
            ai_full = (ai_state.get("full") or "").strip() if isinstance(ai_state, dict) else ""
            if ai_full:
                try:
                    storage.append_chat_message(session_id, resolved_user, "ai", ai_full)
                except Exception:
                    logger.warning(
                        "chat history write (ai) failed session=%s user=%s",
                        session_id,
                        resolved_user,
                        exc_info=True,
                    )
            if not ended:
                yield 'data: {"event": "end"}\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/chat/continue")
def chat_continue(
    thread_id: str,
    action: str,
    user_id: Optional[str] = None,
    provided_token: str = Depends(enforce_rate_limit),
):
    verb = action.strip().lower()
    if verb not in {"continue", "cancel"}:
        raise HTTPException(status_code=400, detail="Unsupported action")

    base_cfg = _build_graph_config(thread_id)
    try:
        snapshot = compiled_graph.get_state(base_cfg)
    except ValueError as exc:  # no checkpoint available
        raise HTTPException(status_code=404, detail="Thread not found") from exc

    state_values = snapshot.values if isinstance(snapshot.values, dict) else {}
    resolved_user = _resolve_user_id(
        thread_id, user_id or state_values.get("user_id"), provided_token  # type: ignore[arg-type]
    )
    _assert_user_scope(resolved_user, provided_token)
    storage.ensure_chat_thread(resolved_user, thread_id=thread_id)
    logger.info(
        "resume request received thread=%s user=%s action=%s",
        thread_id,
        resolved_user,
        action,
    )
    base_cfg = _build_graph_config(thread_id, resolved_user)
    plan: List[Dict[str, Any]] = state_values.get("plan", [])  # type: ignore[assignment]
    if not plan:
        raise HTTPException(status_code=400, detail="No plan to resume")

    current_step = state_values.get("current_step", 0)
    if current_step >= len(plan):
        raise HTTPException(status_code=400, detail="Plan already finished")

    pending = pending_interrupts.get(thread_id) or state_values.get("pending_interrupt")
    if not pending:
        raise HTTPException(status_code=400, detail="No pending confirmation")

    plan_copy = [dict(step) for step in plan]
    step = dict(plan_copy[current_step])
    updates: Dict[str, Any] = {
        "plan": plan_copy,
        "current_step": current_step,
        "active_step_id": step.get("id"),
        "user_id": resolved_user,
    }

    if verb == "cancel":
        step["status"] = "cancelled"
        updates["plan"][current_step] = step
        updates["status"] = "cancelled"
        updates["pending_interrupt"] = None
        updates["last_step_update"] = {
            "step_id": step.get("id"),
            "status": "cancelled",
        }
        compiled_graph.update_state(base_cfg, updates, as_node="executor")
        _clear_interrupt(thread_id)

        def cancel_stream():
            payload = {
                "event": "cancelled",
                "plan": updates["plan"],
                "step_update": updates["last_step_update"],
            }
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            yield 'data: {"event": "end"}\n\n'

        return StreamingResponse(cancel_stream(), media_type="text/event-stream")

    instruction = (
        step.get("prepared_instruction")
        or pending.get("resume_instruction")
        or build_step_instruction(step, current_step, len(plan_copy))
    )
    step["confirmed"] = True
    step["status"] = "in_progress"
    if instruction:
        step["prepared_instruction"] = instruction
    plan_copy[current_step] = dict(step)
    updates["plan"] = [dict(item) for item in plan_copy]
    updates["pending_interrupt"] = None
    updates["status"] = "executing"
    updates["last_step_update"] = {
        "step_id": step.get("id"),
        "status": "in_progress",
        "action": "continue",
    }
    prompt_text = instruction or "继续执行当前步骤，保持节奏。"
    updates["messages"] = [SystemMessage(content=f"[resume] {prompt_text}")]

    logger.info(
        "resume payload thread=%s step=%s keys=%s",
        thread_id,
        step.get("id"),
        sorted(updates.keys()),
    )
    try:
        resume_config = compiled_graph.update_state(base_cfg, updates, as_node="executor")
        logger.info(
            "resume accepted thread=%s step=%s new_config=%s",
            thread_id,
            step.get("id"),
            resume_config,
        )
    except Exception:
        logger.exception("resume failed thread=%s payload=%r", thread_id, updates)
        raise
    _clear_interrupt(thread_id)

    def resume_stream():
        ai_state = {"full": ""}
        ended = False
        try:
            for update in compiled_graph.stream(
                None,
                config=resume_config,
                stream_mode="updates",
            ):
                for node, data in update.items():
                    for chunk in _format_stream_payloads(node, data, thread_id, ai_state):
                        yield "data: " + json.dumps(_to_serializable(chunk), ensure_ascii=False) + "\n\n"
        except Exception as exc:  # pragma: no cover
            logger.exception(
                "resume stream error thread=%s exc=%s",
                thread_id,
                exc,
            )
            if isinstance(exc, KeyError) and exc.args and exc.args[0] == "__end__":
                yield 'data: {"event": "end"}\n\n'
                ended = True
            elif isinstance(exc, RuntimeError) and str(exc) == "Connection error.":
                yield 'data: {"event": "end"}\n\n'
                ended = True
            else:
                yield "data: " + json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False) + "\n\n"
        finally:
            ai_full = (ai_state.get("full") or "").strip() if isinstance(ai_state, dict) else ""
            if ai_full:
                try:
                    storage.append_chat_message(thread_id, resolved_user, "ai", ai_full)
                except Exception:
                    logger.warning(
                        "resume history write (ai) failed thread=%s user=%s",
                        thread_id,
                        resolved_user,
                        exc_info=True,
                    )
            if not ended:
                yield 'data: {"event": "end"}\n\n'

    return StreamingResponse(resume_stream(), media_type="text/event-stream")


@app.get("/chat/threads")
def list_chat_threads(user_id: str, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    return {"items": storage.list_chat_threads(normalized_user)}


@app.post("/chat/threads")
def create_chat_thread(payload: ThreadCreateReq, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (payload.user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    thread_id = storage.ensure_chat_thread(
        normalized_user,
        thread_id=payload.thread_id,
        title=payload.title or "",
        last_message="",
    )
    return {"thread_id": thread_id, "title": payload.title or ""}


@app.patch("/chat/threads/{thread_id}")
def update_chat_thread(thread_id: str, payload: ThreadUpdateReq, user_id: str, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    title = (payload.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    if not storage.update_chat_thread_title(thread_id, normalized_user, title):
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "title": title}


@app.delete("/chat/threads/{thread_id}")
def delete_chat_thread(thread_id: str, user_id: str, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    if not storage.delete_chat_thread(thread_id, normalized_user):
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "status": "deleted"}


@app.get("/chat/threads/{thread_id}/messages")
def chat_thread_messages(
    thread_id: str,
    user_id: str,
    limit: int = 200,
    provided_token: str = Depends(enforce_rate_limit),
):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    items = storage.list_chat_messages(thread_id, normalized_user, limit=limit)
    return {"items": items}


@app.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    user_id: str = Form(...),
    tenant_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    provided_token: str = Depends(enforce_rate_limit),
):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    storage.upsert_user(normalized_user)

    document_id = uuid.uuid4().hex
    filename = _safe_filename(file.filename)
    user_dir = _sanitize_user_folder(normalized_user)
    destination = user_dir / f"{document_id}_{filename}"
    content = await file.read()
    meta = _validate_upload(file, content)
    destination.write_bytes(content)
    size_bytes: Optional[float] = meta.get("size_bytes")
    mime_type = meta.get("mime_type")

    version_id = storage.create_document(
        document_id=document_id,
        tenant_id=resolved_tenant,
        user_id=normalized_user,
        filename=filename,
        path=str(destination),
        status="processing",
        size_bytes=size_bytes,
        mime_type=mime_type,
        content_type=mime_type,
    )
    enqueue_process_document(
        document_id=document_id,
        user_id=normalized_user,
        path=str(destination),
        version_id=version_id,
        publish=True,
        tenant_id=resolved_tenant,
    )
    logger.info(
        "document uploaded doc_id=%s user=%s filename=%s bytes=%s",
        document_id,
        normalized_user,
        filename,
        len(content),
    )
    return {"id": document_id, "status": "processing"}


@app.post("/documents/{document_id}/versions/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document_version(
    request: Request,
    document_id: str,
    user_id: str = Form(...),
    tenant_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    provided_token: str = Depends(enforce_rate_limit),
):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user or (doc.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Document not found")

    filename = _safe_filename(file.filename)
    user_dir = _sanitize_user_folder(normalized_user)
    destination = user_dir / f"{uuid.uuid4().hex}_{filename}"
    content = await file.read()
    meta = _validate_upload(file, content)
    destination.write_bytes(content)
    size_bytes: Optional[float] = meta.get("size_bytes")
    mime_type = meta.get("mime_type")

    version_id = storage.create_document_version(
        document_id=document_id,
        parent_version_id=doc.get("latest_version_id"),
        status="processing",
        path_raw=str(destination),
        path_text=None,
        embedding_model=None,
        published=False,
    )
    enqueue_process_document(
        document_id=document_id,
        user_id=normalized_user,
        path=str(destination),
        version_id=version_id,
        publish=True,
        tenant_id=resolved_tenant,
    )
    logger.info(
        "document new version uploaded doc_id=%s version=%s user=%s filename=%s bytes=%s",
        document_id,
        version_id,
        normalized_user,
        filename,
        len(content),
    )
    return {"document_id": document_id, "version_id": version_id, "status": "processing"}


@app.get("/documents")
def list_documents(request: Request, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    storage.upsert_user(normalized_user)
    return {"items": storage.list_documents(normalized_user, tenant_id=resolved_tenant)}


@app.get("/documents/{document_id}")
def get_document(request: Request, document_id: str, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user or (doc.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/documents/{document_id}/versions")
def list_document_versions(request: Request, document_id: str, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user or (doc.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"items": storage.get_document_versions(document_id)}


@app.post("/documents/{document_id}/versions/{version_id}/publish")
def publish_document_version(request: Request, document_id: str, version_id: str, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user or (doc.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Document not found")
    version = storage.get_document_version(version_id)
    if not version or version.get("document_id") != document_id:
        raise HTTPException(status_code=404, detail="Version not found")
    storage.publish_document_version(document_id, version_id)
    return {"document_id": document_id, "version_id": version_id, "status": "published"}


@app.post("/documents/{document_id}/versions/{version_id}/rollback")
def rollback_document_version(request: Request, document_id: str, version_id: str, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    # rollback is equivalent to publishing a previous version
    return publish_document_version(request, document_id, version_id, user_id, tenant_id, provided_token)


@app.get("/documents/{document_id}/download")
def download_document(request: Request, document_id: str, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user or (doc.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Document not found")
    path = doc.get("path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Original file not found on server")
    media_type = doc.get("mime_type") or doc.get("content_type") or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=doc.get("filename") or "document")


@app.post("/documents/{document_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_document(
    request: Request,
    document_id: str,
    user_id: str,
    tenant_id: Optional[str] = None,
    provided_token: str = Depends(enforce_rate_limit),
):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user or (doc.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Document not found")
    path = doc.get("path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=400, detail="Document file is missing; please re-upload.")
    storage.update_document_status(document_id, status="processing", error=None, path=path)
    version_id = storage.get_latest_version_id(document_id)
    enqueue_process_document(
        document_id=document_id,
        user_id=normalized_user,
        path=path,
        version_id=version_id,
        publish=True,
        tenant_id=resolved_tenant,
    )
    logger.info("document retry queued doc_id=%s user=%s", document_id, normalized_user)
    return {"id": document_id, "status": "processing"}


@app.delete("/documents/{document_id}")
def delete_document(request: Request, document_id: str, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user or (doc.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Document not found")
    path = doc.get("path")
    storage.delete_document(document_id)
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:  # pragma: no cover
            logger.warning("failed to delete document file path=%s", path, exc_info=True)
    logger.info("document deleted doc_id=%s user=%s", document_id, normalized_user)
    return {"id": document_id, "status": "deleted"}


@app.post("/ingestion_jobs", status_code=status.HTTP_202_ACCEPTED)
def create_ingestion_job(request: Request, payload: IngestionJobCreateReq, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (payload.user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    items = payload.items or []
    resolved_tenant = _resolve_tenant_id(request, payload.tenant_id)
    job_id = storage.create_ingestion_job(
        user_id=normalized_user,
        tenant_id=resolved_tenant,
        source=payload.source,
        params=payload.params or {},
        status="queued",
        total=len(items),
    )
    task_ids: List[str] = []
    for item in items:
        is_remote = _is_remote_source(item)
        item_path = Path(item)
        if not is_remote and not item_path.exists():
            storage.update_ingestion_job(job_id, status="failed", error=f"File not found: {item}")
            return {"job_id": job_id, "status": "failed", "error": f"File not found: {item}"}
        filename = _safe_filename(item_path.name if not is_remote else item_path.name or Path(item).name or "remote_file")
        size_bytes = float(item_path.stat().st_size) if not is_remote else None
        mime_type = None
        document_id = uuid.uuid4().hex
        version_id = storage.create_document(
            document_id=document_id,
            tenant_id=resolved_tenant,
            user_id=normalized_user,
            filename=filename,
            path=str(item),
            status="processing",
            size_bytes=size_bytes,
            mime_type=mime_type,
            content_type=mime_type,
        )
        task_ids.append(
            storage.create_ingestion_task(
                job_id,
                document_id=document_id,
                version_id=version_id,
                step="ingest",
                status="queued",
            )
        )
    enqueue_ingestion_job(job_id, normalized_user, task_ids)
    return {"job_id": job_id, "status": "queued", "tasks": len(task_ids)}


@app.get("/ingestion_jobs")
def list_ingestion_jobs(request: Request, user_id: str, tenant_id: Optional[str] = None, limit: int = 50, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    items = [
        job for job in storage.list_ingestion_jobs(normalized_user, limit=limit)
        if job.get("tenant_id") in {None, resolved_tenant}
    ]
    return {"items": items}


@app.get("/ingestion_jobs/{job_id}")
def get_ingestion_job(request: Request, job_id: str, user_id: str, tenant_id: Optional[str] = None, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    job = storage.get_ingestion_job(job_id)
    if not job or job.get("user_id") != normalized_user or (job.get("tenant_id") not in {None, resolved_tenant}):
        raise HTTPException(status_code=404, detail="Job not found")
    tasks = storage.list_ingestion_tasks(job_id)
    return {"job": job, "tasks": tasks}


@app.post("/reindex", status_code=status.HTTP_202_ACCEPTED)
def trigger_reindex(request: Request, payload: ReindexReq, provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (payload.user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, payload.tenant_id)
    user_docs: Dict[str, Dict[str, Any]] = {doc["id"]: doc for doc in storage.list_documents(normalized_user, tenant_id=resolved_tenant)}
    doc_ids = [doc_id for doc_id in (payload.document_ids or []) if doc_id in user_docs]
    if not doc_ids:
        raise HTTPException(status_code=404, detail="No documents found for user")
    for doc_id in doc_ids:
        storage.update_document_status(doc_id, status="processing", error=None, path=user_docs[doc_id].get("path"))
    job_id = storage.create_ingestion_job(
        user_id=normalized_user,
        tenant_id=payload.tenant_id,
        source="reindex",
        params={"embedding_model": payload.embedding_model},
        status="queued",
        total=len(doc_ids),
    )
    task_ids: List[str] = []
    for doc_id in doc_ids:
        version_id = storage.create_document_version(
            doc_id,
            status="pending",
            path_raw=None,
            path_text=None,
            embedding_model=payload.embedding_model,
            published=False,
        )
        task_ids.append(
            storage.create_ingestion_task(
                job_id,
                document_id=doc_id,
                version_id=version_id,
                step="reindex",
                status="queued",
            )
        )
    enqueue_ingestion_job(job_id, normalized_user, task_ids)
    return {"job_id": job_id, "status": "queued", "tasks": len(task_ids)}


@app.get("/ops/vector/health")
def vector_health(request: Request, user_id: str, tenant_id: Optional[str] = None, document_ids: Optional[List[str]] = Query(None), provided_token: str = Depends(enforce_rate_limit)):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    _assert_user_scope(normalized_user, provided_token)
    resolved_tenant = _resolve_tenant_id(request, tenant_id)
    user_docs: Set[str] = {doc["id"] for doc in storage.list_documents(normalized_user, tenant_id=resolved_tenant)}
    ids = document_ids or []
    if ids:
        ids = [doc_id for doc_id in ids if doc_id in user_docs]
        if not ids:
            raise HTTPException(status_code=404, detail="Documents not found for user")
    else:
        ids = list(user_docs)
    items: List[Dict[str, Any]] = []
    index_size = user_index_size(normalized_user)
    index_doc_ids = index_ids_for_user(normalized_user)
    for doc_id in ids:
        versions = storage.get_document_versions(doc_id)
        published_version = next((v for v in versions if v.get("published")), None)
        active_version_id = (published_version or (versions[0] if versions else {})).get("id")
        expected = 0
        stored = 0
        if published_version:
            expected = published_version.get("chunk_count") or storage.count_embeddings_for_version(published_version["id"])
        else:
            latest = versions[0] if versions else None
            if latest:
                expected = latest.get("chunk_count") or storage.count_embeddings_for_version(latest["id"])
        # approximate stored by counting chunks for the published version
        if published_version:
            stored = storage.count_embeddings_for_version(published_version["id"])
        elif active_version_id:
            stored = storage.count_embeddings_for_version(active_version_id)
        health_status = "healthy"
        notes: Optional[str] = None
        if expected == 0:
            health_status = "unknown"
            notes = "no published chunks found."
        elif stored < expected:
            health_status = "degraded"
            notes = f"stored vectors {stored} < expected {expected}."
        if doc_id not in index_doc_ids and expected > 0:
            health_status = "degraded"
            notes = (notes or "") + " index missing document vectors."
        storage.upsert_vector_stats(
            tenant_id=tenant_id,
            document_id=doc_id,
            version_id=active_version_id,
            expected_vectors=expected,
            stored_vectors=stored,
            health_status=health_status,
            notes=notes,
        )
        items.append(
            {
                "document_id": doc_id,
                "version_id": active_version_id,
                "expected_vectors": expected,
                "stored_vectors": stored,
                "health_status": health_status,
                "notes": notes,
                "published": bool(published_version),
                "in_index": doc_id in index_doc_ids,
                "user_index_vectors": index_size,
            }
        )
    return {"items": items}


@app.get("/config/tool-budget", response_model=ToolBudgetConfig)
def read_tool_budget(_: str = Depends(enforce_rate_limit)):
    settings = get_tool_budget_settings()
    return ToolBudgetConfig(**settings.as_dict())


@app.patch("/config/tool-budget", response_model=ToolBudgetConfig)
def patch_tool_budget(payload: ToolBudgetUpdate, _: str = Depends(enforce_rate_limit)):
    updates = payload.dict(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No changes submitted.")
    new_settings = update_tool_budget_settings(**updates)
    logger.info("tool budget patched fields=%s", sorted(updates.keys()))
    return ToolBudgetConfig(**new_settings.as_dict())


@app.get("/metrics")
def metrics():
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
