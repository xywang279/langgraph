# app/tools.py
from __future__ import annotations

import ast
import json
import logging
import operator as op
import os
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Deque, Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field

from .rag import retrieve

load_dotenv()

logger = logging.getLogger(__name__)

USER_AGENT = "langgraph-agent/1.0"
_SESSION = requests.Session()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.5, value)


@dataclass(frozen=True)
class AuthConfig:
    mode: Literal["none", "bearer", "header"] = "none"
    env_var: Optional[str] = None
    header_name: str = "Authorization"
    prefix: Optional[str] = None

    def token(self) -> str:
        if not self.env_var:
            return ""
        return os.getenv(self.env_var, "").strip()

    def headers(self) -> Dict[str, str]:
        if self.mode == "none":
            return {}
        token = self.token()
        if not token:
            return {}
        if self.mode == "bearer":
            return {self.header_name: f"Bearer {token}"}
        prefix = (self.prefix or "").strip()
        value = f"{prefix} {token}".strip() if prefix else token
        return {self.header_name: value}


@dataclass(frozen=True)
class RateLimitPolicy:
    max_calls: int
    per_seconds: float


@dataclass(frozen=True)
class ToolRuntimeMetadata:
    name: str
    auth: AuthConfig
    rate_limit: Optional[RateLimitPolicy] = None
    timeout: float = 8.0


class RateLimiter:
    def __init__(self) -> None:
        self._events: Dict[str, Deque[float]] = {}
        self._locks: Dict[str, Lock] = {}

    def acquire(self, tool_name: str, policy: RateLimitPolicy) -> None:
        now = time.monotonic()
        lock = self._locks.setdefault(tool_name, Lock())
        with lock:
            events = self._events.setdefault(tool_name, deque())
            window = max(policy.per_seconds, 0.5)
            while events and now - events[0] >= window:
                events.popleft()
            if len(events) >= policy.max_calls:
                TOOL_RATE_LIMITED.labels(tool=tool_name).inc()
                raise RuntimeError(
                    f"Rate limit exceeded for '{tool_name}' "
                    f"({policy.max_calls} calls/{window:.0f}s)."
                )
            events.append(now)


TOOL_CALLS = Counter(
    "tool_invocations_total",
    "Total number of tool invocations grouped by status.",
    ("tool", "status"),
)
TOOL_ERRORS = Counter(
    "tool_invocation_errors_total",
    "Tool invocation failures grouped by exception type.",
    ("tool", "error"),
)
TOOL_DURATION = Histogram(
    "tool_invocation_duration_seconds",
    "Latency histogram per tool.",
    ("tool",),
    buckets=(
        0.05,
        0.1,
        0.2,
        0.5,
        1,
        2,
        5,
        10,
        20,
    ),
)
TOOL_RATE_LIMITED = Counter(
    "tool_rate_limited_total",
    "Times a tool call was rejected by its local rate limiter.",
    ("tool",),
)

HTTP_SEARCH_BASE_URL = os.getenv("HTTP_SEARCH_BASE_URL", "").rstrip("/")
HTTP_SEARCH_TIMEOUT = _env_float("HTTP_SEARCH_TIMEOUT", 10.0)
HTTP_SEARCH_RATE = _env_int("HTTP_SEARCH_MAX_CALLS_PER_MINUTE", 12)

INTERNAL_API_BASE_URL = os.getenv("INTERNAL_API_BASE_URL", "").rstrip("/")
INTERNAL_API_TIMEOUT = _env_float("INTERNAL_API_TIMEOUT", 6.0)
INTERNAL_API_RATE = _env_int("INTERNAL_API_MAX_CALLS_PER_MINUTE", 30)

WORKFLOW_API_BASE_URL = os.getenv("WORKFLOW_API_BASE_URL", "").rstrip("/")
WORKFLOW_API_TIMEOUT = _env_float("WORKFLOW_API_TIMEOUT", 6.0)
WORKFLOW_RATE = _env_int("WORKFLOW_TRIGGER_MAX_CALLS_PER_MINUTE", 10)

DOC_INSIGHTS_TIMEOUT = _env_float("DOC_INSIGHTS_TIMEOUT", 8.0)
DOC_INSIGHTS_RATE = _env_int("DOC_INSIGHTS_MAX_CALLS_PER_MINUTE", 30)

CALC_RATE = _env_int("CALC_MAX_CALLS_PER_MINUTE", 60)
NOW_RATE = _env_int("NOW_MAX_CALLS_PER_MINUTE", 60)


def _parse_default_params(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {str(key): str(value) for key, value in parsed.items()}
    params: Dict[str, str] = {}
    for chunk in raw.split("&"):
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
        else:
            key, value = chunk, ""
        params[key.strip()] = value.strip()
    return params


HTTP_SEARCH_AUTH_MODE = os.getenv("HTTP_SEARCH_AUTH_MODE", "bearer").strip().lower()
if HTTP_SEARCH_AUTH_MODE not in {"bearer", "query", "none"}:
    HTTP_SEARCH_AUTH_MODE = "bearer"
HTTP_SEARCH_API_KEY_PARAM = os.getenv("HTTP_SEARCH_API_KEY_PARAM", "api_key").strip() or "api_key"
HTTP_SEARCH_DEFAULT_PARAMS = _parse_default_params(os.getenv("HTTP_SEARCH_DEFAULT_PARAMS", ""))

TOOL_RUNTIME: Dict[str, ToolRuntimeMetadata] = {
    "calc": ToolRuntimeMetadata(
        name="calc",
        auth=AuthConfig(),
        rate_limit=RateLimitPolicy(max_calls=CALC_RATE, per_seconds=60.0),
        timeout=_env_float("CALC_TIMEOUT", 3.0),
    ),
    "now": ToolRuntimeMetadata(
        name="now",
        auth=AuthConfig(),
        rate_limit=RateLimitPolicy(max_calls=NOW_RATE, per_seconds=60.0),
        timeout=_env_float("NOW_TIMEOUT", 2.0),
    ),
    "http_search": ToolRuntimeMetadata(
        name="http_search",
        auth=AuthConfig(mode="bearer", env_var="HTTP_SEARCH_API_KEY"),
        rate_limit=RateLimitPolicy(max_calls=HTTP_SEARCH_RATE, per_seconds=60.0),
        timeout=HTTP_SEARCH_TIMEOUT,
    ),
    "internal_api": ToolRuntimeMetadata(
        name="internal_api",
        auth=AuthConfig(mode="bearer", env_var="INTERNAL_API_TOKEN"),
        rate_limit=RateLimitPolicy(max_calls=INTERNAL_API_RATE, per_seconds=60.0),
        timeout=INTERNAL_API_TIMEOUT,
    ),
    "doc_insights": ToolRuntimeMetadata(
        name="doc_insights",
        auth=AuthConfig(),
        rate_limit=RateLimitPolicy(max_calls=DOC_INSIGHTS_RATE, per_seconds=60.0),
        timeout=DOC_INSIGHTS_TIMEOUT,
    ),
    "workflow_trigger": ToolRuntimeMetadata(
        name="workflow_trigger",
        auth=AuthConfig(mode="bearer", env_var="WORKFLOW_API_TOKEN"),
        rate_limit=RateLimitPolicy(max_calls=WORKFLOW_RATE, per_seconds=60.0),
        timeout=WORKFLOW_API_TIMEOUT,
    ),
}

_RATE_LIMITER = RateLimiter()


def _guarded_call(tool_name: str, func: Callable[[], Any]) -> Any:
    metadata = TOOL_RUNTIME.get(tool_name)
    if metadata and metadata.rate_limit:
        _RATE_LIMITER.acquire(tool_name, metadata.rate_limit)
    start = time.perf_counter()
    try:
        result = func()
    except Exception as exc:
        duration = time.perf_counter() - start
        TOOL_DURATION.labels(tool=tool_name).observe(duration)
        TOOL_CALLS.labels(tool=tool_name, status="error").inc()
        TOOL_ERRORS.labels(tool=tool_name, error=exc.__class__.__name__).inc()
        raise
    else:
        duration = time.perf_counter() - start
        TOOL_DURATION.labels(tool=tool_name).observe(duration)
        TOOL_CALLS.labels(tool=tool_name, status="success").inc()
        return result


def _format_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return str(payload)


class HttpSearchClient:
    def __init__(
        self,
        base_url: str,
        auth: AuthConfig,
        timeout: float,
        *,
        auth_mode: str = "bearer",
        api_key_param: str = "api_key",
        default_params: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        self.auth_mode = auth_mode
        self.api_key_param = api_key_param
        self.default_params = default_params or {}

    def _build_url(self) -> str:
        if not self.base_url:
            raise RuntimeError("HTTP_SEARCH_BASE_URL is not configured.")
        if self.base_url.endswith("/search"):
            return self.base_url
        return f"{self.base_url}/search"

    def search(self, query: str, limit: int) -> List[Dict[str, str]]:
        url = self._build_url()
        headers = {"User-Agent": USER_AGENT}
        params = dict(self.default_params)
        params.update({"q": query, "limit": limit})

        if self.auth_mode == "query":
            token = self.auth.token()
            if token:
                params[self.api_key_param] = token
        else:
            headers.update(self.auth.headers())

        try:
            resp = _SESSION.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network path
            raise RuntimeError(f"HTTP search failed: {exc}") from exc

        try:
            payload = resp.json()
        except ValueError:
            text = resp.text.strip()
            return [
                {
                    "title": "Raw response",
                    "url": url,
                    "snippet": text[:500],
                }
            ]

        items: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            candidate_keys = [
                "results",
                "data",
                "items",
                "organic_results",
                "news_results",
                "inline_videos",
            ]
            for key in candidate_keys:
                value = payload.get(key)
                if isinstance(value, list):
                    items = [item for item in value if isinstance(item, dict)]
                    if items:
                        break
            if not items:
                answer_box = payload.get("answer_box")
                if isinstance(answer_box, dict):
                    items = [answer_box]
                else:
                    items = [payload]
        elif isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        else:
            items = [{"title": "Response", "snippet": str(payload)}]

        results: List[Dict[str, str]] = []
        for item in items[:limit]:
            title = (
                item.get("title")
                or item.get("name")
                or item.get("question")
                or item.get("headline")
                or "Untitled result"
            )
            snippet = (
                item.get("snippet")
                or item.get("summary")
                or item.get("description")
                or item.get("answer")
                or ""
            )
            results.append(
                {
                    "title": str(title),
                    "url": str(item.get("url") or item.get("link") or ""),
                    "snippet": str(snippet),
                }
            )
        return results


class InternalAPIClient:
    def __init__(self, base_url: str, auth: AuthConfig, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout

    def request(
        self,
        *,
        path: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        if not self.base_url:
            raise RuntimeError("INTERNAL_API_BASE_URL is not configured.")
        sanitized_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{sanitized_path}"
        method_upper = method.upper()
        if method_upper not in {"GET", "POST", "PUT", "PATCH"}:
            raise ValueError("method must be GET, POST, PUT or PATCH.")
        headers = {"User-Agent": USER_AGENT}
        headers.update(self.auth.headers())
        if user_id:
            headers["X-User-Id"] = user_id
        try:
            resp = _SESSION.request(
                method_upper,
                url,
                params=params or None,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Internal API request failed: {exc}") from exc

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return _format_json(resp.json())
        return resp.text.strip()


class WorkflowDispatcher:
    def __init__(self, base_url: str, auth: AuthConfig, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout

    def trigger(
        self,
        *,
        workflow: str,
        payload: Dict[str, Any],
        priority: Literal["low", "normal", "high"],
        user_id: Optional[str],
    ) -> str:
        if not self.base_url:
            raise RuntimeError("WORKFLOW_API_BASE_URL is not configured.")
        url = f"{self.base_url}/workflows/trigger"
        headers = {"User-Agent": USER_AGENT, **self.auth.headers()}
        body = {
            "workflow": workflow,
            "payload": payload or {},
            "priority": priority,
            "requested_by": user_id,
        }
        try:
            resp = _SESSION.post(url, json=body, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Workflow trigger failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError:
            return f"Workflow '{workflow}' accepted."
        run_id = data.get("run_id") or data.get("id") or "n/a"
        status = data.get("status") or "accepted"
        return f"Workflow '{workflow}' accepted as run {run_id} (status={status})."


_HTTP_SEARCH_CLIENT = HttpSearchClient(
    HTTP_SEARCH_BASE_URL,
    TOOL_RUNTIME["http_search"].auth,
    HTTP_SEARCH_TIMEOUT,
    auth_mode=HTTP_SEARCH_AUTH_MODE,
    api_key_param=HTTP_SEARCH_API_KEY_PARAM,
    default_params=HTTP_SEARCH_DEFAULT_PARAMS,
)
_INTERNAL_API_CLIENT = InternalAPIClient(
    INTERNAL_API_BASE_URL,
    TOOL_RUNTIME["internal_api"].auth,
    INTERNAL_API_TIMEOUT,
)
_WORKFLOW_DISPATCHER = WorkflowDispatcher(
    WORKFLOW_API_BASE_URL,
    TOOL_RUNTIME["workflow_trigger"].auth,
    WORKFLOW_API_TIMEOUT,
)


# ---- 安全四则运算 ----
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
}


class CalcArgs(BaseModel):
    expression: str = Field(
        ...,
        description="Arithmetic expression using + - * / and parentheses, e.g. '1 + 2*(3-1)'.",
    )


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n  # type: ignore[return-value]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_ast(node.operand))  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](
            _eval_ast(node.left),  # type: ignore[arg-type]
            _eval_ast(node.right),  # type: ignore[arg-type]
        )
    raise ValueError("Only + - * / and parentheses are supported.")


@tool("calc", args_schema=CalcArgs)
def calc(expression: str) -> str:
    """安全计算器，支持 + - * / 和括号。"""

    def _impl() -> str:
        node = ast.parse(expression, mode="eval").body
        return str(_eval_ast(node))

    return _guarded_call("calc", _impl)


class NowArgs(BaseModel):
    fmt: str = Field(
        "%Y-%m-%d %H:%M",
        description="Python strftime format string, defaults to '%Y-%m-%d %H:%M'.",
    )


@tool("now", args_schema=NowArgs)
def now(fmt: str = "%Y-%m-%d %H:%M") -> str:
    """返回当前时间，支持自定义 strftime 格式。"""

    def _impl() -> str:
        return time.strftime(fmt)

    return _guarded_call("now", _impl)


class HttpSearchArgs(BaseModel):
    query: str = Field(..., description="Keyword or sentence to search against online sources.")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of snippets to return.")


@tool("http_search", args_schema=HttpSearchArgs)
def http_search(query: str, max_results: int = 5) -> str:
    """面向互联网/HTTP 服务的检索工具，支持通过配置的 API 获取最新资料。"""

    def _impl() -> str:
        normalized = (query or "").strip()
        if not normalized:
            raise ValueError("query is required.")
        hits = _HTTP_SEARCH_CLIENT.search(normalized, max_results)
        if not hits:
            return "Search completed but no documents were returned."
        lines: List[str] = []
        for idx, item in enumerate(hits, start=1):
            title = item.get("title") or "Untitled"
            url = item.get("url")
            snippet = (item.get("snippet") or "").strip()
            link = f" ({url})" if url else ""
            snippet = snippet[:400]
            lines.append(f"{idx}. {title}{link}\n   {snippet}".rstrip())
        return "\n".join(lines)

    return _guarded_call("http_search", _impl)


class InternalAPIArgs(BaseModel):
    path: str = Field(
        ...,
        description="Relative API path, e.g. '/tickets/INC-123' or 'orders/2023-09'.",
    )
    method: Literal["GET", "POST", "PUT", "PATCH"] = Field(
        "GET", description="HTTP method to use."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query parameters passed to the internal API.",
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON body for write operations.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="自动注入的用户标识，用于打到 X-User-Id header。",
    )


@tool("internal_api", args_schema=InternalAPIArgs)
def internal_api(
    path: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> str:
    """访问企业内部 API（如 CRM/ITSM/库存等），自动加上鉴权与速率保护。"""

    def _impl() -> str:
        normalized_path = (path or "").strip()
        if not normalized_path:
            raise ValueError("path is required.")
        return _INTERNAL_API_CLIENT.request(
            path=normalized_path,
            method=method,
            params=params or {},
            payload=payload,
            user_id=user_id,
        )

    return _guarded_call("internal_api", _impl)


class DocInsightsArgs(BaseModel):
    query: str = Field(..., description="Question or topic to search in uploaded documents.")
    top_k: int = Field(3, ge=1, le=10, description="Maximum snippets to fetch.")
    min_score: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to keep a chunk.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="会话用户，通常由 orchestrator 自动注入。",
    )


@tool("doc_insights", args_schema=DocInsightsArgs)
def doc_insights(
    query: str,
    top_k: int = 3,
    min_score: float = 0.2,
    user_id: Optional[str] = None,
) -> str:
    """结合用户上传文档 / 记忆库做向量检索，返回高分段落。"""

    def _impl() -> str:
        normalized_query = (query or "").strip()
        if not normalized_query:
            raise ValueError("query is required.")
        if not user_id:
            raise ValueError("user_id is required for doc_insights.")
        records = retrieve(
            user_id=user_id,
            query=normalized_query,
            top_k=top_k,
            min_score=min_score,
        )
        if not records:
            return "No matching documents were found for the given query."
        lines: List[str] = []
        for idx, record in enumerate(records, start=1):
            meta = record.get("metadata") or {}
            title = meta.get("filename") or meta.get("document_id") or "document"
            score = record.get("score")
            snippet = (record.get("content") or "").strip()
            snippet = snippet[:500]
            lines.append(
                f"{idx}. {title} (score={score})\n   {snippet}".rstrip()
            )
        return "\n".join(lines)

    return _guarded_call("doc_insights", _impl)


class WorkflowTriggerArgs(BaseModel):
    workflow: str = Field(..., description="Workflow identifier, e.g. 'publish_report'.")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parameters passed downstream.",
    )
    priority: Literal["low", "normal", "high"] = Field(
        "normal", description="Queue priority for the dispatcher."
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Initiator user id, orchestrator 会自动注入。",
    )


@tool("workflow_trigger", args_schema=WorkflowTriggerArgs)
def workflow_trigger(
    workflow: str,
    payload: Optional[Dict[str, Any]] = None,
    priority: Literal["low", "normal", "high"] = "normal",
    user_id: Optional[str] = None,
) -> str:
    """触发下游自动化流程/工单/发布流水线，并记录审计信息。"""

    def _impl() -> str:
        normalized = (workflow or "").strip()
        if not normalized:
            raise ValueError("workflow is required.")
        return _WORKFLOW_DISPATCHER.trigger(
            workflow=normalized,
            payload=payload or {},
            priority=priority,
            user_id=user_id,
        )

    return _guarded_call("workflow_trigger", _impl)


def get_registered_tools():
    """返回一个 Tool 列表供 LangGraph/LangChain 注册。"""
    return [calc, now, http_search, internal_api, doc_insights, workflow_trigger]


TOOL_REGISTRY: Dict[str, Any] = {
    "calc": calc,
    "now": now,
    "http_search": http_search,
    "internal_api": internal_api,
    "doc_insights": doc_insights,
    "workflow_trigger": workflow_trigger,
}
