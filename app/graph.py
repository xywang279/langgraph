from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Annotated, Callable, Dict, List, Literal, Optional, Set, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover - optional dependency
    SqliteSaver = None
from langgraph.types import interrupt, Command
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ValidationError
from prometheus_client import Counter

# ---- tools_condition: 多版本兼容导入（新 → 旧 → 本地兜底）----
try:
    # 常见新版本：langgraph>=0.3
    from langgraph.prebuilt import tools_condition  # type: ignore
except Exception:
    try:
        # 老版本：langgraph 0.1~0.2
        from langgraph.prebuilt.tool_node import tools_condition  # type: ignore
    except Exception:
        # 兜底实现：根据最后一条 AIMessage 是否包含 tool_calls 来路由
        def tools_condition(
            state: Dict[str, Any],
            *,
            messages_key: str = "messages",
        ) -> Literal["tools", "__end__"]:
            msgs: List[Any] = state.get(messages_key, [])
            if not msgs:
                return "__end__"
            last = msgs[-1]
            # LangChain 的 AIMessage 一般有 .tool_calls
            tc = getattr(last, "tool_calls", None)
            if isinstance(tc, list) and tc:
                return "tools"
            # 部分提供商把 tool_calls 放在 additional_kwargs
            ak = getattr(last, "additional_kwargs", {}) or {}
            if isinstance(ak.get("tool_calls"), list) and ak["tool_calls"]:
                return "tools"
            return "__end__"

from . import storage
from .rag import retrieve, store_memory_snippet
from .tools import TOOL_REGISTRY, get_registered_tools

load_dotenv()
logger = logging.getLogger(__name__)
logger.info("HTTP_SEARCH_BASE_URL=%s", os.getenv("HTTP_SEARCH_BASE_URL"))

# ---- LLM & tools registration ----
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY)
TOOLS = get_registered_tools()
llm_with_tools = llm.bind_tools(TOOLS)
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL") or MODEL
summary_llm = ChatOpenAI(model=SUMMARY_MODEL, base_url=BASE_URL, api_key=API_KEY)
logger = logging.getLogger(__name__)

ENABLE_DOCUMENT_RETRIEVAL = os.getenv("ENABLE_DOCUMENT_RETRIEVAL", "true").strip().lower()
ENABLE_DOCUMENT_RETRIEVAL = ENABLE_DOCUMENT_RETRIEVAL not in {"0", "false", "no"}

CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH") or str(
    Path(__file__).resolve().parent.parent / "data" / "checkpoints.sqlite3"
)
CHECKPOINT_DB_PATH = os.path.abspath(CHECKPOINT_DB_PATH)

SYSTEM_PROMPT = (
    "你是一名公众号写作助手。"
    "当用户在指令中点名某个工具（例如“用 now”“调用 calc”）时，"
    "必须为每个被点名的工具分别创建 tool_call，并在需要时并行执行。"
    "若多个工具能同步运行，请一次性返回全部所需的 tool_calls，避免只调用其中之一。"
    "生成最终回答前，务必整理所有工具结果并明确标注其来源。"
)

PLANNER_PROMPT = (
    "请将用户的任务拆解为 3-5 个可执行步骤，"
    "明确每一步需要产出的结果与可能使用的工具。"
    "如果某一步需要用户确认才能继续，请将 requires_confirmation 置为 true。"
    "可调用的工具 ID 包括：now（获取当前时间）、calc（算式计算）、http_search（在线检索）、"
    "doc_insights（文档向量检索）、internal_api（内部系统接口）和 workflow_trigger（触发工作流）。"
    "当某个步骤需要某个工具时，请在 tool_names 字段中填入准确的工具 ID（例如 \"now\"），避免使用概念化的名字。"
    "最终严格输出 JSON 结构，供后续执行器使用。"
)

# 如果项目里未定义 SUMMARY_SYSTEM_PROMPT，这里给一个兜底，避免 NameError
try:
    SUMMARY_SYSTEM_PROMPT
except NameError:
    SUMMARY_SYSTEM_PROMPT = "你是会话摘要器，请基于给定的 summary 与最近一轮对话生成简洁的短期对话摘要。"

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder("context_messages"),
        MessagesPlaceholder("conversation"),
    ]
)

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_PROMPT),
        ("human", "{goal}"),
    ]
)

planner_llm = llm.bind(response_format={"type": "json_object"})


@dataclass(frozen=True)
class ToolExecutionSpec:
    name: str
    priority: int = 5  # smaller => higher priority
    timeout: float = 6.0
    retries: int = 0
    backoff: float = 0.4
    exclusive: bool = False
    fallback: Optional[str] = None


TOOL_SPECS: Dict[str, ToolExecutionSpec] = {
    "http_search": ToolExecutionSpec(
        name="http_search",
        priority=1,
        timeout=10.0,
        retries=1,
        backoff=0.8,
        fallback="doc_insights",
    ),
    "doc_insights": ToolExecutionSpec(
        name="doc_insights",
        priority=2,
        timeout=8.0,
    ),
    "internal_api": ToolExecutionSpec(
        name="internal_api",
        priority=2,
        timeout=8.0,
        retries=1,
        backoff=0.6,
    ),
    "workflow_trigger": ToolExecutionSpec(
        name="workflow_trigger",
        priority=3,
        timeout=6.0,
        exclusive=True,
    ),
    "calc": ToolExecutionSpec(
        name="calc",
        priority=4,
        timeout=2.0,
    ),
    "now": ToolExecutionSpec(
        name="now",
        priority=5,
        timeout=2.0,
    ),
}

MAX_PARALLEL_WORKERS = int(os.getenv("TOOL_MAX_WORKERS", "4"))


@dataclass(frozen=True)
class ToolExecutionBudget:
    max_tasks: int = 6
    max_parallel: int = 3
    total_latency: float = 12.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_tasks": self.max_tasks,
            "max_parallel": self.max_parallel,
            "total_latency": self.total_latency,
        }


def _clamp_positive(value: int, *, floor: int = 1, ceil: Optional[int] = None) -> int:
    cap = ceil if ceil is not None else value
    return max(floor, min(cap, value))


class ToolBudgetManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings = ToolExecutionBudget(
            max_tasks=_clamp_positive(int(os.getenv("TOOL_MAX_TASKS", "6")), floor=1),
            max_parallel=_clamp_positive(
                int(os.getenv("TOOL_MAX_PARALLEL", "3")), floor=1, ceil=MAX_PARALLEL_WORKERS
            ),
            total_latency=float(os.getenv("TOOL_MAX_LATENCY", "12")),
        )

    def get(self) -> ToolExecutionBudget:
        with self._lock:
            return replace(self._settings)

    def update(
        self,
        *,
        max_tasks: Optional[int] = None,
        max_parallel: Optional[int] = None,
        total_latency: Optional[float] = None,
    ) -> ToolExecutionBudget:
        with self._lock:
            settings = self._settings
            if max_tasks is not None:
                settings = replace(settings, max_tasks=_clamp_positive(max_tasks, floor=1))
            if max_parallel is not None:
                settings = replace(
                    settings,
                    max_parallel=_clamp_positive(
                        max_parallel, floor=1, ceil=MAX_PARALLEL_WORKERS
                    ),
                )
            if total_latency is not None:
                safe_latency = max(0.0, float(total_latency))
                settings = replace(settings, total_latency=safe_latency)
            self._settings = settings
            logger.info(
                "tool budget updated max_tasks=%s max_parallel=%s total_latency=%s",
                settings.max_tasks,
                settings.max_parallel,
                settings.total_latency,
            )
            return replace(self._settings)


TOOL_THROTTLE_COUNTER = Counter(
    "tool_throttle_events_total",
    "Number of tool calls skipped due to guardrails",
    ("reason",),
)
TOOL_LATENCY_SECONDS = Counter(
    "tool_latency_seconds_total",
    "Total time spent executing tools (wall-clock seconds)",
)
TOOL_LATENCY_BUDGET_EXHAUSTED = Counter(
    "tool_latency_budget_exhausted_total",
    "Times the orchestration latency budget was exhausted mid-run",
)

tool_budget_manager = ToolBudgetManager()


def get_tool_budget_settings() -> ToolExecutionBudget:
    return tool_budget_manager.get()


def update_tool_budget_settings(**kwargs: Any) -> ToolExecutionBudget:
    return tool_budget_manager.update(**kwargs)


_SHARED_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS)
atexit.register(_SHARED_TOOL_EXECUTOR.shutdown)


def _get_spec(tool_name: str) -> ToolExecutionSpec:
    return TOOL_SPECS.get(tool_name, ToolExecutionSpec(name=tool_name))


TOOL_HINTS = {
    "now": ["now", "当前时间", "现在几点", "现在时间", "time"],
    "calc": ["calc", "计算", "算一下", "结果是多少", "算出", "求值"],
    "http_search": [
        "http_search",
        "搜索",
        "在线检索",
        "新闻",
        "最新信息",
        "查一下",
        "查找",
        "search",
        "lookup",
        "latest update",
        "最新更新",
        "use the tools",
    ],
    "doc_insights": ["文档", "资料", "上传内容", "insight", "记忆", "kb"],
    "internal_api": ["内部接口", "ticket", "订单", "inventory", "internal_api"],
    "workflow_trigger": ["工作流", "触发流程", "自动化", "发布", "workflow"],
}


def _context_with_user(state: "AgentState") -> Dict[str, Any]:
    user_id = state.get("user_id")
    return {"user_id": user_id} if user_id else {}


TOOL_CONTEXT_PROVIDERS: Dict[str, Callable[["AgentState"], Dict[str, Any]]] = {
    "doc_insights": _context_with_user,
    "internal_api": _context_with_user,
    "workflow_trigger": _context_with_user,
}


class PlanStepModel(BaseModel):
    title: str = Field(..., description="步骤的标题")
    description: str = Field(..., description="执行说明，描述要完成的具体子任务")
    requires_confirmation: bool = Field(
        False, description="若需要用户确认后再继续，请置为 true"
    )
    tool_names: List[str] = Field(
        default_factory=list,
        description="建议使用的内置工具名称列表，例如 calc、now、http_search",
    )


class PlanOutline(BaseModel):
    objective: str = Field(..., description="整体目标概述")
    steps: List[PlanStepModel] = Field(..., min_length=1, max_length=6)


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                if "text" in chunk:
                    parts.append(str(chunk["text"]))
                elif "content" in chunk:
                    parts.append(str(chunk["content"]))
            else:
                parts.append(str(chunk))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _normalize_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    objective = normalized.get("objective") or normalized.get("goal") or normalized.get("summary")
    if not isinstance(objective, str) or not objective.strip():
        normalized["objective"] = "Auto generated execution plan"
    else:
        normalized["objective"] = objective.strip()

    raw_steps = normalized.get("steps") or []
    if not isinstance(raw_steps, list):
        normalized["steps"] = []
        return normalized

    cleaned_steps: List[Dict[str, Any]] = []
    for idx, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        title = (
            step.get("title")
            or step.get("name")
            or step.get("summary")
            or step.get("description")
        )
        if not isinstance(title, str) or not title.strip():
            step_number = step.get("step_number")
            if isinstance(step_number, int):
                title = f"Step {step_number}"
            else:
                title = f"Step {idx}"
        step["title"] = str(title).strip()

        description = (
            step.get("description")
            or step.get("description_text")
            or step.get("details")
            or step.get("summary")
        )
        if not isinstance(description, str) or not description.strip():
            description = step["title"]
        step["description"] = str(description).strip()

        requires_confirmation = step.get("requires_confirmation")
        if requires_confirmation is None:
            requires_confirmation = bool(
                step.get("needs_confirmation")
                or step.get("requires_user_input")
                or step.get("await_user")
            )
        step["requires_confirmation"] = bool(requires_confirmation)

        tool_names = step.get("tool_names")
        if isinstance(tool_names, list):
            step["tool_names"] = [str(tool).strip() for tool in tool_names if tool]
        elif isinstance(tool_names, str):
            step["tool_names"] = [tool_names.strip()]
        elif isinstance(step.get("tools"), list):
            step["tool_names"] = [str(tool).strip() for tool in step["tools"] if tool]
        else:
            step["tool_names"] = []

        cleaned_steps.append(step)

    normalized["steps"] = cleaned_steps
    return normalized


def _parse_plan_outline(raw: Any) -> PlanOutline:
    if isinstance(raw, PlanOutline):
        return raw
    if isinstance(raw, dict):
        payload = raw
    else:
        content = getattr(raw, "content", raw)
        text = _stringify_content(content)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner response is not valid JSON: {exc}") from exc
    if isinstance(payload, list):
        payload = {"steps": payload}
    if not isinstance(payload, dict):
        raise ValueError("Planner response is not a JSON object.")
    normalized = _normalize_plan_payload(payload)
    try:
        return PlanOutline.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(f"Planner response missing required fields: {exc}") from exc


StepStatus = Literal[
    "pending",
    "waiting",
    "in_progress",
    "completed",
    "cancelled",
    "failed",
]


class PlanStepState(TypedDict, total=False):
    id: str
    title: str
    description: str
    requires_confirmation: bool
    tool_names: List[str]
    status: StepStatus
    result: str
    confirmed: bool
    result_message_id: Optional[str]
    prepared_instruction: str


class AgentState(TypedDict, total=False):
    user_id: str
    short_term_summary: str
    memory_checkpoint: int
    remember_current: bool
    retrieval_query: Optional[str]
    retrieval_results: List[Dict[str, Any]]
    messages: Annotated[List[AnyMessage], add_messages]
    plan: List[PlanStepState]
    current_step: int
    status: str
    pending_interrupt: Optional[Dict[str, Any]]
    active_step_id: Optional[str]
    last_step_update: Optional[Dict[str, Any]]


planner_chain = planner_prompt | planner_llm
agent_chain = agent_prompt | llm_with_tools

FALLBACK_PLAN_BLUEPRINT: List[Dict[str, Any]] = [
    {
        "title": "澄清并理解用户问题",
        "description": "快速复述用户需求，确认需要解答的信息类型；必要时准备调用相关工具。",
        "requires_confirmation": False,
        "tool_names": [],
    },
    {
        "title": "检索或调用工具获取答案",
        "description": "根据需求选择 now、calc、http_search、doc_insights 或 internal_api 等工具，提取关键事实。",
        "tool_names": ["now", "calc", "http_search", "doc_insights"],
    },
    {
        "title": "整理输出最终回复",
        "description": "把工具结果组织成清晰的回答，并在需要时注明来源或下一步建议。",
        "tool_names": [],
    },
]


def _extract_user_goal(messages: List[AnyMessage]) -> str:
    last_user: Optional[HumanMessage] = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user = msg
            break
    if not last_user:
        return ""

    fragments: List[str] = []
    if isinstance(last_user.content, str):
        fragments.append(last_user.content)
    elif isinstance(last_user.content, list):
        for chunk in last_user.content:  # type: ignore[attr-defined]
            if isinstance(chunk, dict):
                if "text" in chunk:
                    fragments.append(str(chunk["text"]))
                elif "content" in chunk:
                    fragments.append(str(chunk["content"]))
            else:
                fragments.append(str(chunk))
    else:
        fragments.append(str(last_user.content))
    return "\n".join(fragments).strip()


def _stringify_message(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:  # type: ignore[attr-defined]
            if isinstance(chunk, dict):
                if "text" in chunk:
                    parts.append(str(chunk["text"]))
                elif "content" in chunk:
                    parts.append(str(chunk["content"]))
            else:
                parts.append(str(chunk))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def _clone_plan(plan: List[PlanStepState]) -> List[PlanStepState]:
    return [dict(step) for step in plan]


def build_step_instruction(step: PlanStepState, idx: int, total: int) -> str:
    segments = [f"当前总计划进度：第 {idx + 1}/{total} 步。", f"目标：{step['title']}."]
    if step.get("description"):
        segments.append(f"说明：{step['description']}")
    if step.get("tool_names"):
        segments.append(
            "建议优先尝试的工具：" + "、".join(step["tool_names"]) + "。"
        )
    segments.append("请完成本步骤需要的工作，并直接给出阶段性结果。")
    return "\n".join(segments)


def _find_latest_human_message(messages: List[AnyMessage]) -> Optional[HumanMessage]:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def _find_latest_ai_message(
    messages: List[AnyMessage],
    exclude_ids: Optional[Set[str]] = None,
) -> Optional[AIMessage]:
    exclude_ids = exclude_ids or set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.id and msg.id in exclude_ids:
                continue
            return msg
    return None


def memory_bootstrap(state: AgentState) -> AgentState:
    user_id = state.get("user_id")
    if not user_id:
        return {}

    updates: AgentState = {}
    if "short_term_summary" not in state:
        summary = storage.get_user_summary(user_id)
        updates["short_term_summary"] = summary
    if "memory_checkpoint" not in state:
        updates["memory_checkpoint"] = len(state.get("messages", []))
    if "retrieval_results" not in state:
        updates["retrieval_results"] = []
    if "retrieval_query" not in state:
        updates["retrieval_query"] = None
    if "remember_current" not in state:
        updates["remember_current"] = False
    return updates


def prepare_retrieval(state: AgentState) -> AgentState:
    if not ENABLE_DOCUMENT_RETRIEVAL:
        return {"retrieval_query": None, "retrieval_results": []}
    user_id = state.get("user_id")
    if not user_id:
        return {"retrieval_results": []}

    query = _extract_user_goal(state.get("messages", []))
    previous_query = state.get("retrieval_query")
    if not query:
        return {"retrieval_query": None, "retrieval_results": []}
    if previous_query == query and state.get("retrieval_results"):
        return {}

    try:
        results = retrieve(user_id=user_id, query=query, top_k=5)
    except Exception as exc:  # pragma: no cover
        logger.warning("retrieval failed user=%s err=%s", user_id, exc)
        results = []
    return {"retrieval_query": query, "retrieval_results": results}


def memory_writer(state: AgentState) -> AgentState:
    user_id = state.get("user_id")
    if not user_id:
        return {}

    messages = state.get("messages", [])
    checkpoint = int(state.get("memory_checkpoint", 0))
    recent = messages[checkpoint:]
    if not recent:
        return {}
    has_user = any(isinstance(msg, HumanMessage) for msg in recent)
    has_ai = any(isinstance(msg, AIMessage) for msg in recent)
    if not (has_user and has_ai):
        return {}

    latest_human = _find_latest_human_message(messages)
    latest_ai = _find_latest_ai_message(messages)
    if not latest_human or not latest_ai:
        return {}

    existing_summary = state.get("short_term_summary", "")
    payload = {
        "summary": existing_summary,
        "user_message": _stringify_message(latest_human),
        "assistant_message": _stringify_message(latest_ai),
    }
    try:
        response = summary_llm.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        updated_summary = response.content.strip() if isinstance(response.content, str) else existing_summary
    except Exception as exc:  # pragma: no cover
        logger.warning("summary generation failed user=%s err=%s", user_id, exc)
        updated_summary = existing_summary

    try:
        storage.update_user_summary(user_id, updated_summary)
    except Exception as exc:  # pragma: no cover
        logger.warning("summary persistence failed user=%s err=%s", user_id, exc)

    updates: AgentState = {
        "short_term_summary": updated_summary,
        "memory_checkpoint": len(messages),
        "remember_current": False,
    }

    if state.get("remember_current"):
        snippet = f"User: {_stringify_message(latest_human)}\nAssistant: {_stringify_message(latest_ai)}"
        try:
            store_memory_snippet(
                user_id=user_id,
                text=snippet,
                metadata={"source": "conversation"},
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("memory snippet store failed user=%s err=%s", user_id, exc)
    return updates


def planner(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    goal = _extract_user_goal(messages)
    if not goal:
        return {"status": "waiting_for_input"}

    outline: Optional[PlanOutline]
    planner_note: Optional[str] = None
    try:
        raw_outline = planner_chain.invoke({"goal": goal})
        outline = _parse_plan_outline(raw_outline)
    except Exception as exc:  # pragma: no cover
        logger.warning("Planner JSON output failed: %s", exc)
        planner_note = "Planner output could not be parsed; switched to a default four-step template."
        objective = goal or "default objective"
        fallback_steps = [
            PlanStepModel(**step_blueprint) for step_blueprint in FALLBACK_PLAN_BLUEPRINT
        ]
        outline = PlanOutline(objective=objective, steps=fallback_steps[:4])

    plan: List[PlanStepState] = []
    for idx, step in enumerate(outline.steps, start=1):
        plan.append(
            {
                "id": f"step-{idx}-{uuid.uuid4().hex[:6]}",
                "title": step.title,
                "description": step.description,
                "requires_confirmation": step.requires_confirmation,
                "tool_names": step.tool_names,
                "status": "pending",
                "result": "",
                "confirmed": not step.requires_confirmation,
            }
        )

    summary_text = (
        f"Generated a {len(plan)} step execution plan: {outline.objective}\n"
        "Steps will run sequentially with status updates at each stage."
    )
    messages_out: List[AnyMessage] = []
    if planner_note:
        messages_out.append(SystemMessage(content=f"[warning] {planner_note}"))
    summary = SystemMessage(content=summary_text)
    messages_out.append(summary)

    if not plan:
        status_value = "completed"
    elif planner_note:
        status_value = "planning_degraded"
    else:
        status_value = "planning_completed"

    return {
        "plan": plan,
        "current_step": 0,
        "status": status_value,
        "pending_interrupt": None,
        "active_step_id": plan[0]["id"] if plan else None,
        "last_step_update": None,
        "messages": messages_out,
    }


def executor(state: AgentState) -> AgentState:
    plan = _clone_plan(state.get("plan", []))
    idx = state.get("current_step", 0)
    updates: AgentState = {
        "plan": plan,
        "current_step": idx,
    }
    logger.debug("executor invoked idx=%s total_steps=%s", idx, len(plan))

    if not plan:
        updates["status"] = "idle"
        updates["active_step_id"] = None
        updates["pending_interrupt"] = None
        logger.debug("executor idle no-plan updates=%s", updates)
        return updates

    if idx >= len(plan):
        updates["status"] = "completed"
        updates["active_step_id"] = None
        updates["pending_interrupt"] = None
        logger.debug("executor already finished idx=%s", idx)
        return updates

    step = plan[idx]
    if step.get("status") in {"completed", "cancelled", "failed"}:
        next_idx = idx + 1
        updates["current_step"] = next_idx
        updates["status"] = "completed" if next_idx >= len(plan) else "executing"
        updates["active_step_id"] = (
            plan[next_idx]["id"] if next_idx < len(plan) else None
        )
        updates["pending_interrupt"] = None
        logger.debug(
            "executor step already terminal step_id=%s status=%s next_idx=%s",
            step.get("id"),
            step.get("status"),
            next_idx,
        )
        return updates

    if step.get("requires_confirmation") and not step.get("confirmed"):
        instruction = build_step_instruction(step, idx, len(plan))
        step["status"] = "waiting"
        step["prepared_instruction"] = instruction
        plan[idx] = step
        pending_payload = {
            "type": "confirmation",
            "step_id": step["id"],
            "title": step["title"],
            "description": step.get("description", ""),
            "message": f"请确认是否继续执行第 {idx + 1} 步：{step['title']}",
            "options": ["continue", "cancel"],
            "resume_instruction": instruction,
        }
        updates["plan"] = plan
        updates["pending_interrupt"] = pending_payload
        updates["status"] = "waiting"
        updates["active_step_id"] = step["id"]
        updates["last_step_update"] = {
            "step_id": step["id"],
            "status": "waiting",
        }
        logger.info(
            "executor waiting for confirmation step=%s instruction=%s",
            step["id"],
            instruction,
        )
        return updates

    instruction = step.get("prepared_instruction") or build_step_instruction(
        step, idx, len(plan)
    )
    step["status"] = "in_progress"
    step["confirmed"] = True
    step.pop("prepared_instruction", None)
    plan[idx] = step

    updates["plan"] = plan
    updates["pending_interrupt"] = None
    updates["status"] = "executing"
    updates["active_step_id"] = step["id"]
    updates["last_step_update"] = {
        "step_id": step["id"],
        "status": "in_progress",
    }
    updates["messages"] = [SystemMessage(content=instruction)]
    logger.info(
        "executor moving to in_progress step=%s instruction=%s",
        step["id"],
        instruction,
    )
    return updates


def interrupt_gate(state: AgentState) -> AgentState:
    pending = state.get("pending_interrupt")
    if pending:
        logger.info(
            "interrupt_gate triggering interrupt for step=%s",
            pending.get("step_id") if isinstance(pending, dict) else pending,
        )
        # 新写法：暂停执行，外部用 Command(resume=...) 恢复
        interrupt(pending)
    return {"pending_interrupt": None}


def step_reporter(state: AgentState) -> AgentState:
    plan = _clone_plan(state.get("plan", []))
    idx = state.get("current_step", 0)

    if not plan or idx >= len(plan):
        logger.debug("step_reporter nothing to report idx=%s total=%s", idx, len(plan))
        return {
            "plan": plan,
            "status": "completed" if plan else "idle",
            "active_step_id": None,
            "pending_interrupt": None,
        }

    step = plan[idx]
    if step.get("status") != "in_progress":
        logger.debug(
            "step_reporter skip step=%s status=%s",
            step.get("id"),
            step.get("status"),
        )
        return {}

    exclude = {step.get("result_message_id")} if step.get("result_message_id") else set()
    latest_ai = _find_latest_ai_message(state.get("messages", []), exclude)
    if not latest_ai:
        logger.debug("step_reporter no new ai message for step=%s", step.get("id"))
        return {}

    if isinstance(latest_ai.content, str):
        result_text = latest_ai.content
    else:
        result_text = json.dumps(latest_ai.content, ensure_ascii=False)

    step["status"] = "completed"
    step["result"] = result_text
    step["result_message_id"] = latest_ai.id
    plan[idx] = step

    next_idx = idx + 1
    updates: AgentState = {
        "plan": plan,
        "current_step": next_idx,
        "pending_interrupt": None,
        "active_step_id": plan[next_idx]["id"] if next_idx < len(plan) else None,
        "status": "completed" if next_idx >= len(plan) else "executing",
        "last_step_update": {
            "step_id": step["id"],
            "status": "completed",
            "result": step.get("result", ""),
        },
    }
    logger.info(
        "step_reporter completed step=%s next_idx=%s result=%s",
        step["id"],
        next_idx,
        step.get("result", ""),
    )
    return updates


def executor_router(state: AgentState) -> str:
    if state.get("pending_interrupt"):
        return "agent"
    plan = state.get("plan", [])
    idx = state.get("current_step", 0)
    status = state.get("status")
    if not plan or idx >= len(plan) or status in {"completed", "cancelled"}:
        return "done"
    return "agent"


def _tools_already_executed(messages: List[AnyMessage]) -> List[str]:
    last_human_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, HumanMessage):
            last_human_idx = idx
            break

    executed: List[str] = []
    for msg in messages[last_human_idx + 1 :]:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            if isinstance(name, str):
                executed.append(name)
    return executed


def _infer_required_tools(messages: List[AnyMessage]) -> List[str]:
    normalized = _extract_user_goal(messages).lower()
    if not normalized:
        return []
    required: List[str] = []
    for tool_name, hints in TOOL_HINTS.items():
        if any(hint.lower() in normalized for hint in hints):
            required.append(tool_name)
    return required


def _infer_default_args(tool_name: str, messages: List[AnyMessage]) -> Optional[Dict[str, Any]]:
    if tool_name == "now":
        return {}
    if tool_name == "http_search":
        query = _extract_user_goal(messages)
        if query:
            return {"query": query, "max_results": 5}
    return None


def _normalize_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
            return {"input": parsed}
        except json.JSONDecodeError:
            return {"input": raw_args}
    if raw_args is None:
        return {}
    return {"input": raw_args}


def _extract_tool_calls(messages: List[AnyMessage]) -> List[Dict[str, Any]]:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and getattr(message, "tool_calls", None):
            calls: List[Dict[str, Any]] = []
            for idx, call in enumerate(message.tool_calls):
                calls.append(
                    {
                        "index": idx,
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "args": _normalize_args(call.get("args")),
                    }
                )
            return calls
    return []


def _run_tool_without_fallback(
    tool_name: str, args: Dict[str, Any], spec: ToolExecutionSpec
) -> Dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        return {
            "tool": tool_name,
            "status": "error",
            "observation": f"Tool '{tool_name}' is not registered.",
            "latency": 0.0,
            "tries": 0,
            "priority": spec.priority,
            "exclusive": spec.exclusive,
            "error": "unregistered tool",
        }

    tool = TOOL_REGISTRY[tool_name]
    max_attempts = spec.retries + 1
    total_latency = 0.0
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as single_executor:
                future = single_executor.submit(tool.invoke, args)
                observation = future.result(timeout=spec.timeout)
            latency = time.perf_counter() - started
            return {
                "tool": tool_name,
                "status": "ok",
                "observation": observation,
                "latency": round(latency, 3),
                "tries": attempt,
                "priority": spec.priority,
                "exclusive": spec.exclusive,
            }
        except TimeoutError:
            latency = time.perf_counter() - started
            total_latency += latency
            last_error = f"timeout after {spec.timeout}s"
        except Exception as exc:  # pragma: no cover
            latency = time.perf_counter() - started
            total_latency += latency
            last_error = repr(exc)

        if attempt < max_attempts:
            time.sleep(spec.backoff * attempt)

    failure_message = last_error or "unknown failure"
    observation = (
        f"Tool '{tool_name}' failed after {round(total_latency, 3)}s: {failure_message}."
        + ("" if spec.fallback else " No fallback configured.")
    )
    return {
        "tool": tool_name,
        "status": "error",
        "observation": observation,
        "latency": round(total_latency, 3),
        "tries": max_attempts,
        "priority": spec.priority,
        "exclusive": spec.exclusive,
        "error": failure_message,
    }


def _execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    spec: ToolExecutionSpec,
    *,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    result = _run_tool_without_fallback(tool_name, args, spec)
    if result["status"] == "ok" or not allow_fallback or not spec.fallback:
        return result

    fallback_spec = _get_spec(spec.fallback)
    fallback_result = _execute_tool(
        spec.fallback, args, fallback_spec, allow_fallback=False
    )
    result["fallback"] = fallback_result
    result["latency"] = round(
        result.get("latency", 0.0) + fallback_result.get("latency", 0.0), 3
    )

    if fallback_result.get("status") == "ok":
        primary_error = result.get("error") or "unknown failure"
        fallback_obs = fallback_result.get("observation", "")
        result["status"] = "degraded"
        result["observation"] = (
            f"Primary tool '{tool_name}' failed ({primary_error}). "
            f"Fallback '{spec.fallback}' succeeded: {fallback_obs}"
        )
    else:
        result.setdefault("error", "fallback failed")
    return result


def _build_tool_message(
    call: Dict[str, Any], result: Dict[str, Any], step_id: Optional[str]
) -> ToolMessage:
    payload = {**result, "tool_call_id": call.get("id")}
    if step_id:
        payload["step_id"] = step_id
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=call.get("id"),
        name=call.get("name") or result.get("tool"),
    )


def _skip_task_message(
    task: Dict[str, Any],
    reason: str,
    step_id: Optional[str],
    *,
    code: str,
) -> ToolMessage:
    call = task["call"]
    spec: ToolExecutionSpec = task["spec"]
    result = {
        "tool": call.get("name") or "unknown",
        "status": "skipped",
        "observation": reason,
        "latency": 0.0,
        "tries": 0,
        "priority": spec.priority,
        "exclusive": spec.exclusive,
        "error": code,
    }
    return _build_tool_message(call, result, step_id)


def _merge_context_args(
    tool_name: Optional[str],
    args: Dict[str, Any],
    state: AgentState,
) -> Dict[str, Any]:
    if not tool_name:
        return args
    provider = TOOL_CONTEXT_PROVIDERS.get(tool_name)
    if not provider:
        return args
    additions = provider(state) or {}
    if not additions:
        return args
    merged = dict(additions)
    merged.update(args or {})
    return merged


def _run_parallel_batch(batch: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    futures = {
        _SHARED_TOOL_EXECUTOR.submit(
            _execute_tool,
            task["call"]["name"],
            task["args"],
            task["spec"],
        ): task
        for task in batch
    }
    completed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for future in as_completed(futures):
        task = futures[future]
        completed.append((task, future.result()))
    return completed


def tool_orchestrator(state: AgentState) -> Dict[str, List[ToolMessage]]:
    messages = state.get("messages", [])
    tool_calls = _extract_tool_calls(messages)
    user_provided_tool_calls = bool(tool_calls)
    required = set(_infer_required_tools(messages))
    already_executed = set(_tools_already_executed(messages))
    present = {call["name"] for call in tool_calls if call.get("name")}

    extra_calls: List[Dict[str, Any]] = []
    next_index = tool_calls[-1]["index"] + 1 if tool_calls else 0
    for tool_name in sorted(required):
        if tool_name in present or tool_name in already_executed:
            continue
        default_args = _infer_default_args(tool_name, messages)
        if default_args is None:
            continue
        extra_calls.append(
            {
                "index": next_index,
                "id": f"auto_{tool_name}_{uuid.uuid4().hex[:8]}",
                "name": tool_name,
                "args": default_args,
            }
        )
        present.add(tool_name)
        next_index += 1

    tool_calls.extend(extra_calls)
    if not tool_calls:
        return {"messages": []}

    step_id = state.get("active_step_id")
    tasks: List[Dict[str, Any]] = []
    seen_calls = set()
    for call in tool_calls:
        args = call.get("args") or {}
        merged_args = _merge_context_args(call.get("name"), args, state)
        call["args"] = merged_args
        spec = _get_spec(call["name"] or "")
        key = (call.get("name"), json.dumps(call["args"], sort_keys=True))
        if key in seen_calls:
            continue
        seen_calls.add(key)
        tasks.append({"call": call, "spec": spec, "args": call["args"]})

    if not tasks:
        return {"messages": []}

    budget = tool_budget_manager.get()
    latency_budget = budget.total_latency if budget.total_latency > 0 else float("inf")

    key_fn = lambda item: (item["spec"].priority, item["call"]["index"])
    ordered_tasks = sorted(tasks, key=key_fn)
    max_tasks = budget.max_tasks or len(ordered_tasks)
    selected_tasks = ordered_tasks[:max_tasks]
    trimmed_tasks = ordered_tasks[max_tasks:]

    results: Dict[int, ToolMessage] = {}
    synthetic_assistant: Optional[AIMessage] = None
    for task in trimmed_tasks:
        idx = task["call"]["index"]
        results[idx] = _skip_task_message(
            task,
            f"Skipped because tool budget only allows {budget.max_tasks} calls per turn.",
            step_id,
            code="max_calls",
        )
    if trimmed_tasks:
        TOOL_THROTTLE_COUNTER.labels(reason="max_calls").inc(len(trimmed_tasks))
        logger.warning(
            "tool orchestrator trimmed %s tasks due to max_tasks budget=%s user=%s step=%s",
            len(trimmed_tasks),
            budget.max_tasks,
            state.get("user_id"),
            step_id,
            extra={
                "event": "tool_throttle",
                "reason": "max_calls",
                "trimmed": len(trimmed_tasks),
                "budget": budget.max_tasks,
            },
        )

    if not selected_tasks:
        ordered = [results[idx] for idx in sorted(results.keys())]
        if not user_provided_tool_calls and tool_calls:
            synthetic_assistant = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": call["id"],
                        "name": call["name"],
                        "args": call["args"],
                    }
                    for call in tool_calls
                ],
            )
            ordered.insert(0, synthetic_assistant)
        return {"messages": ordered}

    parallel_queue = deque(task for task in selected_tasks if not task["spec"].exclusive)
    serial_queue = deque(task for task in selected_tasks if task["spec"].exclusive)

    latency_consumed = 0.0
    budget_limit = latency_budget
    budget_stop = False
    budget_reason: Optional[str] = None

    def _record_latency(outcome: Dict[str, Any]) -> None:
        nonlocal latency_consumed, budget_stop, budget_reason
        try:
            latency_value = float(outcome.get("latency", 0.0) or 0.0)
        except (TypeError, ValueError):
            latency_value = 0.0
        latency_value = max(0.0, latency_value)
        latency_consumed += latency_value
        if latency_value:
            TOOL_LATENCY_SECONDS.inc(latency_value)
        if budget_limit < float("inf") and not budget_stop and latency_consumed >= budget_limit:
            budget_stop = True
            budget_reason = (
                f"Tool latency budget ({budget.total_latency}s) was exhausted."
            )

    parallel_batch_size = budget.max_parallel or 1
    if parallel_batch_size > MAX_PARALLEL_WORKERS:
        parallel_batch_size = MAX_PARALLEL_WORKERS

    while parallel_queue and not budget_stop:
        batch: List[Dict[str, Any]] = []
        while parallel_queue and len(batch) < parallel_batch_size:
            batch.append(parallel_queue.popleft())
        for task, outcome in _run_parallel_batch(batch):
            if step_id:
                outcome.setdefault("step_id", step_id)
            idx = task["call"]["index"]
            results[idx] = _build_tool_message(task["call"], outcome, step_id)
            _record_latency(outcome)

    while serial_queue and not budget_stop:
        task = serial_queue.popleft()
        outcome = _execute_tool(task["call"]["name"], task["args"], task["spec"])
        if step_id:
            outcome.setdefault("step_id", step_id)
        idx = task["call"]["index"]
        results[idx] = _build_tool_message(task["call"], outcome, step_id)
        _record_latency(outcome)

    if budget_stop:
        pending = list(parallel_queue) + list(serial_queue)
        reason = budget_reason or "Resource budget exhausted before remaining tools could run."
        for task in pending:
            idx = task["call"]["index"]
            if idx in results:
                continue
            results[idx] = _skip_task_message(task, reason, step_id, code="latency_budget")
        TOOL_THROTTLE_COUNTER.labels(reason="latency_budget").inc(len(pending))
        TOOL_LATENCY_BUDGET_EXHAUSTED.inc()
        logger.warning(
            "tool orchestrator latency budget hit user=%s step=%s consumed=%.3f limit=%.3f pending=%s",
            state.get("user_id"),
            step_id,
            latency_consumed,
            budget.total_latency,
            len(pending),
            extra={
                "event": "tool_throttle",
                "reason": "latency_budget",
                "pending": len(pending),
                "consumed": latency_consumed,
                "limit": budget.total_latency,
            },
        )

    ordered = [results[idx] for idx in sorted(results.keys())]
    if not user_provided_tool_calls and tool_calls:
        synthetic_assistant = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": call["id"],
                    "name": call["name"],
                    "args": call["args"],
                }
                for call in tool_calls
            ],
        )
        ordered.insert(0, synthetic_assistant)
    return {"messages": ordered}


def agent(state: AgentState) -> Dict[str, List[AnyMessage]]:
    raw_messages: List[AnyMessage] = list(state.get("messages", []))
    filtered: List[AnyMessage] = [
        msg
        for msg in raw_messages
        if not (isinstance(msg, SystemMessage) and msg.content == SYSTEM_PROMPT)
    ]

    context_messages: List[AnyMessage] = []
    summary_text = (state.get("short_term_summary") or "").strip()
    if summary_text:
        context_messages.append(
            SystemMessage(
                content=f"[memory]\n{summary_text}",
                additional_kwargs={"internal_only": True},
            )
        )

    retrievals = state.get("retrieval_results") or []
    if retrievals:
        context_lines: List[str] = []
        for idx, item in enumerate(retrievals, start=1):
            source = item.get("metadata", {}).get("filename") or item.get("document_id")
            snippet = item.get("content", "")
            context_lines.append(f"[{idx}] {source}: {snippet}")
        context_messages.append(
            SystemMessage(
                content="Relevant knowledge base entries:\n" + "\n".join(context_lines),
                additional_kwargs={
                    "internal_only": True,
                    "retrieval_sources": retrievals,
                },
            )
        )

    # 丰富的配对逻辑：为所有孤立的 ToolMessage 补齐对应的 AIMessage.tool_calls
    filtered_fixed: List[AnyMessage] = []
    seen_call_ids: set = set()
    for msg in filtered:
        if isinstance(msg, AIMessage):
            for call in getattr(msg, "tool_calls", []) or []:
                cid = call.get("id")
                if cid:
                    seen_call_ids.add(cid)
            filtered_fixed.append(msg)
            continue
        if isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None) or f"auto_{uuid.uuid4().hex[:8]}"
            name = getattr(msg, "name", None) or "auto_tool"
            if call_id not in seen_call_ids:
                synthetic_ai = AIMessage(
                    content="",
                    tool_calls=[{"id": call_id, "name": name, "args": {}}],
                )
                filtered_fixed.append(synthetic_ai)
                seen_call_ids.add(call_id)
            filtered_fixed.append(msg)
            continue
        filtered_fixed.append(msg)
    filtered = filtered_fixed

    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "context_messages": context_messages,
        "conversation": filtered,
    }
    response = agent_chain.invoke(payload)
    return {"messages": [response]}


_checkpoint_conn: Optional[sqlite3.Connection] = None


def _build_checkpointer():
    """Prefer persistent SQLite checkpointer; fall back to memory if unavailable."""
    global _checkpoint_conn
    if SqliteSaver is None:
        logger.warning("SqliteSaver not installed; falling back to in-memory checkpoints.")
        return MemorySaver()

    db_path = Path(CHECKPOINT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _checkpoint_conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            _checkpoint_conn.execute("PRAGMA journal_mode=WAL")
            _checkpoint_conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:  # pragma: no cover - best effort
            logger.debug("SQLite checkpointer pragmas not applied", exc_info=True)
        logger.info("Using SQLite checkpointer at %s", db_path)
        return SqliteSaver(_checkpoint_conn)
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "SQLite checkpointer init failed; falling back to memory saver.",
            exc_info=exc,
        )
        _checkpoint_conn = None
        return MemorySaver()


def _close_checkpointer() -> None:
    global _checkpoint_conn
    conn = _checkpoint_conn
    _checkpoint_conn = None
    if conn is not None:
        try:
            conn.close()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to close checkpointer connection: %s", exc)


atexit.register(_close_checkpointer)

graph = StateGraph(AgentState)
graph.add_node("memory_bootstrap", memory_bootstrap)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("interrupt_gate", interrupt_gate)
graph.add_node("retrieval", prepare_retrieval)
graph.add_node("agent", agent)
graph.add_node("tools", tool_orchestrator)
graph.add_node("step_reporter", step_reporter)
graph.add_node("memory_writer", memory_writer)

graph.add_edge(START, "memory_bootstrap")
graph.add_edge("memory_bootstrap", "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", executor_router, {"agent": "interrupt_gate", "done": END})
graph.add_edge("interrupt_gate", "retrieval")
graph.add_edge("retrieval", "agent")

# 关键：tools_condition 的结束分支键是 "__end__"
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "step_reporter",
    },
)

graph.add_edge("tools", "agent")
graph.add_edge("step_reporter", "memory_writer")
graph.add_edge("memory_writer", "executor")

memory = _build_checkpointer()
compiled_graph = graph.compile(checkpointer=memory)
