from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from typing import Any, Annotated, Dict, List, Literal, Optional, Set, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

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
    "返回 JSON 结构，供后续执行器使用。"
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
    "multi_search": ToolExecutionSpec(
        name="multi_search",
        priority=1,
        timeout=8.0,
        retries=1,
        backoff=0.8,
        fallback="kb_search",
    ),
    "kb_search": ToolExecutionSpec(
        name="kb_search",
        priority=2,
        timeout=3.0,
    ),
    "calc": ToolExecutionSpec(
        name="calc",
        priority=3,
        timeout=2.0,
    ),
    "now": ToolExecutionSpec(
        name="now",
        priority=3,
        timeout=1.0,
        exclusive=True,
    ),
    "faq": ToolExecutionSpec(
        name="faq",
        priority=4,
        timeout=2.0,
    ),
    "unstable": ToolExecutionSpec(
        name="unstable",
        priority=2,
        timeout=4.0,
        retries=1,
        backoff=1.0,
        exclusive=True,
    ),
}

MAX_PARALLEL_WORKERS = 4


def _get_spec(tool_name: str) -> ToolExecutionSpec:
    return TOOL_SPECS.get(tool_name, ToolExecutionSpec(name=tool_name))


TOOL_HINTS = {
    "now": ["now", "当前时间", "现在几点", "现在时间", "time"],
    "calc": ["calc", "计算", "算一下", "结果是多少", "算出", "求值"],
    "kb_search": ["kb", "知识库", "资料", "查一下资料"],
    "multi_search": ["multi_search", "多源", "检索", "搜索", "调研"],
    "faq": ["faq", "常见问题", "帮助", "说明", "文档"],
    "unstable": ["unstable", "慢工具", "不稳定", "超时测试", "重试"],
}


class PlanStepModel(BaseModel):
    title: str = Field(..., description="步骤的标题")
    description: str = Field(..., description="执行说明，描述要完成的具体子任务")
    requires_confirmation: bool = Field(
        False, description="若需要用户确认后再继续，请置为 true"
    )
    tool_names: List[str] = Field(
        default_factory=list,
        description="建议使用的内置工具名称列表，例如 calc、now、kb_search",
    )


class PlanOutline(BaseModel):
    objective: str = Field(..., description="整体目标概述")
    steps: List[PlanStepModel] = Field(..., min_length=1, max_length=6)


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


planner_chain = planner_prompt | llm.with_structured_output(PlanOutline)
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
        "description": "根据需求选择 now、calc、kb_search 或 multi_search 等工具，提取关键事实。",
        "tool_names": ["now", "calc", "kb_search", "multi_search"],
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
        outline = planner_chain.invoke({"goal": goal})
    except Exception as exc:  # pragma: no cover
        logger.warning("Planner structured output failed: %s", exc)
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


def tool_orchestrator(state: AgentState) -> Dict[str, List[ToolMessage]]:
    messages = state.get("messages", [])
    tool_calls = _extract_tool_calls(messages)
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
        spec = _get_spec(call["name"] or "")
        key = (call.get("name"), json.dumps(call["args"], sort_keys=True))
        if key in seen_calls:
            continue
        seen_calls.add(key)
        tasks.append({"call": call, "spec": spec, "args": call["args"]})

    parallel_tasks = [task for task in tasks if not task["spec"].exclusive]
    serial_tasks = [task for task in tasks if task["spec"].exclusive]
    key_fn = lambda item: (item["spec"].priority, item["call"]["index"])

    results: Dict[int, ToolMessage] = {}

    if parallel_tasks:
        with ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_WORKERS, len(parallel_tasks))
        ) as executor_pool:
            futures = {
                executor_pool.submit(
                    _execute_tool,
                    task["call"]["name"],
                    task["args"],
                    task["spec"],
                ): task
                for task in sorted(parallel_tasks, key=key_fn)
            }
            for future in as_completed(futures):
                task = futures[future]
                outcome = future.result()
                if step_id:
                    outcome.setdefault("step_id", step_id)
                idx = task["call"]["index"]
                results[idx] = _build_tool_message(task["call"], outcome, step_id)

    for task in sorted(serial_tasks, key=key_fn):
        outcome = _execute_tool(task["call"]["name"], task["args"], task["spec"])
        if step_id:
            outcome.setdefault("step_id", step_id)
        idx = task["call"]["index"]
        results[idx] = _build_tool_message(task["call"], outcome, step_id)

    ordered = [results[idx] for idx in sorted(results.keys())]
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

    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "context_messages": context_messages,
        "conversation": filtered,
    }
    response = agent_chain.invoke(payload)
    return {"messages": [response]}


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

memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)
