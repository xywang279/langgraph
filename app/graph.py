from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import tools_condition

from .tools import calc, faq, kb_search, multi_search, now, unstable

load_dotenv()


# ---- LLM & tools registration ----
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY)
TOOLS = [calc, now, kb_search, multi_search, faq, unstable]
TOOL_REGISTRY = {tool.name: tool for tool in TOOLS}
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "你是一名公众号写作助手。"
    "当用户在指令中点名某个工具（例如“用 now”“调用 calc”）时，"
    "必须为每个被点名的工具分别创建 tool_call，并在需要时并行执行。"
    "若多个工具能同步运行，请一次性返回全部所需的 tool_calls，避免只调用其中之一。"
    "生成最终回答前，务必整理所有工具结果并明确标注其来源。"
)


# ---- Tool arbitration configuration ----
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


def _infer_required_tools(messages: List[AnyMessage]) -> List[str]:
    last_user: Optional[HumanMessage] = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user = msg
            break
    if not last_user:
        return []

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

    normalized = " ".join(fragments).lower()
    required: List[str] = []
    for tool_name, hints in TOOL_HINTS.items():
        if any(hint.lower() in normalized for hint in hints):
            required.append(tool_name)
    return required


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


def _run_tool_without_fallback(tool_name: str, args: Dict[str, Any], spec: ToolExecutionSpec) -> Dict[str, Any]:
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
        except Exception as exc:  # pragma: no cover - demo only
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


def _execute_tool(tool_name: str, args: Dict[str, Any], spec: ToolExecutionSpec, *, allow_fallback: bool = True) -> Dict[str, Any]:
    result = _run_tool_without_fallback(tool_name, args, spec)
    if result["status"] == "ok" or not allow_fallback or not spec.fallback:
        return result

    fallback_spec = _get_spec(spec.fallback)
    fallback_result = _execute_tool(spec.fallback, args, fallback_spec, allow_fallback=False)
    result["fallback"] = fallback_result
    result["latency"] = round(result.get("latency", 0.0) + fallback_result.get("latency", 0.0), 3)

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


def _build_tool_message(call: Dict[str, Any], result: Dict[str, Any]) -> ToolMessage:
    payload = {**result, "tool_call_id": call.get("id")}
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=call.get("id"),
        name=call.get("name") or result.get("tool"),
    )


def tool_orchestrator(state: MessagesState) -> Dict[str, List[ToolMessage]]:
    tool_calls = _extract_tool_calls(state["messages"])
    required = set(_infer_required_tools(state["messages"]))
    already_executed = set(_tools_already_executed(state["messages"]))
    present = {call["name"] for call in tool_calls if call.get("name")}

    extra_calls: List[Dict[str, Any]] = []
    next_index = tool_calls[-1]["index"] + 1 if tool_calls else 0
    for tool_name in sorted(required):
        if tool_name in present or tool_name in already_executed:
            continue
        default_args = _infer_default_args(tool_name, state["messages"])
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
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(parallel_tasks))) as executor:
            futures = {
                executor.submit(
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
                idx = task["call"]["index"]
                results[idx] = _build_tool_message(task["call"], outcome)

    for task in sorted(serial_tasks, key=key_fn):
        outcome = _execute_tool(task["call"]["name"], task["args"], task["spec"])
        idx = task["call"]["index"]
        results[idx] = _build_tool_message(task["call"], outcome)

    ordered = [results[idx] for idx in sorted(results.keys())]
    return {"messages": ordered}


# ---- Agent + Graph definition ----
def agent(state: MessagesState) -> Dict[str, List[AnyMessage]]:
    messages: List[AnyMessage] = list(state["messages"])
    if not any(
        getattr(msg, "type", None) == "system" and getattr(msg, "content", "") == SYSTEM_PROMPT
        for msg in messages
    ):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_orchestrator)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)
