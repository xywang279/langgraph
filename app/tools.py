from __future__ import annotations

import ast
import operator as op
import random
import time
from datetime import datetime
from typing import Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---- 安全四则运算（结构化参数） ----
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
    # 兼容 Python 3.8+ 的 ast.Constant 以及旧版 ast.Num
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers allowed in expressions.")
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
    """安全计算器，只支持 + - * / 和括号，例如 '1 + 2*(3-1)'。"""
    try:
        node = ast.parse(expression, mode="eval").body
        return str(_eval_ast(node))
    except Exception as exc:  # pragma: no cover - 示例无需覆盖
        return f"ERROR: {exc}"


class NowArgs(BaseModel):
    fmt: str = Field(
        "%Y-%m-%d %H:%M",
        description="Python strftime format string, defaults to '%Y-%m-%d %H:%M'.",
    )


@tool("now", args_schema=NowArgs)
def now(fmt: str = "%Y-%m-%d %H:%M") -> str:
    """返回当前时间，支持 strftime 格式化。"""
    return datetime.now().strftime(fmt)


_KB: Dict[str, str] = {
    "langgraph": "LangGraph：用于构建有状态 Agent 的图式执行框架，支持循环、持久化与可观测。",
    "tool arbitration": "多工具场景可按优先级、超时和降级策略调度，保障整体吞吐与稳定性。",
    "wechat drafting": "公众号写作常见流程：选题→调研→数据校验→排版→发布。",
}


class KnowledgeArgs(BaseModel):
    query: str = Field(..., description="Keyword for local knowledge-base lookup.")


def _kb_lookup(query: str) -> str:
    q = query.lower()
    hits: List[str] = []
    for key, value in _KB.items():
        if q in key or q in value.lower():
            hits.append(f"- {key}: {value}")
    if not hits:
        return "未命中本地知识库。"
    return "\n".join(hits)


@tool("kb_search", args_schema=KnowledgeArgs)
def kb_search(query: str) -> str:
    """本地知识库检索，适合兜底或快速查找内置资料。"""
    return _kb_lookup(query)


_FAQ: Dict[str, str] = {
    "LangGraph 是什么": "LangGraph 是一个针对有状态 Agent 的编排框架，支持图式执行、循环控制与持久化存储。",
    "工具降级如何提示": "在工具失败后，observation 字段会写明失败原因以及是否触发 fallback。",
    "并行工具会不会阻塞": "tool_orchestrator 会依据 priority 和 exclusive 配置决定是否并发执行，默认可并发的工具不会互相阻塞。",
}


class FaqArgs(BaseModel):
    question: str = Field(..., description="Question to lookup in local FAQ store.")


def _faq_lookup(question: str) -> str:
    q = question.strip().lower()
    if not q:
        return "请输入有效的问题。"
    for key, value in _FAQ.items():
        if q in key.lower():
            return f"{key}：{value}"
    return "未命中本地 FAQ，可改用 multi_search 获取更多信息。"


@tool("faq", args_schema=FaqArgs)
def faq(question: str) -> str:
    """本地 FAQ 检索，适合快速回答常见问题。"""
    return _faq_lookup(question)


class MultiSearchArgs(BaseModel):
    query: str = Field(..., description="Topic to research for the article.")
    prefer_fresh: bool = Field(
        False,
        description="Set true to prefer simulated online sources even if local KB has an answer.",
    )


def _simulate_online_sources(query: str, prefer_fresh: bool) -> List[str]:
    """模拟外部检索：可能成功，也可能因限流或超时抛出异常。"""
    time.sleep(0.4)  # 模拟网络耗时
    q = query.lower()
    if "timeout" in q or "降级" in q:
        raise TimeoutError("upstream search timed out")
    if "langgraph" in q:
        return [
            "LangGraph 发布 0.2：新增工具并发执行与回溯能力。",
            "社区案例：用 LangGraph 管理多 Agent 写作流水线。",
        ]
    if prefer_fresh and random.random() < 0.2:
        raise RuntimeError("random upstream failure for fresh sources")
    return [f"未找到 {query} 的权威公开结果，可退回内置资料。"]


@tool("multi_search", args_schema=MultiSearchArgs)
def multi_search(query: str, prefer_fresh: bool = False) -> str:
    """多源检索：尝试模拟在线数据，必要时将结果并入本地知识库。"""
    try:
        online_hits = _simulate_online_sources(query, prefer_fresh)
    except Exception as exc:  # 交给编排层做降级
        raise RuntimeError(f"上游检索失败：{exc}") from exc

    kb_hits = _kb_lookup(query)
    kb_section = kb_hits if kb_hits != "未命中本地知识库。" else ""
    combined = "\n".join(online_hits + ([kb_section] if kb_section else []))
    return combined.strip() or "检索完成，但未找到可用信息。"


class UnstableArgs(BaseModel):
    task: str = Field(..., description="Task name used to identify the run.")
    seconds: float = Field(
        3.0,
        ge=0,
        description="Simulated execution time in seconds.",
    )
    fail: bool = Field(
        False,
        description="If true, raise an exception to simulate a failure.",
    )


@tool("unstable", args_schema=UnstableArgs)
def unstable(task: str, seconds: float = 3.0, fail: bool = False) -> str:
    """
    一个“可能很慢 / 会失败”的工具：sleep seconds 后返回。
    设置 fail=True 模拟异常。用于验证超时 / 重试 / 降级路径。
    """
    time.sleep(max(0.0, float(seconds)))
    if fail:
        raise RuntimeError(f"Task '{task}' failed intentionally.")
    return f"Task '{task}' done after {seconds}s."
