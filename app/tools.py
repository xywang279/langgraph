from langchain_core.tools import tool
from datetime import datetime
import ast, operator as op

# --- 安全四则运算计算器（仅 + - * / 和括号） ---
_ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}
def _eval_ast(node):
    # 兼容 Python 3.8+ 的 ast.Constant 与旧的 ast.Num
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers allowed in expressions.")
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ops:
        return _ops[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ops:
        return _ops[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("Only + - * / and parentheses are supported.")

@tool
def calc(expr: str) -> str:
    """安全计算器，只支持 + - * / 和括号，例如 '1 + 2*(3-1)'。"""
    try:
        node = ast.parse(expr, mode="eval").body
        return str(_eval_ast(node))
    except Exception as e:
        return f"ERROR: {e}"

@tool
def now(fmt: str = "%Y-%m-%d %H:%M") -> str:
    """返回当前时间，支持 strftime 格式化。"""
    return datetime.now().strftime(fmt)

_FAQ = {
    "LangGraph 是什么": "一个用于构建有状态 Agent 的编排框架（图式执行、持久化、可控循环）。",
    "如何使用工具": "让模型产生 tool_calls，由工具节点执行，再把 ToolMessage 写回对话。",
    "项目结构": "FastAPI 提供 API，LangGraph 管图，Tools 提供外部能力。"
}

@tool
def faq(query: str) -> str:
    """一个最简单的本地 FAQ 检索（子串匹配），不依赖外网。"""
    for k, v in _FAQ.items():
        if query.lower() in k.lower():
            return f"{k}: {v}"
    return "未命中本地 FAQ。"
