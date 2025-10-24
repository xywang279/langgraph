import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from .tools import calc, now, faq

load_dotenv()

# 1) 模型（可绑定工具）
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("OPENAI_BASE_URL")  # 兼容 DeepSeek/Groq 等
API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY)
TOOLS = [calc, now, faq]
llm_with_tools = llm.bind_tools(TOOLS)

# 2) agent 节点：调用 LLM（让它决定是否使用工具）
def agent(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 3) 预置的工具执行节点
tool_node = ToolNode(TOOLS)

# 4) 搭图：agent -> (如需) tools -> agent 循环，直到不再产生 tool_calls
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

# 5) 会话记忆（demo 用内存；生产可替换为 SQLite/Redis Checkpointer）
memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)
