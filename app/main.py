from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .graph import compiled_graph

app = FastAPI(title="LangGraph Agent · Tool Orchestration Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatReq(BaseModel):
    session_id: str
    message: str


class ChatResp(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResp)
def chat(body: ChatReq) -> ChatResp:
    """单次对话：根据 session_id 复用持久化上下文。"""
    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=body.message)]},
        config={"configurable": {"thread_id": body.session_id}},
    )
    reply = "（暂未生成回复）"
    for message in reversed(result["messages"]):
        if getattr(message, "type", None) == "ai":
            reply = message.content
            break
    return ChatResp(reply=reply)


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


@app.get("/chat/stream")
def chat_stream(session_id: str, message: str):
    """SSE 流：返回节点事件，附带工具执行的可观测字段。"""

    def event_gen():
        cfg = {"configurable": {"thread_id": session_id}}
        ai_acc = ""
        ended = False
        try:
            for update in compiled_graph.stream(
                {"messages": [HumanMessage(content=message)]},
                config=cfg,
                stream_mode="updates",
            ):
                for node, data in update.items():
                    payload: Dict[str, Any] = {"event": node}
                    if isinstance(data, dict) and "messages" in data:
                        for msg_obj in data["messages"]:
                            role = getattr(msg_obj, "type", None) or "unknown"
                            content = getattr(msg_obj, "content", "")
                            payload = {"event": node}

                            if role == "ai" and isinstance(content, str):
                                if content.startswith(ai_acc):
                                    delta = content[len(ai_acc) :]
                                    ai_acc = content
                                else:
                                    delta = content
                                    ai_acc = content
                                payload.update({"role": "ai", "delta": delta})
                            elif role == "tool":
                                tool_name = getattr(msg_obj, "name", "") or "tool"
                                tool_payload = _normalize_tool_payload(content, tool_name)
                                payload.update({"role": "tool", **tool_payload})
                            else:
                                payload.update({"role": role, "content": content})

                            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                        continue

                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        except Exception as exc:  # pragma: no cover - SSE 错误仅用于观测
            if isinstance(exc, KeyError) and exc.args and exc.args[0] == "__end__":
                yield 'data: {"event": "end"}\n\n'
                ended = True
            elif isinstance(exc, RuntimeError) and str(exc) == "Connection error.":
                # 客户端断开或连接中断，静默结束
                yield 'data: {"event": "end"}\n\n'
                ended = True
            else:
                yield "data: " + json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False) + "\n\n"
        finally:
            if not ended:
                yield 'data: {"event": "end"}\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream")
