from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from .graph import compiled_graph
import json

app = FastAPI(title="Minimal LangGraph Agent")

# 允许本地前端（Vite 默认 5173）跨域访问
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
def chat(body: ChatReq):
    """每个 session_id 维度维护一条 LangGraph 会话轨迹（MemorySaver）"""
    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=body.message)]},
        config={"configurable": {"thread_id": body.session_id}},
    )
    # 取最后一条 AI 回复
    msgs = result["messages"]
    reply = "（无回复）"
    for m in reversed(msgs):
        if getattr(m, "type", None) == "ai":
            reply = m.content
            break
    return ChatResp(reply=reply)

@app.get("/chat/stream")
def chat_stream(session_id: str, message: str):
    """SSE 流式：按图中节点更新、以及 AI 回复增量，逐步推送给前端。"""
    def event_gen():
        cfg = {"configurable": {"thread_id": session_id}}
        ai_acc = ""
        try:
            for update in compiled_graph.stream(
                {"messages": [HumanMessage(content=message)]},
                config=cfg,
                stream_mode="updates",
            ):
                # update: {node_name: state_update}
                for node, data in update.items():
                    payload = {"event": node}
                    if isinstance(data, dict) and "messages" in data:
                        msgs = data["messages"]
                        if msgs:
                            last = msgs[-1]
                            role = getattr(last, "type", None) or "unknown"
                            content = getattr(last, "content", "")
                            if role == "ai" and isinstance(content, str):
                                if content.startswith(ai_acc):
                                    delta = content[len(ai_acc):]
                                    ai_acc = content
                                else:
                                    delta = content
                                    ai_acc = content
                                payload.update({"role": "ai", "delta": delta})
                            else:
                                payload.update({"role": role, "content": content})
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": str(e)}, ensure_ascii=False) + "\n\n"
        finally:
            yield "data: {\"event\": \"end\"}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
