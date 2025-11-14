from __future__ import annotations
import os

import json
import logging
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Interrupt
from pydantic import BaseModel
from starlette import status

from . import storage
from .graph import build_step_instruction, compiled_graph
from .rag import process_document_file, retrieve

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

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
DOCS_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "40"))


def _build_graph_config(thread_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    configurable: Dict[str, Any] = {"thread_id": thread_id}
    if user_id:
        configurable["user_id"] = user_id
    return {"configurable": configurable, "recursion_limit": DEFAULT_RECURSION_LIMIT}


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
    user_id: Optional[str] = None
    remember: bool = False


class ChatResp(BaseModel):
    reply: str


pending_interrupts: Dict[str, Dict[str, Any]] = {}


def _register_interrupt(session_id: str, payload: Dict[str, Any]) -> None:
    pending_interrupts[session_id] = payload


def _clear_interrupt(session_id: str) -> None:
    pending_interrupts.pop(session_id, None)


def _resolve_user_id(session_id: str, user_id: Optional[str]) -> str:
    resolved = (user_id or "").strip() or session_id
    storage.upsert_user(resolved)
    return resolved


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

def _safe_filename(name: Optional[str]) -> str:
    if not name:
        return 'upload.txt'
    clean = Path(name).name.strip()
    return clean or 'upload.txt'

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
def chat(body: ChatReq) -> ChatResp:
    logger.info("chat request session=%s message=%s", body.session_id, body.message)

    user_id = _resolve_user_id(body.session_id, body.user_id)
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
    return ChatResp(reply=reply)


def _process_document_background(document_id: str, user_id: str, dest_path: Path) -> None:
    try:
        process_document_file(document_id=document_id, user_id=user_id, file_path=dest_path)
    except Exception as exc:  # pragma: no cover - background worker
        logger.exception("document processing failed doc=%s user=%s", document_id, user_id)
        storage.update_document_status(document_id, status="failed", error=str(exc))


@app.get("/chat/stream")
def chat_stream(
    session_id: str,
    message: str,
    user_id: Optional[str] = None,
    remember: Optional[str] = None,
) -> StreamingResponse:
    resolved_user = _resolve_user_id(session_id, user_id)
    remember_flag = _parse_bool_flag(remember)
    logger.info(
        "chat stream session=%s user=%s message=%s",
        session_id,
        resolved_user,
        message,
    )

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
            if not ended:
                yield 'data: {"event": "end"}\n\n'

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/chat/continue")
def chat_continue(thread_id: str, action: str, user_id: Optional[str] = None):
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
        thread_id, user_id or state_values.get("user_id")  # type: ignore[arg-type]
    )
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
            if not ended:
                yield 'data: {"event": "end"}\n\n'

    return StreamingResponse(resume_stream(), media_type="text/event-stream")

@app.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    storage.upsert_user(normalized_user)

    document_id = uuid.uuid4().hex
    filename = _safe_filename(file.filename)
    user_dir = _sanitize_user_folder(normalized_user)
    destination = user_dir / f"{document_id}_{filename}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    destination.write_bytes(content)

    storage.create_document(
        document_id=document_id,
        user_id=normalized_user,
        filename=filename,
        path=str(destination),
        status="processing",
    )
    background_tasks.add_task(
        _process_document_background,
        document_id,
        normalized_user,
        destination,
    )
    logger.info(
        "document uploaded doc_id=%s user=%s filename=%s bytes=%s",
        document_id,
        normalized_user,
        filename,
        len(content),
    )
    return {"id": document_id, "status": "processing"}


@app.get("/documents")
def list_documents(user_id: str):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    storage.upsert_user(normalized_user)
    return {"items": storage.list_documents(normalized_user)}


@app.get("/documents/{document_id}")
def get_document(document_id: str, user_id: str):
    normalized_user = (user_id or "").strip()
    if not normalized_user:
        raise HTTPException(status_code=400, detail="user_id is required")
    doc = storage.get_document(document_id)
    if not doc or doc.get("user_id") != normalized_user:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
