from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_ROOT_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT_DIR / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DATA_DIR / "langgraph.sqlite3"

_CONNECTION: Optional[sqlite3.Connection] = None
_LOCK = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    global _CONNECTION
    if _CONNECTION is None:
        _CONNECTION = sqlite3.connect(DB_PATH, check_same_thread=False)
        _CONNECTION.row_factory = sqlite3.Row
        _CONNECTION.execute("PRAGMA foreign_keys = ON")
    return _CONNECTION


def init_db() -> None:
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                summary TEXT DEFAULT '',
                updated_at REAL DEFAULT (strftime('%s','now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                path TEXT,
                kind TEXT NOT NULL DEFAULT 'document',
                status TEXT NOT NULL,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
        )
        conn.commit()


def upsert_user(user_id: str) -> None:
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO users (user_id, summary, updated_at)
            VALUES (?, COALESCE((SELECT summary FROM users WHERE user_id = ?), ''), ?)
            ON CONFLICT(user_id) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (user_id, user_id, time.time()),
        )
        conn.commit()


def get_user_summary(user_id: str) -> str:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            "SELECT summary FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            upsert_user(user_id)
            return ""
        return row["summary"] or ""


def update_user_summary(user_id: str, summary: str) -> None:
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO users (user_id, summary, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                summary=excluded.summary,
                updated_at=excluded.updated_at
            """,
            (user_id, summary, time.time()),
        )
        conn.commit()


def create_document(
    *,
    document_id: str,
    user_id: str,
    filename: str,
    path: Optional[str],
    kind: str = "document",
    status: str = "pending",
) -> None:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO documents (id, user_id, filename, path, kind, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, user_id, filename, path, kind, status, now, now),
        )
        conn.commit()


def update_document_status(
    document_id: str, *, status: str, error: Optional[str] = None, path: Optional[str] = None
) -> None:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            """
            UPDATE documents
            SET status = ?, error = ?, updated_at = ?, path = COALESCE(?, path)
            WHERE id = ?
            """,
            (status, error, now, path, document_id),
        )
        conn.commit()


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            """
            SELECT id, user_id, filename, path, kind, status, error, created_at, updated_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
    return dict(row) if row else None


def list_documents(user_id: str) -> List[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT id, filename, path, kind, status, error, created_at, updated_at
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_chunks_for_document(document_id: str) -> None:
    conn = _get_connection()
    with _LOCK:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.commit()


def store_chunks(
    *,
    document_id: str,
    user_id: str,
    chunks: Iterable[Dict[str, Any]],
) -> None:
    conn = _get_connection()
    payload = [
        (
            chunk["id"],
            document_id,
            user_id,
            chunk["chunk_index"],
            chunk["content"],
            json.dumps(chunk["embedding"]),
            json.dumps(chunk.get("metadata", {})),
            time.time(),
        )
        for chunk in chunks
    ]
    with _LOCK:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id,
                document_id,
                user_id,
                chunk_index,
                content,
                embedding,
                metadata,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()


def iter_user_chunks(user_id: str) -> Iterable[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT id, document_id, chunk_index, content, embedding, metadata
            FROM chunks
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
    for row in rows:
        embedding = json.loads(row["embedding"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        yield {
            "id": row["id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "embedding": embedding,
            "metadata": metadata,
        }


def ensure_memory_document(user_id: str) -> str:
    memory_doc_id = f"memory::{user_id}"
    if get_document(memory_doc_id):
        return memory_doc_id
    create_document(
        document_id=memory_doc_id,
        user_id=user_id,
        filename="Conversation memory",
        path=None,
        kind="memory",
        status="ready",
    )
    return memory_doc_id


def clear_connection() -> None:
    global _CONNECTION
    if _CONNECTION is not None:
        _CONNECTION.close()
        _CONNECTION = None
