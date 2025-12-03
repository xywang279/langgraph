from __future__ import annotations

import json
import sqlite3
import threading
import time
import hashlib
import hmac
import secrets
import uuid
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
                updated_at REAL DEFAULT (strftime('%s','now')),
                created_at REAL DEFAULT (strftime('%s','now')),
                password_hash TEXT,
                password_salt TEXT,
                last_login_at REAL
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
                size_bytes REAL,
                mime_type TEXT,
                content_type TEXT,
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                last_message TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created ON chat_messages(thread_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(user_id, updated_at DESC)")

        def _column_exists(table: str, column: str) -> bool:
            cur = conn.execute(f"PRAGMA table_info({table})")
            return any(row[1] == column for row in cur.fetchall())

        for column, ddl in [
            ("size_bytes", "ALTER TABLE documents ADD COLUMN size_bytes REAL"),
            ("mime_type", "ALTER TABLE documents ADD COLUMN mime_type TEXT"),
            ("content_type", "ALTER TABLE documents ADD COLUMN content_type TEXT"),
        ]:
            if not _column_exists("documents", column):
                conn.execute(ddl)

        for column, ddl in [
            ("password_hash", "ALTER TABLE users ADD COLUMN password_hash TEXT"),
            ("password_salt", "ALTER TABLE users ADD COLUMN password_salt TEXT"),
            ("created_at", "ALTER TABLE users ADD COLUMN created_at REAL"),
            ("last_login_at", "ALTER TABLE users ADD COLUMN last_login_at REAL"),
        ]:
            if not _column_exists("users", column):
                conn.execute(ddl)

        conn.commit()


def upsert_user(user_id: str) -> None:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO users (user_id, summary, updated_at, created_at)
            VALUES (
                ?,
                COALESCE((SELECT summary FROM users WHERE user_id = ?), ''),
                ?,
                COALESCE((SELECT created_at FROM users WHERE user_id = ?), ?)
            )
            ON CONFLICT(user_id) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (user_id, user_id, now, user_id, now),
        )
        conn.commit()


def user_exists(user_id: str) -> bool:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return row is not None


def _hash_password(password: str, salt: str) -> str:
    raw = (salt + password).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def set_user_password(user_id: str, password: str) -> None:
    salt = secrets.token_hex(16)
    digest = _hash_password(password, salt)
    now = time.time()
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO users (user_id, summary, password_hash, password_salt, created_at, updated_at, last_login_at)
            VALUES (?, COALESCE((SELECT summary FROM users WHERE user_id = ?), ''), ?, ?, COALESCE((SELECT created_at FROM users WHERE user_id = ?), ?), ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                password_hash=excluded.password_hash,
                password_salt=excluded.password_salt,
                updated_at=excluded.updated_at
            """,
            (user_id, user_id, digest, salt, user_id, now, now, now),
        )
        conn.commit()


def verify_user_password(user_id: str, password: str) -> bool:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            "SELECT password_hash, password_salt FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return False
    stored_hash = row["password_hash"] or ""
    salt = row["password_salt"] or ""
    if not stored_hash or not salt:
        return False
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(stored_hash, candidate)


def record_login(user_id: str) -> None:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            """
            UPDATE users
            SET last_login_at = ?, updated_at = COALESCE(updated_at, ?)
            WHERE user_id = ?
            """,
            (now, now, user_id),
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
    now = time.time()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO users (user_id, summary, updated_at, created_at)
            VALUES (?, ?, ?, COALESCE((SELECT created_at FROM users WHERE user_id = ?), ?))
            ON CONFLICT(user_id) DO UPDATE SET
                summary=excluded.summary,
                updated_at=excluded.updated_at
            """,
            (user_id, summary, now, user_id, now),
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
    size_bytes: Optional[float] = None,
    mime_type: Optional[str] = None,
    content_type: Optional[str] = None,
) -> None:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO documents (
                id,
                user_id,
                filename,
                path,
                kind,
                status,
                size_bytes,
                mime_type,
                content_type,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                user_id,
                filename,
                path,
                kind,
                status,
                size_bytes,
                mime_type,
                content_type,
                now,
                now,
            ),
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


def delete_document(document_id: str) -> None:
    conn = _get_connection()
    with _LOCK:
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            """
            SELECT
                id,
                user_id,
                filename,
                path,
                kind,
                status,
                error,
                size_bytes,
                mime_type,
                content_type,
                created_at,
                updated_at,
                (
                    SELECT COUNT(*) FROM chunks c WHERE c.document_id = documents.id
                ) AS chunk_count,
                (
                    SELECT COUNT(*)
                    FROM chunks c
                    WHERE c.document_id = documents.id
                      AND c.embedding IS NOT NULL
                      AND c.embedding != ''
                ) AS embedding_count
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
            SELECT
                d.id,
                d.user_id,
                d.filename,
                d.path,
                d.kind,
                d.status,
                d.error,
                d.size_bytes,
                d.mime_type,
                d.content_type,
                d.created_at,
                d.updated_at,
                (
                    SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id
                ) AS chunk_count,
                (
                    SELECT COUNT(*)
                    FROM chunks c
                    WHERE c.document_id = d.id
                      AND c.embedding IS NOT NULL
                      AND c.embedding != ''
                ) AS embedding_count
            FROM documents d
            WHERE d.user_id = ?
            ORDER BY d.created_at DESC
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


def ensure_chat_thread(user_id: str, *, thread_id: Optional[str] = None, title: str = "", last_message: str = "") -> str:
    """Create a chat thread if missing or update the metadata for an existing one."""
    upsert_user(user_id)
    tid = thread_id or uuid.uuid4().hex
    now = time.time()
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO chat_threads (thread_id, user_id, title, last_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                user_id = excluded.user_id,
                title = CASE
                    WHEN excluded.title IS NOT NULL AND excluded.title != '' THEN excluded.title
                    ELSE chat_threads.title
                END,
                last_message = CASE
                    WHEN excluded.last_message IS NOT NULL AND excluded.last_message != '' THEN excluded.last_message
                    ELSE chat_threads.last_message
                END,
                updated_at = excluded.updated_at
            """,
            (tid, user_id, title or "", last_message or "", now, now),
        )
        conn.commit()
    return tid


def append_chat_message(
    thread_id: str,
    user_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    if not thread_id or not user_id or not role:
        return ""
    ensure_chat_thread(user_id, thread_id=thread_id)
    now = time.time()
    msg_id = uuid.uuid4().hex
    conn = _get_connection()
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata is not None else None
    snippet = (content or "").strip()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO chat_messages (id, thread_id, user_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, thread_id, user_id, role, content or "", meta_json, now),
        )
        conn.execute(
            """
            INSERT INTO chat_threads (thread_id, user_id, title, last_message, created_at, updated_at)
            VALUES (?, ?, '', ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                last_message = excluded.last_message,
                updated_at = excluded.updated_at
            """,
            (thread_id, user_id, snippet[:500], now, now),
        )
        conn.commit()
    return msg_id


def list_chat_threads(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT thread_id, user_id, title, last_message, created_at, updated_at
            FROM chat_threads
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, max(1, limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def list_chat_messages(thread_id: str, user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT m.id, m.role, m.content, m.metadata, m.created_at
            FROM chat_messages m
            JOIN chat_threads t ON t.thread_id = m.thread_id
            WHERE m.thread_id = ? AND t.user_id = ?
            ORDER BY m.created_at ASC
            LIMIT ?
            """,
            (thread_id, user_id, max(1, limit)),
        ).fetchall()
    items: List[Dict[str, Any]] = []
    for row in rows:
        meta = json.loads(row["metadata"]) if row["metadata"] else None
        items.append(
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": meta,
                "created_at": row["created_at"],
            }
        )
    return items


def update_chat_thread_title(thread_id: str, user_id: str, title: str) -> bool:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        cur = conn.execute(
            """
            UPDATE chat_threads
            SET title = ?, updated_at = ?
            WHERE thread_id = ? AND user_id = ?
            """,
            (title, now, thread_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def delete_chat_thread(thread_id: str, user_id: str) -> bool:
    conn = _get_connection()
    with _LOCK:
        cur = conn.execute(
            "DELETE FROM chat_threads WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


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
