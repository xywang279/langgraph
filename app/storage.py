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
                tenant_id TEXT,
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
                version_id TEXT,
                tenant_id TEXT,
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY(version_id) REFERENCES document_versions(id) ON DELETE SET NULL
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
            ("latest_version_id", "ALTER TABLE documents ADD COLUMN latest_version_id TEXT"),
            ("tenant_id", "ALTER TABLE documents ADD COLUMN tenant_id TEXT"),
        ]:
            if not _column_exists("documents", column):
                conn.execute(ddl)
        if not _column_exists("chunks", "version_id"):
            conn.execute("ALTER TABLE chunks ADD COLUMN version_id TEXT")
        if not _column_exists("chunks", "tenant_id"):
            conn.execute("ALTER TABLE chunks ADD COLUMN tenant_id TEXT")
        if _column_exists("chunks", "version_id"):
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(version_id)"
            )
        if _column_exists("chunks", "tenant_id"):
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id)"
            )

        for column, ddl in [
            ("password_hash", "ALTER TABLE users ADD COLUMN password_hash TEXT"),
            ("password_salt", "ALTER TABLE users ADD COLUMN password_salt TEXT"),
            ("created_at", "ALTER TABLE users ADD COLUMN created_at REAL"),
            ("last_login_at", "ALTER TABLE users ADD COLUMN last_login_at REAL"),
        ]:
            if not _column_exists("users", column):
                conn.execute(ddl)

        if not _column_exists("vector_stats", "tenant_id"):
            conn.execute("ALTER TABLE vector_stats ADD COLUMN tenant_id TEXT")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                parent_version_id TEXT,
                version_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                path_raw TEXT,
                path_text TEXT,
                text_hash TEXT,
                chunk_count INTEGER,
                embedding_model TEXT,
                published INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY(parent_version_id) REFERENCES document_versions(id) ON DELETE SET NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_versions_doc ON document_versions(document_id, version_no DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_versions_published ON document_versions(document_id, published)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                params_json TEXT,
                status TEXT NOT NULL,
                error TEXT,
                total INTEGER DEFAULT 0,
                processed INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user ON ingestion_jobs(user_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_tasks (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                document_id TEXT,
                version_id TEXT,
                step TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                attempts INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL,
                FOREIGN KEY(version_id) REFERENCES document_versions(id) ON DELETE SET NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_tasks_job ON ingestion_tasks(job_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_tasks_status ON ingestion_tasks(status)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_stats (
                id TEXT PRIMARY KEY,
                tenant_id TEXT,
                document_id TEXT,
                version_id TEXT,
                expected_vectors INTEGER,
                stored_vectors INTEGER,
                last_checked_at REAL,
                health_status TEXT,
                notes TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL,
                FOREIGN KEY(version_id) REFERENCES document_versions(id) ON DELETE SET NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_stats_doc ON vector_stats(document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_stats_health ON vector_stats(health_status)")

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
    tenant_id: Optional[str],
    user_id: str,
    filename: str,
    path: Optional[str],
    kind: str = "document",
    status: str = "pending",
    size_bytes: Optional[float] = None,
    mime_type: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO documents (
                id,
                tenant_id,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                tenant_id,
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
        # create initial version placeholder for future ingestion pipeline
        version_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO document_versions (
                id,
                document_id,
                parent_version_id,
                version_no,
                status,
                path_raw,
                path_text,
                created_at,
                updated_at,
                published
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, 0)
            """,
            (
                version_id,
                document_id,
                None,
                1,
                status,
                path,
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE documents SET latest_version_id = ? WHERE id = ?",
            (version_id, document_id),
        )
        conn.commit()
    return version_id


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
                tenant_id,
                user_id,
                filename,
                path,
                kind,
                status,
                error,
                size_bytes,
                mime_type,
                content_type,
                latest_version_id,
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


def list_documents(user_id: str, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _get_connection()
    params: List[Any] = [user_id]
    tenant_clause = ""
    if tenant_id is not None:
        tenant_clause = "AND (d.tenant_id = ? OR d.tenant_id IS NULL)"
        params.append(tenant_id)
    with _LOCK:
        rows = conn.execute(
            f"""
            SELECT
                d.id,
                d.tenant_id,
                d.user_id,
                d.filename,
                d.path,
                d.kind,
                d.status,
                d.error,
                d.size_bytes,
                d.mime_type,
                d.content_type,
                d.latest_version_id,
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
            {tenant_clause}
            ORDER BY d.created_at DESC
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def _next_version_no(document_id: str) -> int:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            "SELECT COALESCE(MAX(version_no), 0) AS max_no FROM document_versions WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return int(row["max_no"] or 0) + 1


def create_document_version(
    document_id: str,
    *,
    parent_version_id: Optional[str] = None,
    status: str = "pending",
    path_raw: Optional[str] = None,
    path_text: Optional[str] = None,
    embedding_model: Optional[str] = None,
    published: bool = False,
) -> str:
    version_id = uuid.uuid4().hex
    version_no = _next_version_no(document_id)
    now = time.time()
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO document_versions (
                id,
                document_id,
                parent_version_id,
                version_no,
                status,
                path_raw,
                path_text,
                embedding_model,
                published,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                document_id,
                parent_version_id,
                version_no,
                status,
                path_raw,
                path_text,
                embedding_model,
                1 if published else 0,
                now,
                now,
            ),
        )
        if published:
            conn.execute(
                "UPDATE documents SET latest_version_id = ? WHERE id = ?",
                (version_id, document_id),
            )
        conn.commit()
    return version_id


def update_document_version(
    version_id: str,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    path_raw: Optional[str] = None,
    path_text: Optional[str] = None,
    chunk_count: Optional[int] = None,
    embedding_model: Optional[str] = None,
    published: Optional[bool] = None,
) -> None:
    fields = []
    params: List[Any] = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if error is not None:
        fields.append("error = ?")
        params.append(error)
    if path_raw is not None:
        fields.append("path_raw = ?")
        params.append(path_raw)
    if path_text is not None:
        fields.append("path_text = ?")
        params.append(path_text)
    if chunk_count is not None:
        fields.append("chunk_count = ?")
        params.append(chunk_count)
    if embedding_model is not None:
        fields.append("embedding_model = ?")
        params.append(embedding_model)
    if published is not None:
        fields.append("published = ?")
        params.append(1 if published else 0)
    if not fields:
        return
    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(version_id)
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            f"UPDATE document_versions SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        conn.commit()


def publish_document_version(document_id: str, version_id: str) -> None:
    conn = _get_connection()
    now = time.time()
    with _LOCK:
        conn.execute(
            "UPDATE document_versions SET published = 0 WHERE document_id = ?",
            (document_id,),
        )
        conn.execute(
            """
            UPDATE document_versions
            SET published = 1, updated_at = ?
            WHERE id = ? AND document_id = ?
            """,
            (now, version_id, document_id),
        )
        conn.execute(
            "UPDATE documents SET latest_version_id = ? WHERE id = ?",
            (version_id, document_id),
        )
        conn.commit()


def get_latest_version_id(document_id: str) -> Optional[str]:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            "SELECT latest_version_id FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    if not row:
        return None
    return row["latest_version_id"]


def get_document_versions(document_id: str) -> List[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT
                id,
                document_id,
                parent_version_id,
                version_no,
                status,
                error,
                path_raw,
                path_text,
                text_hash,
                chunk_count,
                embedding_model,
                published,
                created_at,
                updated_at
            FROM document_versions
            WHERE document_id = ?
            ORDER BY version_no DESC
            """,
            (document_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_document_version(version_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            """
            SELECT
                id,
                document_id,
                parent_version_id,
                version_no,
                status,
                error,
                path_raw,
                path_text,
                text_hash,
                chunk_count,
                embedding_model,
                published,
                created_at,
                updated_at
            FROM document_versions
            WHERE id = ?
            """,
            (version_id,),
        ).fetchone()
    return dict(row) if row else None


def get_published_version_id(document_id: str) -> Optional[str]:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            """
            SELECT id
            FROM document_versions
            WHERE document_id = ? AND published = 1
            ORDER BY version_no DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()
    if not row:
        return None
    return row["id"]


def count_embeddings_for_version(version_id: str) -> int:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM chunks
            WHERE version_id = ?
              AND embedding IS NOT NULL
              AND embedding != ''
            """,
            (version_id,),
        ).fetchone()
    return int(row["cnt"] or 0)


def delete_chunks_for_document(document_id: str) -> None:
    conn = _get_connection()
    with _LOCK:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.commit()


def delete_chunks_for_version(version_id: str) -> None:
    conn = _get_connection()
    with _LOCK:
        conn.execute("DELETE FROM chunks WHERE version_id = ?", (version_id,))
        conn.commit()


def store_chunks(
    *,
    document_id: str,
    user_id: str,
    tenant_id: Optional[str],
    chunks: Iterable[Dict[str, Any]],
) -> None:
    conn = _get_connection()
    payload = [
        (
            chunk["id"],
            chunk.get("version_id"),
            tenant_id,
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
                version_id,
                tenant_id,
                document_id,
                user_id,
                chunk_index,
                content,
                embedding,
                metadata,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()


def iter_user_chunks(user_id: str, tenant_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    conn = _get_connection()
    params: List[Any] = [user_id]
    tenant_clause = ""
    if tenant_id is not None:
        tenant_clause = "AND (tenant_id = ? OR tenant_id IS NULL)"
        params.append(tenant_id)
    with _LOCK:
        rows = conn.execute(
            f"""
            SELECT id, version_id, document_id, chunk_index, content, embedding, metadata
            FROM chunks
            WHERE user_id = ?
            {tenant_clause}
            """,
            params,
        ).fetchall()
    for row in rows:
        embedding = json.loads(row["embedding"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        yield {
            "id": row["id"],
            "version_id": row["version_id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "embedding": embedding,
            "metadata": metadata,
        }


def iter_published_chunks(user_id: str, tenant_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    conn = _get_connection()
    params: List[Any] = [user_id]
    tenant_clause = ""
    if tenant_id is not None:
        tenant_clause = "AND (d.tenant_id = ? OR d.tenant_id IS NULL)"
        params.append(tenant_id)
    with _LOCK:
        rows = conn.execute(
            f"""
            SELECT
                c.id,
                c.version_id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.embedding,
                c.metadata
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN document_versions v ON v.id = c.version_id
            WHERE d.user_id = ?
              AND (c.version_id IS NULL OR v.published = 1)
              {tenant_clause}
            """,
            params,
        ).fetchall()
    for row in rows:
        embedding = json.loads(row["embedding"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        yield {
            "id": row["id"],
            "version_id": row["version_id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "embedding": embedding,
            "metadata": metadata,
        }


def ensure_chat_thread(
    user_id: str,
    *,
    thread_id: Optional[str] = None,
    title: str = "",
    last_message: str = "",
    allow_create: bool = True,
) -> str:
    """Create a chat thread if missing or update the metadata for an existing one."""
    upsert_user(user_id)
    tid = thread_id or uuid.uuid4().hex
    if not allow_create and not tid:
        raise ValueError("thread_id is required when creation is disabled.")
    now = time.time()
    conn = _get_connection()
    with _LOCK:
        if thread_id:
            row = conn.execute(
                "SELECT user_id FROM chat_threads WHERE thread_id = ?",
                (tid,),
            ).fetchone()
            if row:
                existing_user = row["user_id"]
                if existing_user != user_id:
                    raise ValueError("Thread already belongs to a different user.")
            elif not allow_create:
                raise ValueError("Thread not found.")
        elif not allow_create:
            raise ValueError("Thread not found.")
        conn.execute(
            """
            INSERT INTO chat_threads (thread_id, user_id, title, last_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
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
    ensure_chat_thread(user_id, thread_id=thread_id, allow_create=False)
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
            UPDATE chat_threads
            SET last_message = ?, updated_at = ?
            WHERE thread_id = ? AND user_id = ?
            """,
            (snippet[:500], now, thread_id, user_id),
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


def get_chat_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    if not thread_id:
        return None
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            """
            SELECT thread_id, user_id, title, last_message, created_at, updated_at
            FROM chat_threads
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
    return dict(row) if row else None


def list_chat_messages(thread_id: str, user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT m.id, m.role, m.content, m.metadata, m.created_at
            FROM chat_messages m
            JOIN chat_threads t ON t.thread_id = m.thread_id
            WHERE m.thread_id = ? AND t.user_id = ?
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (thread_id, user_id, max(1, limit)),
        ).fetchall()
    rows = list(reversed(rows))
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


def create_ingestion_job(
    *,
    user_id: str,
    tenant_id: Optional[str],
    source: str,
    params: Optional[Dict[str, Any]],
    status: str = "queued",
    total: int = 0,
) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    conn = _get_connection()
    params_json = json.dumps(params or {}, ensure_ascii=False)
    with _LOCK:
        conn.execute(
            """
            INSERT INTO ingestion_jobs (
                id, tenant_id, user_id, source, params_json, status, total, processed, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (job_id, tenant_id, user_id, source, params_json, status, int(total), now, now),
        )
        conn.commit()
    return job_id


def update_ingestion_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    total: Optional[int] = None,
    processed: Optional[int] = None,
) -> None:
    fields = []
    params: List[Any] = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if error is not None:
        fields.append("error = ?")
        params.append(error)
    if total is not None:
        fields.append("total = ?")
        params.append(int(total))
    if processed is not None:
        fields.append("processed = ?")
        params.append(int(processed))
    if not fields:
        return
    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(job_id)
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            f"UPDATE ingestion_jobs SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        conn.commit()


def create_ingestion_task(
    job_id: str,
    *,
    document_id: Optional[str],
    version_id: Optional[str],
    step: str,
    status: str = "queued",
) -> str:
    task_id = uuid.uuid4().hex
    now = time.time()
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO ingestion_tasks (
                id, job_id, document_id, version_id, step, status, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (task_id, job_id, document_id, version_id, step, status, now, now),
        )
        conn.commit()
    return task_id


def update_ingestion_task(
    task_id: str, *, status: Optional[str] = None, error: Optional[str] = None, attempts: Optional[int] = None
) -> None:
    fields = []
    params: List[Any] = []
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if error is not None:
        fields.append("error = ?")
        params.append(error)
    if attempts is not None:
        fields.append("attempts = ?")
        params.append(int(attempts))
    if not fields:
        return
    fields.append("updated_at = ?")
    params.append(time.time())
    params.append(task_id)
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            f"UPDATE ingestion_tasks SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        conn.commit()


def get_ingestion_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        row = conn.execute(
            """
            SELECT id, tenant_id, user_id, source, params_json, status, error, total, processed, created_at, updated_at
            FROM ingestion_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    if not row:
        return None
    return {
        **dict(row),
        "params": json.loads(row["params_json"] or "{}"),
    }


def list_ingestion_jobs(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT id, tenant_id, user_id, source, params_json, status, error, total, processed, created_at, updated_at
            FROM ingestion_jobs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, max(1, limit)),
        ).fetchall()
    return [
        {
            **dict(row),
            "params": json.loads(row["params_json"] or "{}"),
        }
        for row in rows
    ]


def list_ingestion_tasks(job_id: str) -> List[Dict[str, Any]]:
    conn = _get_connection()
    with _LOCK:
        rows = conn.execute(
            """
            SELECT id, job_id, document_id, version_id, step, status, error, attempts, created_at, updated_at
            FROM ingestion_tasks
            WHERE job_id = ?
            ORDER BY created_at ASC
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_vector_stats(
    *,
    tenant_id: Optional[str],
    document_id: Optional[str],
    version_id: Optional[str],
    expected_vectors: Optional[int],
    stored_vectors: Optional[int],
    health_status: str,
    notes: Optional[str] = None,
) -> str:
    stat_id = uuid.uuid4().hex
    now = time.time()
    conn = _get_connection()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO vector_stats (
                id, tenant_id, document_id, version_id, expected_vectors, stored_vectors, last_checked_at, health_status, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stat_id,
                tenant_id,
                document_id,
                version_id,
                expected_vectors,
                stored_vectors,
                now,
                health_status,
                notes,
                now,
                now,
            ),
        )
        conn.commit()
    return stat_id


def list_vector_stats(document_ids: Optional[List[str]] = None, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _get_connection()
    params: List[Any] = []
    where_clauses: List[str] = []
    if document_ids:
        placeholders = ", ".join("?" for _ in document_ids)
        where_clauses.append(f"document_id IN ({placeholders})")
        params.extend(document_ids)
    if tenant_id is not None:
        where_clauses.append("(tenant_id = ? OR tenant_id IS NULL)")
        params.append(tenant_id)
    where = ""
    if where_clauses:
        where = "WHERE " + " AND ".join(where_clauses)
    with _LOCK:
        rows = conn.execute(
            f"""
            SELECT id, tenant_id, document_id, version_id, expected_vectors, stored_vectors, last_checked_at, health_status, notes, created_at, updated_at
            FROM vector_stats
            {where}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def ensure_memory_document(user_id: str, tenant_id: Optional[str] = None) -> str:
    memory_doc_id = f"memory::{user_id}"
    if get_document(memory_doc_id):
        return memory_doc_id
    create_document(
        document_id=memory_doc_id,
        tenant_id=tenant_id,
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
