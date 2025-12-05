import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "langgraph.sqlite3"


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    with conn:
        if not column_exists(conn, "documents", "tenant_id"):
            conn.execute("ALTER TABLE documents ADD COLUMN tenant_id TEXT")
        if not column_exists(conn, "chunks", "tenant_id"):
            conn.execute("ALTER TABLE chunks ADD COLUMN tenant_id TEXT")
        if not column_exists(conn, "vector_stats", "tenant_id"):
            conn.execute("ALTER TABLE vector_stats ADD COLUMN tenant_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id)")
        print("Migration completed.")


if __name__ == "__main__":
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found at {DB_PATH}")
    main()
