import re
import sqlite3
from pathlib import Path
from typing import Sequence

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, kind: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {kind} name: {name!r}")


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    _validate_identifier(table, "table")
    _validate_identifier(column, "column")
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row["name"] == column for row in cur.fetchall())


def ensure_column(
    conn: sqlite3.Connection, table: str, column: str, column_type: str
) -> bool:
    if column_exists(conn, table, column):
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    return True


def index_exists(conn: sqlite3.Connection, table: str, index_name: str) -> bool:
    _validate_identifier(table, "table")
    _validate_identifier(index_name, "index")
    cur = conn.execute(f"PRAGMA index_list({table})")
    return any(row["name"] == index_name for row in cur.fetchall())


def ensure_index(
    conn: sqlite3.Connection, index_name: str, table: str, columns: Sequence[str]
) -> bool:
    for column in columns:
        _validate_identifier(column, "column")
    if index_exists(conn, table, index_name):
        return False
    columns_sql = ", ".join(columns)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({columns_sql})"
    )
    return True
