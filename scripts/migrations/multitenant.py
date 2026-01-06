from dataclasses import dataclass
from typing import Sequence

from .db_utils import ensure_column, ensure_index


@dataclass(frozen=True)
class ColumnSpec:
    table: str
    column: str
    column_type: str


@dataclass(frozen=True)
class IndexSpec:
    name: str
    table: str
    columns: Sequence[str]


TENANT_COLUMN = "tenant_id"
COLUMN_SPECS = (
    ColumnSpec("documents", TENANT_COLUMN, "TEXT"),
    ColumnSpec("chunks", TENANT_COLUMN, "TEXT"),
    ColumnSpec("vector_stats", TENANT_COLUMN, "TEXT"),
)
INDEX_SPECS = (IndexSpec("idx_chunks_tenant", "chunks", (TENANT_COLUMN,)),)


def run(conn) -> list[str]:
    changes: list[str] = []
    for spec in COLUMN_SPECS:
        if ensure_column(conn, spec.table, spec.column, spec.column_type):
            changes.append(f"added column {spec.table}.{spec.column}")
    for spec in INDEX_SPECS:
        if ensure_index(conn, spec.name, spec.table, spec.columns):
            changes.append(f"created index {spec.name} on {spec.table}")
    return changes
