import argparse
import sqlite3
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = SCRIPTS_DIR.parent / "data" / "langgraph.sqlite3"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from migrations.db_utils import connect_db
from migrations.multitenant import run as run_multitenant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate tenant_id columns.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db
    if not db_path.exists():
        print(f"DB not found at {db_path}", file=sys.stderr)
        return 1
    try:
        conn = connect_db(db_path)
    except sqlite3.Error as exc:
        print(f"Failed to connect to DB: {exc}", file=sys.stderr)
        return 1

    try:
        with conn:
            changes = run_multitenant(conn)
    except sqlite3.Error as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1

    if changes:
        print("Migration completed.")
        for change in changes:
            print(f"- {change}")
    else:
        print("Migration completed. No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
