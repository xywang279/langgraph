"""
Ingest generated questions markdown into the RAG store.

Usage:
  python scripts/ingest_questions.py --md data/ingest/questions/foo.md --user-id u1
  python scripts/ingest_questions.py --pdf input.pdf --out-dir data/ingest --user-id u1
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional

from app import storage
from app.rag import process_document_file


def _resolve_md_path(pdf_path: Optional[Path], out_dir: Path, md_path: Optional[Path]) -> Path:
    if md_path:
        return md_path
    if not pdf_path:
        raise ValueError("Either --md or --pdf is required.")
    pdf_stem = pdf_path.stem
    return out_dir / "questions" / f"{pdf_stem}.md"


def ingest_questions(
    *,
    md_path: Path,
    user_id: str,
    tenant_id: Optional[str],
    document_id: Optional[str],
    title: Optional[str],
    publish: bool,
) -> dict:
    if not md_path.exists():
        raise FileNotFoundError(f"markdown file not found: {md_path}")

    document_id = document_id or uuid.uuid4().hex
    filename = title or md_path.name
    size_bytes = md_path.stat().st_size
    doc = storage.get_document(document_id)

    if doc is None:
        version_id = storage.create_document(
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            filename=filename,
            path=str(md_path),
            kind="questions",
            status="pending",
            size_bytes=size_bytes,
            mime_type="text/markdown",
            content_type="text/markdown",
        )
    else:
        version_id = storage.create_document_version(
            document_id=document_id,
            status="pending",
            path_raw=str(md_path),
            published=False,
        )

    process_document_file(
        document_id=document_id,
        user_id=user_id,
        file_path=md_path,
        version_id=version_id,
        publish=publish,
        tenant_id=tenant_id,
    )

    return {
        "document_id": document_id,
        "version_id": version_id,
        "path": str(md_path),
        "published": publish,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest questions markdown into the RAG store.")
    parser.add_argument("--md", type=Path, help="Path to questions markdown.")
    parser.add_argument("--pdf", type=Path, help="Source PDF path to derive markdown filename.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "ingest",
        help="Base output directory used by pdf_to_questions.py.",
    )
    parser.add_argument("--user-id", required=True, help="User id for vector store isolation.")
    parser.add_argument("--tenant-id", help="Tenant id if applicable.")
    parser.add_argument("--document-id", help="Existing document id to append a new version.")
    parser.add_argument("--title", help="Document title/filename override.")
    parser.add_argument("--no-publish", action="store_true", help="Do not publish the version.")
    args = parser.parse_args()

    md_path = _resolve_md_path(args.pdf, args.out_dir, args.md)
    result = ingest_questions(
        md_path=md_path,
        user_id=args.user_id,
        tenant_id=args.tenant_id,
        document_id=args.document_id,
        title=args.title,
        publish=not args.no_publish,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
