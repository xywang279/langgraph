from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore

from . import storage
from .rag import process_document_file

logger = logging.getLogger(__name__)


def process_document_inline(
    *,
    document_id: str,
    user_id: str,
    path: str,
    version_id: Optional[str],
    publish: bool = True,
    embedding_model: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    file_path, cleanup = _materialize_source(path, document_id)
    try:
        process_document_file(
            document_id=document_id,
            user_id=user_id,
            file_path=file_path,
            version_id=version_id,
            publish=publish,
            embedding_model=embedding_model,
            tenant_id=tenant_id,
        )
    finally:
        if cleanup:
            try:
                file_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("cleanup temp file failed path=%s", file_path, exc_info=True)


def run_ingestion_job(job_id: str, user_id: str, task_ids: List[str]) -> None:
    job = storage.get_ingestion_job(job_id) or {}
    params: Dict[str, Any] = job.get("params") or {}
    tasks = storage.list_ingestion_tasks(job_id)
    processed = job.get("processed") or 0
    storage.update_ingestion_job(job_id, status="running")
    tenant_id = job.get("tenant_id")
    for task in tasks:
        if task["id"] not in task_ids:
            continue
        if task.get("status") == "completed":
            continue
        attempts = (task.get("attempts") or 0) + 1
        storage.update_ingestion_task(task["id"], status="running", attempts=attempts)
        try:
            step = task.get("step")
            doc_id = task.get("document_id")
            version_id = task.get("version_id")
            doc = storage.get_document(doc_id) if doc_id else None
            if not doc or not doc.get("path"):
                raise RuntimeError("Document missing or file path unavailable.")
            if tenant_id and doc.get("tenant_id") not in {None, tenant_id}:
                raise RuntimeError("Tenant mismatch for document.")

            if step in {"reindex", "ingest"}:
                process_document_inline(
                    document_id=doc_id,
                    user_id=user_id,
                    path=doc["path"],
                    version_id=version_id,
                    publish=True,
                    embedding_model=params.get("embedding_model"),
                    tenant_id=tenant_id,
                )
            else:
                logger.info("unknown ingestion step=%s job_id=%s task_id=%s", step, job_id, task["id"])
            storage.update_ingestion_task(task["id"], status="completed", error=None)
            processed += 1
            storage.update_ingestion_job(job_id, processed=processed)
        except Exception as task_exc:  # pragma: no cover - worker path
            logger.exception("ingestion task failed job_id=%s task_id=%s", job_id, task["id"])
            if task.get("version_id"):
                storage.update_document_version(task["version_id"], status="failed", error=str(task_exc))
            if task.get("document_id"):
                storage.update_document_status(task["document_id"], status="failed", error=str(task_exc))
            storage.update_ingestion_task(
                task["id"],
                status="failed",
                error=str(task_exc),
                attempts=attempts,
            )
            storage.update_ingestion_job(job_id, status="failed", error=str(task_exc), processed=processed)
            storage.upsert_vector_stats(
                tenant_id=tenant_id,
                document_id=doc_id,
                version_id=version_id,
                expected_vectors=None,
                stored_vectors=None,
                health_status="failed",
                notes=str(task_exc),
            )
            return
    storage.update_ingestion_job(job_id, status="completed", processed=processed)
    logger.info("ingestion job completed job_id=%s user=%s processed=%s", job_id, user_id, processed)


def _materialize_source(path: str, document_id: str) -> Tuple[Path, bool]:
    """Return a local Path for the source. If downloaded, cleanup flag is True."""
    if path.lower().startswith(("http://", "https://")):
        return _download_http(path, document_id), True
    if path.lower().startswith("s3://"):
        return _download_s3(path, document_id), True
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"source not found: {path}")
    return p, False


def _download_http(url: str, document_id: str) -> Path:
    tmp_dir = Path(tempfile.gettempdir())
    suffix = Path(url).suffix
    target = tmp_dir / f"ingest_{document_id}{suffix}"
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with target.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return target


def _download_s3(url: str, document_id: str) -> Path:
    if boto3 is None:
        raise RuntimeError("boto3 not installed; cannot fetch s3 object.")
    parts = url.replace("s3://", "", 1).split("/", 1)
    if len(parts) != 2:
        raise RuntimeError(f"invalid s3 url: {url}")
    bucket, key = parts
    tmp_dir = Path(tempfile.gettempdir())
    suffix = Path(key).suffix
    target = tmp_dir / f"ingest_{document_id}{suffix}"
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    )
    s3.download_file(bucket, key, str(target))
    return target
