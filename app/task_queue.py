from __future__ import annotations

import logging
import os
from typing import List, Optional

try:
    from celery import Celery
except ImportError:  # pragma: no cover
    Celery = None  # type: ignore

logger = logging.getLogger(__name__)

BROKER_URL = os.getenv("CELERY_BROKER_URL", "").strip()
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "").strip() or BROKER_URL
ENABLE_CELERY = bool(BROKER_URL and Celery is not None)

celery_app: Optional[Celery] = None
if ENABLE_CELERY:
    celery_app = Celery(
        "langgraph",
        broker=BROKER_URL,
        backend=RESULT_BACKEND or None,
    )
    celery_app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_default_queue="default",
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "600")),
        task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "540")),
    )
else:  # pragma: no cover
    logger.warning("Celery not enabled (missing broker or dependency); tasks will run inline.")


def enqueue_process_document(
    document_id: str,
    user_id: str,
    path: str,
    version_id: Optional[str],
    publish: bool = True,
    embedding_model: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    from .ingestion_worker import process_document_inline

    if celery_app:
        return process_document_task.delay(
            document_id=document_id,
            user_id=user_id,
            path=path,
            version_id=version_id,
            publish=publish,
            embedding_model=embedding_model,
            tenant_id=tenant_id,
        )
    return process_document_inline(
        document_id=document_id,
        user_id=user_id,
        path=path,
        version_id=version_id,
        publish=publish,
        embedding_model=embedding_model,
        tenant_id=tenant_id,
    )


def enqueue_ingestion_job(job_id: str, user_id: str, task_ids: List[str]):
    from .ingestion_worker import run_ingestion_job

    if celery_app:
        return run_ingestion_job_task.delay(job_id=job_id, user_id=user_id, task_ids=task_ids)
    return run_ingestion_job(job_id=job_id, user_id=user_id, task_ids=task_ids)


if celery_app:

    @celery_app.task(
        bind=True,
        autoretry_for=(Exception,),
        retry_backoff=True,
        retry_kwargs={"max_retries": 3},
    )
    def process_document_task(  # type: ignore
        self,
        *,
        document_id: str,
        user_id: str,
        path: str,
        version_id: Optional[str],
        publish: bool = True,
        embedding_model: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        from .ingestion_worker import process_document_inline

        process_document_inline(
            document_id=document_id,
            user_id=user_id,
            path=path,
            version_id=version_id,
            publish=publish,
            embedding_model=embedding_model,
            tenant_id=tenant_id,
        )
        return document_id

    @celery_app.task(
        bind=True,
        autoretry_for=(Exception,),
        retry_backoff=True,
        retry_kwargs={"max_retries": 3},
    )
    def run_ingestion_job_task(self, *, job_id: str, user_id: str, task_ids: List[str]) -> str:  # type: ignore
        from .ingestion_worker import run_ingestion_job

        run_ingestion_job(job_id=job_id, user_id=user_id, task_ids=task_ids)
        return job_id
