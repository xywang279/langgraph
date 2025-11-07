from __future__ import annotations

import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

from sentence_transformers import SentenceTransformer

from . import storage

logger = logging.getLogger(__name__)


EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

_EMBEDDINGS: Optional[SentenceTransformer] = None


def _get_embeddings() -> SentenceTransformer:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info("loading embedding model %s", EMBED_MODEL)
        _EMBEDDINGS = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDINGS


def _split_text(
    text: str, *, chunk_size: int = 800, chunk_overlap: int = 200
) -> List[Tuple[int, int, str]]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    length = len(cleaned)
    if not cleaned.strip():
        return []

    window = max(chunk_size, 200)
    stride = max(window - chunk_overlap, 100)

    segments: List[Tuple[int, int, str]] = []
    start = 0
    index = 0
    while start < length:
        stop = min(length, start + window)
        chunk = cleaned[start:stop]
        if chunk.strip():
            segments.append((start, stop, chunk.strip()))
        start += stride
        index += 1
    return segments


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    model = _get_embeddings()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (norm_a * norm_b)


def _format_snippet(text: str, *, max_length: int = 300) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= max_length:
        return snippet
    return snippet[: max_length - 1].rstrip() + "..."


def _ensure_user(user_id: str) -> None:
    if not user_id:
        raise ValueError("user_id is required")
    storage.upsert_user(user_id)


def process_document_file(
    *,
    document_id: str,
    user_id: str,
    file_path: Path,
) -> None:
    _ensure_user(user_id)
    logger.info("processing document doc_id=%s user=%s path=%s", document_id, user_id, file_path)

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

    segments = _split_text(text)
    if not segments:
        storage.update_document_status(
            document_id, status="empty", error="Document contains no readable content."
        )
        logger.warning("document empty doc_id=%s user=%s", document_id, user_id)
        return

    chunk_payloads: List[Dict[str, Any]] = []
    embeddings = embed_texts([segment for _, _, segment in segments])
    for (start, stop, chunk_text), embedding in zip(segments, embeddings):
        chunk_payloads.append(
            {
                "id": uuid.uuid4().hex,
                "chunk_index": len(chunk_payloads),
                "content": chunk_text,
                "embedding": embedding,
                "metadata": {
                    "start": start,
                    "end": stop,
                    "filename": file_path.name,
                },
            }
        )

    storage.delete_chunks_for_document(document_id)
    storage.store_chunks(document_id=document_id, user_id=user_id, chunks=chunk_payloads)
    storage.update_document_status(document_id, status="ready")
    logger.info(
        "document processed doc_id=%s user=%s chunks=%s", document_id, user_id, len(chunk_payloads)
    )


def store_memory_snippet(
    *,
    user_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    cleaned = text.strip()
    if not cleaned:
        return None
    _ensure_user(user_id)
    embeddings = embed_texts([cleaned])
    if not embeddings:
        return None
    document_id = storage.ensure_memory_document(user_id)
    chunk_id = uuid.uuid4().hex
    storage.store_chunks(
        document_id=document_id,
        user_id=user_id,
        chunks=[
            {
                "id": chunk_id,
                "chunk_index": int(time.time()),
                "content": cleaned,
                "embedding": embeddings[0],
                "metadata": metadata or {"source": "memory"},
            }
        ],
    )
    return chunk_id


def retrieve(
    *,
    user_id: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.2,
) -> List[Dict[str, Any]]:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    _ensure_user(user_id)
    embeddings = embed_texts([normalized_query])
    if not embeddings:
        return []
    query_vector = embeddings[0]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in storage.iter_user_chunks(user_id):
        score = _cosine_similarity(query_vector, chunk["embedding"])
        if score < min_score:
            continue
        snippet = _format_snippet(chunk["content"])
        metadata = dict(chunk.get("metadata") or {})
        scored.append(
            (
                score,
                {
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "content": snippet,
                    "score": round(score, 4),
                    "metadata": metadata,
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in scored[:top_k]]
