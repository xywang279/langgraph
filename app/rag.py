from __future__ import annotations

import logging
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pdfplumber
import pytesseract
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from . import storage

logger = logging.getLogger(__name__)


EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

_ROOT_DIR = Path(__file__).resolve().parent.parent
_VECTOR_DIR = _ROOT_DIR / "data" / "vectorstores"
_VECTOR_DIR.mkdir(parents=True, exist_ok=True)

_TEXT_SPLITTER: Optional[RecursiveCharacterTextSplitter] = None
_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_OCR_AVAILABLE = True
try:
    pytesseract.get_tesseract_version()
except Exception:
    _OCR_AVAILABLE = False


class UserVectorStoreRetriever(BaseRetriever):
    """LangChain 1.0 compatible retriever bound to a user's FAISS index."""

    user_id: str
    top_k: int = 5
    min_score: float = 0.2

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        normalized = (query or "").strip()
        if not normalized:
            return []

        _ensure_user(self.user_id)
        store = _ensure_vector_store(self.user_id)
        if store is None:
            return _legacy_similarity_scan(
                user_id=self.user_id,
                query=normalized,
                top_k=self.top_k,
                min_score=self.min_score,
            )

        documents: List[Document] = []
        for doc, distance in store.similarity_search_with_score(
            normalized, k=self.top_k
        ):
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < self.min_score:
                continue
            metadata = dict(doc.metadata or {})
            metadata["score"] = similarity
            documents.append(
                Document(
                    page_content=_format_snippet(doc.page_content),
                    metadata=metadata,
                )
            )
        return documents


def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    global _TEXT_SPLITTER
    if _TEXT_SPLITTER is None:
        _TEXT_SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
    return _TEXT_SPLITTER


def _get_embedding_model() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info("loading embedding model %s", EMBED_MODEL)
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
    if not texts:
        return []
    model = _get_embedding_model()
    return model.embed_documents(list(texts))


def _sanitize_user_id(user_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (user_id or ""))
    return safe or "_default"


def _vector_store_dir(user_id: str) -> Path:
    safe = _sanitize_user_id(user_id)
    target = _VECTOR_DIR / safe
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_vector_store(user_id: str) -> Optional[FAISS]:
    base = _vector_store_dir(user_id)
    index_path = base / "index.faiss"
    store_path = base / "index.pkl"
    if not index_path.exists() or not store_path.exists():
        return None
    try:
        return FAISS.load_local(
            str(base),
            embeddings=_get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to load vector index for user=%s err=%s", user_id, exc)
        return None


def _persist_vector_store(user_id: str, store: FAISS) -> None:
    target = _vector_store_dir(user_id)
    store.save_local(str(target))


def _delete_vector_store(user_id: str) -> None:
    base = _vector_store_dir(user_id)
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)


def _append_to_vector_index(user_id: str, documents: List[Document]) -> None:
    if not documents:
        return
    store = _load_vector_store(user_id)
    embeddings = _get_embedding_model()
    if store is None:
        store = FAISS.from_documents(documents, embeddings)
    else:
        store.add_documents(documents)
    _persist_vector_store(user_id, store)


def _rebuild_vector_index(user_id: str) -> None:
    chunks = list(storage.iter_user_chunks(user_id))
    if not chunks:
        _delete_vector_store(user_id)
        return
    documents = _documents_from_chunks(chunks)
    embeddings = _get_embedding_model()
    store = FAISS.from_documents(documents, embeddings)
    _persist_vector_store(user_id, store)


def _ensure_vector_store(user_id: str) -> Optional[FAISS]:
    store = _load_vector_store(user_id)
    if store is not None:
        return store
    _rebuild_vector_index(user_id)
    return _load_vector_store(user_id)


def _documents_from_chunks(chunks: Iterable[Dict[str, Any]]) -> List[Document]:
    documents: List[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        metadata.setdefault("chunk_id", chunk["id"])
        metadata.setdefault("document_id", chunk["document_id"])
        metadata.setdefault("chunk_index", chunk["chunk_index"])
        documents.append(Document(page_content=chunk["content"], metadata=metadata))
    return documents


def _format_snippet(text: str, *, max_length: int = 300) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= max_length:
        return snippet
    return snippet[: max_length - 1].rstrip() + "..."


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (norm_a * norm_b)


def _legacy_similarity_scan(
    *, user_id: str, query: str, top_k: int, min_score: float
) -> List[Document]:
    embeddings = _embed_texts([query])
    if not embeddings:
        return []
    query_vector = embeddings[0]

    scored: List[Tuple[float, Document]] = []
    for chunk in storage.iter_user_chunks(user_id):
        score = _cosine_similarity(query_vector, chunk["embedding"])
        if score < min_score:
            continue
        metadata = dict(chunk.get("metadata") or {})
        metadata.setdefault("document_id", chunk["document_id"])
        metadata.setdefault("chunk_id", chunk["id"])
        metadata["score"] = score
        doc = Document(page_content=_format_snippet(chunk["content"]), metadata=metadata)
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


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
    logger.info(
        "processing document doc_id=%s user=%s path=%s", document_id, user_id, file_path
    )

    text = _extract_text(file_path)

    splitter = _get_text_splitter()
    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
    if not chunks:
        storage.update_document_status(
            document_id, status="empty", error="Document contains no readable content."
        )
        logger.warning("document empty doc_id=%s user=%s", document_id, user_id)
        return

    embeddings = _embed_texts(chunks)
    chunk_payloads: List[Dict[str, Any]] = []
    documents: List[Document] = []
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = uuid.uuid4().hex
        metadata = {
            "filename": file_path.name,
            "chunk_index": idx,
            "document_id": document_id,
            "chunk_id": chunk_id,
        }
        chunk_payloads.append(
            {
                "id": chunk_id,
                "chunk_index": idx,
                "content": chunk_text,
                "embedding": embedding,
                "metadata": metadata,
            }
        )
        documents.append(Document(page_content=chunk_text, metadata=metadata))

    storage.delete_chunks_for_document(document_id)
    storage.store_chunks(document_id=document_id, user_id=user_id, chunks=chunk_payloads)
    storage.update_document_status(document_id, status="ready")
    _rebuild_vector_index(user_id)
    logger.info(
        "document processed doc_id=%s user=%s chunks=%s", document_id, user_id, len(documents)
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
    embeddings = _embed_texts([cleaned])
    if not embeddings:
        return None
    document_id = storage.ensure_memory_document(user_id)
    chunk_id = uuid.uuid4().hex
    chunk_metadata = {"source": "memory", **(metadata or {})}
    chunk_metadata.update({"document_id": document_id, "chunk_id": chunk_id})
    payload = {
        "id": chunk_id,
        "chunk_index": int(time.time()),
        "content": cleaned,
        "embedding": embeddings[0],
        "metadata": chunk_metadata,
    }
    storage.store_chunks(
        document_id=document_id,
        user_id=user_id,
        chunks=[payload],
    )
    _append_to_vector_index(
        user_id,
        [Document(page_content=cleaned, metadata=chunk_metadata)],
    )
    return chunk_id


def _extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            with pdfplumber.open(file_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(page for page in pages if page)
            if text.strip():
                return text
            logger.warning("pdf text empty after extraction path=%s", file_path)
        except Exception as exc:
            logger.warning("pdf extraction failed path=%s err=%s", file_path, exc)
        # Fallback to OCR for scanned / image-only PDFs.
        ocr_text = _ocr_pdf(file_path)
        if ocr_text.strip():
            return ocr_text
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8", errors="ignore")


def _ocr_pdf(file_path: Path) -> str:
    """Use Tesseract OCR as a best-effort fallback for image-only PDFs."""
    if not _OCR_AVAILABLE:
        logger.warning("tesseract not available; skipping OCR path=%s", file_path)
        return ""
    try:
        with pdfplumber.open(file_path) as pdf:
            texts: List[str] = []
            for idx, page in enumerate(pdf.pages):
                try:
                    image = page.to_image(resolution=200).original.convert("RGB")
                    ocr = pytesseract.image_to_string(image)
                    if ocr.strip():
                        texts.append(ocr)
                except Exception as exc:
                    logger.warning("ocr failed page=%s path=%s err=%s", idx, file_path, exc)
            return "\n\n".join(chunk.strip() for chunk in texts if chunk.strip())
    except Exception as exc:
        logger.warning("ocr open pdf failed path=%s err=%s", file_path, exc)
        return ""


def retrieve(
    *,
    user_id: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.2,
) -> List[Dict[str, Any]]:
    retriever = UserVectorStoreRetriever(user_id=user_id, top_k=top_k, min_score=min_score)
    documents = retriever.invoke(query)

    results: List[Dict[str, Any]] = []
    for doc in documents:
        metadata = dict(doc.metadata or {})
        results.append(
            {
                "chunk_id": metadata.get("chunk_id"),
                "document_id": metadata.get("document_id"),
                "content": doc.page_content,
                "score": round(float(metadata.get("score", 0.0)), 4),
                "metadata": metadata,
            }
        )
    return results

