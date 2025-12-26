from __future__ import annotations

import logging
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pdfplumber
import requests
import pytesseract
from PIL import Image
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

_CV2_AVAILABLE = False
try:  # optional dependency for table OCR on scanned PDFs/images
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
DEFAULT_TRANSCRIBE_MODEL = os.getenv("AUDIO_TRANSCRIBE_MODEL", "whisper-1")
DEFAULT_TRANSCRIBE_LANGUAGE = os.getenv("AUDIO_TRANSCRIBE_LANGUAGE")
DEFAULT_TRANSCRIBE_PROVIDER = os.getenv("AUDIO_TRANSCRIBE_PROVIDER", "openai").strip().lower()
def _parse_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


ENABLE_TABLE_OCR = (os.getenv("ENABLE_TABLE_OCR") or "").strip().lower() in {"1", "true", "yes", "on"}
TABLE_OCR_MAX_CELLS = max(1, _parse_int_env("TABLE_OCR_MAX_CELLS", 200))
TABLE_OCR_MIN_PIXELS = max(0, _parse_int_env("TABLE_OCR_MIN_PIXELS", 1200))
TABLE_OCR_MIN_LINE_FRAC = min(0.9, max(0.05, _parse_float_env("TABLE_OCR_MIN_LINE_FRAC", 0.25)))
PDF_TABLE_FALLBACK_STRATEGIES = [
    strategy.strip().lower()
    for strategy in (os.getenv("PDF_TABLE_FALLBACK_STRATEGIES") or "lines,text").split(",")
    if strategy.strip()
]

if ENABLE_TABLE_OCR and not _OCR_AVAILABLE:
    logger.warning("ENABLE_TABLE_OCR=1 but Tesseract is unavailable; table OCR disabled.")
if ENABLE_TABLE_OCR and not _CV2_AVAILABLE:
    logger.warning("ENABLE_TABLE_OCR=1 but OpenCV is unavailable; table OCR disabled.")


@dataclass
class ExtractedBlock:
    type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExtractedDocument:
    text: str
    blocks: List[ExtractedBlock]


class UserVectorStoreRetriever(BaseRetriever):
    """LangChain 1.0 compatible retriever bound to a user's FAISS index."""

    user_id: str
    tenant_id: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.2
    published_only: bool = True

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
                tenant_id=self.tenant_id,
                query=normalized,
                top_k=self.top_k,
                min_score=self.min_score,
                published_only=self.published_only,
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


def user_index_size(user_id: str) -> int:
    store = _load_vector_store(user_id)
    if store is None:
        return 0
    try:
        return int(getattr(store, "index", None).ntotal)
    except Exception:
        return 0


def index_ids_for_user(user_id: str) -> List[str]:
    store = _load_vector_store(user_id)
    if store is None:
        return []
    try:
        return [metadata.get("document_id", "") for metadata in store.docstore._dict.values()]  # type: ignore
    except Exception:
        return []


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


def _rebuild_vector_index(user_id: str, tenant_id: Optional[str] = None) -> None:
    chunks = list(storage.iter_published_chunks(user_id, tenant_id))
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
    *, user_id: str, tenant_id: Optional[str], query: str, top_k: int, min_score: float, published_only: bool
) -> List[Document]:
    embeddings = _embed_texts([query])
    if not embeddings:
        return []
    query_vector = embeddings[0]

    scored: List[Tuple[float, Document]] = []
    chunks_iter = storage.iter_published_chunks(user_id, tenant_id) if published_only else storage.iter_user_chunks(user_id, tenant_id)
    for chunk in chunks_iter:
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
    version_id: Optional[str] = None,
    publish: bool = True,
    embedding_model: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    _ensure_user(user_id)
    logger.info(
        "processing document doc_id=%s user=%s path=%s", document_id, user_id, file_path
    )

    extracted = _extract_text(file_path)

    splitter = _get_text_splitter()
    chunk_entries = _build_chunks_from_blocks(extracted, splitter)
    if not chunk_entries:
        storage.update_document_status(
            document_id, status="empty", error="Document contains no readable content."
        )
        if version_id:
            storage.update_document_version(
                version_id,
                status="empty",
                error="Document contains no readable content.",
            )
        logger.warning("document empty doc_id=%s user=%s", document_id, user_id)
        return

    embeddings = _embed_texts([chunk["content"] for chunk in chunk_entries])
    used_model = embedding_model or EMBED_MODEL
    chunk_payloads: List[Dict[str, Any]] = []
    documents: List[Document] = []
    for idx, (chunk_entry, embedding) in enumerate(zip(chunk_entries, embeddings)):
        chunk_text = chunk_entry["content"]
        chunk_id = uuid.uuid4().hex
        metadata = {
            "filename": file_path.name,
            "chunk_index": idx,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "version_id": version_id,
        }
        metadata.update(chunk_entry.get("metadata") or {})
        chunk_payloads.append(
            {
                "id": chunk_id,
                "version_id": version_id,
                "chunk_index": idx,
                "content": chunk_text,
                "embedding": embedding,
                "metadata": metadata,
            }
        )
        documents.append(Document(page_content=chunk_text, metadata=metadata))

    if version_id:
        storage.delete_chunks_for_version(version_id)
    else:
        storage.delete_chunks_for_document(document_id)
    storage.store_chunks(document_id=document_id, user_id=user_id, tenant_id=tenant_id, chunks=chunk_payloads)
    storage.update_document_status(document_id, status="ready", error=None, path=str(file_path))
    if version_id:
        storage.update_document_version(
            version_id,
            status="ready",
            path_raw=str(file_path),
            chunk_count=len(chunk_payloads),
            embedding_model=used_model,
        )
        if publish:
            storage.publish_document_version(document_id, version_id)
        storage.upsert_vector_stats(
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
            expected_vectors=len(chunk_payloads),
            stored_vectors=len(chunk_payloads),
            health_status="healthy",
            notes=None,
        )
    _rebuild_vector_index(user_id, tenant_id)
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
    document_id = storage.ensure_memory_document(user_id, tenant_id=None)
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
        tenant_id=None,
        chunks=[payload],
    )
    _append_to_vector_index(
        user_id,
        [Document(page_content=cleaned, metadata=chunk_metadata)],
    )
    return chunk_id


def _build_chunks_from_blocks(
    extracted: ExtractedDocument, splitter: RecursiveCharacterTextSplitter
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for block in extracted.blocks:
        base_meta = dict(block.metadata or {})
        base_meta.setdefault("structure_type", block.type)
        parts = _split_block_content(block, splitter)
        for part in parts:
            if block.type in {"code", "table"}:
                cleaned = part
            else:
                cleaned = part.strip()
            if not cleaned.strip():
                continue
            chunks.append({"content": cleaned, "metadata": dict(base_meta)})
    return chunks


def _split_block_content(
    block: ExtractedBlock, splitter: RecursiveCharacterTextSplitter
) -> List[str]:
    content = block.content or ""
    if not content.strip():
        return []
    if block.type in {"code", "table"}:
        return [content]
    return [chunk for chunk in splitter.split_text(content) if chunk.strip()]


def _make_extracted_document(blocks: List[ExtractedBlock]) -> ExtractedDocument:
    cleaned: List[ExtractedBlock] = []
    for block in blocks:
        content = block.content or ""
        if not content.strip():
            continue
        cleaned.append(ExtractedBlock(block.type, content, block.metadata))
    text = "\n\n".join(block.content for block in cleaned)
    return ExtractedDocument(text=text, blocks=cleaned)


def _split_structured_text(
    text: str, base_meta: Optional[Dict[str, Any]] = None
) -> List[ExtractedBlock]:
    if not text:
        return []
    lines = text.splitlines()
    blocks: List[ExtractedBlock] = []
    buffer: List[str] = []
    idx = 0
    base = dict(base_meta or {})

    def _flush_buffer() -> None:
        if buffer:
            payload = "\n".join(buffer).strip()
            if payload:
                blocks.append(ExtractedBlock("text", payload, dict(base) or None))
            buffer.clear()

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped.startswith("```"):
            _flush_buffer()
            fence = stripped[3:].strip()
            code_lines = [line]
            idx += 1
            while idx < len(lines):
                code_lines.append(lines[idx])
                if lines[idx].strip().startswith("```"):
                    break
                idx += 1
            meta = dict(base)
            if fence:
                meta["language"] = fence
            blocks.append(ExtractedBlock("code", "\n".join(code_lines).strip(), meta or None))
            idx += 1
            continue

        if _looks_like_table_header(lines, idx):
            _flush_buffer()
            table_lines = [lines[idx]]
            idx += 1
            while idx < len(lines) and _looks_like_table_line(lines[idx]):
                table_lines.append(lines[idx])
                idx += 1
            meta = dict(base)
            meta["rows"] = _parse_markdown_table(table_lines)
            blocks.append(ExtractedBlock("table", "\n".join(table_lines).strip(), meta or None))
            continue

        buffer.append(line)
        idx += 1

    _flush_buffer()
    return blocks


def _looks_like_table_header(lines: List[str], idx: int) -> bool:
    if idx + 1 >= len(lines):
        return False
    header = lines[idx].strip()
    divider = lines[idx + 1].strip()
    return _looks_like_table_line(header) and _is_table_divider(divider)


def _looks_like_table_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.count("|") < 2:
        return False
    cells = [cell for cell in stripped.split("|") if cell.strip()]
    return len(cells) >= 2


def _is_table_divider(line: str) -> bool:
    stripped = line.strip().strip("|").replace(":", "").replace("-", "")
    return stripped == ""


def _parse_markdown_table(lines: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned or _is_table_divider(cleaned):
            continue
        cells = [cell.strip() for cell in cleaned.strip("|").split("|")]
        rows.append(cells)
    return rows


def _table_to_markdown(rows: Sequence[Sequence[Any]]) -> str:
    normalized = [
        [("" if cell is None else str(cell)).strip() for cell in row] for row in (rows or [])
    ]
    if not normalized:
        return ""
    header = normalized[0]
    body = normalized[1:] or []
    if not body:
        body = [header]
    if not any(header):
        header = [f"col_{idx + 1}" for idx in range(len(body[0]) if body else 0)]
    separator = "| " + " | ".join("---" for _ in header) + " |"
    lines = ["| " + " | ".join(header) + " |", separator]
    for row in body:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _table_ocr_from_pil(
    image: "Image.Image",
    *,
    base_metadata: Dict[str, Any],
) -> Optional[ExtractedBlock]:
    """Best-effort table extraction from raster images using OpenCV line detection + Tesseract per cell."""
    if not (ENABLE_TABLE_OCR and _OCR_AVAILABLE and _CV2_AVAILABLE):
        return None
    if cv2 is None or np is None:  # pragma: no cover
        return None

    try:
        rgb = image.convert("RGB")
        arr = np.array(rgb)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        height, width = bw.shape[:2]
        if height < 50 or width < 50:
            return None

        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(15, width // 40), 1)
        )
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(15, height // 40))
        )
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)

        table_mask = cv2.bitwise_or(h_lines, v_lines)
        ys, xs = np.where(table_mask > 0)
        if xs.size < TABLE_OCR_MIN_PIXELS or ys.size < TABLE_OCR_MIN_PIXELS:
            return None

        def _group_indices(indices: "np.ndarray") -> List[int]:
            if indices.size == 0:
                return []
            grouped: List[int] = []
            start = int(indices[0])
            prev = int(indices[0])
            for val in indices[1:]:
                cur = int(val)
                if cur == prev + 1:
                    prev = cur
                    continue
                grouped.append(int((start + prev) / 2))
                start = prev = cur
            grouped.append(int((start + prev) / 2))
            return grouped

        def _extract_line_coords(mask: "np.ndarray", *, axis: int, min_frac: float) -> List[int]:
            if axis == 0:
                strength = (mask > 0).sum(axis=0)
                threshold = int(mask.shape[0] * min_frac)
            else:
                strength = (mask > 0).sum(axis=1)
                threshold = int(mask.shape[1] * min_frac)
            indices = np.where(strength >= threshold)[0]
            coords = _group_indices(indices)
            coords = sorted(set(coords))
            pruned: List[int] = []
            for coord in coords:
                if pruned and abs(coord - pruned[-1]) < 6:
                    continue
                pruned.append(coord)
            return pruned

        row_lines = _extract_line_coords(h_lines, axis=1, min_frac=TABLE_OCR_MIN_LINE_FRAC)
        col_lines = _extract_line_coords(v_lines, axis=0, min_frac=TABLE_OCR_MIN_LINE_FRAC)
        if len(row_lines) < 2 or len(col_lines) < 2:
            return None

        cells = (len(row_lines) - 1) * (len(col_lines) - 1)
        if cells <= 0 or cells > TABLE_OCR_MAX_CELLS:
            logger.info(
                "table ocr skipped: cells=%s exceeds limit=%s meta=%s",
                cells,
                TABLE_OCR_MAX_CELLS,
                {k: base_metadata.get(k) for k in ("filename", "page")},
            )
            return None

        def _ocr_cell(crop: "Image.Image") -> str:
            try:
                text = pytesseract.image_to_string(crop, config="--psm 6")
                return (text or "").strip()
            except Exception:  # pragma: no cover - best effort
                return ""

        rows: List[List[str]] = []
        padding = 2
        for r in range(len(row_lines) - 1):
            y0 = max(0, row_lines[r] + padding)
            y1 = min(height, row_lines[r + 1] - padding)
            if y1 <= y0:
                continue
            row: List[str] = []
            for c in range(len(col_lines) - 1):
                x0 = max(0, col_lines[c] + padding)
                x1 = min(width, col_lines[c + 1] - padding)
                if x1 <= x0:
                    row.append("")
                    continue
                crop = rgb.crop((x0, y0, x1, y1))
                row.append(_ocr_cell(crop))
            rows.append(row)

        # Drop empty rows/cols.
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            return None
        col_count = max(len(row) for row in rows)
        padded = [row + [""] * (col_count - len(row)) for row in rows]
        keep_cols = [
            idx
            for idx in range(col_count)
            if any(padded[ridx][idx].strip() for ridx in range(len(padded)))
        ]
        normalized = [[row[idx] for idx in keep_cols] for row in padded]
        if not normalized or len(keep_cols) < 2:
            return None

        markdown = _table_to_markdown(normalized)
        meta = dict(base_metadata)
        meta.update(
            {
                "source": "table_ocr",
                "method": "opencv_grid",
                "rows": normalized,
            }
        )
        logger.info(
            "table ocr extracted rows=%s cols=%s meta=%s",
            len(normalized),
            len(normalized[0]) if normalized else 0,
            {k: base_metadata.get(k) for k in ("filename", "page")},
        )
        return ExtractedBlock("table", markdown, meta)
    except Exception:  # pragma: no cover - best effort
        logger.warning(
            "table ocr failed meta=%s", base_metadata, exc_info=True
        )
        return None


def _extract_pdf_document(file_path: Path) -> ExtractedDocument:
    blocks: List[ExtractedBlock] = []
    table_keys: set = set()
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                meta = {"page": page_idx + 1, "filename": file_path.name}
                text = page.extract_text() or ""
                if text.strip():
                    blocks.extend(_split_structured_text(text, base_meta=meta))
                table_found = False

                def _push_tables(tables: List[List[List[Any]]], source: str) -> None:
                    nonlocal table_found
                    for table_idx, table in enumerate(tables):
                        normalized = [
                            [("" if cell is None else str(cell)).strip() for cell in row]
                            for row in (table or [])
                        ]
                        if not normalized:
                            continue
                        key = tuple(tuple(row) for row in normalized)
                        if key in table_keys:
                            continue
                        table_keys.add(key)
                        markdown = _table_to_markdown(normalized)
                        if not markdown.strip():
                            continue
                        table_meta = {**meta, "table_index": table_idx, "rows": normalized, "table_source": source}
                        blocks.append(ExtractedBlock("table", markdown, table_meta))
                        table_found = True

                try:
                    tables = page.extract_tables() or []
                    _push_tables(tables, "pdf_default")
                except Exception as table_exc:
                    logger.warning("table extraction failed page=%s path=%s err=%s", page_idx, file_path, table_exc)

                if not table_found and PDF_TABLE_FALLBACK_STRATEGIES:
                    for strategy in PDF_TABLE_FALLBACK_STRATEGIES:
                        if strategy not in {"lines", "text"}:
                            continue
                        try:
                            tables = page.extract_tables(
                                table_settings={
                                    "vertical_strategy": strategy,
                                    "horizontal_strategy": strategy,
                                }
                            ) or []
                            _push_tables(tables, f"pdf_{strategy}")
                        except Exception as table_exc:
                            logger.warning(
                                "table fallback failed strategy=%s page=%s path=%s err=%s",
                                strategy,
                                page_idx,
                                file_path,
                                table_exc,
                            )

                if ENABLE_TABLE_OCR and not table_found:
                    try:
                        image = page.to_image(resolution=200).original.convert("RGB")
                    except Exception as exc:
                        logger.warning("pdf rasterize failed page=%s path=%s err=%s", page_idx, file_path, exc)
                        image = None
                    if image is not None:
                        table_block = _table_ocr_from_pil(
                            image,
                            base_metadata={"page": page_idx + 1, "filename": file_path.name, "source": "pdf_ocr"},
                        )
                        if table_block:
                            blocks.append(table_block)
            if blocks:
                return _make_extracted_document(blocks)
    except Exception as exc:
        logger.warning("pdf extraction failed path=%s err=%s", file_path, exc)
    ocr_blocks = _ocr_pdf_blocks(file_path)
    if ocr_blocks:
        return _make_extracted_document(ocr_blocks)
    return _make_extracted_document([])


def _transcribe_audio(file_path: Path) -> str:
    provider = (DEFAULT_TRANSCRIBE_PROVIDER or "openai").lower()
    base_url_override = os.getenv("AUDIO_TRANSCRIBE_BASE_URL")
    # If the base URL looks like a direct /transcribe endpoint, prefer fastwhisper even if provider is unset.
    if provider == "openai" and base_url_override and "transcribe" in base_url_override:
        logger.info("AUDIO_TRANSCRIBE_BASE_URL contains 'transcribe'; routing to fastwhisper provider.")
        provider = "fastwhisper"
    if provider == "fastwhisper":
        base_url = base_url_override
        if not base_url:
            raise RuntimeError("AUDIO_TRANSCRIBE_BASE_URL is required for fastwhisper provider.")
        endpoint = base_url.rstrip("/")
        data: Dict[str, Any] = {}
        if DEFAULT_TRANSCRIBE_LANGUAGE:
            data["language"] = DEFAULT_TRANSCRIBE_LANGUAGE
        logger.info("audio transcription provider=fastwhisper endpoint=%s file=%s", endpoint, file_path.name)
        with file_path.open("rb") as audio_f:
            resp = requests.post(endpoint, files={"file": audio_f}, data=data, timeout=60)
        if resp.status_code >= 300:
            raise RuntimeError(f"FastWhisper transcription failed: HTTP {resp.status_code} body={resp.text[:500]}")
        try:
            payload = resp.json()
        except Exception as exc:
            raise RuntimeError(f"FastWhisper transcription returned non-JSON response: {exc}") from exc
        transcript = (payload.get("text") or "").strip()
        if not transcript:
            raise RuntimeError("FastWhisper transcription returned empty text.")
        return transcript

    if provider != "openai":
        raise RuntimeError(f"Unsupported audio transcription provider '{provider}'.")

    api_key = os.getenv("AUDIO_TRANSCRIBE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("AUDIO_TRANSCRIBE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("Audio transcription requires OPENAI_API_KEY or AUDIO_TRANSCRIBE_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("openai package is required for audio transcription.") from exc

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    logger.info("audio transcription provider=openai base_url=%s file=%s", base_url or "https://api.openai.com/v1", file_path.name)
    with file_path.open("rb") as audio_f:
        response = client.audio.transcriptions.create(
            model=DEFAULT_TRANSCRIBE_MODEL or "whisper-1",
            file=audio_f,
            language=DEFAULT_TRANSCRIBE_LANGUAGE or None,
            response_format="text",
        )
    transcript = response if isinstance(response, str) else getattr(response, "text", None) or ""
    if not transcript.strip():
        raise RuntimeError("Audio transcription returned empty result.")
    return transcript


def _extract_text(file_path: Path) -> ExtractedDocument:
    suffix = file_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        blocks: List[ExtractedBlock] = []
        try:
            with Image.open(file_path) as img:
                rgb = img.convert("RGB")
                table_block = _table_ocr_from_pil(
                    rgb,
                    base_metadata={"filename": file_path.name, "source": "image"},
                )
                if table_block:
                    blocks.append(table_block)
                if _OCR_AVAILABLE:
                    text = pytesseract.image_to_string(rgb)
                else:
                    text = ""
                blocks.append(
                    ExtractedBlock(
                        "image_ocr",
                        text,
                        {"source": "image", "filename": file_path.name},
                    )
                )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("image extraction failed path=%s err=%s", file_path, exc)
            blocks.append(
                ExtractedBlock(
                    "image_ocr",
                    _ocr_image(file_path),
                    {"source": "image", "filename": file_path.name},
                )
            )
        return _make_extracted_document(blocks)

    if suffix in AUDIO_EXTENSIONS:
        transcript = _transcribe_audio(file_path)
        meta: Dict[str, Any] = {
            "source": "audio",
            "filename": file_path.name,
            "provider": DEFAULT_TRANSCRIBE_PROVIDER or "openai",
            "model": DEFAULT_TRANSCRIBE_MODEL,
        }
        if DEFAULT_TRANSCRIBE_LANGUAGE:
            meta["language"] = DEFAULT_TRANSCRIBE_LANGUAGE
        return _make_extracted_document([ExtractedBlock("transcript", transcript, meta)])

    if suffix == ".pdf":
        pdf_doc = _extract_pdf_document(file_path)
        if pdf_doc.text.strip():
            return pdf_doc

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    blocks = _split_structured_text(raw_text, base_meta={"filename": file_path.name})
    return _make_extracted_document(blocks)


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


def _ocr_pdf_blocks(file_path: Path) -> List[ExtractedBlock]:
    """OCR a PDF into text blocks, with optional table OCR for scanned tables."""
    if not _OCR_AVAILABLE:
        logger.warning("tesseract not available; skipping OCR path=%s", file_path)
        return []
    blocks: List[ExtractedBlock] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_meta = {"page": idx + 1, "filename": file_path.name, "source": "pdf_ocr"}
                try:
                    image = page.to_image(resolution=200).original.convert("RGB")
                except Exception as exc:
                    logger.warning("pdf rasterize failed page=%s path=%s err=%s", idx, file_path, exc)
                    continue

                table_block = _table_ocr_from_pil(image, base_metadata=dict(page_meta))
                if table_block:
                    blocks.append(table_block)

                try:
                    text = pytesseract.image_to_string(image)
                except Exception as exc:
                    logger.warning("pdf ocr failed page=%s path=%s err=%s", idx, file_path, exc)
                    text = ""
                if text.strip():
                    blocks.extend(_split_structured_text(text, base_meta=page_meta))
    except Exception as exc:
        logger.warning("ocr open pdf failed path=%s err=%s", file_path, exc)
        return []
    return blocks


def _ocr_image(file_path: Path) -> str:
    if not _OCR_AVAILABLE:
        logger.warning("tesseract not available; skipping OCR path=%s", file_path)
        return ""
    try:
        with Image.open(file_path) as img:
            text = pytesseract.image_to_string(img.convert("RGB"))
            return text or ""
    except Exception as exc:
        logger.warning("image ocr failed path=%s err=%s", file_path, exc)
        return ""


def retrieve(
    *,
    user_id: str,
    tenant_id: Optional[str] = None,
    query: str,
    top_k: int = 5,
    min_score: float = 0.2,
    published_only: bool = True,
) -> List[Dict[str, Any]]:
    retriever = UserVectorStoreRetriever(
        user_id=user_id,
        tenant_id=tenant_id,
        top_k=top_k,
        min_score=min_score,
        published_only=published_only,
    )
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
