"""
Generate enrichment fields (analysis/knowledge/difficulty/tags) for core questions.

Usage:
  python scripts/enrich_questions.py --core data/ingest/questions/foo.core.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = (
    "You are an educational assistant. Produce enrichment only. "
    "Output a JSON object ONLY with keys: analysis, knowledge_points, difficulty, tags. "
    "analysis: string, knowledge_points: list of strings, difficulty: integer 1-5, tags: list of strings. "
    "If uncertain, return empty lists and difficulty 3."
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_enrich_index(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for record in _load_jsonl(path):
        qid = record.get("question_id")
        if qid:
            index[qid] = record
    return index


def _needs_enrichment(existing: Dict[str, Any]) -> bool:
    if not existing:
        return True
    if "analysis" not in existing:
        return True
    if "knowledge_points" not in existing:
        return True
    if "difficulty" not in existing:
        return True
    if "tags" not in existing:
        return True
    return False


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        cleaned = cleaned.rstrip("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _build_prompt(record: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Subject: {record.get('subject', 'unknown')}")
    if record.get("year"):
        lines.append(f"Year: {record['year']}")
    if record.get("grade"):
        lines.append(f"Grade: {record['grade']}")
    lines.append(f"Question stem: {record.get('question_stem', '')}")
    options = record.get("options") or []
    if options:
        opt_text = " ".join(f"{opt.get('key')}) {opt.get('text')}" for opt in options if opt.get("key"))
        lines.append(f"Options: {opt_text}")
    answer = record.get("answer") or {}
    if answer.get("value"):
        lines.append(f"Answer: {answer.get('value')}")
    lines.append("Generate enrichment fields for this question.")
    return "\n".join(lines).strip()


def _normalize_difficulty(value: Any) -> Optional[int]:
    try:
        num = int(value)
    except Exception:
        return None
    return max(1, min(5, num))


def _should_enrich(
    record: Dict[str, Any],
    *,
    mode: str,
    min_ocr_conf: float,
    min_seg_conf: float,
) -> bool:
    if mode == "all":
        return True
    if mode == "review":
        return bool(record.get("review_required"))
    if mode == "low_conf":
        ocr_conf = float(record.get("ocr_confidence") or 0.0)
        seg_conf = float(record.get("segmentation_confidence") or 0.0)
        return ocr_conf < min_ocr_conf or seg_conf < min_seg_conf
    return True


def enrich_questions(
    *,
    core_path: Path,
    out_path: Optional[Path] = None,
    mode: str = "review",
    min_ocr_conf: float = 0.7,
    min_seg_conf: float = 0.6,
    max_items: Optional[int] = None,
    overwrite: bool = False,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 2,
    sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    core_records = _load_jsonl(core_path)
    out_path = out_path or core_path.with_suffix(".enrich.jsonl")
    existing = _load_enrich_index(out_path)

    llm = ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
    )

    enriched: Dict[str, Dict[str, Any]] = {}
    processed = 0
    skipped = 0
    errors = 0

    for record in core_records:
        qid = record.get("question_id")
        if not qid:
            continue
        if not overwrite and qid in existing and not _needs_enrichment(existing[qid]):
            enriched[qid] = existing[qid]
            skipped += 1
            continue
        if not _should_enrich(record, mode=mode, min_ocr_conf=min_ocr_conf, min_seg_conf=min_seg_conf):
            if qid in existing:
                enriched[qid] = existing[qid]
            skipped += 1
            continue
        if max_items is not None and processed >= max_items:
            break

        prompt = _build_prompt(record)
        payload: Dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
                data = _extract_json(response.content)
                payload = {
                    "question_id": qid,
                    "analysis": {"text": data.get("analysis", ""), "generated": True},
                    "knowledge_points": {"items": data.get("knowledge_points", []) or [], "generated": True},
                    "difficulty": {
                        "value": _normalize_difficulty(data.get("difficulty")),
                        "generated": True,
                    },
                    "tags": {"items": data.get("tags", []) or [], "generated": True},
                }
                break
            except Exception:
                if attempt >= max_retries:
                    errors += 1
                    payload = {}
                else:
                    time.sleep(0.5)
        if payload:
            if qid in existing and not overwrite:
                merged = dict(existing[qid])
                merged.update(payload)
                payload = merged
            enriched[qid] = payload
            processed += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)

    with out_path.open("w", encoding="utf-8") as f:
        for record in core_records:
            qid = record.get("question_id")
            if not qid:
                continue
            if qid in enriched:
                f.write(json.dumps(enriched[qid], ensure_ascii=False) + "\n")
            elif qid in existing:
                f.write(json.dumps(existing[qid], ensure_ascii=False) + "\n")

    return {
        "core": str(core_path),
        "out": str(out_path),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich core question records with LLM-generated fields.")
    parser.add_argument("--core", type=Path, required=True, help="Path to core jsonl.")
    parser.add_argument("--out", type=Path, help="Output enrich jsonl path.")
    parser.add_argument(
        "--mode",
        choices=["review", "low_conf", "all"],
        default="review",
        help="Which records to enrich.",
    )
    parser.add_argument("--min-ocr-conf", type=float, default=0.7, help="OCR confidence threshold.")
    parser.add_argument("--min-seg-conf", type=float, default=0.6, help="Segmentation confidence threshold.")
    parser.add_argument("--max-items", type=int, help="Maximum records to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing enrich records.")
    parser.add_argument("--model", help="LLM model name.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries for LLM calls.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between requests.")
    args = parser.parse_args()

    result = enrich_questions(
        core_path=args.core,
        out_path=args.out,
        mode=args.mode,
        min_ocr_conf=args.min_ocr_conf,
        min_seg_conf=args.min_seg_conf,
        max_items=args.max_items,
        overwrite=args.overwrite,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
