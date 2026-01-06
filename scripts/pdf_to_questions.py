"""
Quick-and-dirty pipeline to turn scanned PDFs into extracted images + questions JSON.

Usage:
  python scripts/pdf_to_questions.py --pdf input.pdf --out-dir data/ingest --dpi 300
  # by default renders a full-page PNG when no embedded images exist; disable via --no-render-page-when-empty
  # per-question crops (vertical slices) are enabled by default; disable via --no-crop-questions

Outputs (inside --out-dir):
  - images/<pdf-stem>/page-001-img-001.png (raw images extracted from PDF)
  - images/<pdf-stem>/page-001-q001.png (per-question crops when detected)
  - questions/<pdf-stem>.json (legacy list of question dicts with text and image paths)
  - questions/<pdf-stem>.core.jsonl (extract-only core question records)
  - questions/<pdf-stem>.enrich.jsonl (optional enrichment stub records)
  - questions/<pdf-stem>.md (markdown for RAG ingestion)
  - questions/review_queue.jsonl (low-confidence review queue)

Prereqs:
  - Tesseract installed locally; set env TESSERACT_CMD if it's not on PATH.
  - Python deps already in requirements.txt: pdfplumber, pillow, pytesseract.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pdfplumber
import pytesseract
from PIL import Image, ImageOps

_CV2_AVAILABLE = False
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


def configure_tesseract() -> None:
    """Use TESSERACT_CMD env if provided."""
    custom_cmd = os.getenv("TESSERACT_CMD")
    if custom_cmd:
        pytesseract.pytesseract.tesseract_cmd = custom_cmd


def _preprocess_for_ocr(
    image: Image.Image,
    *,
    enable: bool,
    denoise: bool,
    binarize: bool,
    sharpen: bool,
    deskew: bool,
) -> Image.Image:
    if not enable or not _CV2_AVAILABLE:
        return image
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if denoise:
        gray = cv2.medianBlur(gray, 3)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
    if binarize:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if deskew:
        coords = np.column_stack(np.where(gray < 255))
        if coords.size:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.1:
                height, width = gray.shape[:2]
                matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
                gray = cv2.warpAffine(
                    gray,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
    return Image.fromarray(gray).convert("RGB")


def extract_page_images(
    page: pdfplumber.page.Page, base_dir: Path, page_idx: int, render_if_empty: bool, dpi: int
) -> List[Dict]:
    """
    Extract embedded images in a PDF page to disk.
    If no embedded images are found and render_if_empty=True, rasterizes the whole page as a fallback.
    Returns list of dicts containing image path and optional bbox.
    """
    saved: List[Dict] = []
    if page.images:
        for img_idx, img in enumerate(page.images, start=1):
            try:
                extracted = page.extract_image(img["object_id"])
                if not extracted:
                    continue
                img_bytes = extracted["image"]
                img_ext = extracted.get("ext", "png")
                img_out = base_dir / f"page-{page_idx:03d}-img-{img_idx:03d}.{img_ext}"
                with open(img_out, "wb") as f:
                    f.write(img_bytes)
                bbox = _pdf_image_bbox_to_ratio(img, page)
                saved.append({"path": str(img_out), "bbox": bbox, "role": "embedded"})
            except Exception:
                # Best-effort: skip images that fail to extract.
                continue

    if not saved and render_if_empty:
        # Fallback: render whole page to an image so scanned PDFs still get a visual.
        pil_img: Image.Image = page.to_image(resolution=dpi).original
        img_out = base_dir / f"page-{page_idx:03d}-full.png"
        pil_img.save(img_out)
        saved.append({"path": str(img_out), "bbox": [0.0, 0.0, 1.0, 1.0], "role": "page_image"})

    return saved


def ocr_page_to_text(
    page: pdfplumber.page.Page,
    dpi: int = 300,
    *,
    ocr_lang: str = "chi_sim+eng",
    preprocess: bool = False,
    preprocess_denoise: bool = False,
    preprocess_binarize: bool = False,
    preprocess_sharpen: bool = False,
    preprocess_deskew: bool = False,
    save_preprocessed_path: Optional[Path] = None,
) -> Dict:
    """Rasterize a page and run OCR. Returns raw text and data with boxes."""
    pil_img: Image.Image = page.to_image(resolution=dpi).original
    ocr_img = _preprocess_for_ocr(
        pil_img,
        enable=preprocess,
        denoise=preprocess_denoise,
        binarize=preprocess_binarize,
        sharpen=preprocess_sharpen,
        deskew=preprocess_deskew,
    )
    if save_preprocessed_path:
        ocr_img.save(save_preprocessed_path)
    text = pytesseract.image_to_string(ocr_img, lang=ocr_lang)
    data = pytesseract.image_to_data(ocr_img, lang=ocr_lang, output_type=pytesseract.Output.DICT)
    return {"text": text, "data": data, "image": pil_img, "ocr_image": ocr_img}


def split_questions(raw_text: str) -> List[str]:
    """
    Naive question splitter for K-12 style materials.
    Splits on lines that look like numbered questions: 1. / 1、/ 1．
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    questions: List[str] = []
    buffer: List[str] = []
    pattern = re.compile(r"^\d+\s*[\.、\)]")

    for line in lines:
        if pattern.match(line) and buffer:
            questions.append("\n".join(buffer).strip())
            buffer = [line]
        else:
            buffer.append(line)

    if buffer:
        questions.append("\n".join(buffer).strip())
    return questions


def find_question_boundaries(ocr_data: Dict, page_height: int) -> List[Dict]:
    """
    Use OCR line data to estimate vertical spans for each question.
    Heuristic: a line starting with a question number marks the start; end is before next start.
    """
    pattern = re.compile(r"^\d+\s*[\.、\)]")
    starts = []
    n = len(ocr_data["text"])
    seen_lines = set()
    for i in range(n):
        txt = ocr_data["text"][i].strip()
        if not txt:
            continue
        line_key = (
            ocr_data.get("block_num", [None])[i],
            ocr_data.get("par_num", [None])[i],
            ocr_data.get("line_num", [None])[i],
        )
        if line_key in seen_lines:
            continue
        seen_lines.add(line_key)
        if pattern.match(txt):
            top = ocr_data["top"][i]
            height = ocr_data["height"][i]
            starts.append({"y": top, "h": height})

    if not starts:
        return []

    boundaries: List[Dict] = []
    for idx, start in enumerate(starts):
        y0 = max(0, start["y"] - 10)
        if idx + 1 < len(starts):
            next_start = starts[idx + 1]["y"]
            y1 = max(y0 + start["h"] + 10, next_start - 5)
        else:
            y1 = page_height
        boundaries.append({"y0": y0, "y1": y1})
    return boundaries


def crop_question_images(page_image: Image.Image, boundaries: List[Dict], base_dir: Path, page_idx: int) -> List[str]:
    """Crop the page into regions per question based on vertical boundaries."""
    saved = []
    width, height = page_image.size
    for q_idx, b in enumerate(boundaries, start=1):
        y0 = max(0, b["y0"])
        y1 = min(height, b["y1"])
        if y1 - y0 < 20:
            continue
        crop = page_image.crop((0, y0, width, y1))
        crop = ImageOps.expand(crop, border=4, fill="white")
        out_path = base_dir / f"page-{page_idx:03d}-q{q_idx:03d}.png"
        crop.save(out_path)
        saved.append(str(out_path))
    return saved


def _normalize_whitespace(text: str) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _build_ocr_lines(ocr_data: Dict) -> List[Dict]:
    lines: Dict[Tuple, Dict] = {}
    n = len(ocr_data.get("text", []))
    for i in range(n):
        txt = (ocr_data["text"][i] or "").strip()
        if not txt:
            continue
        conf = ocr_data.get("conf", [None])[i]
        try:
            conf_val = int(conf)
        except Exception:
            conf_val = -1
        key = (
            ocr_data.get("block_num", [None])[i],
            ocr_data.get("par_num", [None])[i],
            ocr_data.get("line_num", [None])[i],
        )
        entry = lines.setdefault(
            key,
            {
                "words": [],
                "left": ocr_data["left"][i],
                "top": ocr_data["top"][i],
                "right": ocr_data["left"][i] + ocr_data["width"][i],
                "bottom": ocr_data["top"][i] + ocr_data["height"][i],
                "confs": [],
            },
        )
        entry["words"].append(txt)
        entry["left"] = min(entry["left"], ocr_data["left"][i])
        entry["top"] = min(entry["top"], ocr_data["top"][i])
        entry["right"] = max(entry["right"], ocr_data["left"][i] + ocr_data["width"][i])
        entry["bottom"] = max(entry["bottom"], ocr_data["top"][i] + ocr_data["height"][i])
        if conf_val >= 0:
            entry["confs"].append(conf_val)
    result = []
    for entry in lines.values():
        text = " ".join(entry["words"]).strip()
        if not text:
            continue
        avg_conf = sum(entry["confs"]) / len(entry["confs"]) if entry["confs"] else 0.0
        result.append(
            {
                "text": text,
                "bbox": [entry["left"], entry["top"], entry["right"], entry["bottom"]],
                "conf": avg_conf,
            }
        )
    return result


def _mean_conf(ocr_data: Dict, bounds: Optional[Tuple[int, int]] = None) -> float:
    n = len(ocr_data.get("text", []))
    confs: List[int] = []
    y0 = y1 = None
    if bounds:
        y0, y1 = bounds
    for i in range(n):
        txt = (ocr_data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf_val = int(ocr_data.get("conf", [None])[i])
        except Exception:
            conf_val = -1
        if conf_val < 0:
            continue
        if y0 is not None and y1 is not None:
            center = ocr_data["top"][i] + ocr_data["height"][i] / 2
            if center < y0 or center > y1:
                continue
        confs.append(conf_val)
    if not confs:
        return 0.0
    return max(0.0, min(1.0, sum(confs) / len(confs) / 100.0))


def _calc_segmentation_confidence(
    *,
    has_boundary: bool,
    stem_len: int,
    options_count: int,
    has_answer: bool,
    min_stem_chars: int,
) -> float:
    score = 0.3
    if has_boundary:
        score += 0.2
    if stem_len >= min_stem_chars:
        score += 0.2
    if options_count >= 2:
        score += 0.2
    if has_answer:
        score += 0.1
    return max(0.0, min(1.0, score))


def _options_incomplete(options: Sequence[Dict]) -> bool:
    if not options:
        return False
    keys = [opt.get("key") for opt in options if opt.get("key")]
    if not keys:
        return False
    if keys[0] != "A":
        return True
    expected = [chr(ord("A") + i) for i in range(len(keys))]
    return keys != expected


def _extract_options(question_text: str) -> Tuple[str, List[Dict]]:
    lines = [ln.strip() for ln in question_text.splitlines() if ln.strip()]
    options: List[Dict] = []
    stem_lines: List[str] = []
    option_pattern = re.compile(r"^\s*([A-D])\s*[\.\)）\]、．]\s*(.+)$")
    answer_pattern = re.compile(r"(答案|正确答案|参考答案|解答|解)[:：]")
    option_started = False
    for line in lines:
        if answer_pattern.search(line):
            continue
        match = option_pattern.match(line)
        if match:
            option_started = True
            options.append({"key": match.group(1), "text": match.group(2).strip()})
            continue
        if option_started and options:
            options[-1]["text"] = f"{options[-1]['text']} {line}".strip()
            continue
        stem_lines.append(line)
    stem = "\n".join(stem_lines).strip()
    if not stem:
        stem = "\n".join(lines).strip()
    return stem, options


def _extract_answer(question_text: str, ocr_lines: Sequence[Dict]) -> Optional[Dict]:
    answer_pattern = re.compile(
        r"(答案|正确答案|参考答案|解答|解)[:：]?\s*([A-D]|对|错|正确|错误|T|F)",
        re.IGNORECASE,
    )
    for line in question_text.splitlines():
        match = answer_pattern.search(line)
        if match:
            return {"value": match.group(2).upper(), "evidence": line.strip(), "source_bbox": None}
    for line in ocr_lines:
        match = answer_pattern.search(line["text"])
        if match:
            return {
                "value": match.group(2).upper(),
                "evidence": line["text"],
                "source_bbox": line["bbox"],
            }
    return None


def _to_ratio_bbox(bbox: Sequence[float], width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = bbox
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]


def _pdf_image_bbox_to_ratio(img: Dict, page: pdfplumber.page.Page) -> Optional[List[float]]:
    try:
        x0 = float(img.get("x0", 0.0))
        x1 = float(img.get("x1", 0.0))
        top = float(img.get("top", 0.0))
        bottom = float(img.get("bottom", 0.0))
        if page.width <= 0 or page.height <= 0:
            return None
        return [
            max(0.0, min(1.0, x0 / page.width)),
            max(0.0, min(1.0, top / page.height)),
            max(0.0, min(1.0, x1 / page.width)),
            max(0.0, min(1.0, bottom / page.height)),
        ]
    except Exception:
        return None


def _detect_subject(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    subject_map = [
        ("math", ["math", "数学"]),
        ("physics", ["physics", "物理"]),
        ("chemistry", ["chemistry", "化学"]),
        ("biology", ["biology", "生物"]),
        ("english", ["english", "英语"]),
        ("chinese", ["chinese", "语文", "中文"]),
        ("history", ["history", "历史"]),
        ("geography", ["geography", "地理"]),
        ("politics", ["politics", "政治", "思想品德", "道德与法治"]),
        ("science", ["science", "科学"]),
        ("cs", ["computer", "计算机", "信息技术"]),
    ]
    for subject, keywords in subject_map:
        for kw in keywords:
            if kw.lower() in lowered:
                return subject
    return None


def _detect_year(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _detect_grade(text: str) -> Optional[str]:
    if not text:
        return None
    grade_patterns = [
        (r"grade\s*(1[0-2]|[1-9])", "grade_{}"),
        (r"(一年级|一年級)", "grade_1"),
        (r"(二年级|二年級)", "grade_2"),
        (r"(三年级|三年級)", "grade_3"),
        (r"(四年级|四年級)", "grade_4"),
        (r"(五年级|五年級)", "grade_5"),
        (r"(六年级|六年級)", "grade_6"),
        (r"(七年级|七年級|初一)", "grade_7"),
        (r"(八年级|八年級|初二)", "grade_8"),
        (r"(九年级|九年級|初三)", "grade_9"),
        (r"(高一)", "grade_10"),
        (r"(高二)", "grade_11"),
        (r"(高三)", "grade_12"),
    ]
    for pattern, label in grade_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        if "{}" in label:
            return label.format(match.group(1))
        return label
    return None


def pdf_to_questions(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
    render_if_empty: bool = True,
    crop_questions: bool = True,
    *,
    ocr_lang: str = "chi_sim+eng",
    preprocess: bool = False,
    preprocess_denoise: bool = False,
    preprocess_binarize: bool = False,
    preprocess_sharpen: bool = False,
    preprocess_deskew: bool = False,
    save_preprocessed: bool = False,
    doc_meta_pages: int = 2,
    subject: Optional[str] = None,
    year: Optional[int] = None,
    grade: Optional[str] = None,
    ocr_min_conf: float = 0.7,
    seg_min_conf: float = 0.6,
    min_stem_chars: int = 20,
) -> Dict:
    """Process a PDF into OCR text, extracted images, and naive question list."""
    pdf_stem = pdf_path.stem
    images_dir = out_dir / "images" / pdf_stem
    images_dir.mkdir(parents=True, exist_ok=True)
    questions_dir = out_dir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    doc_text_samples: List[str] = []
    all_questions: List[Dict] = []
    core_records: List[Dict] = []
    review_queue: List[Dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_images = extract_page_images(
                page, images_dir, idx, render_if_empty=render_if_empty, dpi=dpi
            )
            page_image_paths = [item.get("path") for item in page_images if item.get("path")]
            preprocessed_path = None
            if save_preprocessed:
                preprocessed_path = images_dir / f"page-{idx:03d}-preprocessed.png"
            ocr_out = ocr_page_to_text(
                page,
                dpi=dpi,
                ocr_lang=ocr_lang,
                preprocess=preprocess,
                preprocess_denoise=preprocess_denoise,
                preprocess_binarize=preprocess_binarize,
                preprocess_sharpen=preprocess_sharpen,
                preprocess_deskew=preprocess_deskew,
                save_preprocessed_path=preprocessed_path,
            )
            page_text = ocr_out["text"]
            print(
                f"[page {idx}] embedded/fallback images saved: {len(page_images)} (fallback={render_if_empty})"
            )

            if idx <= max(1, doc_meta_pages):
                doc_text_samples.append(page_text)

            question_chunks = split_questions(page_text) or [page_text.strip()]

            question_crops: List[str] = []
            boundaries: List[Dict] = []
            if crop_questions:
                boundaries = find_question_boundaries(ocr_out["data"], ocr_out["image"].height)
                if boundaries:
                    question_crops = crop_question_images(ocr_out["image"], boundaries, images_dir, idx)
                    print(f"[page {idx}] question crops: {len(question_crops)}")
                else:
                    print(f"[page {idx}] no question starts detected for cropping")

            ocr_lines = _build_ocr_lines(ocr_out["data"])
            page_conf = _mean_conf(ocr_out["data"])
            page_width, page_height = ocr_out["image"].size

            for q_idx, text in enumerate(question_chunks, start=1):
                question_id = f"{pdf_stem}-p{idx:03d}-q{q_idx:03d}"
                boundary = boundaries[q_idx - 1] if q_idx - 1 < len(boundaries) else None
                crop_path = question_crops[q_idx - 1] if q_idx - 1 < len(question_crops) else None

                images_for_question: List[str] = []
                if crop_path:
                    images_for_question = [crop_path]
                elif page_image_paths:
                    images_for_question = page_image_paths

                all_questions.append(
                    {
                        "id": question_id,
                        "source_pdf": str(pdf_path),
                        "page": idx,
                        "text": text,
                        "images": images_for_question,
                        "status": "draft",
                    }
                )

                question_stem, options = _extract_options(text)
                question_stem = _normalize_whitespace(question_stem)
                if not question_stem:
                    question_stem = _normalize_whitespace(text)
                for opt in options:
                    opt["text"] = _normalize_whitespace(opt.get("text", ""))

                answer = _extract_answer(text, ocr_lines)
                if answer and isinstance(answer.get("source_bbox"), list):
                    answer["source_bbox"] = _to_ratio_bbox(
                        answer["source_bbox"], page_width, page_height
                    )

                figures: List[Dict] = []
                if crop_path and boundary:
                    bbox = _to_ratio_bbox(
                        [0, boundary["y0"], page_width, boundary["y1"]],
                        page_width,
                        page_height,
                    )
                    figures.append(
                        {
                            "page": idx,
                            "bbox": bbox,
                            "path": crop_path,
                            "role": "question_crop",
                        }
                    )
                elif page_images:
                    for img in page_images:
                        figures.append(
                            {
                                "page": idx,
                                "bbox": img.get("bbox") or [0.0, 0.0, 1.0, 1.0],
                                "path": img.get("path"),
                                "role": img.get("role", "image"),
                            }
                        )

                bounds = None
                if boundary:
                    bounds = (int(boundary["y0"]), int(boundary["y1"]))
                ocr_conf = _mean_conf(ocr_out["data"], bounds=bounds) if bounds else page_conf
                seg_conf = _calc_segmentation_confidence(
                    has_boundary=bool(boundary),
                    stem_len=len(question_stem),
                    options_count=len(options),
                    has_answer=bool(answer),
                    min_stem_chars=min_stem_chars,
                )

                review_reasons: List[str] = []
                if ocr_conf < ocr_min_conf:
                    review_reasons.append("low_ocr_conf")
                if seg_conf < seg_min_conf:
                    review_reasons.append("low_segmentation_conf")
                if len(question_stem) < min_stem_chars:
                    review_reasons.append("short_stem")
                if _options_incomplete(options):
                    review_reasons.append("options_incomplete")

                review_required = bool(review_reasons)
                core_record = {
                    "question_id": question_id,
                    "source_pdf": str(pdf_path),
                    "page": idx,
                    "question_stem": question_stem,
                    "options": options,
                    "figures": figures,
                    "formula_latex": [],
                    "answer": answer,
                    "ocr_confidence": round(float(ocr_conf), 4),
                    "segmentation_confidence": round(float(seg_conf), 4),
                    "review_required": review_required,
                }
                core_records.append(core_record)

                if review_required:
                    review_queue.append(
                        {
                            "question_id": question_id,
                            "page": idx,
                            "reasons": review_reasons,
                            "ocr_confidence": round(float(ocr_conf), 4),
                            "segmentation_confidence": round(float(seg_conf), 4),
                        }
                    )

    subject_override = (subject or "").strip() or None
    grade_override = (grade or "").strip() or None
    doc_text = "\n".join([pdf_path.name] + doc_text_samples)
    doc_subject = subject_override or _detect_subject(doc_text) or "unknown"
    doc_year = year if year is not None else _detect_year(doc_text)
    doc_grade = grade_override or _detect_grade(doc_text)

    for record in core_records:
        record["subject"] = doc_subject
        if doc_year is not None:
            record["year"] = doc_year
        if doc_grade:
            record["grade"] = doc_grade

    out_json = questions_dir / f"{pdf_stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    core_jsonl = questions_dir / f"{pdf_stem}.core.jsonl"
    with open(core_jsonl, "w", encoding="utf-8") as f:
        for record in core_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    enrich_records: List[Dict] = []
    if not doc_grade:
        for record in core_records:
            enrich_records.append(
                {"question_id": record["question_id"], "grade": {"value": "unknown", "generated": False}}
            )
    enrich_jsonl = None
    if enrich_records:
        enrich_jsonl = questions_dir / f"{pdf_stem}.enrich.jsonl"
        with open(enrich_jsonl, "w", encoding="utf-8") as f:
            for record in enrich_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    md_out = questions_dir / f"{pdf_stem}.md"
    md_blocks: List[str] = []
    for record in core_records:
        lines = []
        meta_parts = [f"Subject: {record.get('subject', 'unknown')}"]
        if "year" in record:
            meta_parts.append(f"Year: {record['year']}")
        if "grade" in record:
            meta_parts.append(f"Grade: {record['grade']}")
        lines.append(" ".join(meta_parts))
        lines.append(f"Source: {Path(record['source_pdf']).name} page {record['page']}")
        lines.append(f"Q: {record['question_stem']}")
        if record.get("options"):
            options_text = " ".join(
                f"{opt.get('key')}) {opt.get('text')}" for opt in record["options"] if opt.get("key")
            )
            if options_text:
                lines.append(f"Options: {options_text}")
        if record.get("answer"):
            evidence = record["answer"].get("evidence")
            if evidence:
                lines.append(f"Answer: {record['answer'].get('value')} (evidence: {evidence})")
            else:
                lines.append(f"Answer: {record['answer'].get('value')}")
        if record.get("figures"):
            fig_refs = "; ".join(
                f"{fig.get('path')}@{fig.get('bbox')}" for fig in record["figures"] if fig.get("path")
            )
            if fig_refs:
                lines.append(f"Figures: {fig_refs}")
        if record.get("formula_latex"):
            formula_text = " ".join(item.get("latex", "") for item in record["formula_latex"])
            if formula_text.strip():
                lines.append(f"Formula: {formula_text}")
        md_blocks.append("\n".join(lines))
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(md_blocks))

    review_path = None
    if review_queue:
        review_path = questions_dir / "review_queue.jsonl"
        with open(review_path, "w", encoding="utf-8") as f:
            for record in review_queue:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "pdf": str(pdf_path),
        "questions_file": str(out_json),
        "core_questions_file": str(core_jsonl),
        "enrich_questions_file": str(enrich_jsonl) if enrich_jsonl else None,
        "markdown_file": str(md_out),
        "review_queue_file": str(review_path) if review_path else None,
        "num_questions": len(all_questions),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract OCR text/images from scanned PDFs.")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to a scanned PDF.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "ingest",
        help="Base output directory for images/questions JSON.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for OCR.")
    parser.add_argument(
        "--ocr-lang",
        default="chi_sim+eng",
        help="Tesseract language packs to use, e.g. chi_sim+eng.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable OCR pre-processing (denoise+binarize+sharpen+deskew).",
    )
    parser.add_argument("--preprocess-denoise", action="store_true", help="Apply median blur denoise.")
    parser.add_argument("--preprocess-binarize", action="store_true", help="Apply Otsu binarization.")
    parser.add_argument("--preprocess-sharpen", action="store_true", help="Apply sharpening filter.")
    parser.add_argument("--preprocess-deskew", action="store_true", help="Apply deskew correction.")
    parser.add_argument("--save-preprocessed", action="store_true", help="Save preprocessed page images.")
    parser.add_argument(
        "--no-render-page-when-empty",
        action="store_false",
        dest="render_page_when_empty",
        help="Disable saving a full-page PNG when no embedded images are found.",
    )
    parser.add_argument(
        "--no-crop-questions",
        action="store_false",
        dest="crop_questions",
        help="Disable per-question vertical crops; fallback to page-level images.",
    )
    parser.add_argument("--subject", help="Override document subject (e.g., math).")
    parser.add_argument("--year", type=int, help="Override document year (e.g., 2022).")
    parser.add_argument("--grade", help="Override document grade (e.g., grade_8).")
    parser.add_argument(
        "--doc-meta-pages",
        type=int,
        default=2,
        help="How many initial pages to scan for doc-level subject/year/grade hints.",
    )
    parser.add_argument(
        "--ocr-min-conf",
        type=float,
        default=0.7,
        help="OCR confidence threshold for review queue.",
    )
    parser.add_argument(
        "--seg-min-conf",
        type=float,
        default=0.6,
        help="Segmentation confidence threshold for review queue.",
    )
    parser.add_argument(
        "--min-stem-chars",
        type=int,
        default=20,
        help="Minimum question stem length for quality checks.",
    )
    parser.set_defaults(render_page_when_empty=True, crop_questions=True)
    args = parser.parse_args()

    configure_tesseract()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preprocess = args.preprocess or any(
        [args.preprocess_denoise, args.preprocess_binarize, args.preprocess_sharpen, args.preprocess_deskew]
    )
    preprocess_denoise = args.preprocess_denoise or args.preprocess
    preprocess_binarize = args.preprocess_binarize or args.preprocess
    preprocess_sharpen = args.preprocess_sharpen or args.preprocess
    preprocess_deskew = args.preprocess_deskew or args.preprocess
    if preprocess and not _CV2_AVAILABLE:
        print("[warn] preprocess requested but OpenCV is unavailable; skipping preprocessing.")
        preprocess = False

    result = pdf_to_questions(
        args.pdf,
        args.out_dir,
        dpi=args.dpi,
        render_if_empty=args.render_page_when_empty,
        crop_questions=args.crop_questions,
        ocr_lang=args.ocr_lang,
        preprocess=preprocess,
        preprocess_denoise=preprocess_denoise,
        preprocess_binarize=preprocess_binarize,
        preprocess_sharpen=preprocess_sharpen,
        preprocess_deskew=preprocess_deskew,
        save_preprocessed=args.save_preprocessed,
        doc_meta_pages=args.doc_meta_pages,
        subject=args.subject,
        year=args.year,
        grade=args.grade,
        ocr_min_conf=args.ocr_min_conf,
        seg_min_conf=args.seg_min_conf,
        min_stem_chars=args.min_stem_chars,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
