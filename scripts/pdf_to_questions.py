"""
Quick-and-dirty pipeline to turn scanned math PDFs into structured question JSON + extracted images.

Usage:
  python scripts/pdf_to_questions.py --pdf input.pdf --out-dir data/ingest --dpi 300
  # by default renders a full-page PNG when no embedded images exist; disable via --no-render-page-when-empty
  # per-question crops (vertical slices) are enabled by default; disable via --no-crop-questions

Outputs (inside --out-dir):
  - images/<pdf-stem>/page-001-img-001.png (raw images extracted from PDF)
  - images/<pdf-stem>/page-001-q001.png (per-question crops when detected)
  - questions/<pdf-stem>.json (list of question dicts with text and image paths)

Prereqs:
  - Tesseract installed locally; set env TESSERACT_CMD if it's not on PATH.
  - Python deps already in requirements.txt: pdfplumber, pillow, pytesseract.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pdfplumber
import pytesseract
from PIL import Image, ImageOps


def configure_tesseract() -> None:
    """Use TESSERACT_CMD env if provided."""
    custom_cmd = os.getenv("TESSERACT_CMD")
    if custom_cmd:
        pytesseract.pytesseract.tesseract_cmd = custom_cmd


def extract_page_images(
    page: pdfplumber.page.Page, base_dir: Path, page_idx: int, render_if_empty: bool, dpi: int
) -> List[str]:
    """
    Extract embedded images in a PDF page to disk.
    If no embedded images are found and render_if_empty=True, rasterizes the whole page as a fallback.
    Returns list of relative file paths.
    """
    saved_paths: List[str] = []
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
                saved_paths.append(str(img_out))
            except Exception:
                # Best-effort: skip images that fail to extract.
                continue

    if not saved_paths and render_if_empty:
        # Fallback: render whole page to an image so scanned PDFs still get a visual.
        pil_img: Image.Image = page.to_image(resolution=dpi).original
        img_out = base_dir / f"page-{page_idx:03d}-full.png"
        pil_img.save(img_out)
        saved_paths.append(str(img_out))

    return saved_paths


def ocr_page_to_text(page: pdfplumber.page.Page, dpi: int = 300) -> Dict:
    """Rasterize a page and run OCR. Returns raw text and data with boxes."""
    pil_img: Image.Image = page.to_image(resolution=dpi).original
    text = pytesseract.image_to_string(pil_img, lang="chi_sim+eng")
    data = pytesseract.image_to_data(pil_img, lang="chi_sim+eng", output_type=pytesseract.Output.DICT)
    return {"text": text, "data": data, "image": pil_img}


def split_questions(raw_text: str) -> List[str]:
    """
    Naive question splitter for K-12 style materials.
    Splits on lines that look like numbered questions: 1. / 1、/ 1．
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    questions: List[str] = []
    buffer: List[str] = []
    pattern = re.compile(r"^\d+\s*[\.、．)]")

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
    pattern = re.compile(r"^\d+\s*[\.、．)]")
    starts = []
    n = len(ocr_data["text"])
    for i in range(n):
        txt = ocr_data["text"][i].strip()
        if not txt:
            continue
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


def pdf_to_questions(
    pdf_path: Path, out_dir: Path, dpi: int = 300, render_if_empty: bool = True, crop_questions: bool = True
) -> Dict:
    """Process a PDF into OCR text, extracted images, and naive question list."""
    pdf_stem = pdf_path.stem
    images_dir = out_dir / "images" / pdf_stem
    images_dir.mkdir(parents=True, exist_ok=True)
    questions_dir = out_dir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    all_questions = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_images = extract_page_images(page, images_dir, idx, render_if_empty=render_if_empty, dpi=dpi)
            ocr_out = ocr_page_to_text(page, dpi=dpi)
            page_text = ocr_out["text"]
            print(f"[page {idx}] embedded/fallback images saved: {len(page_images)} (fallback={render_if_empty})")

            question_chunks = split_questions(page_text) or [page_text.strip()]

            question_crops: List[str] = []
            if crop_questions:
                boundaries = find_question_boundaries(ocr_out["data"], ocr_out["image"].height)
                if boundaries:
                    question_crops = crop_question_images(ocr_out["image"], boundaries, images_dir, idx)
                    print(f"[page {idx}] question crops: {len(question_crops)}")
                else:
                    print(f"[page {idx}] no question starts detected for cropping")

            for q_idx, text in enumerate(question_chunks, start=1):
                images_for_question: List[str] = []
                if question_crops:
                    if q_idx - 1 < len(question_crops):
                        images_for_question = [question_crops[q_idx - 1]]
                elif page_images:
                    images_for_question = page_images

                all_questions.append(
                    {
                        "id": f"{pdf_stem}-p{idx:03d}-q{q_idx:03d}",
                        "source_pdf": str(pdf_path),
                        "page": idx,
                        "text": text,
                        "images": images_for_question,
                        "status": "draft",
                    }
                )

    out_json = questions_dir / f"{pdf_stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    return {
        "pdf": str(pdf_path),
        "questions_file": str(out_json),
        "num_questions": len(all_questions),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract OCR text/images from scanned math PDFs.")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to a scanned PDF.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "ingest",
        help="Base output directory for images/questions JSON.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI for OCR.")
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
    parser.set_defaults(render_page_when_empty=True, crop_questions=True)
    args = parser.parse_args()

    configure_tesseract()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = pdf_to_questions(
        args.pdf,
        args.out_dir,
        dpi=args.dpi,
        render_if_empty=args.render_page_when_empty,
        crop_questions=args.crop_questions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
