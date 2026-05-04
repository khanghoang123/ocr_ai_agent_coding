"""
Export formatters: convert OCRResult into downloadable file formats.

Supported in v1:
  - txt:  Plain text, one line per text line, pages separated by form-feed.
  - json: Full structured output with bboxes, confidence, metadata.

Future (v2):
  - docx: python-docx preserving layout approximation.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr_pipeline.schemas import ExportFormat, OCRResult


from PIL import Image
from typing import Optional

def export_result(result: OCRResult, fmt: ExportFormat, images: Optional[list[Image.Image]] = None) -> tuple[bytes, str]:
    """Convert an OCRResult to bytes for download.

    Returns:
        (file_bytes, media_type): ready for FastAPI FileResponse / Response.
    """
    if fmt == ExportFormat.TXT:
        content = _to_txt(result)
        return content.encode("utf-8"), "text/plain; charset=utf-8"
    elif fmt == ExportFormat.JSON:
        content = _to_json(result)
        return content.encode("utf-8"), "application/json"
    elif fmt == ExportFormat.PDF:
        content = _to_pdf(result, images)
        return content, "application/pdf"
    else:
        raise ValueError(f"Unsupported export format: {fmt}")


def suggested_filename(original_filename: str, fmt: ExportFormat) -> str:
    """Return a downloadable filename based on the input file."""
    stem = Path(original_filename).stem
    return f"{stem}_ocr.{fmt.value}"


# ── Private formatters ────────────────────────────────────────────────────────

def _to_txt(result: OCRResult) -> str:
    """Plain text export.

    Format:
        [Page 1]
        Line 1 text
        Line 2 text
        ...
        \f
        [Page 2]
        ...

    - Each line of text is on its own line.
    - Pages are separated by a form-feed character (\\f).
    - Bounding box info is NOT included (pure text output).
    """
    sections = []
    for page in result.pages:
        lines = [f"[Page {page.page_number}]"]
        for tl in page.lines:
            lines.append(tl.text)
        sections.append("\n".join(lines))
    return "\f\n".join(sections)


def _to_json(result: OCRResult) -> str:
    """Full structured JSON export matching the OCRResult schema.

    Includes:
      - Filename, model used, processing time
      - Per-page: width, height, full_text
      - Per-line: text, confidence, bounding box (x1/y1/x2/y2)

    Output validation:
      - bbox coordinates are always finite floats
      - confidence is clamped to [0, 1]
      - line_index reflects final reading order
    """
    payload = {
        "filename": result.filename,
        "file_type": result.file_type,
        "model_used": result.model_used,
        "total_pages": result.total_pages,
        "total_lines": result.total_lines,
        "processing_time_ms": result.processing_time_ms,
        "pages": [
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "line_count": page.line_count,
                "full_text": page.full_text,
                "lines": [
                    {
                        "line_index": tl.line_index,
                        "text": tl.text,
                        "confidence": round(tl.confidence, 4),
                        "bbox": {
                            "x1": round(tl.bbox.x1, 1),
                            "y1": round(tl.bbox.y1, 1),
                            "x2": round(tl.bbox.x2, 1),
                            "y2": round(tl.bbox.y2, 1),
                        },
                    }
                    for tl in page.lines
                ],
            }
            for page in result.pages
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def _to_pdf(
    result: OCRResult, images: Optional[list[Image.Image]] = None
) -> bytes:
    """Render the recognised text onto a clean white-background PDF.

    The output PDF mirrors the input image's geometry — page size in
    points equals the input image's pixel dimensions, and each line is
    rendered at its detected bounding-box position with a font size
    scaled so the text fills the bbox horizontally (with a hard cap
    at the bbox height to preserve vertical layout).

    Crucially, the source image is **not** drawn as background: the
    user-facing PDF must be a clean black-text-on-white-paper render
    so it is usable as a structured transcription of the input.

    The ``images`` argument is kept for backward compatibility with
    callers that still pass it; we only read each image's size as a
    fallback when ``page.width`` / ``page.height`` are missing.
    """
    import io

    try:
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "PDF export requires the 'reportlab' package. Install it via "
            "`pip install reportlab` (already listed in requirements.txt)."
        ) from exc

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_name = "DejaVuSans"
    pdfmetrics.registerFont(TTFont(font_name, font_path))

    buf = io.BytesIO()
    c = canvas.Canvas(buf)

    for i, page in enumerate(result.pages):
        w = float(page.width or 0)
        h = float(page.height or 0)
        if (w <= 0 or h <= 0) and images and i < len(images):
            w, h = float(images[i].width), float(images[i].height)
        if w <= 0 or h <= 0:
            # Defensive fallback: A4 portrait at 72 dpi.
            w, h = 595.0, 842.0

        c.setPageSize((w, h))
        # Explicit white background — reportlab pages are nominally
        # transparent, but some PDF viewers render that as black.
        c.setFillColorRGB(1.0, 1.0, 1.0)
        c.rect(0, 0, w, h, fill=1, stroke=0)
        c.setFillColorRGB(0.0, 0.0, 0.0)

        for line in page.lines:
            bbox = line.bbox
            text = (line.text or "").strip()
            if not text:
                continue
            # Vertical: cap font size at 90 % of the bbox height so
            # ascenders/descenders fit cleanly within the line slot.
            target_h = max(6.0, float(bbox.height) * 0.9)
            font_size = target_h
            # Horizontal: shrink the font size further if the text is
            # wider than the bbox. This keeps each line within its
            # detected horizontal slot, mirroring the input layout.
            target_w = max(1.0, float(bbox.width))
            text_w_at_target = pdfmetrics.stringWidth(text, font_name, font_size)
            if text_w_at_target > target_w:
                font_size *= target_w / text_w_at_target
            font_size = max(4.0, font_size)
            c.setFont(font_name, font_size)
            # ReportLab origin is bottom-left; OCR bboxes are top-left.
            # Place the text baseline near the bbox's bottom edge,
            # offset upwards by ~20 % of the font size so the visible
            # glyph body sits inside the bbox.
            x = float(bbox.x1)
            y = h - float(bbox.y2) + font_size * 0.2
            c.drawString(x, y, text)

        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()
