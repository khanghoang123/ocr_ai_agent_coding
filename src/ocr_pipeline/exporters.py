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
    """Render recognised text as a clean white-background PDF.

    Layout strategy:
    1. Use A4 page size for readability (not the raw image pixel size).
    2. Compute a **uniform** font size from the median detected line
       height so all body text has consistent sizing.
    3. Place each line at a y-position derived from its detected bbox
       centre, scaled proportionally onto the PDF page. A minimum gap
       between consecutive lines prevents overlap.
    4. Horizontally, text starts at a proportionally-scaled x offset
       from the left margin, clamped so text stays within margins.

    The source image is **not** drawn as background: the PDF is a
    clean black-text-on-white-paper transcription.

    ``images`` is accepted for backward compatibility; image sizes are
    used as a fallback when ``page.width`` / ``page.height`` are missing.
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
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        pdfmetrics.registerFont(TTFont(font_name, font_path))

    # A4 page dimensions in points.
    PAGE_W, PAGE_H = 595.0, 842.0
    MARGIN_TOP = 50.0
    MARGIN_BOTTOM = 50.0
    MARGIN_LEFT = 50.0
    MARGIN_RIGHT = 50.0
    usable_w = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
    usable_h = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM

    buf = io.BytesIO()
    c = canvas.Canvas(buf)

    for page_idx, page in enumerate(result.pages):
        src_w = float(page.width or 0)
        src_h = float(page.height or 0)
        if (src_w <= 0 or src_h <= 0) and images and page_idx < len(images):
            src_w = float(images[page_idx].width)
            src_h = float(images[page_idx].height)
        if src_w <= 0 or src_h <= 0:
            src_w, src_h = PAGE_W, PAGE_H

        non_empty = [ln for ln in page.lines if (ln.text or "").strip()]
        if not non_empty:
            c.setPageSize((PAGE_W, PAGE_H))
            c.setFillColorRGB(1.0, 1.0, 1.0)
            c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
            c.showPage()
            continue

        # Compute uniform font size from median bbox height,
        # scaled to the PDF coordinate space and clamped to a
        # readable range.
        bbox_heights = [float(ln.bbox.height) for ln in non_empty]
        median_h = float(sorted(bbox_heights)[len(bbox_heights) // 2])
        scale_y = usable_h / src_h
        font_size = max(8.0, min(median_h * scale_y * 0.55, 14.0))
        line_spacing = font_size * 1.45

        c.setPageSize((PAGE_W, PAGE_H))
        c.setFillColorRGB(1.0, 1.0, 1.0)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColorRGB(0.0, 0.0, 0.0)
        c.setFont(font_name, font_size)

        # Place lines at scaled y-positions with overlap prevention.
        scale_x = usable_w / src_w
        cursor_y = PAGE_H - MARGIN_TOP  # top of usable area

        for line in non_empty:
            text = (line.text or "").strip()
            bbox = line.bbox

            # Proportional x from the source page.
            x = MARGIN_LEFT + float(bbox.x1) * scale_x
            x = max(MARGIN_LEFT, min(x, PAGE_W - MARGIN_RIGHT - 20))

            # Proportional y from the source page (top-left origin).
            target_y_from_top = MARGIN_TOP + float(bbox.y1) * scale_y
            target_y = PAGE_H - target_y_from_top

            # Prevent overlap: never place above previous cursor.
            y = min(target_y, cursor_y - line_spacing * 0.15)

            # Page overflow: start new page if below bottom margin.
            if y < MARGIN_BOTTOM:
                c.showPage()
                c.setPageSize((PAGE_W, PAGE_H))
                c.setFillColorRGB(1.0, 1.0, 1.0)
                c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
                c.setFillColorRGB(0.0, 0.0, 0.0)
                c.setFont(font_name, font_size)
                cursor_y = PAGE_H - MARGIN_TOP
                y = cursor_y

            # Shrink font if text is wider than available space.
            avail_w = PAGE_W - MARGIN_RIGHT - x
            text_w = pdfmetrics.stringWidth(text, font_name, font_size)
            if text_w > avail_w and avail_w > 0:
                adjusted = font_size * avail_w / text_w
                c.setFont(font_name, max(6.0, adjusted))
                c.drawString(x, y, text)
                c.setFont(font_name, font_size)
            else:
                c.drawString(x, y, text)

            cursor_y = y - line_spacing

        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()
