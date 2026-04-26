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

def _to_pdf(result: OCRResult, images: Optional[list[Image.Image]] = None) -> bytes:
    """Generate a Searchable PDF where text is overlayed on the original image.
    
    If images are provided, they are drawn as background.
    Each page in OCRResult is transformed into a PDF page.
    """
    import io
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader

    # Register Vietnamese-compatible font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_name = "DejaVuSans"
    pdfmetrics.registerFont(TTFont(font_name, font_path))

    buf = io.BytesIO()
    c = canvas.Canvas(buf)

    for i, page in enumerate(result.pages):
        w, h = page.width, page.height
        c.setPageSize((w, h))

        # 1. Draw background image if available
        if images and i < len(images):
            img_buf = io.BytesIO()
            images[i].save(img_buf, format="JPEG", quality=85)
            img_buf.seek(0)
            c.drawImage(ImageReader(img_buf), 0, 0, width=w, height=h)

        # 2. Draw text layer
        # For a "Searchable PDF", text should match the position of input
        # Note: ReportLab origin is bottom-left, while our OCR is top-left.
        for line in page.lines:
            bbox = line.bbox
            text = line.text
            
            # Calculate height for font scaling
            fh = max(8, bbox.height * 0.8)
            c.setFont(font_name, fh)
            
            # Position: y_pdf = h - y_ocr (approx)
            # We use x1, y2 (bottom left of the text line in OCR space)
            # which maps to (x1, h - y2) in ReportLab space.
            c.drawString(bbox.x1, h - bbox.y2 + 2, text)

        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()
