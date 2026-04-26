"""
File handler: converts input files (images / PDFs) into lists of PIL Images
ready for the OCR pipeline.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

from PIL import Image


# Supported extensions (lowercase)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DOC_EXTS = {".pdf"}


def load_file(file_path: str | Path) -> tuple[str, list[Image.Image]]:
    """Load an image or PDF and return (file_type, list_of_PIL_images).

    Returns:
        file_type: "image" or "pdf"
        pages:     List of RGB PIL Images (one per page for PDF, one for image)

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext in IMAGE_EXTS:
        return "image", [_load_image(path)]
    elif ext in DOC_EXTS:
        return "pdf", list(_load_pdf_pages(path))
    else:
        supported = sorted(IMAGE_EXTS | DOC_EXTS)
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )


def load_bytes(data: bytes, filename: str) -> tuple[str, list[Image.Image]]:
    """Same as load_file but from raw bytes (for FastAPI UploadFile)."""
    ext = Path(filename).suffix.lower()

    if ext in IMAGE_EXTS:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return "image", [img]
    elif ext in DOC_EXTS:
        return "pdf", list(_load_pdf_pages_from_bytes(data))
    else:
        supported = sorted(IMAGE_EXTS | DOC_EXTS)
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_image(path: Path) -> Image.Image:
    """Open an image file and convert to RGB."""
    try:
        img = Image.open(path)
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image '{path}': {e}") from e


def _load_pdf_pages(path: Path) -> Iterator[Image.Image]:
    """Yield each PDF page as a 300-DPI RGB PIL Image using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. Install it with:\n"
            "  pip install pymupdf"
        )

    doc = fitz.open(str(path))
    try:
        for page in doc:
            yield _fitz_page_to_pil(page)
    finally:
        doc.close()


def _load_pdf_pages_from_bytes(data: bytes) -> Iterator[Image.Image]:
    """Yield each PDF page from raw bytes as a 300-DPI RGB PIL Image."""
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. Install it with:\n"
            "  pip install pymupdf"
        )

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        for page in doc:
            yield _fitz_page_to_pil(page)
    finally:
        doc.close()


def _fitz_page_to_pil(page, dpi: int = 300) -> Image.Image:
    """Render a fitz page to a PIL RGB Image at the given DPI."""
    scale = dpi / 72.0  # fitz default is 72 DPI
    mat = __import__("fitz").Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=__import__("fitz").csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
