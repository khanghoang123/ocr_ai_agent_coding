"""
Pydantic schemas for OCR pipeline input/output.

These are the canonical data contracts used by:
  - The OCR pipeline internally
  - The FastAPI request/response models
  - The export formatters
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, computed_field
from typing import Optional
from PIL import Image


# ── Enums ────────────────────────────────────────────────────────────────────

class ExportFormat(str, Enum):
    TXT = "txt"
    JSON = "json"
    PDF = "pdf"


# ── Bounding box ─────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (top-left origin)."""
    x1: float = Field(..., description="Left edge (pixels)")
    y1: float = Field(..., description="Top edge (pixels)")
    x2: float = Field(..., description="Right edge (pixels)")
    y2: float = Field(..., description="Bottom edge (pixels)")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


# ── Line-level result ─────────────────────────────────────────────────────────

class TextLine(BaseModel):
    """OCR result for a single detected text line."""
    line_index: int = Field(..., description="Reading order index (0-based)")
    text: str = Field(..., description="Recognised text")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Detection confidence from PaddleOCR"
    )
    bbox: BoundingBox = Field(..., description="Bounding box of the line")


# ── Page-level result ─────────────────────────────────────────────────────────

class PageResult(BaseModel):
    """OCR result for one page (or one image)."""
    page_number: int = Field(..., ge=1, description="1-based page index")
    width: int = Field(..., description="Page image width (pixels)")
    height: int = Field(..., description="Page image height (pixels)")
    lines: list[TextLine] = Field(default_factory=list)
    postprocessed: Optional[dict[str, str]] = None
    postprocessed_lines: Optional[dict[str, list[str]]] = None

    @computed_field(return_type=str)
    def full_text(self) -> str:
        """All lines joined in reading order."""
        return "\n".join(line.text for line in self.lines)

    @computed_field(return_type=int)
    def line_count(self) -> int:
        return len(self.lines)


# ── Document-level result ─────────────────────────────────────────────────────

class OCRResult(BaseModel):
    """Top-level OCR result for a single input file (image or PDF)."""
    model_config = {"arbitrary_types_allowed": True}
    
    filename: str
    file_type: str          # "image" | "pdf"
    model_used: str         # e.g. "experiment_B"
    total_pages: int
    pages: list[PageResult]
    processing_time_ms: float = Field(..., description="End-to-end latency in ms")
    page_images: list[Image.Image] = Field(default_factory=list, exclude=True)

    @computed_field(return_type=str)
    def full_text(self) -> str:
        """All pages, separated by form-feed character."""
        return "\f".join(page.full_text for page in self.pages)

    @computed_field(return_type=int)
    def total_lines(self) -> int:
        return sum(p.line_count for p in self.pages)


# ── API request / response wrappers ──────────────────────────────────────────

class OCRRequest(BaseModel):
    """Query parameters for POST /ocr."""
    model: str = "experiment_B_50k"
    export_format: ExportFormat = ExportFormat.JSON


class OCRResponse(BaseModel):
    """Response body for POST /ocr."""
    status: str = "success"
    results: list[OCRResult]


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    model_loaded: str
    device: str
    gpu_available: bool
    version: str


class ModelInfo(BaseModel):
    key: str
    name: str
    display_name: str
    description: str
    architecture: str
    training_iters: Optional[int] = None
    cer: Optional[float] = None
    wer: Optional[float] = None
    exact_match: Optional[float] = None
    status: str
    recommended: bool = False
    is_default: bool = False


class ModelCatalogResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]


class DebugDetectionDecision(BaseModel):
    source_index: int
    action: str
    note: Optional[str] = None
    decision_stage: str = "full_page"
    fallback_reason: Optional[str] = None
    raw_polygon: list[list[float]] = Field(default_factory=list)
    refined_boxes: list[BoundingBox] = Field(default_factory=list)
    tight_bbox: Optional[BoundingBox] = None
    projection_bbox: Optional[BoundingBox] = None
    clamped_bbox: Optional[BoundingBox] = None
    curve_score: float = 0.0
    neighbor_strategy: Optional[str] = None
    projection_threshold: Optional[float] = None
    notebook_mode: bool = False
    ruled_line_mode: bool = False
    original_text: Optional[str] = None
    original_score: Optional[float] = None
    split_texts: list[str] = Field(default_factory=list)
    split_scores: list[float] = Field(default_factory=list)
    mask_preview_base64: Optional[str] = None
    rectified_preview_base64: Optional[str] = None


class DebugCropPreview(BaseModel):
    line_index: int
    text: str
    detection_confidence: float
    recognition_score: Optional[float] = None
    bbox: BoundingBox
    preview_base64: str
    before_preview_base64: Optional[str] = None
    crop_flags: list[str] = Field(default_factory=list)
    decode_mode: Optional[str] = None
    pre_norm_ratio: Optional[float] = None
    post_norm_ratio: Optional[float] = None
    baseline_confidence: Optional[float] = None
    baseline_offset: Optional[float] = None
    normalization_preview_base64: Optional[str] = None


class DebugPageResult(BaseModel):
    page_number: int
    raw_polygons: list[list[list[float]]] = Field(default_factory=list)
    refined_boxes: list[BoundingBox] = Field(default_factory=list)
    decisions: list[DebugDetectionDecision] = Field(default_factory=list)
    final_crops: list[DebugCropPreview] = Field(default_factory=list)


class OCRDebugItem(BaseModel):
    result: OCRResult
    debug_pages: list[DebugPageResult]


class OCRDebugResponse(BaseModel):
    status: str = "success"
    results: list[OCRResult]
    debug_results: list[OCRDebugItem]


# ── Performance metrics ───────────────────────────────────────────────────────

class PerformanceMetrics(BaseModel):
    """Latency breakdown for a single image/page — used in testing."""
    filename: str
    detection_ms: float
    recognition_ms: float
    total_ms: float
    num_lines: int
    image_width: int
    image_height: int
