"""
Layout reconstructor: sorts detected text lines into reading order
(top-to-bottom, left-to-right) and assembles structured page output.

This module handles the critical step of converting unordered detection
results into a coherent document structure that preserves the original
reading flow of Vietnamese handwritten text.
"""

from __future__ import annotations

import logging

import numpy as np

from ocr_pipeline.cropper.line_cropper import CropResult
from ocr_pipeline.schemas import BoundingBox, PageResult, TextLine

logger = logging.getLogger(__name__)


def polygon_to_bbox(polygon: np.ndarray) -> BoundingBox:
    """Convert a 4×2 polygon array to an axis-aligned BoundingBox."""
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    return BoundingBox(
        x1=float(np.min(xs)),
        y1=float(np.min(ys)),
        x2=float(np.max(xs)),
        y2=float(np.max(ys)),
    )


class LayoutReconstructor:
    """Reconstructs page layout from unordered line detections.

    Algorithm (proven for Vietnamese handwritten documents):
    1. Compute median line height across all detections.
    2. Group lines into "rows" — lines are in the same row if their
       vertical centres are within (row_tolerance_ratio × median_height).
    3. Within each row, sort lines left-to-right by bbox x1.
    4. Sort rows top-to-bottom by the minimum y1 of the row.

    This tolerates slight vertical misalignment within handwritten rows
    while maintaining correct inter-row ordering.
    """

    def __init__(self, row_tolerance_ratio: float = 0.5):
        """
        Args:
            row_tolerance_ratio: Lines within this fraction of the median
                line height are considered on the same row. 0.5 = 50%.
        """
        self.row_tolerance_ratio = row_tolerance_ratio

    def reconstruct(
        self,
        crops: list[CropResult],
        texts: list[str],
        image_width: int,
        image_height: int,
        page_number: int = 1,
    ) -> PageResult:
        """Build a PageResult from crops + recognised texts.

        Args:
            crops:        CropResult list from LineCropper.crop_all().
            texts:        Recognised text strings, same length as crops.
            image_width:  Width of the original full-page image.
            image_height: Height of the original full-page image.
            page_number:  1-based page index.

        Returns:
            PageResult with lines sorted in reading order.
        """
        if len(crops) != len(texts):
            raise ValueError(
                f"crops ({len(crops)}) and texts ({len(texts)}) must be the same length."
            )

        if not crops:
            return PageResult(
                page_number=page_number,
                width=image_width,
                height=image_height,
                lines=[],
            )

        # ── 1. Compute bounding boxes ─────────────────────────────────────────
        bboxes = [polygon_to_bbox(c.polygon) for c in crops]

        # ── 2. Compute median line height ─────────────────────────────────────
        heights = [bb.height for bb in bboxes]
        median_h = float(np.median(heights))
        tolerance = self.row_tolerance_ratio * median_h
        logger.debug("Median line height=%.1fpx  tolerance=%.1fpx", median_h, tolerance)

        # ── 3. Group into rows ────────────────────────────────────────────────
        # Sort by centre-y first for greedy grouping
        items = list(zip(bboxes, texts, crops))
        items.sort(key=lambda x: x[0].center_y)

        rows: list[list[tuple[BoundingBox, str, CropResult]]] = []
        for bbox, text, crop in items:
            placed = False
            for row in rows:
                row_center_y = np.mean([r[0].center_y for r in row])
                if abs(bbox.center_y - row_center_y) <= tolerance:
                    row.append((bbox, text, crop))
                    placed = True
                    break
            if not placed:
                rows.append([(bbox, text, crop)])

        # ── 4. Sort rows top-to-bottom, lines left-to-right ──────────────────
        rows.sort(key=lambda row: min(r[0].y1 for r in row))
        for row in rows:
            row.sort(key=lambda r: r[0].x1)

        # ── 5. Build TextLine list with final reading-order indices ───────────
        text_lines: list[TextLine] = []
        line_idx = 0
        for row in rows:
            for bbox, text, crop in row:
                text_lines.append(TextLine(
                    line_index=line_idx,
                    text=text,
                    confidence=crop.confidence,
                    bbox=bbox,
                ))
                line_idx += 1

        logger.debug(
            "Page %d: %d lines in %d rows reconstructed.",
            page_number, len(text_lines), len(rows),
        )

        return PageResult(
            page_number=page_number,
            width=image_width,
            height=image_height,
            lines=text_lines,
        )

    @classmethod
    def from_settings(cls) -> "LayoutReconstructor":
        from ocr_pipeline.config import settings
        return cls(row_tolerance_ratio=settings.row_tolerance_ratio)
