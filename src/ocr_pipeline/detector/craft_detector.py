"""CRAFT word-level detector + row-clustering merger (Tier-1 candidate E2).

CRAFT (clovaai/CRAFT-pytorch) emits a per-character heatmap (region map)
and a between-character link map. Post-processing yields **word-level**
polygons. For VietOCR, however, we need **line-level** polygons. This
backend therefore runs CRAFT and then merges the words into lines using
a y-coordinate row-clustering heuristic that respects the median word
height (so half a line worth of slant doesn't bleed into a neighbour).

Why this choice:
    The audit (ocr_audit_phase3_bottleneck.md) showed PaddleOCR DBNet
    fails on handwriting because its objective is text-mask, which is
    fragile under cursive / connected glyphs. CRAFT's character-region
    heatmap is more general and survives handwriting at the cost of
    requiring a merging step. We absorb that step here so VietOCR sees
    the same per-line polygons as before.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from ocr_pipeline.detector.base import empty_detection_result
from ocr_pipeline.detector.paddle_detector import DetectionResult

logger = logging.getLogger(__name__)


# Default location for the bundled CRAFT MLT weights (~80MB). The runner
# downloads these once via easyocr if missing; production deployments can
# point at any other CRAFT checkpoint by passing ``weights_path``.
DEFAULT_WEIGHTS = Path(
    os.environ.get(
        "CRAFT_WEIGHTS",
        "models/detector_weights/craft_mlt_25k.pth",
    )
)


class CraftDetector:
    """CRAFT + row-clustering line merger."""

    BACKEND_NAME = "craft"

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        row_overlap_ratio: float = 0.4,
        x_pad_ratio: float = 0.02,
        y_pad_ratio: float = 0.10,
        **_: Any,
    ):
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self.device = torch.device(device)
        self.canvas_size = int(canvas_size)
        self.mag_ratio = float(mag_ratio)
        self.text_threshold = float(text_threshold)
        self.link_threshold = float(link_threshold)
        self.low_text = float(low_text)
        # Row-clustering knobs (chosen on tests/test/ during Phase 3 dev).
        self.row_overlap_ratio = float(row_overlap_ratio)
        self.x_pad_ratio = float(x_pad_ratio)
        self.y_pad_ratio = float(y_pad_ratio)
        self._net: torch.nn.Module | None = None

    def _ensure_loaded(self) -> None:
        if self._net is not None:
            return
        if not self.weights_path.exists():
            raise RuntimeError(
                f"CRAFT weights not found at {self.weights_path}. "
                f"Either download `craft_mlt_25k.pth` (e.g. via easyocr's "
                f"first-run downloader) or set the CRAFT_WEIGHTS env var."
            )
        from ocr_pipeline.detector._craft.craft import CRAFT
        net = CRAFT()
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        net.load_state_dict(_strip_module_prefix(state_dict))
        net.to(self.device)
        net.eval()
        self._net = net
        logger.info("CRAFT detector ready (device=%s).", self.device)

    # ── Public API ──────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> DetectionResult:
        return self.detect_with_notebook_fallback(image)

    def detect_with_notebook_fallback(self, image: Image.Image) -> DetectionResult:
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.warning("CRAFT init failed: %s", exc)
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason=f"craft_init_error:{type(exc).__name__}",
            )

        try:
            word_polys = self._run_word_detection(image)
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("CRAFT inference raised: %s", exc)
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason=f"craft_runtime_error:{type(exc).__name__}",
            )

        if not word_polys:
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason="craft_no_words_above_threshold",
            )

        line_polys = cluster_words_into_lines(
            word_polys,
            row_overlap_ratio=self.row_overlap_ratio,
            x_pad_ratio=self.x_pad_ratio,
            y_pad_ratio=self.y_pad_ratio,
            image_width=image.width,
            image_height=image.height,
        )

        if not line_polys:  # pragma: no cover - defensive
            return empty_detection_result(
                image,
                backend=self.BACKEND_NAME,
                reason="craft_clustering_empty",
            )

        polygons = [poly for poly, _ in line_polys]
        confidences = [conf for _, conf in line_polys]
        sources = ["craft_line"] * len(polygons)
        diagnostics = _diagnose_polygons(polygons, image.height)
        diagnostics.update(
            {
                "backend": self.BACKEND_NAME,
                "low_detector_recall": False,
                "raw_word_count": len(word_polys),
                "line_count": len(polygons),
                "patchwise_used": False,
            }
        )
        return DetectionResult(
            polygons=polygons,
            confidences=confidences,
            sources=sources,
            diagnostics=diagnostics,
        )

    # ── Internals ───────────────────────────────────────────────────────────

    def _run_word_detection(self, image: Image.Image) -> list[np.ndarray]:
        """Run CRAFT on a PIL image and return a list of word polygons (4×2 float32)."""
        from ocr_pipeline.detector._craft import craft_utils, imgproc

        img_rgb = np.asarray(image.convert("RGB"))
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            img_rgb, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio
        )
        ratio_h = ratio_w = 1.0 / target_ratio

        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        assert self._net is not None
        with torch.no_grad():
            y, _ = self._net(x)

        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()

        boxes, _ = craft_utils.getDetBoxes(
            score_text,
            score_link,
            self.text_threshold,
            self.link_threshold,
            self.low_text,
            poly=False,
        )
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        word_polys: list[np.ndarray] = []
        for box in boxes:
            poly = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            if poly.shape[0] != 4:
                continue
            word_polys.append(poly)
        return word_polys


# ── Row clustering ──────────────────────────────────────────────────────────


def cluster_words_into_lines(
    word_polys: list[np.ndarray],
    row_overlap_ratio: float = 0.4,
    x_pad_ratio: float = 0.02,
    y_pad_ratio: float = 0.10,
    image_width: int = 0,
    image_height: int = 0,
) -> list[tuple[np.ndarray, float]]:
    """Group word-level polygons into line-level polygons.

    Algorithm:
      1. Compute axis-aligned bbox per word.
      2. Sort by y-center.
      3. Greedily assign each word to an existing row whose vertical extent
         overlaps the word by at least ``row_overlap_ratio`` of the word
         height; otherwise start a new row.
      4. For each row, take the union bbox (x_min .. x_max, y_min .. y_max)
         and pad slightly to give VietOCR's cropper some breathing room.

    Returns a list of ``(quad, confidence)`` where ``quad`` is the 4-point
    polygon and ``confidence`` is the mean word area (unitless proxy —
    CRAFT does not emit per-word confidence in this code path).
    """
    if not word_polys:
        return []

    word_bboxes: list[tuple[float, float, float, float]] = []
    for poly in word_polys:
        pts = poly.reshape(-1, 2)
        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        word_bboxes.append((x1, y1, x2, y2))

    if not word_bboxes:
        return []

    word_bboxes.sort(key=lambda b: (b[1] + b[3]) * 0.5)

    rows: list[list[tuple[float, float, float, float]]] = []
    for bbox in word_bboxes:
        _, y1, _, y2 = bbox
        bh = max(y2 - y1, 1.0)
        placed = False
        for row in rows:
            ry1 = min(b[1] for b in row)
            ry2 = max(b[3] for b in row)
            inter = max(0.0, min(ry2, y2) - max(ry1, y1))
            if inter / bh >= row_overlap_ratio:
                row.append(bbox)
                placed = True
                break
        if not placed:
            rows.append([bbox])

    line_polys: list[tuple[np.ndarray, float]] = []
    for row in rows:
        x1 = min(b[0] for b in row)
        y1 = min(b[1] for b in row)
        x2 = max(b[2] for b in row)
        y2 = max(b[3] for b in row)
        h = y2 - y1
        x_pad = max(2.0, h * x_pad_ratio)
        y_pad = max(1.0, h * y_pad_ratio)
        x1 = max(0.0, x1 - x_pad)
        y1 = max(0.0, y1 - y_pad)
        if image_width:
            x2 = min(float(image_width), x2 + x_pad)
        else:
            x2 = x2 + x_pad
        if image_height:
            y2 = min(float(image_height), y2 + y_pad)
        else:
            y2 = y2 + y_pad
        quad = np.asarray(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        # Confidence proxy: number of merged words (normalised). CRAFT's
        # post-processing strips per-word scores, so this is the cheapest
        # honest signal.
        line_polys.append((quad, float(min(1.0, len(row) / 8.0 + 0.2))))
    line_polys.sort(key=lambda item: float(np.min(item[0][:, 1])))
    return line_polys


# ── Helpers ──────────────────────────────────────────────────────────────────


def _strip_module_prefix(state_dict: dict) -> dict:
    """Strip the leading ``module.`` prefix that DataParallel adds."""
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    start = 1 if keys[0].startswith("module") else 0
    out = OrderedDict()
    for k, v in state_dict.items():
        out[".".join(k.split(".")[start:])] = v
    return out


def _diagnose_polygons(polygons: list[np.ndarray], image_height: int) -> dict:
    if not polygons:
        return {"median_height": 0.0, "estimated_rows": 0.0, "coverage_ratio": 0.0}
    heights = []
    for poly in polygons:
        pts = poly.reshape(-1, 2)
        h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        heights.append(max(h, 1.0))
    median_height = float(np.median(heights))
    estimated_rows = float(image_height / max(median_height, 1.0))
    coverage_ratio = float(len(polygons) / max(estimated_rows, 1.0))
    return {
        "median_height": median_height,
        "estimated_rows": estimated_rows,
        "coverage_ratio": coverage_ratio,
    }
