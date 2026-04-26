"""
Text detection module using PaddleOCR.

Extracted and refactored from notebooks/05_crawl_and_detect_CLEANED.ipynb.
Returns axis-aligned bounding boxes with confidence scores.
"""

from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Raw detection output for a single image."""
    # Each box: 4 corner points [[x,y], [x,y], [x,y], [x,y]] (polygon)
    polygons: list[np.ndarray] = field(default_factory=list)
    # Confidence per box (from PaddleOCR)
    confidences: list[float] = field(default_factory=list)
    # Provenance per polygon (e.g. full_page, patch_0)
    sources: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def num_boxes(self) -> int:
        return len(self.polygons)


class PaddleDetector:
    """Wrapper around PaddleOCR for text line detection only (no recognition).

    Configured to match the pseudo-labeling pipeline settings from notebook 05.
    GPU is disabled by default to avoid paddlepaddle-gpu dependency in Docker.
    Set use_gpu=True only if paddlepaddle-gpu is installed.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        db_thresh: float = 0.3,
        db_box_thresh: float = 0.2,
        unclip_ratio: float = 1.6,
        limit_side_len: int = 1920,
        limit_type: str = "max",
        score_mode: str = "slow",
        box_type: str = "poly",
        detect_on_upscaled_image: bool = True,
        upscale_factor: float = 1.0,
        **kwargs,
    ):
        self.use_gpu = use_gpu
        self.db_thresh = db_thresh
        self.db_box_thresh = db_box_thresh
        self.unclip_ratio = unclip_ratio
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.score_mode = score_mode
        self.box_type = box_type
        self.detect_on_upscaled_image = detect_on_upscaled_image
        self.upscale_factor = float(upscale_factor)
        self._ocr = None  # Lazy init — heavy import
        self._paddle_runtime_failed = False

    def _ensure_loaded(self) -> None:
        """Lazy-initialize PaddleOCR on first use."""
        if self._ocr is not None:
            return

        self._patch_numpy_sctypes_for_imgaug()

        # Suppress PaddleOCR verbose logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        logging.getLogger("paddle").setLevel(logging.ERROR)

        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR detector (gpu=%s) ...", self.use_gpu)
        signature = inspect.signature(PaddleOCR)
        params = signature.parameters
        if "use_gpu" in params:
            kwargs = {
                "use_angle_cls": False,
                "lang": "vi",
                "use_det": True,
                "use_gpu": self.use_gpu,
                "use_rec": False,
                "det_limit_side_len": self.limit_side_len,
                "det_limit_type": self.limit_type,
                "det_db_thresh": self.db_thresh,
                "det_db_box_thresh": self.db_box_thresh,
                "det_db_unclip_ratio": self.unclip_ratio,
                "det_db_score_mode": self.score_mode,
                "det_db_box_type": self.box_type,
                "show_log": False,
            }
        else:
            kwargs = {
                "lang": "vi",
                "device": "gpu" if self.use_gpu else "cpu",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "text_det_limit_side_len": self.limit_side_len,
                "text_det_limit_type": self.limit_type,
                "text_det_thresh": self.db_thresh,
                "text_det_box_thresh": self.db_box_thresh,
                "text_det_unclip_ratio": self.unclip_ratio,
            }

        self._ocr = PaddleOCR(**kwargs)
        logger.info("PaddleOCR detector ready.")

    @staticmethod
    def _patch_numpy_sctypes_for_imgaug() -> None:
        """Provide numpy.sctypes compatibility for imgaug on NumPy >= 2.0."""
        if hasattr(np, "sctypes"):
            return
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.bytes_, np.str_, np.object_],
        }

    # ── Public API ────────────────────────────────────────────────────────────

    # ── Public API ────────────────────────────────────────────────────────────
    
    def detect(self, image: Image.Image) -> DetectionResult:
        """Detect text lines in a PIL RGB Image using baseline notebook logic.
        
        Includes pre-processing (upscale & sharpen) as used in notebook 05.
        """
        self._ensure_loaded()

        original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed, scale = self._preprocess_image(original)
        detections = self._run_detection(processed)
        if not detections:
            return DetectionResult()
            
        polygons: list[np.ndarray] = []
        confidences: list[float] = []
        sources: list[str] = []
        
        for pts, conf in detections:
            if scale != 1.0:
                pts /= scale
            polygons.append(pts)
            confidences.append(conf)
            sources.append("full_page")

        diagnostics = self._diagnose_detection(polygons, image.height)
        return DetectionResult(
            polygons=polygons,
            confidences=confidences,
            sources=sources,
            diagnostics=diagnostics,
        )

    def detect_with_notebook_fallback(self, image: Image.Image) -> DetectionResult:
        """Run full-page detection and optionally patch-wise fallback for notebook photos."""
        full = self.detect(image)
        if not self._looks_undersegmented(full, image.height):
            full.diagnostics["patchwise_used"] = False
            return full

        patchwise = self.detect_patchwise(image)
        if patchwise.num_boxes == 0:
            full.diagnostics["patchwise_used"] = False
            return full

        merged = self._merge_results(full, patchwise)
        if merged.num_boxes <= full.num_boxes:
            full.diagnostics["patchwise_used"] = False
            return full

        merged.diagnostics = self._diagnose_detection(merged.polygons, image.height)
        merged.diagnostics["patchwise_used"] = True
        return merged

    def detect_patchwise(
        self,
        image: Image.Image,
        overlap_ratio: float = 0.18,
    ) -> DetectionResult:
        self._ensure_loaded()

        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height = image_np.shape[0]
        windows = [
            (0.00, 0.40),
            (0.20, 0.62),
            (0.44, 0.86),
            (0.62, 1.00),
        ]

        polygons: list[np.ndarray] = []
        confidences: list[float] = []
        sources: list[str] = []

        for patch_index, (start_ratio, end_ratio) in enumerate(windows):
            start_y = max(0, int(height * start_ratio))
            end_y = min(height, int(height * end_ratio))
            if end_y - start_y < 40:
                continue
            patch = image_np[start_y:end_y, :]
            processed_patch, scale = self._preprocess_image(patch)
            detections = self._run_detection(processed_patch, allow_estimated_fallback=False)
            if not detections:
                continue
            for pts, confidence in detections:
                if scale != 1.0:
                    pts /= scale
                pts[:, 1] += start_y
                polygons.append(pts)
                confidences.append(confidence)
                sources.append(f"patch_{patch_index}")

        diagnostics = self._diagnose_detection(polygons, image.height)
        diagnostics["patchwise_only"] = True
        return DetectionResult(
            polygons=polygons,
            confidences=confidences,
            sources=sources,
            diagnostics=diagnostics,
        )

    def _preprocess_image(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        scale = 1.0
        if self.detect_on_upscaled_image:
            img, scale = self._upscale_for_detection(img)
        img = self._sharpen(img)
        return img, scale

    def _upscale_for_detection(
        self,
        img: np.ndarray,
        min_side: int = 800,
        max_scale: float = 2.0,
    ) -> tuple[np.ndarray, float]:
        """Upscale images for better detection.

        `upscale_factor=1.0` preserves the previous behavior: only small images
        are enlarged up to `min_side`. Values above 1.0 force that scale for
        experiments while still respecting `max_scale`.
        """
        h, w = img.shape[:2]
        m = max(h, w)
        if self.upscale_factor > 1.0:
            scale = min(self.upscale_factor, max_scale)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            return img, scale
        if m < min_side:
            scale = min(min_side / m, max_scale)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            return img, scale
        return img, 1.0

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Sharpen image for better OCR (from notebook 05)."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    def _run_detection(
        self,
        img_np: np.ndarray,
        allow_estimated_fallback: bool = True,
    ) -> list[tuple[np.ndarray, float]]:
        """Run PaddleOCR and return normalized (polygon, confidence) items."""
        if self._paddle_runtime_failed:
            return self._opencv_line_fallback(
                img_np,
                allow_estimated_fallback=allow_estimated_fallback,
            )

        if hasattr(self._ocr, "predict"):
            try:
                preds = self._ocr.predict(img_np)
                return self._parse_predict_result(preds)
            except Exception as e:
                return self._handle_detection_exception(
                    e,
                    img_np=img_np,
                    allow_estimated_fallback=allow_estimated_fallback,
                    api_name="PaddleOCR predict",
                )

        if hasattr(self._ocr, "ocr"):
            try:
                res = self._ocr.ocr(img_np, cls=False, rec=False, det=True)
                return self._parse_legacy_ocr_result(res)
            except TypeError as e:
                try:
                    res = self._ocr.ocr(img_np)
                    return self._parse_legacy_ocr_result(res)
                except Exception as fallback_error:
                    return self._handle_detection_exception(
                        fallback_error,
                        img_np=img_np,
                        allow_estimated_fallback=allow_estimated_fallback,
                        api_name="PaddleOCR legacy ocr",
                    )
            except Exception as e:
                return self._handle_detection_exception(
                    e,
                    img_np=img_np,
                    allow_estimated_fallback=allow_estimated_fallback,
                    api_name="PaddleOCR legacy ocr",
                )

        logger.warning("PaddleOCR object has neither predict() nor ocr(); using fallback detector.")
        return self._opencv_line_fallback(
            img_np,
            allow_estimated_fallback=allow_estimated_fallback,
        )

    def _handle_detection_exception(
        self,
        exc: Exception,
        img_np: np.ndarray,
        allow_estimated_fallback: bool,
        api_name: str,
    ) -> list[tuple[np.ndarray, float]]:
        if "truth value of an array with more than one element is ambiguous" in str(exc):
            logger.debug("%s returned ambiguous result, likely empty.", api_name)
            return []
        if isinstance(exc, AttributeError) and "predict" in str(exc):
            logger.warning("%s failed because predict() is unavailable: %s", api_name, exc)
            return []
        if self._is_paddle_runtime_attribute_error(exc):
            self._paddle_runtime_failed = True
            logger.warning(
                "PaddleOCR runtime is incompatible in this environment; "
                "using OpenCV line-detection fallback: %s",
                exc,
            )
            return self._opencv_line_fallback(
                img_np,
                allow_estimated_fallback=allow_estimated_fallback,
            )
        logger.warning("%s detection failed: %s", api_name, exc)
        return self._opencv_line_fallback(
            img_np,
            allow_estimated_fallback=allow_estimated_fallback,
        )

    @staticmethod
    def _is_paddle_runtime_attribute_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "ConvertPirAttribute2RuntimeAttribute" in message
            and "pir::ArrayAttribute<pir::DoubleAttribute>" in message
        )

    def _opencv_line_fallback(
        self,
        img_np: np.ndarray,
        allow_estimated_fallback: bool = True,
    ) -> list[tuple[np.ndarray, float]]:
        """Conservative non-model fallback when Paddle runtime cannot execute.

        This is intentionally limited to line-like rectangles so experiments can
        still produce crops/debug metrics in environments where PaddleOCR 3.x
        fails before returning any detection results.
        """
        if img_np is None or img_np.size == 0:
            return []

        h, w = img_np.shape[:2]
        if h < 20 or w < 20:
            return []

        mask = self._build_text_mask(img_np)
        boxes = self._boxes_from_row_projection(mask, image_width=w, image_height=h)
        if allow_estimated_fallback and self._needs_estimated_line_grid(boxes, h):
            boxes = self._estimated_line_grid(mask, image_width=w, image_height=h)

        return [
            (np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32), 0.25)
            for x1, y1, x2, y2 in boxes
        ]

    @staticmethod
    def _build_text_mask(img_np: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        background = cv2.GaussianBlur(gray, (0, 0), 25)
        normalized = cv2.divide(gray, background, scale=255)
        adaptive = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )
        colored_ink = ((saturation > 25) & (gray < 245)).astype(np.uint8) * 255
        dark_ink = (gray < 160).astype(np.uint8) * 255
        mask = cv2.bitwise_or(adaptive, cv2.bitwise_or(colored_ink, dark_ink))
        kernel_width = max(9, img_np.shape[1] // 60)
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3)),
        )

    @staticmethod
    def _boxes_from_row_projection(
        mask: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int, int, int]]:
        projection = (mask > 0).sum(axis=1).astype(np.float32)
        smooth_window = max(3, image_height // 180)
        smoothed = np.convolve(
            projection,
            np.ones(smooth_window, dtype=np.float32) / smooth_window,
            mode="same",
        )
        threshold = max(0.006 * image_width, float(np.percentile(smoothed, 70)) * 0.6)
        active = smoothed > threshold

        bands: list[list[int]] = []
        start = None
        for row, is_active in enumerate(active):
            if is_active and start is None:
                start = row
            elif not is_active and start is not None:
                if row - start >= 4:
                    bands.append([start, row])
                start = None
        if start is not None and image_height - start >= 4:
            bands.append([start, image_height])

        merged: list[list[int]] = []
        for y1, y2 in bands:
            if not merged or y1 - merged[-1][1] > 8:
                merged.append([y1, y2])
            else:
                merged[-1][1] = y2

        boxes: list[tuple[int, int, int, int]] = []
        for y1, y2 in merged:
            yy1 = max(0, y1 - 4)
            yy2 = min(image_height, y2 + 4)
            submask = mask[yy1:yy2]
            column_projection = (submask > 0).sum(axis=0)
            min_column_ink = max(1, int((yy2 - yy1) * 0.008))
            xs = np.where(column_projection > min_column_ink)[0]
            if xs.size == 0:
                continue
            x1 = max(0, int(xs[0] - 8))
            x2 = min(image_width, int(xs[-1] + 8))
            box_height = yy2 - yy1
            if (x2 - x1) < image_width * 0.04:
                continue
            if box_height < 4 or box_height > image_height * 0.18:
                continue
            boxes.append((x1, yy1, x2, yy2))
        return boxes

    @staticmethod
    def _needs_estimated_line_grid(
        boxes: list[tuple[int, int, int, int]],
        image_height: int,
    ) -> bool:
        if image_height < 700:
            return False
        return len(boxes) < max(6, int(round(image_height / 150)))

    @staticmethod
    def _estimated_line_grid(
        mask: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int, int, int]]:
        target_count = int(np.clip(round(image_height / 48), 8, 24))
        top_margin = int(image_height * 0.04)
        bottom_margin = int(image_height * 0.04)
        usable_height = max(image_height - top_margin - bottom_margin, target_count)
        step = usable_height / target_count
        line_height = max(12, int(step * 0.62))
        boxes: list[tuple[int, int, int, int]] = []

        for index in range(target_count):
            center_y = int(top_margin + (index + 0.5) * step)
            y1 = max(0, center_y - line_height // 2)
            y2 = min(image_height, center_y + line_height // 2)
            submask = mask[y1:y2]
            column_projection = (submask > 0).sum(axis=0)
            xs = np.where(column_projection > 0)[0]
            if xs.size:
                x1 = max(0, int(xs[0] - 12))
                x2 = min(image_width, int(xs[-1] + 12))
            else:
                x1 = int(image_width * 0.06)
                x2 = int(image_width * 0.94)
            if (x2 - x1) < image_width * 0.08:
                x1 = int(image_width * 0.06)
                x2 = int(image_width * 0.94)
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _parse_legacy_ocr_result(self, res) -> list[tuple[np.ndarray, float]]:
        if not res or not isinstance(res, list):
            return []
        page = res[0] if res else None
        if page is None:
            return []

        items: list[tuple[np.ndarray, float]] = []
        for row in page:
            if row is None:
                continue
            pts, conf = self._parse_legacy_row(row)
            if pts is not None:
                items.append((pts, conf))
        return items

    def _parse_legacy_row(self, row) -> tuple[np.ndarray | None, float]:
        row_array = np.array(row, dtype=object)
        if row_array.ndim == 2 and row_array.shape[1] == 2:
            pts = np.array(row, dtype=np.float32)
            if pts.shape[0] >= 4:
                return pts, 1.0

        if isinstance(row, np.ndarray):
            pts = np.array(row, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] >= 4:
                return pts, 1.0
            return None, 1.0

        if isinstance(row, (tuple, list)) and len(row) > 0:
            pts = np.array(row[0], dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] >= 4:
                conf = self._parse_confidence(row[1] if len(row) > 1 else 1.0)
                return pts, conf

        return None, 1.0

    def _parse_predict_result(self, preds) -> list[tuple[np.ndarray, float]]:
        if preds is None:
            return []
        if not isinstance(preds, list):
            preds = [preds]

        items: list[tuple[np.ndarray, float]] = []
        for pred in preds:
            payload = pred
            if hasattr(pred, "to_dict"):
                try:
                    payload = pred.to_dict()
                except Exception:
                    payload = pred

            polys = None
            scores = None
            if isinstance(payload, dict):
                polys = payload.get("dt_polys")
                scores = payload.get("dt_scores")
            else:
                polys = getattr(payload, "dt_polys", None)
                scores = getattr(payload, "dt_scores", None)

            if polys is None:
                continue
            if scores is None:
                scores = [1.0] * len(polys)

            for poly, score in zip(polys, scores):
                pts = np.array(poly, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 4:
                    continue
                conf = self._parse_confidence(score)
                items.append((pts, conf))

        return items

    @staticmethod
    def _parse_confidence(conf_field) -> float:
        if isinstance(conf_field, (int, float, np.floating)):
            return float(conf_field)
        if isinstance(conf_field, (tuple, list)) and len(conf_field) >= 2:
            try: return float(conf_field[1])
            except (TypeError, ValueError): return 1.0
        return 1.0

    @staticmethod
    def _bbox_from_polygon(poly: np.ndarray) -> tuple[float, float, float, float]:
        pts = poly.reshape(-1, 2)
        return (
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
        )

    def _diagnose_detection(self, polygons: list[np.ndarray], image_height: int) -> dict:
        if not polygons:
            return {"median_height": 0.0, "estimated_rows": 0.0, "coverage_ratio": 0.0}

        heights = []
        for poly in polygons:
            _, y1, _, y2 = self._bbox_from_polygon(poly)
            heights.append(max(y2 - y1, 1.0))

        median_height = float(np.median(heights))
        estimated_rows = float(image_height / max(median_height, 1.0))
        coverage_ratio = float(len(polygons) / max(estimated_rows, 1.0))
        return {
            "median_height": median_height,
            "estimated_rows": estimated_rows,
            "coverage_ratio": coverage_ratio,
        }

    def _looks_undersegmented(self, result: DetectionResult, image_height: int) -> bool:
        if result.num_boxes == 0:
            return False
        median_height = float(result.diagnostics.get("median_height", 0.0))
        estimated_rows = float(result.diagnostics.get("estimated_rows", 0.0))
        coverage_ratio = float(result.diagnostics.get("coverage_ratio", 1.0))

        if image_height < 700:
            return False
        if result.num_boxes <= 14:
            return True
        if median_height > image_height * 0.045:
            return True
        if estimated_rows > 0 and coverage_ratio < 0.72:
            return True
        return False

    def _merge_results(
        self,
        primary: DetectionResult,
        secondary: DetectionResult,
    ) -> DetectionResult:
        merged_polygons = list(primary.polygons)
        merged_confidences = list(primary.confidences)
        merged_sources = list(primary.sources or ["full_page"] * len(primary.polygons))

        for polygon, confidence, source in zip(
            secondary.polygons,
            secondary.confidences,
            secondary.sources or ["patch"] * len(secondary.polygons),
        ):
            candidate_bbox = self._bbox_from_polygon(polygon)
            duplicate_index = self._find_duplicate(merged_polygons, candidate_bbox)
            if duplicate_index is None:
                merged_polygons.append(polygon)
                merged_confidences.append(confidence)
                merged_sources.append(source)
                continue

            existing_bbox = self._bbox_from_polygon(merged_polygons[duplicate_index])
            existing_height = existing_bbox[3] - existing_bbox[1]
            candidate_height = candidate_bbox[3] - candidate_bbox[1]
            if confidence > merged_confidences[duplicate_index] and candidate_height <= existing_height * 1.12:
                merged_polygons[duplicate_index] = polygon
                merged_confidences[duplicate_index] = confidence
                merged_sources[duplicate_index] = source

        return DetectionResult(
            polygons=merged_polygons,
            confidences=merged_confidences,
            sources=merged_sources,
        )

    def _find_duplicate(
        self,
        polygons: list[np.ndarray],
        candidate_bbox: tuple[float, float, float, float],
    ) -> int | None:
        for index, polygon in enumerate(polygons):
            existing_bbox = self._bbox_from_polygon(polygon)
            if self._bbox_iou(existing_bbox, candidate_bbox) >= 0.65:
                return index
            if self._same_text_line(existing_bbox, candidate_bbox):
                return index
        return None

    @staticmethod
    def _bbox_iou(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1.0)
        area_b = max((bx2 - bx1) * (by2 - by1), 1.0)
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _same_text_line(
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
    ) -> bool:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        a_center_y = (ay1 + ay2) / 2.0
        b_center_y = (by1 + by2) / 2.0
        min_height = max(min(ay2 - ay1, by2 - by1), 1.0)
        vertical_close = abs(a_center_y - b_center_y) <= min_height * 0.3
        overlap_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        min_width = max(min(ax2 - ax1, bx2 - bx1), 1.0)
        return vertical_close and overlap_width >= min_width * 0.65

    @classmethod
    def from_settings(cls) -> "PaddleDetector":
        """Create a PaddleDetector from global settings."""
        from ocr_pipeline.config import settings
        return cls(
            use_gpu=settings.det_use_gpu,
            db_thresh=settings.det_db_thresh,
            db_box_thresh=settings.det_db_box_thresh,
            unclip_ratio=settings.det_db_unclip_ratio,
            limit_side_len=settings.det_limit_side_len,
            limit_type=settings.det_limit_type,
            score_mode=settings.det_db_score_mode,
            box_type=settings.det_db_box_type,
        )
