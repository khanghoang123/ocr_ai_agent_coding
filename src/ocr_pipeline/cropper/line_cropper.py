"""
Text line cropper: converts polygon bounding boxes into axis-aligned
rectangular crops, with perspetive correction for skewed/rotated text.

Extracted and refactored from notebooks/05_crawl_and_detect_CLEANED.ipynb.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CropResult:
    """A single cropped text line image with metadata."""
    image: Image.Image          # RGB PIL crop
    polygon: np.ndarray         # original 4-point polygon (float32, shape 4×2)
    confidence: float
    line_index: int             # index in original detection order (pre-sort)
    flags: list[str] = field(default_factory=list)
    before_image: Image.Image | None = None


@dataclass
class WarpedPolygonCrop:
    """Intermediate warped crop with source geometry metadata."""
    image: np.ndarray
    polygon: np.ndarray
    source_bbox: tuple[float, float, float, float]
    crop_to_page: np.ndarray | None = None
    page_to_crop: np.ndarray | None = None


class LineCropper:
    """Crops detected text line polygons from a full-page image.

    Handles:
    - Axis-aligned boxes (simple rect crop)
    - Rotated / skewed text (perspective transform)
    - Very long lines (preserved as-is)
    - Very short lines (height/width threshold filtering)
    - Padding to avoid cutting off ascenders/descenders
    """

    def __init__(
        self,
        min_height: int = 8,
        min_width: int = 20,
        padding: int = 4,
        padding_ratio: float | None = None,
        enable_rotated_crop: bool = True,
        max_dynamic_padding: int = 64,
        min_valid_height: int = 18,
        max_valid_aspect_ratio: float = 55.0,
        min_ink_density: float = 0.006,
        crop_strategy: str = "basic",
        enable_line_deskew: bool = False,
        min_crop_height_ratio: float = 0.75,
        vertical_padding_ratio: float = 0.35,
        horizontal_padding_ratio: float = 0.60,
        max_deskew_angle: float = 8.0,
    ):
        self.min_height = min_height
        self.min_width = min_width
        self.padding = padding
        self.padding_ratio = padding_ratio
        self.enable_rotated_crop = enable_rotated_crop
        self.max_dynamic_padding = max_dynamic_padding
        self.min_valid_height = min_valid_height
        self.max_valid_aspect_ratio = max_valid_aspect_ratio
        self.min_ink_density = min_ink_density
        self.crop_strategy = crop_strategy
        self.enable_line_deskew = enable_line_deskew
        self.min_crop_height_ratio = min_crop_height_ratio
        self.vertical_padding_ratio = vertical_padding_ratio
        self.horizontal_padding_ratio = horizontal_padding_ratio
        self.max_deskew_angle = max_deskew_angle
        self.page_median_line_height: float | None = None

    def crop_all(
        self,
        image: Image.Image,
        polygons: list[np.ndarray],
        confidences: list[float],
    ) -> list[CropResult]:
        """Crop all detected polygons from the image.
        
        Supports both straight (quad) and curved (poly) polygons adaptiveley.
        """
        img_np = np.array(image)   # H×W×3 RGB
        h, w = img_np.shape[:2]

        results: list[CropResult] = []

        for idx, (poly, conf) in enumerate(zip(polygons, confidences)):
            warped = self.warp_polygon(image, poly)
            if warped is None:
                continue

            crop, flags, before_crop = self.finalize_crop_with_metadata(warped.image)
                
            results.append(CropResult(
                image=Image.fromarray(crop),
                polygon=warped.polygon,
                confidence=conf,
                line_index=idx,
                flags=flags,
                before_image=Image.fromarray(before_crop),
            ))

        logger.debug("Cropped %d/%d valid lines.", len(results), len(polygons))
        return results

    def warp_polygon(
        self,
        image: Image.Image,
        polygon: np.ndarray,
    ) -> WarpedPolygonCrop | None:
        """Warp a detected polygon into a local crop without OCR-specific padding."""
        img_np = np.array(image)
        img_h, img_w = img_np.shape[:2]
        poly = polygon.reshape(-1, 2).astype(np.float32)

        crop_to_page = None
        page_to_crop = None
        if len(poly) == 4 and self.enable_rotated_crop:
            quad = self._crop_quad(img_np, poly, img_h, img_w)
            if quad is None:
                return None
            crop, crop_to_page, page_to_crop = quad
        elif len(poly) == 4:
            crop = self._crop_axis_aligned(img_np, poly, img_h, img_w)
            if crop is None:
                return None
        else:
            curve_score = self._calculate_curvature(poly)
            crop = None
            if curve_score > 0.05:
                # First try the polynomial unwarp; if it fails (e.g. the
                # polynomial fit is ill-conditioned for closed boundaries
                # where ascender/descender points dominate the edge fit),
                # fall back to the minAreaRect-based path so the line is
                # not silently dropped.
                crop = self._crop_unwarp(img_np, poly, img_h, img_w)
            if crop is None:
                rect = cv2.minAreaRect(poly)
                box = cv2.boxPoints(rect)
                if self.enable_rotated_crop:
                    quad = self._crop_quad(img_np, box, img_h, img_w)
                    if quad is not None:
                        crop, crop_to_page, page_to_crop = quad
                else:
                    crop = self._crop_axis_aligned(img_np, box, img_h, img_w)

        if crop is None:
            return None

        xs = poly[:, 0]
        ys = poly[:, 1]
        return WarpedPolygonCrop(
            image=crop,
            polygon=poly,
            source_bbox=(
                float(np.min(xs)),
                float(np.min(ys)),
                float(np.max(xs)),
                float(np.max(ys)),
            ),
            crop_to_page=crop_to_page,
            page_to_crop=page_to_crop,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _calculate_curvature(self, poly: np.ndarray) -> float:
        """Measure curvature by deviation from a baseline. Simple heuristic.

        ``poly`` may come from Paddle (CW-ordered 4-point quad), CRAFT
        (top-edge then bottom-edge, point-symmetric), or Kraken BLLA (a
        closed boundary tracing the entire line). For the first two, the
        first-half-by-index *is* the top edge. For Kraken, the index-based
        split mixes top and bottom points and produces a wildly inflated
        curvature score, which then routes the polygon into ``_crop_unwarp``
        where the same broken split makes the polynomial fit return
        ``height < min_height`` and silently drops the line. Use the shared
        ``_split_top_bottom`` helper to fall back to a y-median split for
        closed boundaries.
        """
        top_pts, _ = self._split_top_bottom(poly)
        if top_pts is None or len(top_pts) < 3:
            return 0.0

        x = top_pts[:, 0]
        y = top_pts[:, 1]

        # Linear fit (straight line).
        line_params = np.polyfit(x, y, 1)
        y_fit = np.polyval(line_params, x)

        # Mean absolute error normalised by height.
        mae = np.mean(np.abs(y - y_fit))
        height = np.max(y) - np.min(y) + 1e-6
        return mae / height

    @staticmethod
    def _split_top_bottom(
        poly: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Split a line polygon into top and bottom edge points.

        For polygons that already store the top edge in the first half and
        the bottom edge in the second half (Paddle, CRAFT, generic
        unwarp-friendly polygons), return that index-based split. For
        polygons that trace a closed boundary (Kraken BLLA, possibly other
        baseline-aware segmenters), the index-based split mixes top and
        bottom points; detect this by checking whether the index-based top
        half stays strictly above the index-based bottom half, and fall
        back to a y-median split when it doesn't.

        Returns ``(top_pts, bottom_pts)`` sorted by x in both halves, or
        ``(None, None)`` if either half ends up with fewer than two points.
        """
        n = len(poly)
        if n < 4:
            return None, None

        if n % 2 == 0:
            top_idx = poly[: n // 2]
            bot_idx = poly[n // 2 :][::-1]
            top_max_y = float(np.max(top_idx[:, 1]))
            bot_min_y = float(np.min(bot_idx[:, 1]))
            # Allow a small tolerance for ascender/descender overshoot.
            tolerance = 0.1 * max(
                float(np.max(poly[:, 1]) - np.min(poly[:, 1])),
                1.0,
            )
            if top_max_y <= bot_min_y + tolerance:
                # Sort each side by x so polynomial fits are well-conditioned.
                top_idx = top_idx[np.argsort(top_idx[:, 0])]
                bot_idx = bot_idx[np.argsort(bot_idx[:, 0])]
                return top_idx, bot_idx
            # Otherwise fall through to the y-median split below.

        # Closed boundary or odd-cardinality polygon: split by y-median.
        y_median = float(np.median(poly[:, 1]))
        top_pts = poly[poly[:, 1] <= y_median]
        bot_pts = poly[poly[:, 1] > y_median]
        if len(top_pts) < 2 or len(bot_pts) < 2:
            return None, None
        top_pts = top_pts[np.argsort(top_pts[:, 0])]
        bot_pts = bot_pts[np.argsort(bot_pts[:, 0])]
        return top_pts, bot_pts

    def _crop_quad(
        self,
        img_np: np.ndarray,
        pts: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Standard perspective transform for 4-point quadrilaterals."""
        pts = self._order_points(pts.reshape(4, 2))

        # Output dims
        width  = max(int(np.linalg.norm(pts[1] - pts[0])), int(np.linalg.norm(pts[2] - pts[3])))
        height = max(int(np.linalg.norm(pts[3] - pts[0])), int(np.linalg.norm(pts[2] - pts[1])))

        if height < self.min_height or width < self.min_width:
            return None

        # Padding & perspective warp (simplified from original version for robustness)
        dst_pts = np.array([[0,0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
        page_to_crop = cv2.getPerspectiveTransform(pts.astype(np.float32), dst_pts)
        crop = cv2.warpPerspective(img_np, page_to_crop, (width, height))
        crop_to_page = cv2.getPerspectiveTransform(dst_pts, pts.astype(np.float32))
        return crop, crop_to_page, page_to_crop

    def _crop_axis_aligned(
        self,
        img_np: np.ndarray,
        pts: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> np.ndarray | None:
        pts = pts.reshape(-1, 2).astype(np.float32)
        x0 = max(0, int(np.floor(np.min(pts[:, 0]))))
        y0 = max(0, int(np.floor(np.min(pts[:, 1]))))
        x1 = min(img_w, int(np.ceil(np.max(pts[:, 0]))))
        y1 = min(img_h, int(np.ceil(np.max(pts[:, 1]))))
        if y1 - y0 < self.min_height or x1 - x0 < self.min_width:
            return None
        return img_np[y0:y1, x0:x1]

    def _crop_unwarp(
        self,
        img_np: np.ndarray,
        poly: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> np.ndarray | None:
        """Polynomial unwarping for curved multi-point polygons.

        Uses the shared ``_split_top_bottom`` helper which falls back to a
        y-median split for closed boundaries (Kraken BLLA). The previous
        even-cardinality branch always split by index, which silently
        produced a near-zero ``height`` when the polygon was a closed loop
        and dropped the line via the ``height < min_height`` guard below.
        """
        top_pts, bot_pts = self._split_top_bottom(poly)
        if top_pts is None or bot_pts is None:
            return None

        # 2. Fit polynomials to top and bottom
        t_x, t_y = top_pts[:, 0], top_pts[:, 1]
        b_x, b_y = bot_pts[:, 0], bot_pts[:, 1]

        # Degree 2 is usually enough for paper curvature
        t_poly = np.polyfit(t_x, t_y, 2)
        b_poly = np.polyfit(b_x, b_y, 2)

        # 3. Define output rectangle grid. Use mean y-difference on a shared
        # x-grid rather than ``b_y - t_y`` directly, which fails when top and
        # bottom have different point counts.
        width = int(np.max(t_x) - np.min(t_x))
        common_x = np.linspace(
            max(np.min(t_x), np.min(b_x)),
            min(np.max(t_x), np.max(b_x)),
            max(width, 8),
        )
        height = int(np.mean(np.polyval(b_poly, common_x) - np.polyval(t_poly, common_x)))
        
        if height < self.min_height or width < self.min_width:
            return None
            
        # 4. Generate map for cv2.remap
        target_x = np.linspace(np.min(t_x), np.max(t_x), width)
        target_y = np.linspace(0, 1, height)
        
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        for i, ty in enumerate(target_y):
            # Linearly interpolate between top and bottom polynomials
            curr_y_poly = (1 - ty) * t_poly + ty * b_poly
            map_x[i, :] = target_x
            map_y[i, :] = np.polyval(curr_y_poly, target_x)
            
        unwarped = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC)
        return unwarped

    def finalize_crop(self, crop: np.ndarray) -> np.ndarray:
        """Apply notebook-inspired cleanup before recognition."""
        finalized, _, _ = self.finalize_crop_with_metadata(crop)
        return finalized

    def finalize_crop_with_metadata(self, crop: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Apply cleanup and return crop-quality flags plus the pre-cleanup crop."""
        if crop.size == 0:
            return crop, ["empty_crop"], crop

        flags: list[str] = []
        before_crop = crop.copy()
        h, w = crop.shape[:2]
        if h > w:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            flags.append("vertical_crop_rotated")
            h, w = crop.shape[:2]

        page_median = float(self.page_median_line_height or 0.0)
        skip_validated_geometry = page_median > 0 and h > page_median * 1.65
        if self.crop_strategy == "validated" and not skip_validated_geometry:
            crop, validation_flags = self._validated_geometry_crop(crop)
            flags.extend(validation_flags)
        elif self.crop_strategy == "validated" and skip_validated_geometry:
            flags.append("validated_geometry_skipped_tall_crop")

        crop, trim_flags = self._trim_horizontal_empty_margins(crop)
        flags.extend(trim_flags)
        h, w = crop.shape[:2]

        if self.enable_line_deskew:
            crop, deskew_flags = self._deskew_line_crop(crop)
            flags.extend(deskew_flags)
            h, w = crop.shape[:2]

        base_padding = self.padding
        if self.padding_ratio is not None:
            base_padding = int(round(h * float(self.padding_ratio)))
        base_padding = max(0, min(base_padding, self.max_dynamic_padding))
        if self.crop_strategy == "validated":
            vertical_padding = max(
                base_padding,
                int(round(np.clip(h * self.vertical_padding_ratio, 3, 28))),
            )
            horizontal_padding = max(
                1 if base_padding > 0 else 0,
                int(round(np.clip(h * self.horizontal_padding_ratio, 4, 36))),
            )
        else:
            vertical_padding = max(base_padding, int(round(np.clip(h * 0.18, 2, 18))))
            horizontal_padding = max(1 if base_padding > 0 else 0, int(round(np.clip(h * 0.08, 1, 10))))

        if vertical_padding > 0 or horizontal_padding > 0:
            crop = cv2.copyMakeBorder(
                crop,
                vertical_padding,
                vertical_padding,
                horizontal_padding,
                horizontal_padding,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )

        flags.extend(self._crop_quality_flags(crop))
        return crop, sorted(set(flags)), before_crop

    def _validated_geometry_crop(self, crop: np.ndarray) -> tuple[np.ndarray, list[str]]:
        flags: list[str] = []
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return crop, flags

        mask = self._foreground_mask(crop)
        component_bbox = self._component_text_bbox(mask)
        if component_bbox is None:
            flags.append("low_ink_density")
            return crop, flags

        x0, y0, x1, y1 = component_bbox
        text_h = max(y1 - y0, 1)
        page_median = float(self.page_median_line_height or 0.0)
        min_height = int(round(max(self.min_valid_height, page_median * self.min_crop_height_ratio)))

        top_pad = int(round(np.clip(text_h * self.vertical_padding_ratio * 0.85, 3, 20)))
        bottom_pad = int(round(np.clip(text_h * self.vertical_padding_ratio * 0.70, 3, 18)))
        side_pad = int(round(np.clip(text_h * self.horizontal_padding_ratio * 0.65, 5, 34)))

        cy = (y0 + y1) / 2.0
        target_y0 = y0 - top_pad
        target_y1 = y1 + bottom_pad
        if target_y1 - target_y0 < min_height:
            target_y0 = int(round(cy - min_height * 0.52))
            target_y1 = int(round(cy + min_height * 0.48))
            flags.append("min_height_expanded")

        clipped_y0 = max(0, int(np.floor(target_y0)))
        clipped_y1 = min(h, int(np.ceil(target_y1)))
        clipped_x0 = max(0, int(np.floor(x0 - side_pad)))
        clipped_x1 = min(w, int(np.ceil(x1 + side_pad)))

        if clipped_y0 == 0 and target_y0 < 0:
            flags.append("vertical_boundary_clamped_top")
        if clipped_y1 == h and target_y1 > h:
            flags.append("vertical_boundary_clamped_bottom")
        if (clipped_y1 - clipped_y0) < min_height:
            flags.append("source_band_height_limited")

        if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
            return crop, flags

        full_width = (clipped_x1 - clipped_x0) >= w * 0.94
        component_span = (x1 - x0) / max(w, 1)
        if full_width and component_span < 0.82:
            flags.append("full_width_background_risk")
        if clipped_y0 > 0 or clipped_y1 < h or clipped_x0 > 0 or clipped_x1 < w:
            flags.append("validated_geometry_crop")
        return crop[clipped_y0:clipped_y1, clipped_x0:clipped_x1], flags

    def _component_text_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        h, w = mask.shape[:2]
        accepted: list[tuple[int, int, int, int]] = []
        min_area = max(3, int(round(h * w * 0.00012)))
        for label in range(1, num_labels):
            x, y, comp_w, comp_h, area = stats[label]
            if area < min_area:
                continue
            if comp_w >= w * 0.88 and comp_h <= max(3, h * 0.12):
                continue
            if comp_h <= 1 or comp_w <= 1:
                continue
            accepted.append((x, y, x + comp_w, y + comp_h))

        if not accepted:
            ys, xs = np.where(mask > 0)
            if xs.size == 0 or ys.size == 0:
                return None
            return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

        x0 = min(item[0] for item in accepted)
        y0 = min(item[1] for item in accepted)
        x1 = max(item[2] for item in accepted)
        y1 = max(item[3] for item in accepted)
        return x0, y0, x1, y1

    def _deskew_line_crop(self, crop: np.ndarray) -> tuple[np.ndarray, list[str]]:
        flags: list[str] = []
        h, w = crop.shape[:2]
        if h < 8 or w < max(32, h * 2):
            return crop, flags

        mask = self._foreground_mask(crop)
        ys, xs = np.where(mask > 0)
        if xs.size < 12:
            return crop, flags

        points = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(points)
        angle = float(rect[-1])
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) > self.max_deskew_angle:
            flags.append("deskew_angle_clamped")
            angle = float(np.clip(angle, -self.max_deskew_angle, self.max_deskew_angle))
        if abs(angle) < 0.6:
            return crop, flags

        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        rotated = cv2.warpAffine(
            crop,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        flags.append("line_deskewed")
        flags.append(f"deskew_angle={angle:.2f}")
        return rotated, flags

    def _trim_horizontal_empty_margins(self, crop: np.ndarray) -> tuple[np.ndarray, list[str]]:
        flags: list[str] = []
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return crop, flags

        mask = self._foreground_mask(crop)
        column_projection = (mask > 0).sum(axis=0)
        min_ink = max(1, int(round(h * 0.02)))
        xs = np.where(column_projection > min_ink)[0]
        if xs.size == 0:
            flags.append("low_ink_density")
            return crop, flags

        pad = int(round(np.clip(h * 0.45, 4, 28)))
        x0 = max(0, int(xs[0]) - pad)
        x1 = min(w, int(xs[-1]) + pad + 1)
        if x1 <= x0:
            return crop, flags

        removed = (x0 > 2) or (x1 < w - 2)
        min_width = max(self.min_width, int(round(w * 0.08)))
        if removed and (x1 - x0) >= min_width:
            crop = crop[:, x0:x1]
            flags.append("ink_horizontal_trimmed")
        return crop, flags

    def _crop_quality_flags(self, crop: np.ndarray) -> list[str]:
        h, w = crop.shape[:2]
        flags: list[str] = []
        if h < self.min_valid_height:
            flags.append("too_thin_crop")
        aspect = w / max(h, 1)
        if aspect > self.max_valid_aspect_ratio:
            flags.append("abnormal_aspect_ratio")

        mask = self._foreground_mask(crop)
        density = float(np.count_nonzero(mask)) / float(max(h * w, 1))
        if density < self.min_ink_density:
            flags.append("low_ink_density")
        return flags

    @staticmethod
    def _foreground_mask(crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31 if min(crop.shape[:2]) >= 31 else max(3, min(crop.shape[:2]) // 2 * 2 + 1),
            12,
        )
        colored_or_dark = (((saturation > 25) & (gray < 245)) | (gray < 165)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(adaptive, colored_or_dark)
        h, w = mask.shape[:2]
        if w >= 40:
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 2), 1))
            horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(horizontal_lines))
        return mask

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    @classmethod
    def from_settings(cls) -> "LineCropper":
        from ocr_pipeline.config import settings
        return cls(
            min_height=settings.min_line_height,
            min_width=settings.min_line_width,
            padding=settings.crop_padding,
        )
