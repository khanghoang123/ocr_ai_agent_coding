"""Classical 4-corner perspective + light curvature correction with OpenCV.

Algorithm:
  1. Downscale a copy for fast contour search.
  2. Convert to grayscale, blur, Canny edges, dilate.
  3. Find external contours, keep ones whose convex hull has ≥4 dominant
     corners and whose area is ≥ ``min_quad_area_ratio`` of the image.
  4. From the best candidate, take the 4 most extreme corners, sort them
     consistently (top-left, top-right, bottom-right, bottom-left).
  5. Apply a perspective warp to a target rectangle whose aspect ratio is
     estimated from the source quad.
  6. If no plausible quad is found, return the input unchanged with
     ``applied=False`` — never mangles a clean already-flat scan.

This is intentionally conservative: it only fires when there is clear
evidence of a paper boundary. On real hand-held photos that's the dominant
case; on already-cropped images or scans it leaves things alone.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from ocr_pipeline.rectifier.base import (
    Rectifier,
    RectifierResult,
    from_numpy_rgb,
    to_numpy_rgb,
)

logger = logging.getLogger(__name__)


class OpenCVRectifier(Rectifier):
    """Robust corner-quad perspective correction.

    Does not handle full curvature (it's a planar warp), but on photographed
    notebook paper the dominant distortion is a 4-corner perspective tilt,
    which this corrects well.
    """

    name = "opencv"

    def __init__(
        self,
        min_quad_area_ratio: float = 0.25,
        max_quad_area_ratio: float = 0.95,
        max_downscale_dim: int = 1024,
        canny_low: int = 50,
        canny_high: int = 200,
        dilate_iter: int = 2,
        epsilon_ratio: float = 0.02,
        target_max_dim: int = 2200,
        # Quad is only considered a real *paper boundary* if the area outside
        # is meaningfully darker than the area inside (paper is brighter than
        # the surrounding desk / shadow). This is the key check that prevents
        # the rectifier from latching onto the *printed inner rectangle* of an
        # already-rectangular scan, which would falsely "rectify" a clean page
        # by cropping off its margins (and potentially valid text in those
        # margins).
        min_paper_vs_background_contrast: float = 18.0,
        require_quad_touches_border: bool = False,
        border_margin_ratio: float = 0.02,
        **_: object,
    ):
        self.min_quad_area_ratio = float(min_quad_area_ratio)
        self.max_quad_area_ratio = float(max_quad_area_ratio)
        self.max_downscale_dim = int(max_downscale_dim)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.dilate_iter = int(dilate_iter)
        self.epsilon_ratio = float(epsilon_ratio)
        self.target_max_dim = int(target_max_dim)
        self.min_paper_vs_background_contrast = float(min_paper_vs_background_contrast)
        self.require_quad_touches_border = bool(require_quad_touches_border)
        self.border_margin_ratio = float(border_margin_ratio)

    # ── Public API ────────────────────────────────────────────────────────

    def rectify(self, image: Image.Image) -> RectifierResult:
        rgb = to_numpy_rgb(image)
        h, w = rgb.shape[:2]
        quad = self._find_paper_quad(rgb)
        if quad is None:
            return RectifierResult(
                image=image,
                backend_used=self.name,
                applied=False,
                confidence=0.0,
                diagnostics={"reason": "no_quad_found", "image_size": [w, h]},
            )

        # ── Paper-vs-background contrast gate ─────────────────────────────
        # If the area *inside* the detected quad is not noticeably brighter
        # than the area *outside*, the quad is most likely the printed inner
        # rectangle of an already-flat scan, not a real paper boundary.
        # In that case, *do not* apply rectification — we'd just lose margin
        # text without fixing any real distortion.
        gate = self._paper_background_gate(rgb, quad)
        contrast = gate["paper_vs_background_contrast"]
        too_close = contrast < self.min_paper_vs_background_contrast
        too_full = gate["quad_coverage"] >= self.max_quad_area_ratio
        skipped_for_inner_rect = too_close or too_full
        if skipped_for_inner_rect:
            return RectifierResult(
                image=image,
                backend_used=self.name,
                applied=False,
                confidence=0.0,
                diagnostics={
                    "reason": "quad_likely_inner_content_rect",
                    "contrast": round(contrast, 2),
                    "min_required": self.min_paper_vs_background_contrast,
                    "quad_coverage": round(gate["quad_coverage"], 4),
                    "quad": quad.tolist(),
                    "image_size": [w, h],
                },
            )

        warped = self._warp_to_rectangle(rgb, quad)
        if warped is None:
            # Quad was found but the implied target rectangle is degenerate
            # (≤32px on a side). Treat as "not applied" so downstream
            # consumers don't think a no-op warp counts as a real rectification.
            return RectifierResult(
                image=image,
                backend_used=self.name,
                applied=False,
                confidence=0.0,
                diagnostics={
                    "reason": "warp_target_too_small",
                    "quad": quad.tolist(),
                    "image_size": [w, h],
                },
            )
        result_img = from_numpy_rgb(warped)
        return RectifierResult(
            image=result_img,
            backend_used=self.name,
            applied=True,
            confidence=self._quad_confidence(quad, (w, h)),
            diagnostics={
                "quad": quad.tolist(),
                "input_size": [w, h],
                "output_size": [warped.shape[1], warped.shape[0]],
                "paper_vs_background_contrast": round(contrast, 2),
                "quad_coverage": round(gate["quad_coverage"], 4),
            },
        )

    def _paper_background_gate(
        self, rgb: np.ndarray, quad: np.ndarray
    ) -> dict:
        """Compare brightness inside vs outside the detected quad.

        Returns:
          - paper_vs_background_contrast: mean(inside_grey) - mean(outside_grey).
            Positive ⇒ paper-on-darker-background (the case rectification helps).
            Near zero ⇒ scan with printed inner rectangle (don't rectify).
          - quad_coverage: quad_area / image_area.
        """
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        # Erode/dilate so we strictly compare *inner* paper to *outer* background.
        kernel = np.ones((11, 11), np.uint8)
        inside_mask = cv2.erode(mask, kernel, iterations=2)
        outside_mask = cv2.bitwise_not(cv2.dilate(mask, kernel, iterations=2))
        inside_pixels = gray[inside_mask > 0]
        outside_pixels = gray[outside_mask > 0]
        inside_mean = float(inside_pixels.mean()) if inside_pixels.size > 100 else float(gray.mean())
        outside_mean = (
            float(outside_pixels.mean()) if outside_pixels.size > 100 else inside_mean
        )
        contrast = inside_mean - outside_mean
        quad_area = float(mask.sum() / 255.0)
        return {
            "paper_vs_background_contrast": contrast,
            "quad_coverage": quad_area / max(float(h * w), 1.0),
            "inside_mean": inside_mean,
            "outside_mean": outside_mean,
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _find_paper_quad(self, rgb: np.ndarray) -> np.ndarray | None:
        h, w = rgb.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_downscale_dim:
            scale = self.max_downscale_dim / max(h, w)
            small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = rgb

        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=self.dilate_iter)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        small_h, small_w = small.shape[:2]
        small_area = float(small_h * small_w)
        candidates: list[tuple[float, np.ndarray]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_quad_area_ratio * small_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append((area, approx.reshape(-1, 2).astype(np.float32)))
            elif len(approx) >= 4:
                # Fallback: take the 4 extreme corners of the convex hull.
                hull = cv2.convexHull(cnt)
                if cv2.contourArea(hull) >= self.min_quad_area_ratio * small_area:
                    quad = self._four_corners_from_hull(hull.reshape(-1, 2).astype(np.float32))
                    if quad is not None:
                        candidates.append((cv2.contourArea(hull), quad))

        if not candidates:
            return None

        # Pick the largest plausible quad.
        candidates.sort(key=lambda x: x[0], reverse=True)
        quad_small = candidates[0][1]
        quad = quad_small / scale  # back to original coords
        return self._order_corners(quad)

    @staticmethod
    def _four_corners_from_hull(hull: np.ndarray) -> np.ndarray | None:
        if hull.shape[0] < 4:
            return None
        # Use the 4 extrema along the (x+y) and (x-y) diagonals.
        s = hull.sum(axis=1)
        d = np.diff(hull, axis=1).reshape(-1)
        try:
            tl = hull[np.argmin(s)]
            br = hull[np.argmax(s)]
            tr = hull[np.argmin(d)]
            bl = hull[np.argmax(d)]
        except (ValueError, IndexError):
            return None
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        # Return corners in order: top-left, top-right, bottom-right, bottom-left.
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]   # tl
        ordered[2] = pts[np.argmax(s)]   # br
        ordered[1] = pts[np.argmin(d)]   # tr
        ordered[3] = pts[np.argmax(d)]   # bl
        return ordered

    def _warp_to_rectangle(
        self, rgb: np.ndarray, quad: np.ndarray
    ) -> np.ndarray | None:
        """Warp ``rgb`` to a flat rectangle inferred from ``quad``.

        Returns ``None`` when the implied target rectangle would be smaller
        than 32px on a side — the caller treats this as ``applied=False``
        rather than silently passing the original image through.
        """
        tl, tr, br, bl = quad
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        target_w = int(round(max(width_top, width_bottom)))
        target_h = int(round(max(height_left, height_right)))
        if target_w < 32 or target_h < 32:
            return None

        # Cap at a sensible size so we don't blow up downstream.
        if max(target_w, target_h) > self.target_max_dim:
            scale = self.target_max_dim / max(target_w, target_h)
            target_w = int(round(target_w * scale))
            target_h = int(round(target_h * scale))

        dst = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(rgb, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
        return warped

    @staticmethod
    def _quad_confidence(quad: np.ndarray, image_size: tuple[int, int]) -> float:
        """Heuristic confidence: how rectangular and how large is the quad."""
        w, h = image_size
        area_image = float(w * h)
        if area_image <= 0:
            return 0.0
        # Quad area via shoelace.
        x, y = quad[:, 0], quad[:, 1]
        area_quad = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        coverage = min(area_quad / area_image, 1.0)

        # Rectangularity: ratio of min vs max side length.
        sides = [
            float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)
        ]
        if min(sides) <= 1.0:
            return 0.0
        rect_score = min(min(sides[0], sides[2]) / max(sides[0], sides[2]),
                         min(sides[1], sides[3]) / max(sides[1], sides[3]))
        return float(0.5 * coverage + 0.5 * rect_score)
