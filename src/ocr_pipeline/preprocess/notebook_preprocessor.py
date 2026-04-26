from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MaskArtifacts:
    gray: np.ndarray
    enhanced: np.ndarray
    analysis: np.ndarray
    binary: np.ndarray
    text_mask: np.ndarray
    line_mask: np.ndarray
    notebook_mode: bool
    ruled_line_mode: bool


class NotebookPreprocessor:
    """Local crop preprocessor tuned for notebook-page handwriting photos."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_tile_size: tuple[int, int] = (8, 8),
        adaptive_block_size: int = 31,
        adaptive_c: int = 11,
        line_kernel_divisor: int = 10,
        min_foreground_pixels: int = 24,
        curve_threshold: float = 0.04,
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.line_kernel_divisor = line_kernel_divisor
        self.min_foreground_pixels = min_foreground_pixels
        self.curve_threshold = curve_threshold

    def build_mask(
        self,
        crop_rgb: np.ndarray,
        notebook_mode: bool | None = None,
    ) -> MaskArtifacts:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size,
        )
        enhanced = clahe.apply(gray)
        if notebook_mode is None:
            notebook_mode = self.detect_notebook_mode(enhanced)

        analysis = enhanced.copy()
        ruled_line_mode = False
        if notebook_mode:
            analysis, ruled_line_mode = self._remove_ruled_lines(enhanced)

        block_size = max(15, self.adaptive_block_size)
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            analysis,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.adaptive_c,
        )

        width = crop_rgb.shape[1]
        kernel_len = max(16, width // self.line_kernel_divisor)
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        line_candidates = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, line_kernel)
        line_mask = cv2.morphologyEx(line_candidates, cv2.MORPH_OPEN, line_kernel)
        line_mask = cv2.dilate(line_mask, np.ones((1, 3), dtype=np.uint8), iterations=1)

        text_mask = cv2.subtract(binary, line_mask)
        text_mask = cv2.morphologyEx(
            text_mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )
        text_mask = cv2.morphologyEx(
            text_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        binary_count = int(np.count_nonzero(binary))
        text_count = int(np.count_nonzero(text_mask))
        if binary_count > 0 and text_count < max(self.min_foreground_pixels, int(binary_count * 0.25)):
            line_mask = np.zeros_like(binary)
            text_mask = binary.copy()
        return MaskArtifacts(
            gray=gray,
            enhanced=enhanced,
            analysis=analysis,
            binary=binary,
            text_mask=text_mask,
            line_mask=line_mask,
            notebook_mode=bool(notebook_mode),
            ruled_line_mode=bool(ruled_line_mode),
        )

    def detect_notebook_mode(self, gray_or_enhanced: np.ndarray) -> bool:
        inverted = cv2.bitwise_not(gray_or_enhanced)
        width = max(gray_or_enhanced.shape[1], 1)
        kernel_len = max(24, width // 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        line_response = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        coverage = float(np.count_nonzero(line_response > 18)) / float(max(line_response.size, 1))
        active_rows = line_response.mean(axis=1) > 10
        long_rows = np.sum(active_rows) / float(max(gray_or_enhanced.shape[0], 1))
        groups = 0
        group_lengths: list[int] = []
        in_group = False
        current_len = 0
        for is_active in active_rows.tolist():
            if is_active and not in_group:
                groups += 1
                in_group = True
                current_len = 1
            elif is_active and in_group:
                current_len += 1
            elif not is_active and in_group:
                in_group = False
                group_lengths.append(current_len)
        if in_group:
            group_lengths.append(current_len)

        median_thickness = float(np.median(group_lengths)) if group_lengths else 0.0
        return (
            groups >= 2
            and median_thickness <= 5.0
            and (coverage > 0.02 or long_rows > 0.08)
        )

    def _remove_ruled_lines(self, enhanced: np.ndarray) -> tuple[np.ndarray, bool]:
        inverted = cv2.bitwise_not(enhanced)
        width = max(enhanced.shape[1], 1)
        kernel_len = max(24, width // 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        line_mask = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        response = float(np.count_nonzero(line_mask > 18)) / float(max(line_mask.size, 1))

        if response < 0.02:
            hough_mask = np.zeros_like(enhanced)
            edges = cv2.Canny(enhanced, 60, 150)
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180.0,
                threshold=50,
                minLineLength=max(width * 0.6, 40),
                maxLineGap=8,
            )
            if lines is not None:
                for segment in lines[:, 0]:
                    x1, y1, x2, y2 = segment.tolist()
                    if abs(y2 - y1) <= 3:
                        cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)
            if np.count_nonzero(hough_mask) > 0:
                line_mask = cv2.max(line_mask, hough_mask)
                response = float(np.count_nonzero(line_mask > 0)) / float(max(line_mask.size, 1))

        if response <= 0.0:
            return enhanced, False

        cleaned_inv = cv2.subtract(inverted, line_mask)
        return cv2.bitwise_not(cleaned_inv), True

    def compute_tight_bounds(
        self,
        text_mask: np.ndarray,
        min_padding: int = 2,
    ) -> tuple[int, int, int, int] | None:
        points = cv2.findNonZero(text_mask)
        if points is None or len(points) < self.min_foreground_pixels:
            return None
        x, y, width, height = cv2.boundingRect(points)
        x0 = max(0, x - min_padding)
        y0 = max(0, y - min_padding)
        x1 = min(text_mask.shape[1], x + width + min_padding)
        y1 = min(text_mask.shape[0], y + height + min_padding)
        return x0, y0, x1, y1

    def compute_tight_bounds_robust(
        self,
        text_mask: np.ndarray,
        min_padding: int = 2,
        dilate_px: int = 1,
        clip_percentile: float = 0.01,
    ) -> tuple[int, int, int, int] | None:
        """Compute bounds while being resilient to small outliers/noise.

        This variant is intended for refining loose/misaligned detector boxes:
        - Optional dilation helps preserve ascenders/descenders lost in thresholding.
        - Percentile clipping ignores a small fraction of far-away pixels caused by noise.
        """
        if text_mask.size == 0:
            return None

        mask = text_mask
        if dilate_px and dilate_px > 0:
            kernel_size = dilate_px * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        ys, xs = np.where(mask > 0)
        if xs.size < self.min_foreground_pixels:
            return None

        if clip_percentile and 0.0 < clip_percentile < 0.5 and xs.size >= 64:
            lo = clip_percentile * 100.0
            hi = 100.0 - lo
            x0 = int(np.percentile(xs, lo))
            x1 = int(np.percentile(xs, hi)) + 1
            y0 = int(np.percentile(ys, lo))
            y1 = int(np.percentile(ys, hi)) + 1
        else:
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1

        x0 = max(0, x0 - min_padding)
        y0 = max(0, y0 - min_padding)
        x1 = min(text_mask.shape[1], x1 + min_padding)
        y1 = min(text_mask.shape[0], y1 + min_padding)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def crop_to_bounds(
        self,
        crop_rgb: np.ndarray,
        text_mask: np.ndarray,
        bounds: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        if bounds is None:
            return crop_rgb, text_mask, (0, 0, crop_rgb.shape[1], crop_rgb.shape[0])
        x0, y0, x1, y1 = bounds
        return crop_rgb[y0:y1, x0:x1], text_mask[y0:y1, x0:x1], bounds

    def estimate_curve_score(self, text_mask: np.ndarray) -> float:
        tops: list[int] = []
        bottoms: list[int] = []
        xs: list[int] = []
        height = max(text_mask.shape[0], 1)

        for col_idx in range(text_mask.shape[1]):
            ys = np.where(text_mask[:, col_idx] > 0)[0]
            if ys.size < 3:
                continue
            xs.append(col_idx)
            tops.append(int(ys[0]))
            bottoms.append(int(ys[-1]))

        if len(xs) < 8:
            return 0.0

        xs_arr = np.array(xs, dtype=np.float32)
        midline = (np.array(tops, dtype=np.float32) + np.array(bottoms, dtype=np.float32)) / 2.0
        coeffs = np.polyfit(xs_arr, midline, 1)
        baseline = np.polyval(coeffs, xs_arr)
        deviation = float(np.mean(np.abs(midline - baseline)))
        return deviation / float(height)

    def rectify_curved_crop(
        self,
        crop_rgb: np.ndarray,
        text_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        column_stats = []
        for col_idx in range(text_mask.shape[1]):
            ys = np.where(text_mask[:, col_idx] > 0)[0]
            if ys.size < 3:
                continue
            column_stats.append((col_idx, int(ys[0]), int(ys[-1])))

        if len(column_stats) < 8:
            return crop_rgb, text_mask

        xs = np.array([item[0] for item in column_stats], dtype=np.float32)
        tops = np.array([item[1] for item in column_stats], dtype=np.float32)
        bottoms = np.array([item[2] for item in column_stats], dtype=np.float32)
        smooth_kernel = max(5, (len(xs) // 12) * 2 + 1)
        kernel = np.ones(smooth_kernel, dtype=np.float32) / smooth_kernel
        tops = np.convolve(tops, kernel, mode="same")
        bottoms = np.convolve(bottoms, kernel, mode="same")

        width = crop_rgb.shape[1]
        grid_x = np.arange(width, dtype=np.float32)
        top_interp = np.interp(grid_x, xs, tops, left=tops[0], right=tops[-1])
        bottom_interp = np.interp(grid_x, xs, bottoms, left=bottoms[0], right=bottoms[-1])
        target_height = max(12, int(np.median(bottom_interp - top_interp)))

        map_x = np.tile(grid_x, (target_height, 1)).astype(np.float32)
        row_positions = np.linspace(0.0, 1.0, target_height, dtype=np.float32)[:, None]
        map_y = top_interp[None, :] + row_positions * np.maximum(bottom_interp - top_interp, 1.0)[None, :]
        map_y = np.clip(map_y, 0, crop_rgb.shape[0] - 1).astype(np.float32)

        rectified_rgb = cv2.remap(
            crop_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rectified_mask = cv2.remap(
            text_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rectified_rgb, rectified_mask

    def split_rows(
        self,
        text_mask: np.ndarray,
        min_segment_height: int,
        gap_merge_px: int,
    ) -> list[tuple[int, int]]:
        projection, smooth, threshold = self.compute_projection_profile(text_mask)
        if smooth.max() <= 0 or projection.max() <= 0:
            return []
        raw_threshold = max(
            0.15 * float(np.max(projection)),
            0.35 * float(np.percentile(projection, 75)),
        )
        active = projection > raw_threshold

        segments: list[tuple[int, int]] = []
        start = None
        for idx, is_active in enumerate(active.tolist()):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                segments.append((start, idx))
                start = None
        if start is not None:
            segments.append((start, len(active)))

        merged: list[tuple[int, int]] = []
        for seg_start, seg_end in segments:
            if seg_end - seg_start < min_segment_height:
                continue
            if merged and seg_start - merged[-1][1] <= gap_merge_px:
                merged[-1] = (merged[-1][0], seg_end)
            else:
                merged.append((seg_start, seg_end))
        return merged

    def compute_projection_profile(
        self,
        text_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        projection = text_mask.sum(axis=1).astype(np.float32)
        if projection.size == 0:
            return projection, projection, 0.0

        row_axis = np.arange(text_mask.shape[0], dtype=np.float32)
        center = 0.6 * max(text_mask.shape[0] - 1, 1)
        sigma = max(1.0, 0.2 * text_mask.shape[0])
        gaussian = np.exp(-0.5 * ((row_axis - center) / sigma) ** 2)
        weights = 0.35 + 0.65 * gaussian
        weighted = projection * weights

        window = max(3, (text_mask.shape[0] // 20) * 2 + 1)
        kernel = np.ones(window, dtype=np.float32) / window
        smooth = np.convolve(weighted, kernel, mode="same")
        threshold = max(
            0.2 * float(np.max(smooth)),
            0.5 * float(np.percentile(smooth, 75)),
        ) if np.max(smooth) > 0 else 0.0
        return projection, smooth, float(threshold)

    def compute_projection_bounds(
        self,
        text_mask: np.ndarray,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        projection, smooth, threshold = self.compute_projection_profile(text_mask)
        if smooth.size == 0 or np.max(smooth) <= 0:
            return None, threshold

        active = smooth > threshold
        if not np.any(active):
            return None, threshold

        ys = np.where(active)[0]
        top = int(ys[0])
        bottom = int(ys[-1]) + 1
        if bottom - top < 2:
            return None, threshold

        x_projection = text_mask.sum(axis=0).astype(np.float32)
        xs = np.where(x_projection > max(1.0, np.max(x_projection) * 0.05))[0]
        if xs.size == 0:
            return None, threshold

        return (int(xs[0]), top, int(xs[-1]) + 1, bottom), threshold

    @staticmethod
    def blank_margin_ratio(
        mask_shape: tuple[int, int],
        bounds: tuple[int, int, int, int],
    ) -> float:
        height, width = mask_shape
        x0, y0, x1, y1 = bounds
        content_area = max((x1 - x0) * (y1 - y0), 1)
        full_area = max(height * width, 1)
        return 1.0 - (content_area / full_area)

    @staticmethod
    def occupancy_ratio(text_mask: np.ndarray) -> float:
        return float(np.count_nonzero(text_mask)) / float(max(text_mask.size, 1))
