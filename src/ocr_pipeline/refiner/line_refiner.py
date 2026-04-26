"""
Geometry-safe line refinement for OCR inference.

This stage consumes PaddleOCR polygons, builds local warped crops, tightens the
geometry using hybrid mask/projection bounds, optionally rectifies curved
baselines, and only attempts split-line recovery when there is evidence of
merged lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ocr_pipeline.cropper.line_cropper import CropResult, LineCropper, WarpedPolygonCrop
from ocr_pipeline.detector.paddle_detector import DetectionResult
from ocr_pipeline.preprocess.notebook_preprocessor import NotebookPreprocessor


@dataclass
class RefinementDecision:
    source_index: int
    action: str
    note: Optional[str]
    raw_polygon: np.ndarray
    refined_polygons: list[np.ndarray] = field(default_factory=list)
    original_text: Optional[str] = None
    original_score: Optional[float] = None
    split_texts: list[str] = field(default_factory=list)
    split_scores: list[float] = field(default_factory=list)
    tight_bbox: Optional[tuple[float, float, float, float]] = None
    projection_bbox: Optional[tuple[float, float, float, float]] = None
    clamped_bbox: Optional[tuple[float, float, float, float]] = None
    curve_score: float = 0.0
    decision_stage: str = "full_page"
    fallback_reason: Optional[str] = None
    neighbor_strategy: Optional[str] = None
    projection_threshold: Optional[float] = None
    notebook_mode: bool = False
    ruled_line_mode: bool = False
    mask_preview: Optional[np.ndarray] = None
    rectified_preview: Optional[np.ndarray] = None


@dataclass
class RefinementResult:
    crops: list[CropResult] = field(default_factory=list)
    decisions: list[RefinementDecision] = field(default_factory=list)

    @property
    def polygons(self) -> list[np.ndarray]:
        return [crop.polygon for crop in self.crops]

    @property
    def confidences(self) -> list[float]:
        return [crop.confidence for crop in self.crops]


@dataclass
class _PageEntry:
    index: int
    polygon: np.ndarray
    bbox: tuple[float, float, float, float]
    center_y: float
    height: float


class LineRefiner:
    def __init__(
        self,
        preprocessor: Optional[NotebookPreprocessor] = None,
        split_score_margin: float = 0.05,
        min_segment_height_ratio: float = 0.25,
        gap_merge_px: int = 5,
        min_segment_count: int = 2,
        rectify_line: bool = True,
        enable_noise_box_filter: bool = False,
        tiny_box_height_ratio: float = 0.45,
        tiny_box_width_ratio: float = 0.08,
        tiny_box_min_ink_occupancy: float = 0.01,
    ):
        self.preprocessor = preprocessor or NotebookPreprocessor()
        self.split_score_margin = split_score_margin
        self.min_segment_height_ratio = min_segment_height_ratio
        self.gap_merge_px = gap_merge_px
        self.min_segment_count = min_segment_count
        self.rectify_line = rectify_line
        self.enable_noise_box_filter = enable_noise_box_filter
        self.tiny_box_height_ratio = tiny_box_height_ratio
        self.tiny_box_width_ratio = tiny_box_width_ratio
        self.tiny_box_min_ink_occupancy = tiny_box_min_ink_occupancy

    def refine(
        self,
        image: Image.Image,
        detection: DetectionResult,
        cropper: LineCropper,
        recognizer=None,
    ) -> RefinementResult:
        if not detection.polygons:
            return RefinementResult()

        page_entries = self._build_page_entries(detection.polygons)
        initial_median_height = float(np.median([entry.height for entry in page_entries])) if page_entries else 0.0
        page_entries = self._regroup_and_split_page_entries(image, page_entries, initial_median_height)
        page_median_height = float(np.median([entry.height for entry in page_entries])) if page_entries else 0.0
        page_median_width = float(
            np.median([max(entry.bbox[2] - entry.bbox[0], 1.0) for entry in page_entries])
        ) if page_entries else 0.0
        if hasattr(cropper, "page_median_line_height"):
            cropper.page_median_line_height = page_median_height
        neighbors = self._resolve_neighbors(page_entries, page_median_height)

        result = RefinementResult()
        next_line_index = 0

        for entry in page_entries:
            source_index = entry.index
            confidence = detection.confidences[source_index]
            source_stage = (
                detection.sources[source_index]
                if detection.sources and source_index < len(detection.sources)
                else "full_page"
            )
            decision = RefinementDecision(
                source_index=source_index,
                action="fallback",
                note=None,
                raw_polygon=entry.polygon,
                decision_stage=source_stage,
            )

            warped = cropper.warp_polygon(image, entry.polygon)
            if warped is None:
                decision.note = "warp_failed"
                decision.fallback_reason = "warp_failed"
                result.decisions.append(decision)
                continue

            artifacts = self.preprocessor.build_mask(warped.image)
            decision.notebook_mode = artifacts.notebook_mode
            decision.ruled_line_mode = artifacts.ruled_line_mode
            decision.mask_preview = artifacts.text_mask
            noise_reason = self._tiny_noise_reason(
                entry=entry,
                warped=warped,
                text_mask=artifacts.text_mask,
                page_median_height=page_median_height,
                page_median_width=page_median_width,
            )
            if noise_reason is not None:
                decision.note = noise_reason
                decision.fallback_reason = noise_reason
                if self.enable_noise_box_filter:
                    decision.action = "filtered"
                    result.decisions.append(decision)
                    continue

            mask_bounds = self.preprocessor.compute_tight_bounds_robust(
                artifacts.text_mask,
                min_padding=1,
                dilate_px=1,
                clip_percentile=0.01,
            )
            projection_bounds, projection_threshold = self.preprocessor.compute_projection_bounds(
                artifacts.text_mask
            )
            decision.projection_threshold = projection_threshold

            bounds = self._combine_bounds(
                mask_bounds=mask_bounds,
                projection_bounds=projection_bounds,
                mask_shape=artifacts.text_mask.shape,
            )
            if bounds is None:
                full_crop, crop_flags, before_crop = cropper.finalize_crop_with_metadata(warped.image)
                result.crops.append(
                    CropResult(
                        image=Image.fromarray(full_crop),
                        polygon=entry.polygon,
                        confidence=confidence,
                        line_index=next_line_index,
                        flags=crop_flags,
                        before_image=Image.fromarray(before_crop),
                    )
                )
                next_line_index += 1
                decision.action = "fallback"
                decision.note = "no_foreground_mask"
                decision.fallback_reason = "no_foreground_mask"
                decision.rectified_preview = full_crop
                decision.refined_polygons = [entry.polygon]
                result.decisions.append(decision)
                continue

            bounds = self._apply_dynamic_padding(bounds, artifacts.text_mask.shape)
            tight_page_bbox = self._map_bounds_to_page_bbox(bounds, warped)
            projection_page_bbox = (
                self._map_bounds_to_page_bbox(projection_bounds, warped)
                if projection_bounds is not None
                else None
            )
            decision.tight_bbox = tight_page_bbox
            decision.projection_bbox = projection_page_bbox

            clamped_bbox, neighbor_strategy = self._clamp_to_neighbors(
                bbox=tight_page_bbox,
                current=entry,
                previous=neighbors[source_index][0],
                next_entry=neighbors[source_index][1],
                page_median_height=page_median_height,
            )
            decision.clamped_bbox = clamped_bbox
            decision.neighbor_strategy = neighbor_strategy

            local_bounds = self._map_page_bbox_to_crop_bounds(clamped_bbox, warped) or bounds
            if getattr(cropper, "crop_strategy", "basic") == "validated":
                local_bounds = self._expand_validated_local_bounds(
                    local_bounds,
                    artifacts.text_mask.shape,
                    page_median_height,
                    cropper,
                )
            crop_rgb, crop_mask, local_bounds = self.preprocessor.crop_to_bounds(
                warped.image,
                artifacts.text_mask,
                local_bounds,
            )

            curve_score = self.preprocessor.estimate_curve_score(crop_mask)
            decision.curve_score = curve_score
            rectified_rgb = crop_rgb
            rectified_mask = crop_mask
            full_action = "kept"
            if self.rectify_line and curve_score >= self.preprocessor.curve_threshold:
                rectified_rgb, rectified_mask = self.preprocessor.rectify_curved_crop(crop_rgb, crop_mask)
                full_action = "curve_rectify"
            decision.rectified_preview = rectified_rgb

            full_crop_final, crop_flags, before_crop = cropper.finalize_crop_with_metadata(rectified_rgb)
            full_text, full_prob, full_score = self._score_candidate(
                recognizer,
                full_crop_final,
                rectified_mask,
                curve_bonus=0.02 if full_action == "curve_rectify" else 0.0,
            )
            decision.original_text = full_text
            decision.original_score = full_score

            full_polygon = self._bbox_to_polygon(clamped_bbox)
            merge_suspected = self._should_try_split(
                rectified_mask=rectified_mask,
                local_height=local_bounds[3] - local_bounds[1],
                page_median_height=page_median_height,
                full_prob=full_prob,
                notebook_mode=artifacts.notebook_mode,
            )
            split_candidates = (
                self._build_split_candidates(
                    rectified_rgb=rectified_rgb,
                    rectified_mask=rectified_mask,
                    full_page_bbox=clamped_bbox,
                    cropper=cropper,
                    confidence=confidence,
                    next_line_index=next_line_index,
                )
                if merge_suspected
                else []
            )

            if len(split_candidates) >= self.min_segment_count:
                split_texts: list[str] = []
                split_scores: list[float] = []
                split_crops: list[CropResult] = []
                for split_crop, split_mask in split_candidates:
                    split_text, _, split_score = self._score_candidate(
                        recognizer,
                        np.array(split_crop.image),
                        split_mask,
                    )
                    split_texts.append(split_text)
                    split_scores.append(split_score)
                    split_crops.append(split_crop)

                decision.split_texts = split_texts
                decision.split_scores = split_scores
                split_aggregate = float(np.mean(split_scores)) + min(len(split_scores), 3) * 0.02
                if split_aggregate >= full_score + self.split_score_margin:
                    decision.action = "split"
                    decision.note = "merge_suspected_split_won"
                    decision.refined_polygons = [crop.polygon for crop in split_crops]
                    result.crops.extend(split_crops)
                    next_line_index += len(split_crops)
                    result.decisions.append(decision)
                    continue

                if decision.note != "tiny_noise_candidate":
                    decision.fallback_reason = "split_score_lower"
                    decision.note = "merge_suspected_full_won"
            else:
                if decision.note != "tiny_noise_candidate":
                    decision.fallback_reason = "split_not_triggered" if merge_suspected else "merge_not_suspected"
                    decision.note = "single_candidate_selected"

            result.crops.append(
                CropResult(
                    image=Image.fromarray(full_crop_final),
                    polygon=full_polygon,
                    confidence=confidence,
                    line_index=next_line_index,
                    flags=crop_flags,
                    before_image=Image.fromarray(before_crop),
                )
            )
            next_line_index += 1
            decision.action = full_action
            decision.refined_polygons = [full_polygon]
            result.decisions.append(decision)

        return result

    def _regroup_and_split_page_entries(
        self,
        image: Image.Image,
        entries: list[_PageEntry],
        page_median_height: float,
    ) -> list[_PageEntry]:
        """Conservative cleanup for fallback detector bands before crop refinement."""
        if not entries or page_median_height <= 0:
            return entries

        merged = self._merge_near_duplicate_bands(entries, page_median_height)
        split: list[_PageEntry] = []
        img_np = np.array(image)
        for entry in merged:
            split.extend(self._split_tall_entry(img_np, entry, page_median_height))
        split.sort(key=lambda item: (item.center_y, item.bbox[0]))
        return split

    def _merge_near_duplicate_bands(
        self,
        entries: list[_PageEntry],
        page_median_height: float,
    ) -> list[_PageEntry]:
        out: list[_PageEntry] = []
        for entry in sorted(entries, key=lambda item: item.center_y):
            if not out:
                out.append(entry)
                continue
            previous = out[-1]
            gap = entry.bbox[1] - previous.bbox[3]
            overlap = max(0.0, min(previous.bbox[3], entry.bbox[3]) - max(previous.bbox[1], entry.bbox[1]))
            min_height = max(min(previous.height, entry.height), 1.0)
            combined_height = max(previous.bbox[3], entry.bbox[3]) - min(previous.bbox[1], entry.bbox[1])
            overlap_ratio = overlap / min_height
            too_close_duplicate = (
                gap >= 0
                and gap <= max(2.0, page_median_height * 0.08)
                and combined_height <= page_median_height * 1.45
            )
            if overlap_ratio >= 0.35 or gap < -page_median_height * 0.15 or too_close_duplicate:
                out[-1] = self._merge_entries(previous, entry)
            else:
                out.append(entry)
        return out

    @staticmethod
    def _merge_entries(left: _PageEntry, right: _PageEntry) -> _PageEntry:
        x0 = min(left.bbox[0], right.bbox[0])
        y0 = min(left.bbox[1], right.bbox[1])
        x1 = max(left.bbox[2], right.bbox[2])
        y1 = max(left.bbox[3], right.bbox[3])
        bbox = (x0, y0, x1, y1)
        return _PageEntry(
            index=left.index,
            polygon=LineRefiner._bbox_to_polygon(bbox),
            bbox=bbox,
            center_y=(y0 + y1) / 2.0,
            height=max(y1 - y0, 1.0),
        )

    def _split_tall_entry(
        self,
        img_np: np.ndarray,
        entry: _PageEntry,
        page_median_height: float,
    ) -> list[_PageEntry]:
        if entry.height <= page_median_height * 2.0:
            return [entry]
        x0, y0, x1, y1 = entry.bbox
        h, w = img_np.shape[:2]
        ix0 = max(0, int(np.floor(x0)))
        iy0 = max(0, int(np.floor(y0)))
        ix1 = min(w, int(np.ceil(x1)))
        iy1 = min(h, int(np.ceil(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            return [entry]

        crop = img_np[iy0:iy1, ix0:ix1]
        if crop.size == 0:
            return [entry]
        mask = self._entry_foreground_mask(crop)
        projection = (mask > 0).sum(axis=1).astype(np.float32)
        if projection.size == 0 or float(projection.max()) <= 0:
            return [entry]

        smooth_window = max(3, min(9, int(round(page_median_height * 0.2)) // 2 * 2 + 1))
        smoothed = np.convolve(
            projection,
            np.ones(smooth_window, dtype=np.float32) / smooth_window,
            mode="same",
        )
        threshold = max(1.0, float(np.percentile(smoothed, 70)) * 0.55)
        active = smoothed > threshold
        segments = self._active_segments(active, min_height=max(5, int(page_median_height * 0.45)))
        segments = self._merge_close_segments(segments, gap=max(2, int(page_median_height * 0.15)))
        if len(segments) < 2 or len(segments) > 3:
            return [entry]

        split_entries: list[_PageEntry] = []
        for offset, (top, bottom) in enumerate(segments):
            if bottom - top < page_median_height * 0.45:
                return [entry]
            submask = mask[top:bottom]
            col_projection = (submask > 0).sum(axis=0)
            xs = np.where(col_projection > 0)[0]
            if xs.size:
                sx0 = max(0, int(xs[0]) - 4)
                sx1 = min(ix1 - ix0, int(xs[-1]) + 5)
            else:
                sx0 = 0
                sx1 = ix1 - ix0
            bbox = (
                float(ix0 + sx0),
                float(iy0 + top),
                float(ix0 + sx1),
                float(iy0 + bottom),
            )
            split_entries.append(
                _PageEntry(
                    index=entry.index,
                    polygon=self._bbox_to_polygon(bbox),
                    bbox=bbox,
                    center_y=(bbox[1] + bbox[3]) / 2.0,
                    height=max(bbox[3] - bbox[1], 1.0),
                )
            )

        original_height = max(entry.height, 1.0)
        recovered_height = sum(item.height for item in split_entries)
        if recovered_height < original_height * 0.45:
            return [entry]
        return split_entries

    @staticmethod
    def _entry_foreground_mask(crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        block = 31 if min(crop.shape[:2]) >= 31 else max(3, min(crop.shape[:2]) // 2 * 2 + 1)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
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
    def _active_segments(active: np.ndarray, min_height: int) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        start = None
        for index, value in enumerate(active):
            if value and start is None:
                start = index
            elif not value and start is not None:
                if index - start >= min_height:
                    segments.append((start, index))
                start = None
        if start is not None and len(active) - start >= min_height:
            segments.append((start, len(active)))
        return segments

    @staticmethod
    def _merge_close_segments(
        segments: list[tuple[int, int]],
        gap: int,
    ) -> list[tuple[int, int]]:
        if not segments:
            return []
        merged = [segments[0]]
        for top, bottom in segments[1:]:
            prev_top, prev_bottom = merged[-1]
            if top - prev_bottom <= gap:
                merged[-1] = (prev_top, bottom)
            else:
                merged.append((top, bottom))
        return merged

    @staticmethod
    def _build_page_entries(polygons: list[np.ndarray]) -> list[_PageEntry]:
        entries: list[_PageEntry] = []
        for idx, polygon in enumerate(polygons):
            pts = polygon.reshape(-1, 2).astype(np.float32)
            xs = pts[:, 0]
            ys = pts[:, 1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            entries.append(
                _PageEntry(
                    index=idx,
                    polygon=pts,
                    bbox=bbox,
                    center_y=(bbox[1] + bbox[3]) / 2.0,
                    height=max(bbox[3] - bbox[1], 1.0),
                )
            )
        entries.sort(key=lambda item: item.center_y)
        return entries

    def _resolve_neighbors(
        self,
        entries: list[_PageEntry],
        page_median_height: float,
    ) -> dict[int, tuple[_PageEntry | None, _PageEntry | None]]:
        indexed = {entry.index: position for position, entry in enumerate(entries)}
        out: dict[int, tuple[_PageEntry | None, _PageEntry | None]] = {}
        for entry in entries:
            pos = indexed[entry.index]
            prev_entry = entries[pos - 1] if pos > 0 else None
            next_entry = entries[pos + 1] if pos + 1 < len(entries) else None
            out[entry.index] = (
                self._eligible_neighbor(entry, prev_entry, page_median_height),
                self._eligible_neighbor(entry, next_entry, page_median_height),
            )
        return out

    @staticmethod
    def _eligible_neighbor(
        current: _PageEntry,
        neighbor: _PageEntry | None,
        page_median_height: float,
    ) -> _PageEntry | None:
        if neighbor is None:
            return None
        overlap = max(
            0.0,
            min(current.bbox[2], neighbor.bbox[2]) - max(current.bbox[0], neighbor.bbox[0]),
        )
        min_width = max(min(current.bbox[2] - current.bbox[0], neighbor.bbox[2] - neighbor.bbox[0]), 1.0)
        overlap_ratio = overlap / min_width
        center_distance = abs(current.center_y - neighbor.center_y)
        if overlap_ratio >= 0.25 or center_distance <= 1.5 * max(page_median_height, 1.0):
            return neighbor
        return None

    def _combine_bounds(
        self,
        mask_bounds: tuple[int, int, int, int] | None,
        projection_bounds: tuple[int, int, int, int] | None,
        mask_shape: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        if mask_bounds is None and projection_bounds is None:
            return None
        if mask_bounds is None:
            return projection_bounds
        if projection_bounds is None:
            return mask_bounds

        x0 = max(0, min(mask_bounds[0], projection_bounds[0]))
        x1 = min(mask_shape[1], max(mask_bounds[2], projection_bounds[2]))
        local_height = max(mask_bounds[3] - mask_bounds[1], projection_bounds[3] - projection_bounds[1], 1)
        tolerance = max(2, int(round(0.08 * local_height)))
        y0 = max(mask_bounds[1], projection_bounds[1] - tolerance)
        y1 = min(mask_bounds[3], projection_bounds[3] + tolerance)
        if y1 <= y0:
            y0 = min(mask_bounds[1], projection_bounds[1])
            y1 = max(mask_bounds[3], projection_bounds[3])
        if x1 <= x0 or y1 <= y0:
            return None
        return int(x0), int(y0), int(x1), int(y1)

    @staticmethod
    def _apply_dynamic_padding(
        bounds: tuple[int, int, int, int],
        shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, width = shape
        x0, y0, x1, y1 = bounds
        box_h = max(y1 - y0, 1)
        box_w = max(x1 - x0, 1)
        top_pad = int(round(np.clip(0.15 * box_h, 2, 18)))
        bottom_pad = int(round(np.clip(0.25 * box_h, 2, 24)))
        side_pad = int(round(np.clip(0.02 * box_w, 1, 12)))
        return (
            max(0, x0 - side_pad),
            max(0, y0 - top_pad),
            min(width, x1 + side_pad),
            min(height, y1 + bottom_pad),
        )

    @staticmethod
    def _map_bounds_to_page_bbox(
        bounds: tuple[int, int, int, int],
        warped: WarpedPolygonCrop,
    ) -> tuple[float, float, float, float]:
        if warped.crop_to_page is None:
            return warped.source_bbox
        x0, y0, x1, y1 = bounds
        corners = np.array(
            [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(corners, warped.crop_to_page).reshape(-1, 2)
        return (
            float(mapped[:, 0].min()),
            float(mapped[:, 1].min()),
            float(mapped[:, 0].max()),
            float(mapped[:, 1].max()),
        )

    def _clamp_to_neighbors(
        self,
        bbox: tuple[float, float, float, float],
        current: _PageEntry,
        previous: _PageEntry | None,
        next_entry: _PageEntry | None,
        page_median_height: float,
    ) -> tuple[tuple[float, float, float, float], str]:
        margin = float(np.clip(0.08 * max(page_median_height, 1.0), 2.0, 8.0))
        x0, y0, x1, y1 = bbox
        clamped_top = y0
        clamped_bottom = y1
        strategy = "overlap_or_distance" if previous is not None or next_entry is not None else "none"

        if previous is not None:
            candidate = previous.bbox[3] + margin
            if candidate > clamped_top:
                clamped_top = candidate
        if next_entry is not None:
            candidate = next_entry.bbox[1] - margin
            if candidate < clamped_bottom:
                clamped_bottom = candidate

        if clamped_bottom <= clamped_top:
            return bbox, strategy
        if (clamped_bottom - clamped_top) < 0.6 * max(current.height, 1.0):
            return bbox, strategy
        return (x0, clamped_top, x1, clamped_bottom), strategy

    @staticmethod
    def _map_page_bbox_to_crop_bounds(
        bbox: tuple[float, float, float, float],
        warped: WarpedPolygonCrop,
    ) -> tuple[int, int, int, int] | None:
        if warped.page_to_crop is None:
            return None
        x0, y0, x1, y1 = bbox
        corners = np.array(
            [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(corners, warped.page_to_crop).reshape(-1, 2)
        mx0 = int(max(0, np.floor(mapped[:, 0].min())))
        my0 = int(max(0, np.floor(mapped[:, 1].min())))
        mx1 = int(min(warped.image.shape[1], np.ceil(mapped[:, 0].max())))
        my1 = int(min(warped.image.shape[0], np.ceil(mapped[:, 1].max())))
        if mx1 <= mx0 or my1 <= my0:
            return None
        return mx0, my0, mx1, my1

    @staticmethod
    def _expand_validated_local_bounds(
        bounds: tuple[int, int, int, int],
        mask_shape: tuple[int, int],
        page_median_height: float,
        cropper: LineCropper,
    ) -> tuple[int, int, int, int]:
        height, width = mask_shape
        x0, y0, x1, y1 = bounds
        box_h = max(y1 - y0, 1)
        min_height = int(round(max(cropper.min_valid_height, page_median_height * cropper.min_crop_height_ratio)))
        base_h = max(box_h, 1.0)
        top_pad = int(round(np.clip(base_h * cropper.vertical_padding_ratio * 0.65, 3, 18)))
        bottom_pad = int(round(np.clip(base_h * cropper.vertical_padding_ratio * 0.55, 3, 16)))
        side_pad = int(round(np.clip(base_h * cropper.horizontal_padding_ratio * 0.45, 4, 30)))

        y0 -= top_pad
        y1 += bottom_pad
        x0 -= side_pad
        x1 += side_pad
        if y1 - y0 < min_height:
            center = (y0 + y1) / 2.0
            y0 = int(round(center - min_height * 0.52))
            y1 = int(round(center + min_height * 0.48))
        return (
            max(0, int(x0)),
            max(0, int(y0)),
            min(width, int(x1)),
            min(height, int(y1)),
        )

    def _should_try_split(
        self,
        rectified_mask: np.ndarray,
        local_height: int,
        page_median_height: float,
        full_prob: float,
        notebook_mode: bool,
    ) -> bool:
        segments = self.preprocessor.split_rows(
            rectified_mask,
            min_segment_height=self._min_segment_height(rectified_mask.shape[0]),
            gap_merge_px=self.gap_merge_px,
        )
        height_suspicious = local_height > 1.35 * max(page_median_height, 1.0)
        prob_suspicious = full_prob < 0.65
        projection_suspicious = len(segments) >= self.min_segment_count
        return projection_suspicious or height_suspicious or prob_suspicious or notebook_mode

    def _build_split_candidates(
        self,
        rectified_rgb: np.ndarray,
        rectified_mask: np.ndarray,
        full_page_bbox: tuple[float, float, float, float],
        cropper: LineCropper,
        confidence: float,
        next_line_index: int,
    ) -> list[tuple[CropResult, np.ndarray]]:
        rows = self.preprocessor.split_rows(
            rectified_mask,
            min_segment_height=self._min_segment_height(rectified_mask.shape[0]),
            gap_merge_px=self.gap_merge_px,
        )
        if len(rows) < self.min_segment_count:
            return []

        page_x0, page_y0, page_x1, page_y1 = full_page_bbox
        page_height = max(page_y1 - page_y0, 1.0)
        out: list[tuple[CropResult, np.ndarray]] = []

        for offset, (seg_top, seg_bottom) in enumerate(rows):
            segment_rgb = rectified_rgb[seg_top:seg_bottom, :]
            segment_mask = rectified_mask[seg_top:seg_bottom, :]
            if segment_rgb.size == 0 or np.count_nonzero(segment_mask) == 0:
                continue
            segment_rgb, crop_flags, before_crop = cropper.finalize_crop_with_metadata(segment_rgb)
            top_ratio = seg_top / float(max(rectified_mask.shape[0], 1))
            bottom_ratio = seg_bottom / float(max(rectified_mask.shape[0], 1))
            seg_bbox = (
                page_x0,
                page_y0 + top_ratio * page_height,
                page_x1,
                page_y0 + bottom_ratio * page_height,
            )
            out.append(
                (
                    CropResult(
                        image=Image.fromarray(segment_rgb),
                        polygon=self._bbox_to_polygon(seg_bbox),
                        confidence=confidence,
                        line_index=next_line_index + offset,
                        flags=crop_flags,
                        before_image=Image.fromarray(before_crop),
                    ),
                    segment_mask,
                )
            )
        return out

    def _min_segment_height(self, mask_height: int) -> int:
        return max(4, min(18, int(mask_height * self.min_segment_height_ratio)))

    def _score_candidate(
        self,
        recognizer,
        crop_rgb: np.ndarray,
        text_mask: np.ndarray,
        curve_bonus: float = 0.0,
    ) -> tuple[str, float, float]:
        occupancy = self.preprocessor.occupancy_ratio(text_mask)
        candidate_bounds = self.preprocessor.compute_tight_bounds_robust(
            text_mask,
            min_padding=0,
            dilate_px=0,
            clip_percentile=0.0,
        )
        blank_margin = (
            self.preprocessor.blank_margin_ratio(text_mask.shape, candidate_bounds)
            if candidate_bounds is not None
            else 1.0
        )
        if recognizer is None:
            return "", 0.0, float(occupancy * 0.25 + curve_bonus - blank_margin * 0.10)

        pil_crop = Image.fromarray(crop_rgb)
        if hasattr(recognizer, "recognize_batch_detailed"):
            detailed = recognizer.recognize_batch_detailed([pil_crop], include_preview=False)[0]
            text = detailed.text
            prob = float(detailed.probability)
        else:
            text, prob = recognizer.recognize_batch([pil_crop], return_prob=True)[0]
            prob = float(prob)

        score = float(prob + occupancy * 0.18 - blank_margin * 0.10 + curve_bonus)
        return text, prob, score

    def _tiny_noise_reason(
        self,
        entry: _PageEntry,
        warped: WarpedPolygonCrop,
        text_mask: np.ndarray,
        page_median_height: float,
        page_median_width: float,
    ) -> str | None:
        bbox_width = max(entry.bbox[2] - entry.bbox[0], 1.0)
        bbox_height = max(entry.bbox[3] - entry.bbox[1], 1.0)
        occupancy = self.preprocessor.occupancy_ratio(text_mask)
        tiny_height = (
            page_median_height > 0
            and bbox_height < page_median_height * self.tiny_box_height_ratio
        )
        tiny_width = (
            page_median_width > 0
            and bbox_width < page_median_width * self.tiny_box_width_ratio
        )
        tiny_crop = warped.image.shape[1] < max(48, page_median_width * self.tiny_box_width_ratio)

        if (tiny_width or tiny_crop) and occupancy <= max(self.tiny_box_min_ink_occupancy, 0.0):
            return "tiny_noise_filtered" if self.enable_noise_box_filter else "tiny_noise_candidate"
        if tiny_width and tiny_height:
            return "tiny_noise_filtered" if self.enable_noise_box_filter else "tiny_noise_candidate"
        return None

    @staticmethod
    def _bbox_to_polygon(bbox: tuple[float, float, float, float]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        return np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            dtype=np.float32,
        )
