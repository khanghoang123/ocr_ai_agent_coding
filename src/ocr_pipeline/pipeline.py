"""
OCRPipeline: the main orchestrator that chains all modules end-to-end.

Input:  A file path (image / PDF) or raw bytes
Output: OCRResult (structured, ready for API response or export)

Pipeline stages:
  1. FileHandler  → list of PIL Images (one per page)
  2. Detector     → polygon bounding boxes per page
  3. Cropper      → rectangular line-level PIL crops
  4. Recognizer   → recognized text per crop
  5. Reconstructor → sorted PageResult with reading order preserved
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from ocr_pipeline.cropper.line_cropper import CropResult, LineCropper
from ocr_pipeline.detector.base import LineDetector
from ocr_pipeline.detector.paddle_detector import PaddleDetector
from ocr_pipeline.file_handlers import load_bytes, load_file
from ocr_pipeline.layout.reconstructor import LayoutReconstructor
from ocr_pipeline.recognizer.vietocr_recognizer import VietOCRRecognizer
from ocr_pipeline.recognizer.vietnamese_postprocess import safe_postprocess_lines
from ocr_pipeline.rectifier import IdentityRectifier, Rectifier, build_rectifier
from ocr_pipeline.refiner.line_refiner import LineRefiner
from ocr_pipeline.schemas import (
    BoundingBox,
    DebugCropPreview,
    DebugDetectionDecision,
    DebugPageResult,
    OCRDebugItem,
    OCRResult,
    PageResult,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


class OCRPipeline:
    """End-to-end OCR pipeline. Designed to be loaded once and reused."""

    def __init__(
        self,
        detector: LineDetector | PaddleDetector,
        cropper: LineCropper,
        recognizer: VietOCRRecognizer,
        reconstructor: LayoutReconstructor,
        refiner: LineRefiner,
        model_key: str = "experiment_B",
        enable_vietnamese_postprocess: bool = False,
        flag_digit_noise: bool = False,
        unsupported_options: list[str] | None = None,
        rectifier: Rectifier | None = None,
        save_rectifier_debug: bool = True,
    ):
        self.detector = detector
        self.cropper = cropper
        self.recognizer = recognizer
        self.reconstructor = reconstructor
        self.refiner = refiner
        self.model_key = model_key
        self.enable_vietnamese_postprocess = enable_vietnamese_postprocess
        self.flag_digit_noise = flag_digit_noise
        self.unsupported_options = unsupported_options or []
        self.rectifier: Rectifier = rectifier or IdentityRectifier()
        self.save_rectifier_debug = save_rectifier_debug

    def process_file(self, file_path: str | Path) -> OCRResult:
        path = Path(file_path)
        t_start = time.perf_counter()
        file_type, pages = load_file(path)
        return self._process_pages(pages, file_type, path.name, t_start)

    def process_bytes(self, data: bytes, filename: str) -> OCRResult:
        t_start = time.perf_counter()
        file_type, pages = load_bytes(data, filename)
        return self._process_pages(pages, file_type, filename, t_start)

    def process_bytes_debug(self, data: bytes, filename: str) -> OCRDebugItem:
        t_start = time.perf_counter()
        file_type, pages = load_bytes(data, filename)
        result, debug_pages = self._process_pages(
            pages,
            file_type,
            filename,
            t_start,
            include_debug=True,
        )
        return OCRDebugItem(result=result, debug_pages=debug_pages)

    def process_image(self, image: Image.Image, filename: str = "image.jpg") -> OCRResult:
        t_start = time.perf_counter()
        return self._process_pages([image], "image", filename, t_start)

    def benchmark(self, images: list[Image.Image]) -> list[PerformanceMetrics]:
        metrics_list: list[PerformanceMetrics] = []
        for idx, img in enumerate(images):
            t0 = time.perf_counter()
            det_result = self.detector.detect_with_notebook_fallback(img)
            det_ms = (time.perf_counter() - t0) * 1000

            crops = self.cropper.crop_all(img, det_result.polygons, det_result.confidences)

            t1 = time.perf_counter()
            self.recognizer.recognize_batch([c.image for c in crops])
            rec_ms = (time.perf_counter() - t1) * 1000

            metrics_list.append(
                PerformanceMetrics(
                    filename=f"bench_image_{idx}.jpg",
                    detection_ms=round(det_ms, 2),
                    recognition_ms=round(rec_ms, 2),
                    total_ms=round(det_ms + rec_ms, 2),
                    num_lines=len(crops),
                    image_width=img.width,
                    image_height=img.height,
                )
            )

        return metrics_list

    def _process_pages(
        self,
        pages: list[Image.Image],
        file_type: str,
        filename: str,
        t_start: float,
        include_debug: bool = False,
    ) -> OCRResult | tuple[OCRResult, list[DebugPageResult]]:
        page_results: list[PageResult] = []
        debug_results: list[DebugPageResult] = []

        for page_num, page_img in enumerate(pages, start=1):
            logger.info(
                "Processing page %d/%d of '%s' (%dx%d) ...",
                page_num,
                len(pages),
                filename,
                page_img.width,
                page_img.height,
            )
            page_output = self._process_single_page(
                page_img,
                page_num,
                include_debug=include_debug,
            )
            if include_debug:
                page_result, page_debug = page_output
                debug_results.append(page_debug)
            else:
                page_result = page_output
            page_results.append(page_result)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Completed '%s' — %d pages, %d lines, %.0f ms total.",
            filename,
            len(page_results),
            sum(page.line_count for page in page_results),
            elapsed_ms,
        )

        ocr_result = OCRResult(
            filename=filename,
            file_type=file_type,
            model_used=self.model_key,
            total_pages=len(page_results),
            pages=page_results,
            processing_time_ms=round(elapsed_ms, 2),
            page_images=pages,
        )
        if include_debug:
            return ocr_result, debug_results
        return ocr_result

    def _process_single_page(
        self,
        image: Image.Image,
        page_number: int = 1,
        include_debug: bool = False,
    ) -> PageResult | tuple[PageResult, DebugPageResult]:
        # Phase 2C: page-level rectification BEFORE detection.
        # The IdentityRectifier (default) is a no-op so this is free when
        # `enable_document_perspective_correction=False`.
        original_image = image
        rect_result = self.rectifier.rectify(image)
        rectified_image = rect_result.image
        if rect_result.applied:
            logger.info(
                "Page %d: rectifier=%s applied=True confidence=%.3f size=%s→%s",
                page_number,
                rect_result.backend_used,
                rect_result.confidence,
                original_image.size,
                rectified_image.size,
            )
        else:
            logger.info(
                "Page %d: rectifier=%s applied=False (%s)",
                page_number,
                rect_result.backend_used,
                rect_result.diagnostics.get("reason") or rect_result.diagnostics.get("reason_for_fallback"),
            )

        # Detection runs on the rectified image; all downstream stages use
        # rectified-image coordinates.
        det_result = self.detector.detect_with_notebook_fallback(rectified_image)
        det_diagnostics = dict(det_result.diagnostics or {})
        debug_page = DebugPageResult(
            page_number=page_number,
            raw_polygons=[
                [[float(x), float(y)] for x, y in poly.reshape(-1, 2).tolist()]
                for poly in det_result.polygons
            ],
            rectifier_backend=rect_result.backend_used,
            rectifier_applied=rect_result.applied,
            rectifier_confidence=float(rect_result.confidence),
            rectifier_diagnostics=dict(rect_result.diagnostics),
            detector_backend=det_diagnostics.get("backend"),
            detector_diagnostics=det_diagnostics,
            low_detector_recall=bool(det_diagnostics.get("low_detector_recall", False)),
        )
        if include_debug and self.save_rectifier_debug:
            debug_page.original_image_base64 = self._encode_preview(original_image)
            if rect_result.applied:
                debug_page.rectified_image_base64 = self._encode_preview(rectified_image)
        # Subsequent stages should see the (possibly rectified) image.
        image = rectified_image

        if det_result.num_boxes == 0:
            logger.warning("Page %d: No text lines detected.", page_number)
            empty_page = PageResult(
                page_number=page_number,
                width=image.width,
                height=image.height,
                lines=[],
            )
            if include_debug:
                return empty_page, debug_page
            return empty_page

        refined = self.refiner.refine(
            image=image,
            detection=det_result,
            cropper=self.cropper,
            recognizer=self.recognizer,
        )
        debug_page.refined_boxes = [self._polygon_to_bbox(crop.polygon) for crop in refined.crops]
        debug_page.decisions = [self._decision_to_schema(decision) for decision in refined.decisions]

        crops = refined.crops
        if not crops:
            empty_page = PageResult(
                page_number=page_number,
                width=image.width,
                height=image.height,
                lines=[],
            )
            if include_debug:
                return empty_page, debug_page
            return empty_page

        if self.recognizer is None:
            recognized = None
            texts = [""] * len(crops)
        elif include_debug:
            recognized = self.recognizer.recognize_batch_detailed(
                [crop.image for crop in crops],
                include_preview=True,
            )
            texts = [item.text for item in recognized]
        else:
            recognized = None
            texts = self.recognizer.recognize_batch([crop.image for crop in crops])

        page_result = self.reconstructor.reconstruct(
            crops=crops,
            texts=texts,
            image_width=image.width,
            image_height=image.height,
            page_number=page_number,
        )
        page_result = self._apply_safe_postprocess(page_result)

        if include_debug:
            debug_page.final_crops = []
            for line in page_result.lines:
                crop_idx = self._match_line_to_crop_index(line, crops)
                if crop_idx is None:
                    continue
                rec_item = recognized[crop_idx] if recognized is not None else None
                debug_page.final_crops.append(
                    DebugCropPreview(
                        line_index=line.line_index,
                        text=line.text,
                        detection_confidence=line.confidence,
                        recognition_score=float(rec_item.probability) if rec_item is not None else 0.0,
                        bbox=line.bbox,
                        preview_base64=self._encode_preview(crops[crop_idx].image),
                        before_preview_base64=(
                            self._encode_preview(crops[crop_idx].before_image)
                            if crops[crop_idx].before_image is not None
                            else None
                        ),
                        crop_flags=list(crops[crop_idx].flags),
                        decode_mode=rec_item.decode_mode if rec_item is not None else "unavailable",
                        pre_norm_ratio=rec_item.pre_norm_ratio if rec_item is not None else 1.0,
                        post_norm_ratio=rec_item.post_norm_ratio if rec_item is not None else 1.0,
                        baseline_confidence=rec_item.baseline_confidence if rec_item is not None else 0.0,
                        baseline_offset=rec_item.baseline_offset if rec_item is not None else 0.0,
                        normalization_preview_base64=self._encode_array_preview(
                            rec_item.normalized_preview if rec_item is not None else None
                        ),
                    )
                )
            return page_result, debug_page

        return page_result

    @classmethod
    def from_settings(cls, model_key: Optional[str] = None) -> "OCRPipeline":
        """Build an ``OCRPipeline`` from environment-driven Settings.

        The detector backend is selected by ``Settings.detector_backend``
        (env var ``DETECTOR_BACKEND``). Phase 3 default is ``kraken``;
        when that fails to initialise (e.g. weights cannot be downloaded
        in a sealed CI environment) we fall back to PaddleDetector so the
        legacy code path keeps working.
        """
        from ocr_pipeline.config import settings
        from ocr_pipeline.detector import build_detector

        key = model_key or settings.get_default_model_key()
        backend_name = (settings.detector_backend or "paddle").lower()
        if backend_name in ("paddle", "paddleocr", "dbnet"):
            detector = PaddleDetector.from_settings()
        else:
            try:
                detector = build_detector(backend_name)
            except Exception as exc:  # pragma: no cover - graceful fallback
                logger.warning(
                    "Configured detector backend %r failed to initialise (%s); "
                    "falling back to PaddleDetector.",
                    backend_name,
                    exc,
                )
                detector = PaddleDetector.from_settings()
        return cls(
            detector=detector,
            cropper=LineCropper.from_settings(),
            recognizer=VietOCRRecognizer.from_settings(key),
            reconstructor=LayoutReconstructor.from_settings(),
            refiner=LineRefiner(),
            model_key=key,
        )

    @classmethod
    def from_experiment_config(cls, config) -> "OCRPipeline":
        from ocr_pipeline.config import settings
        from ocr_pipeline.detector import build_detector

        key = config.model_key or settings.get_default_model_key()
        model_cfg = settings.get_model_config(key)
        rectifier = build_rectifier(
            backend=getattr(config, "rectifier_backend", "hybrid"),
            enabled=bool(getattr(config, "enable_document_perspective_correction", False)),
            weights_path=getattr(config, "rectifier_weights_path", None),
            device=getattr(config, "rectifier_device", "cpu"),
            min_quad_area_ratio=float(getattr(config, "rectifier_min_quad_area_ratio", 0.25)),
            min_confidence=float(getattr(config, "rectifier_min_confidence", 0.5)),
            min_paper_vs_background_contrast=float(
                getattr(config, "rectifier_min_paper_background_contrast", 18.0)
            ),
            max_quad_area_ratio=float(
                getattr(config, "rectifier_max_quad_area_ratio", 0.95)
            ),
        )
        # Phase 3 default: Kraken BLLA (winner of the Tier-1 leaderboard).
        # ``ExperimentConfig`` already defaults to ``"kraken"``; this
        # ``or settings.detector_backend`` fallback only matters when the
        # caller hands us a config object that doesn't define the field.
        default_backend = settings.detector_backend or "kraken"
        backend_name = (
            getattr(config, "detector_backend", default_backend) or default_backend
        ).lower()
        if backend_name in ("paddle", "paddleocr", "dbnet"):
            detector = PaddleDetector(
                use_gpu=settings.det_use_gpu,
                db_thresh=settings.det_db_thresh,
                db_box_thresh=settings.det_db_box_thresh,
                unclip_ratio=settings.det_db_unclip_ratio,
                limit_side_len=settings.det_limit_side_len,
                limit_type=settings.det_limit_type,
                score_mode=settings.det_db_score_mode,
                box_type=settings.det_db_box_type,
                detect_on_upscaled_image=config.detect_on_upscaled_image,
                upscale_factor=config.upscale_factor,
                allow_grid_fallback=bool(
                    getattr(config, "detector_allow_grid_fallback", False)
                ),
            )
        else:
            # Phase 3 detectors: Surya / CRAFT / Kraken BLLA. They share a
            # tiny init surface; experiments override knobs via the
            # ``detector_kwargs`` map on ``ExperimentConfig`` if needed.
            detector = build_detector(
                backend_name,
                **dict(getattr(config, "detector_kwargs", {}) or {}),
            )
        return cls(
            detector=detector,
            cropper=LineCropper(
                min_height=settings.min_line_height,
                min_width=settings.min_line_width,
                padding=settings.crop_padding if config.crop_padding is None else config.crop_padding,
                padding_ratio=config.crop_padding_ratio,
                enable_rotated_crop=config.enable_rotated_crop,
                crop_strategy=config.crop_strategy,
                enable_line_deskew=config.enable_line_deskew,
                min_crop_height_ratio=config.min_crop_height_ratio,
                vertical_padding_ratio=config.vertical_padding_ratio,
                horizontal_padding_ratio=config.horizontal_padding_ratio,
                max_deskew_angle=config.max_deskew_angle,
            ),
            recognizer=cls._build_recognizer(config, model_cfg, settings),
            reconstructor=LayoutReconstructor.from_settings(),
            refiner=LineRefiner(
                rectify_line=config.rectify_line,
                enable_noise_box_filter=config.enable_noise_box_filter,
                tiny_box_height_ratio=config.tiny_box_height_ratio,
                tiny_box_width_ratio=config.tiny_box_width_ratio,
                tiny_box_min_ink_occupancy=config.tiny_box_min_ink_occupancy,
            ),
            model_key=key,
            enable_vietnamese_postprocess=config.enable_vietnamese_postprocess,
            flag_digit_noise=config.flag_digit_noise,
            unsupported_options=config.unsupported_options,
            rectifier=rectifier,
            save_rectifier_debug=bool(getattr(config, "rectifier_save_debug", True)),
        )

    @classmethod
    def _build_recognizer(
        cls,
        config,
        model_cfg: dict,
        settings,
    ) -> "VietOCRRecognizer":
        """Construct VietOCRRecognizer with per-experiment overrides.

        KenLM rescoring is enabled per-experiment via ``rec_kenlm_*``
        fields on ``ExperimentConfig`` which fall back to the global
        ``Settings`` values when left as ``None``. When the resolved
        beam width is 1 the rescorer is not attached — the recognizer
        falls through to the legacy greedy (or no-repeat-ngram)
        decode path with no extra cost.
        """
        from ocr_pipeline.recognizer.kenlm_rescorer import KenLMRescorer

        def pick(cfg_val, settings_val):
            return cfg_val if cfg_val is not None else settings_val

        beam_width = int(
            pick(config.rec_kenlm_beam_width, settings.rec_kenlm_beam_width) or 1
        )
        rescorer = None
        if beam_width > 1:
            kenlm_path = pick(config.rec_kenlm_path, settings.rec_kenlm_path)
            rescorer = KenLMRescorer(
                model_path=kenlm_path,
                alpha=float(
                    pick(config.rec_kenlm_alpha, settings.rec_kenlm_alpha)
                ),
                beta=float(
                    pick(config.rec_kenlm_beta, settings.rec_kenlm_beta)
                ),
                gamma=float(
                    pick(config.rec_kenlm_gamma, settings.rec_kenlm_gamma)
                ),
            )

        return VietOCRRecognizer(
            weights_path=model_cfg["weights_path"],
            architecture=model_cfg["architecture"],
            image_height=model_cfg["image_height"],
            image_max_width=model_cfg["image_max_width"],
            image_min_width=model_cfg["image_min_width"],
            device=settings.rec_device,
            enable_local_contrast=config.enable_local_contrast,
            no_repeat_ngram_size=(
                config.rec_no_repeat_ngram_size
                if config.rec_no_repeat_ngram_size is not None
                else settings.rec_no_repeat_ngram_size
            ),
            kenlm_rescorer=rescorer,
            kenlm_beam_width=beam_width,
        )

    def _apply_safe_postprocess(self, page_result: PageResult) -> PageResult:
        if not (self.enable_vietnamese_postprocess or self.flag_digit_noise):
            return page_result
        processed = safe_postprocess_lines(
            [line.text for line in page_result.lines],
            flag_digit_noise=self.flag_digit_noise,
        )
        updates = {}
        if self.enable_vietnamese_postprocess:
            updates["postprocessed"] = {"safe": "\n".join(item.cleaned_text for item in processed)}
            updates["postprocessed_lines"] = {"safe": [item.cleaned_text for item in processed]}
        return page_result.model_copy(update=updates)

    @staticmethod
    def _encode_preview(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _polygon_to_bbox(polygon) -> BoundingBox:
        pts = polygon.reshape(-1, 2)
        return BoundingBox(
            x1=float(pts[:, 0].min()),
            y1=float(pts[:, 1].min()),
            x2=float(pts[:, 0].max()),
            y2=float(pts[:, 1].max()),
        )

    def _decision_to_schema(self, decision) -> DebugDetectionDecision:
        return DebugDetectionDecision(
            source_index=decision.source_index,
            action=decision.action,
            note=decision.note,
            decision_stage=decision.decision_stage,
            fallback_reason=decision.fallback_reason,
            raw_polygon=[
                [float(x), float(y)]
                for x, y in decision.raw_polygon.reshape(-1, 2).tolist()
            ],
            refined_boxes=[self._polygon_to_bbox(poly) for poly in decision.refined_polygons],
            tight_bbox=(
                self._bbox_tuple_to_model(decision.tight_bbox)
                if decision.tight_bbox
                else None
            ),
            projection_bbox=(
                self._bbox_tuple_to_model(decision.projection_bbox)
                if decision.projection_bbox
                else None
            ),
            clamped_bbox=(
                self._bbox_tuple_to_model(decision.clamped_bbox)
                if decision.clamped_bbox
                else None
            ),
            curve_score=float(decision.curve_score),
            neighbor_strategy=decision.neighbor_strategy,
            projection_threshold=decision.projection_threshold,
            notebook_mode=decision.notebook_mode,
            ruled_line_mode=decision.ruled_line_mode,
            original_text=decision.original_text,
            original_score=decision.original_score,
            split_texts=decision.split_texts,
            split_scores=decision.split_scores,
            mask_preview_base64=self._encode_array_preview(decision.mask_preview),
            rectified_preview_base64=self._encode_array_preview(decision.rectified_preview),
        )

    @staticmethod
    def _bbox_tuple_to_model(
        bbox: tuple[float, float, float, float],
    ) -> BoundingBox:
        return BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

    @staticmethod
    def _encode_array_preview(image_array) -> str | None:
        if image_array is None:
            return None
        buffer = io.BytesIO()
        Image.fromarray(image_array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _match_line_to_crop_index(line, crops: list[CropResult]) -> int | None:
        line_key = (
            round(line.bbox.x1, 2),
            round(line.bbox.y1, 2),
            round(line.bbox.x2, 2),
            round(line.bbox.y2, 2),
        )
        for index, crop in enumerate(crops):
            crop_bbox = OCRPipeline._polygon_to_bbox(crop.polygon)
            crop_key = (
                round(crop_bbox.x1, 2),
                round(crop_bbox.y1, 2),
                round(crop_bbox.x2, 2),
                round(crop_bbox.y2, 2),
            )
            if crop_key == line_key:
                return index
        return None
