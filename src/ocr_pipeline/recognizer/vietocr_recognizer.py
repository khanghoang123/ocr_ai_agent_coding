"""
VietOCR text recognition module.

Wraps the VietOCR Predictor with:
- Lazy model loading (load once, infer many)
- Batch inference support
- Graceful handling of edge-case crops (very long, very short lines)
- Performance timing

Extracted and refactored from notebooks/08_train_experiment_B.ipynb
and notebooks/04_evaluate_baseline.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    text: str
    probability: float
    decode_mode: str
    pre_norm_ratio: float
    post_norm_ratio: float
    baseline_confidence: float
    baseline_offset: float
    normalized_preview: np.ndarray | None = None


@dataclass
class RecognitionPreprocessResult:
    image: Image.Image
    preview_rgb: np.ndarray
    pre_norm_ratio: float
    post_norm_ratio: float
    baseline_confidence: float
    baseline_offset: float


class RecognitionPreprocessor:
    def __init__(
        self,
        target_height: int = 32,
        target_ratio: float = 4.0,
        min_ratio: float = 3.5,
        max_ratio: float = 5.0,
        baseline_target_ratio: float = 0.70,
        min_anchor_width: int = 64,
        enable_local_contrast: bool = True,
    ):
        self.target_height = target_height
        self.target_ratio = target_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.baseline_target_ratio = baseline_target_ratio
        self.min_anchor_width = min_anchor_width
        self.enable_local_contrast = enable_local_contrast

    def preprocess(
        self,
        image: Image.Image,
        max_width: int,
    ) -> RecognitionPreprocessResult:
        crop_rgb = np.array(image.convert("RGB"))
        if crop_rgb.size == 0:
            blank = np.full((self.target_height, self.target_height, 3), 255, dtype=np.uint8)
            return RecognitionPreprocessResult(
                image=Image.fromarray(blank),
                preview_rgb=blank,
                pre_norm_ratio=1.0,
                post_norm_ratio=1.0,
                baseline_confidence=0.0,
                baseline_offset=0.0,
            )

        pre_norm_ratio = float(crop_rgb.shape[1]) / float(max(crop_rgb.shape[0], 1))
        contrast_rgb = self._normalize_contrast(crop_rgb) if self.enable_local_contrast else crop_rgb
        resized = self._resize_height(contrast_rgb)
        baseline_confidence, baseline_position = self._estimate_baseline(resized)
        padded = self._pad_to_ratio(resized, max_width=max_width)
        baseline_offset = 0.0

        if (
            baseline_confidence > 0.7
            and resized.shape[1] >= self.min_anchor_width
            and baseline_position is not None
        ):
            target_baseline = self.baseline_target_ratio * float(padded.shape[0] - 1)
            baseline_offset = float(target_baseline - baseline_position)
            padded = self._shift_vertical(padded, baseline_offset)

        post_norm_ratio = float(padded.shape[1]) / float(max(padded.shape[0], 1))
        normalized = Image.fromarray(padded)
        return RecognitionPreprocessResult(
            image=normalized,
            preview_rgb=padded,
            pre_norm_ratio=pre_norm_ratio,
            post_norm_ratio=post_norm_ratio,
            baseline_confidence=float(baseline_confidence),
            baseline_offset=float(baseline_offset),
        )

    def _normalize_contrast(self, crop_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    def _resize_height(self, crop_rgb: np.ndarray) -> np.ndarray:
        height, width = crop_rgb.shape[:2]
        if height <= 0:
            return crop_rgb
        scale = self.target_height / float(height)
        target_width = max(1, int(round(width * scale)))
        return cv2.resize(
            crop_rgb,
            (target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA,
        )

    def _pad_to_ratio(self, crop_rgb: np.ndarray, max_width: int) -> np.ndarray:
        height, width = crop_rgb.shape[:2]
        ratio = width / float(max(height, 1))
        target_width = width
        if ratio < self.min_ratio:
            target_width = int(round(self.min_ratio * height))
        elif ratio < self.target_ratio:
            target_width = int(round(self.target_ratio * height))

        target_width = min(max(target_width, width), max_width)
        if target_width <= width:
            return crop_rgb

        pad_total = target_width - width
        left = pad_total // 2
        right = pad_total - left
        return cv2.copyMakeBorder(
            crop_rgb,
            0,
            0,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

    def _estimate_baseline(self, crop_rgb: np.ndarray) -> tuple[float, float | None]:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )

        bottoms = []
        for col_idx in range(binary.shape[1]):
            ys = np.where(binary[:, col_idx] > 0)[0]
            if ys.size < 2:
                continue
            bottoms.append(float(ys[-1]))

        if len(bottoms) < max(12, int(binary.shape[1] * 0.3)):
            return 0.0, None

        bottoms_arr = np.array(bottoms, dtype=np.float32)
        kernel_size = max(5, min(21, (len(bottoms_arr) // 8) * 2 + 1))
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smooth = np.convolve(bottoms_arr, kernel, mode="same")
        residual = float(np.mean(np.abs(bottoms_arr - smooth)))
        coverage = len(bottoms_arr) / float(max(binary.shape[1], 1))
        confidence = np.clip(coverage * np.exp(-residual / 3.5), 0.0, 1.0)
        return float(confidence), float(np.median(smooth))

    @staticmethod
    def _shift_vertical(crop_rgb: np.ndarray, offset: float) -> np.ndarray:
        if abs(offset) < 0.5:
            return crop_rgb
        transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, offset]], dtype=np.float32)
        return cv2.warpAffine(
            crop_rgb,
            transform,
            (crop_rgb.shape[1], crop_rgb.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )


class VietOCRRecognizer:
    """Singleton-friendly VietOCR recognizer.

    Usage:
        rec = VietOCRRecognizer.from_settings()        # or
        rec = VietOCRRecognizer(weights_path="...", device="cuda")
        text = rec.recognize(pil_image)
        texts = rec.recognize_batch([img1, img2, ...])
    """

    def __init__(
        self,
        weights_path: str | Path,
        architecture: str = "vgg_seq2seq",
        image_height: int = 32,
        image_max_width: int = 690,
        image_min_width: int = 32,
        device: str = "cpu",
        enable_local_contrast: bool = True,
    ):
        self.weights_path = str(weights_path)
        self.architecture = architecture
        self.image_height = image_height
        self.image_max_width = image_max_width
        self.image_min_width = image_min_width
        self.device = device
        self._preprocessor = RecognitionPreprocessor(
            target_height=image_height,
            enable_local_contrast=enable_local_contrast,
        )
        self._predictor = None   # Lazy init
        self._decode_mode = "greedy"

    # ── Loading ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return

        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}\n"
                "Download the weights and place them at the path above,\n"
                "or update models/model_registry.yaml."
            )

        # Apply Pillow ≥10 compatibility patch BEFORE vietocr imports
        self._patch_pillow()

        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        logger.info(
            "Loading VietOCR model '%s' from %s (device=%s) ...",
            self.architecture, self.weights_path, self.device,
        )

        cfg = Cfg.load_config_from_name(self.architecture)
        cfg["weights"] = self.weights_path
        cfg["device"] = self.device
        cfg["cnn"]["pretrained"] = False
        cfg["dataset"]["image_height"]    = self.image_height
        cfg["dataset"]["image_max_width"] = self.image_max_width
        cfg["dataset"]["image_min_width"] = self.image_min_width
        predictor_cfg = cfg.setdefault("predictor", {})
        beam_enabled = isinstance(predictor_cfg, dict)
        if beam_enabled:
            predictor_cfg["beamsearch"] = True
        try:
            self._predictor = Predictor(cfg)
            self._decode_mode = "beam" if beam_enabled else "greedy"
        except Exception:
            if not beam_enabled:
                raise
            predictor_cfg["beamsearch"] = False
            self._predictor = Predictor(cfg)
            self._decode_mode = "greedy"
        logger.info("VietOCR model loaded successfully.")

    @staticmethod
    def _patch_pillow() -> None:
        """Patch PIL._util for Pillow ≥10 compatibility.

        Pillow ≥10.0 removed `is_path` and `is_directory` from PIL._util.
        VietOCR/torchvision may call these at import time.
        This patch (from notebook 08) must run BEFORE any vietocr import.
        """
        try:
            from PIL._util import is_directory as _test  # noqa: F401
        except ImportError:
            import os
            import PIL._util as _pu
            if not hasattr(_pu, "is_path"):
                _pu.is_path = lambda f: isinstance(f, (str, os.PathLike))
            if not hasattr(_pu, "is_directory"):
                _pu.is_directory = lambda f: os.path.isdir(f)

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize(self, image: Image.Image) -> str:
        """Recognize text in a single PIL RGB Image crop.

        Edge cases:
          - Very long lines: VietOCR truncates at image_max_width; we feed as-is
            (the model was trained with max_width=690, longer crops are rescaled)
          - Empty/blank crops: returns empty string
          - Grayscale inputs: converted to RGB automatically
        """
        self._ensure_loaded()
        try:
            prepared = self._preprocessor.preprocess(image, max_width=self.image_max_width)
            text, _ = self._predict_with_optional_probability(prepared.image)
            return text
        except Exception as e:
            logger.warning("Recognition failed for image (%s): %s", image.size, e)
            return ""

    def recognize_batch(
        self,
        images: list[Image.Image],
        return_prob: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        """Recognize text for a batch of PIL Images.

        Args:
            images:      List of RGB PIL crops.
            return_prob: If True, return (text, confidence) tuples.

        Returns:
            List of strings, or list of (string, float) if return_prob=True.

        Note:
            VietOCR's Predictor does not expose native batch inference cleanly.
            We iterate with predict() here; the model still benefits from GPU
            through torch's internal batching when images are similar sizes.
        """
        detailed = self.recognize_batch_detailed(images, include_preview=False)
        if return_prob:
            return [(item.text, item.probability) for item in detailed]
        return [item.text for item in detailed]

    def recognize_batch_detailed(
        self,
        images: list[Image.Image],
        include_preview: bool = False,
    ) -> list[RecognitionResult]:
        self._ensure_loaded()
        results: list[RecognitionResult] = []
        for img in images:
            prepared = self._preprocessor.preprocess(img, max_width=self.image_max_width)
            preview = prepared.preview_rgb if include_preview else None
            try:
                text, prob = self._predict_with_optional_probability(prepared.image)
                results.append(
                    RecognitionResult(
                        text=text,
                        probability=prob,
                        decode_mode=self._decode_mode,
                        pre_norm_ratio=prepared.pre_norm_ratio,
                        post_norm_ratio=prepared.post_norm_ratio,
                        baseline_confidence=prepared.baseline_confidence,
                        baseline_offset=prepared.baseline_offset,
                        normalized_preview=preview,
                    )
                )
            except Exception as e:
                logger.warning("Batch recognition failed for one crop: %s", e)
                results.append(
                    RecognitionResult(
                        text="",
                        probability=0.0,
                        decode_mode=self._decode_mode,
                        pre_norm_ratio=prepared.pre_norm_ratio,
                        post_norm_ratio=prepared.post_norm_ratio,
                        baseline_confidence=prepared.baseline_confidence,
                        baseline_offset=prepared.baseline_offset,
                        normalized_preview=preview,
                    )
                )
        return results

    def _predict_with_optional_probability(
        self,
        image: Image.Image,
    ) -> tuple[str, float]:
        prediction = self._predictor.predict(image, return_prob=True)
        if isinstance(prediction, tuple):
            text = prediction[0] if len(prediction) > 0 else ""
            probability = prediction[1] if len(prediction) > 1 else None
        else:
            text = prediction
            probability = None

        if text is None:
            text = ""

        if probability is None:
            logger.debug(
                "Predictor returned no probability in %s mode; keeping text and using 0.0 confidence.",
                self._decode_mode,
            )
            return str(text), 0.0

        try:
            return str(text), float(probability)
        except (TypeError, ValueError):
            logger.debug(
                "Predictor returned non-numeric probability %r in %s mode; using 0.0 confidence.",
                probability,
                self._decode_mode,
            )
            return str(text), 0.0

    def recognize_timed(
        self, images: list[Image.Image]
    ) -> tuple[list[str], float]:
        """Recognize a list of images and return (texts, elapsed_ms)."""
        t0 = time.perf_counter()
        texts = self.recognize_batch(images)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return texts, elapsed_ms

    @classmethod
    def from_settings(cls, model_key: Optional[str] = None) -> "VietOCRRecognizer":
        """Create a recognizer from the global settings + model registry."""
        from ocr_pipeline.config import settings
        model_cfg = settings.get_model_config(model_key)
        return cls(
            weights_path=model_cfg["weights_path"],
            architecture=model_cfg["architecture"],
            image_height=model_cfg["image_height"],
            image_max_width=model_cfg["image_max_width"],
            image_min_width=model_cfg["image_min_width"],
            device=settings.rec_device,
        )
