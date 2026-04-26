from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ocr_pipeline.recognizer.vietocr_recognizer import RecognitionPreprocessor, VietOCRRecognizer


def _make_line_image(width: int = 120, height: int = 24) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([12, 6, width - 14, 17], fill="black")
    return image


def test_recognition_preprocessor_normalizes_height_and_ratio():
    preprocessor = RecognitionPreprocessor(target_height=32)

    result = preprocessor.preprocess(_make_line_image(width=48, height=18), max_width=320)

    assert result.image.height == 32
    assert result.post_norm_ratio >= 3.5
    assert result.post_norm_ratio <= 5.0


def test_recognition_preprocessor_baseline_anchor_requires_confident_wide_crop():
    preprocessor = RecognitionPreprocessor(target_height=32, min_anchor_width=64)
    narrow = _make_line_image(width=40, height=18)

    result = preprocessor.preprocess(narrow, max_width=320)

    assert result.baseline_offset == 0.0


def test_recognition_preprocessor_preserves_preview_array():
    preprocessor = RecognitionPreprocessor(target_height=32)
    result = preprocessor.preprocess(_make_line_image(width=120, height=24), max_width=320)

    preview = np.array(result.image.convert("RGB"))
    assert result.preview_rgb.shape == preview.shape
    assert result.preview_rgb.dtype == np.uint8


class _PredictorNoneProbStub:
    def predict(self, image, return_prob=False):
        if return_prob:
            return ("stub text", None)
        return "stub text"


def test_vietocr_recognizer_tolerates_missing_probability():
    recognizer = VietOCRRecognizer(weights_path="dummy.pth")
    recognizer._predictor = _PredictorNoneProbStub()
    recognizer._decode_mode = "beam"

    result = recognizer.recognize_batch_detailed([_make_line_image()], include_preview=False)[0]

    assert result.text == "stub text"
    assert result.probability == 0.0
    assert result.decode_mode == "beam"
