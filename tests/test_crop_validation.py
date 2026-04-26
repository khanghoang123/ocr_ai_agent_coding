from __future__ import annotations

import cv2
import numpy as np

from ocr_pipeline.cropper.line_cropper import LineCropper


def test_validated_crop_trims_horizontal_margins_and_adds_padding():
    crop = np.full((28, 420, 3), 255, dtype=np.uint8)
    cv2.putText(crop, "abc", (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cropper = LineCropper(
        crop_strategy="validated",
        vertical_padding_ratio=0.35,
        horizontal_padding_ratio=0.6,
    )

    final, flags, before = cropper.finalize_crop_with_metadata(crop)

    assert before.shape[:2] == crop.shape[:2]
    assert final.shape[1] < crop.shape[1]
    assert "validated_geometry_crop" in flags


def test_validated_crop_flags_too_thin_crop():
    crop = np.full((8, 240, 3), 255, dtype=np.uint8)
    cv2.line(crop, (40, 4), (190, 4), (0, 0, 0), 1)
    cropper = LineCropper(crop_strategy="validated", min_valid_height=18)

    final, flags, _ = cropper.finalize_crop_with_metadata(crop)

    assert final.shape[0] >= crop.shape[0]
    assert any(flag in flags for flag in ("too_thin_crop", "too_thin_after_validation"))


def test_line_deskew_flags_skewed_crop():
    crop = np.full((60, 260, 3), 255, dtype=np.uint8)
    cv2.line(crop, (30, 42), (225, 20), (0, 0, 0), 3)
    cropper = LineCropper(
        crop_strategy="validated",
        enable_line_deskew=True,
        max_deskew_angle=8.0,
    )

    _, flags, _ = cropper.finalize_crop_with_metadata(crop)

    assert "line_deskewed" in flags
    assert any(flag.startswith("deskew_angle=") for flag in flags)
