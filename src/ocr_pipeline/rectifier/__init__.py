"""Page-level document rectification (dewarping) module.

This module is intentionally separate from `preprocess.notebook_preprocessor`,
which only handles *line-level* curve rectification. The need it fills is
*page-level* rectification: input is a raw photograph of a document (curved,
perspective-distorted), output is a flattened image suitable for line
detection.

Backends
--------
- ``opencv``  : classical 4-corner perspective correction. No extra weights.
- ``doctrpp`` : pretrained DocTr++ (Deep Unrestricted Document Image
                Rectification, TMM 2023). Falls back to ``opencv`` when
                weights are not available.
- ``hybrid``  : tries DocTr++ first, falls back to OpenCV when DocTr++ is
                not loadable or its output looks degenerate.
- ``identity``: no-op (used to disable rectification while keeping the same
                pipeline plumbing).

The chosen backend is wired through `ExperimentConfig` via the existing
``enable_document_perspective_correction`` flag.
"""
from ocr_pipeline.rectifier.base import (
    Rectifier,
    RectifierResult,
    IdentityRectifier,
    build_rectifier,
)
from ocr_pipeline.rectifier.opencv_rectifier import OpenCVRectifier
from ocr_pipeline.rectifier.doctrpp_rectifier import DocTrPlusRectifier

__all__ = [
    "Rectifier",
    "RectifierResult",
    "IdentityRectifier",
    "OpenCVRectifier",
    "DocTrPlusRectifier",
    "build_rectifier",
]
