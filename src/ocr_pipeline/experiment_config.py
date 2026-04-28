from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    description: str = "Baseline OCR pipeline configuration"
    model_key: str | None = None
    crop_padding: int | None = None
    crop_padding_ratio: float | None = None
    rectify_line: bool = True
    enable_deskew: bool = False
    enable_document_perspective_correction: bool = False
    enable_local_contrast: bool = True
    detect_on_upscaled_image: bool = True
    upscale_factor: float = 1.0
    enable_vietnamese_postprocess: bool = False
    flag_digit_noise: bool = False
    debug: bool = True
    enable_noise_box_filter: bool = False
    tiny_box_height_ratio: float = 0.45
    tiny_box_width_ratio: float = 0.08
    tiny_box_min_ink_occupancy: float = 0.01
    enable_rotated_crop: bool = True
    debug_save_intermediate_crops: bool = False
    crop_strategy: str = "basic"
    enable_line_deskew: bool = False
    min_crop_height_ratio: float = 0.75
    vertical_padding_ratio: float = 0.35
    horizontal_padding_ratio: float = 0.60
    max_deskew_angle: float = 8.0
    # Page-level document rectification (Phase 2C).
    # When `enable_document_perspective_correction=True`, the pipeline runs
    # a dewarping pass *before* line detection. `rectifier_backend` selects
    # the algorithm; `rectifier_weights_path` points to a DocTr++ checkpoint
    # if available (TorchScript / ONNX / state_dict). Missing weights
    # silently degrade to the OpenCV fallback — they never crash the run.
    rectifier_backend: str = "hybrid"
    rectifier_weights_path: str | None = None
    rectifier_device: str = "cpu"
    rectifier_min_quad_area_ratio: float = 0.25
    rectifier_max_quad_area_ratio: float = 0.95
    rectifier_min_confidence: float = 0.5
    # Paper-vs-background contrast gate (mean luminance difference, 0–255).
    # Prevents the rectifier from latching onto the *printed inner rectangle*
    # of an already-flat scan, which would falsely "rectify" the page by
    # cropping off its margins.
    rectifier_min_paper_background_contrast: float = 18.0
    rectifier_save_debug: bool = True
    # Phase 3 default: Kraken BLLA — winner of the Tier-1 leaderboard on
    # tests/test/ (best full_width_band_rate, best recall on cursive
    # handwriting, baseline-aware polygons that match the cropper's
    # input shape). ``detector_backend`` picks one of
    # {paddle, surya, craft, kraken}. The silent OpenCV grid fallback in
    # PaddleDetector is OFF by default; pass
    # ``detector_allow_grid_fallback=True`` only for the legacy comparison
    # control. Backend-specific knobs go in ``detector_kwargs``.
    detector_backend: str = "kraken"
    detector_allow_grid_fallback: bool = False
    detector_kwargs: dict[str, Any] = field(default_factory=dict)
    unsupported_options: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")

    allowed = set(ExperimentConfig.__dataclass_fields__) - {"raw", "unsupported_options"}
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    config = ExperimentConfig(**kwargs)
    config.raw = dict(payload)

    unsupported: list[str] = []
    if config.enable_deskew:
        unsupported.append("enable_deskew")
    # `enable_document_perspective_correction` is now implemented by the
    # `ocr_pipeline.rectifier` module (Phase 2C). It is no longer marked as
    # unsupported. If the user requests `rectifier_backend="doctrpp"` but the
    # weights file is missing, the HybridRectifier silently falls back to
    # OpenCV at runtime; we don't fail config validation for that.
    config.unsupported_options = unsupported
    return config


def dump_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)
